"""One local GPU controller shared by notebook submissions; no automatic failure retries."""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from filelock import Timeout

from .durable import (
    OwnedLock,
    atomic_json,
    journal,
    locked,
    read_json,
    run_folder,
    sha256,
    verified_load,
)

CONTROL = Path("artifacts/comparison/controller")


def gpu_memory():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    total, used, free = [float(x.strip()) for x in result.stdout.splitlines()[0].split(",")]
    return {"total_mib": total, "used_mib": used, "free_mib": free}


def detached(command, log, env=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        (
            subprocess.CREATE_NO_WINDOW
            | subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
        )
        if os.name == "nt"
        else 0
    )
    with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
            env=env,
            creationflags=flags,
            start_new_session=os.name != "nt",
        )
    return process


def launch(frozen_path, resume=False):
    frozen_path = Path(frozen_path).resolve()
    protocol = read_json(frozen_path)
    if sha256(frozen_path) != read_json(frozen_path.with_name("frozen.sha256.json"))["sha256"]:
        raise ValueError("Frozen configuration changed")
    CONTROL.mkdir(parents=True, exist_ok=True)
    queue = CONTROL / "requests"
    queue.mkdir(exist_ok=True)
    request = queue / f"{protocol['run_id']}.json"
    with OwnedLock(CONTROL / "submission.lock"):
        if request.exists():
            previous = read_json(request)
            if not resume:
                return {
                    "request": str(request),
                    "state": "already_submitted",
                    "detail": previous["models"],
                }
            if any(
                v["state"] in ("running", "starting", "queued") for v in previous["models"].values()
            ):
                raise ValueError("Run has active/queued workers; do not submit duplicates")
            journal(CONTROL / "request-history.jsonl", previous)
            models = {
                k: ({"state": "queued", "resume": True} if v["state"] != "complete" else v)
                for k, v in previous["models"].items()
            }
            for name, state in models.items():
                if (
                    state["state"] == "queued"
                    and (run_folder(protocol["config"], name) / "PAUSE").exists()
                ):
                    (run_folder(protocol["config"], name) / "PAUSE").unlink()
        else:
            if resume:
                models = {
                    k: {"state": "queued", "resume": True} for k in protocol["config"]["models"]
                }
            else:
                models = {
                    k: {"state": "queued", "resume": False} for k in protocol["config"]["models"]
                }
        atomic_json(
            request,
            {
                "frozen": str(frozen_path),
                "frozen_sha256": sha256(frozen_path),
                "submitted_unix": time.time(),
                "models": models,
            },
        )
    if not locked(CONTROL / "gpu.lock"):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(frozen_path.parent / "source")
        env["PYTHONUNBUFFERED"] = "1"
        child = detached(
            [sys.executable, "-m", "latent_text.comparison", "controller"],
            CONTROL / f"controller-{uuid.uuid4().hex[:8]}.log",
            env,
        )
        controller_pid = child.pid
    else:
        controller_pid = read_json(CONTROL / "gpu.owner.json")["pid"]
    return {"request": str(request), "controller_pid": controller_pid, "state": "submitted"}


def required_mib(protocol, name):
    c = protocol["config"]
    r = protocol["resources"][name]
    return (
        max(r["peak_reserved_mib"], r["device_free_change_mib"]) * 1.2 + c["process_overhead_mib"]
    )


def status(output):
    output = Path(output)
    protocol = read_json(output / "frozen.json") if (output / "frozen.json").exists() else None
    names = protocol["config"]["models"] if protocol else ()
    result = {"models": {}, "gpu": gpu_memory()}
    for name in names:
        folder = run_folder(protocol["config"], name)
        value = (
            read_json(folder / "status.json")
            if (folder / "status.json").exists()
            else {"state": "not_started"}
        )
        value["worker_lock_held"] = locked(folder / "worker.lock") if folder.exists() else False
        result["models"][name] = value
    return result


def comparison_summary(protocol):
    root = Path(protocol["config"]["output_dir"])
    reports, fingerprints = {}, {}
    for name in protocol["config"]["models"]:
        folder = run_folder(protocol["config"], name)
        if not (folder / "final-verification.json").exists():
            return
        pointer = read_json(folder / "last.json")
        state, meta = verified_load(folder / "checkpoints" / pointer["metadata"])
        fingerprints[name] = [row["batch_sha256"] for row in state["history"]]
        reports[name] = {
            "parameters": protocol["resources"][name]["parameters"],
            "updates": state["step"],
            "training_tokens": state["training_tokens"],
            "training_update_seconds": sum(row["seconds"] for row in state["history"]),
            "elapsed_seconds_including_validation_and_checkpointing": state["elapsed_seconds"],
            "memory": state["memory"],
            "last_sha256": meta["sha256"],
            "evaluation": read_json(folder / "final-verification.json"),
        }
        del state
    matched = (
        fingerprints.get("A") == fingerprints.get("B") if {"A", "B"} <= set(fingerprints) else None
    )
    if matched is False:
        raise ValueError("A/B batch fingerprints differ; comparison cannot be reported as matched")
    atomic_json(
        root / "comparison-results.json",
        {
            "models": reports,
            "AB_batches_identical": matched,
            "caution": "A/B encoder-depth screening at one seed. C is a different context/ratio/depth scaling task. Historical warm-start scores are not baselines.",
        },
    )


def controller():
    CONTROL.mkdir(parents=True, exist_ok=True)
    try:
        with OwnedLock(CONTROL / "gpu.lock"):
            _controller()
    except Timeout:
        print(
            "An existing controller owns the GPU queue; exiting duplicate controller.", flush=True
        )


def _controller():
    children = {}
    baseline_free = gpu_memory()["free_mib"]
    idle_since = None
    while True:
        # Submissions are atomic. Only this controller changes worker lifecycle states.
        requests = [
            (path, read_json(path)) for path in sorted((CONTROL / "requests").glob("*.json"))
        ]
        active_required, queued = 0.0, []
        for path, request in requests:
            if sha256(request["frozen"]) != request["frozen_sha256"]:
                raise ValueError("Submitted frozen configuration changed")
            protocol = read_json(request["frozen"])
            c = protocol["config"]
            for name, entry in request["models"].items():
                key = (str(path), name)
                folder = run_folder(c, name)
                if entry["state"] in ("starting", "running"):
                    child = children.get(key)
                    alive = (
                        (child.poll() is None)
                        if child is not None
                        else (folder.exists() and locked(folder / "worker.lock"))
                    )
                    if alive:
                        active_required += required_mib(protocol, name)
                        entry["state"] = "running"
                    else:
                        observed = (
                            read_json(folder / "status.json")
                            if (folder / "status.json").exists()
                            else {}
                        )
                        if (
                            observed.get("state") == "complete"
                            and (folder / "final-verification.json").exists()
                        ):
                            entry["state"] = "complete"
                        elif observed.get("state") == "paused":
                            entry["state"] = "paused"
                        else:
                            entry.update(
                                {
                                    "state": "failed",
                                    "exit_code": child.returncode if child else None,
                                    "reason": "Worker exited/interrupted; inspect logs and evidence before explicit resume",
                                    "automatic_retry": False,
                                }
                            )
                            atomic_json(
                                folder / f"controller-observed-failure-{uuid.uuid4().hex[:8]}.json",
                                entry,
                            )
                        children.pop(key, None)
                        atomic_json(path, request)
                if entry["state"] == "queued":
                    queued.append((path, request, protocol, name))
            if all(e["state"] == "complete" for e in request["models"].values()):
                result = Path(c["output_dir"]) / "comparison-results.json"
                if not result.exists():
                    comparison_summary(protocol)
        for path, request, protocol, name in queued:
            need = required_mib(protocol, name)
            headroom = protocol["config"]["gpu_headroom_mib"]
            physical = gpu_memory()
            available = min(baseline_free - active_required, physical["free_mib"])
            if need + headroom > available:
                atomic_json(
                    Path(protocol["config"]["output_dir"]) / f"queued-{name}.json",
                    {
                        "reason": "Measured GPU memory admission limit",
                        "required_mib": need,
                        "headroom_mib": headroom,
                        "available_mib": available,
                        "gpu": physical,
                    },
                )
                continue
            folder = run_folder(protocol["config"], name)
            folder.mkdir(parents=True, exist_ok=True)
            if locked(folder / "worker.lock"):
                request["models"][name].update(
                    {"state": "running", "adopted_existing_worker": True}
                )
                atomic_json(path, request)
                active_required += need
                continue
            entry = request["models"][name]
            command = [
                sys.executable,
                "-m",
                "latent_text.comparison",
                "worker",
                "--frozen",
                request["frozen"],
                "--model",
                name,
            ]
            if entry["resume"]:
                command.append("--resume")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(request["frozen"]).parent / "source")
            env["PYTHONUNBUFFERED"] = "1"
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            entry.update({"state": "starting", "started_unix": time.time(), "required_mib": need})
            atomic_json(path, request)
            child = detached(command, folder / f"worker-{uuid.uuid4().hex[:8]}.log", env)
            children[(str(path), name)] = child
            entry.update({"state": "running", "pid": child.pid})
            atomic_json(path, request)
            active_required += need
            print(f"Started {name} pid={child.pid}; reservation={need:.1f} MiB", flush=True)
        physical = gpu_memory()
        journal(
            CONTROL / "gpu-history.jsonl",
            {"time": time.time(), **physical, "reserved_for_workers_mib": active_required},
        )
        atomic_json(
            CONTROL / "status.json",
            {
                "pid": os.getpid(),
                "gpu": physical,
                "worker_reservations_mib": active_required,
                "requests": [r for _, r in requests],
                "time": time.time(),
            },
        )
        if not active_required and not queued:
            if idle_since is None:
                idle_since = time.monotonic()
            # Brief idle grace makes near-simultaneous notebook submissions reliable.
            if time.monotonic() - idle_since > 20:
                return
        else:
            idle_since = None
        time.sleep(5)
