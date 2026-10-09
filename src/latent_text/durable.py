"""Verified generations and OS-owned locks for long, restartable experiments."""

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import torch
from filelock import FileLock, Timeout


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    if os.name != "nt":
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_folder(config, name):
    root = Path(config["output_dir"])
    return root if config.get("flat_output") else root / name


def journal(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as f:
        f.write((json.dumps(value, allow_nan=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())


def locked(path):
    lock = FileLock(str(path))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    lock.release()
    return False


class OwnedLock:
    def __init__(self, path, timeout=0):
        self.path = Path(path)
        self.lock = FileLock(str(path))
        self.timeout = timeout

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock.acquire(timeout=self.timeout)
        atomic_json(
            self.path.with_suffix(".owner.json"),
            {"pid": os.getpid(), "started_unix": time.time(), "lock": str(self.path.resolve())},
        )
        return self

    def __exit__(self, *args):
        self.lock.release()


def verified_load(metadata_path):
    metadata_path = Path(metadata_path)
    meta = read_json(metadata_path)
    path = metadata_path.parent / meta["file"]
    if path.parent.resolve() != metadata_path.parent.resolve():
        raise ValueError("Invalid checkpoint filename")
    if sha256(path) != meta["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["step"] != meta["step"] or state["run_id"] != meta["run_id"]:
        raise ValueError("Checkpoint metadata/state mismatch")
    return state, meta


def checkpoint_candidates(folder):
    folder = Path(folder)
    good, invalid = [], []
    for meta in (folder / "checkpoints").glob("step-*.json"):
        try:
            header = read_json(meta)
            good.append((header["step"], header["created_unix"], meta))
        except Exception as exc:
            invalid.append({"path": str(meta), "error": repr(exc)})
    return sorted(good, reverse=True), invalid


def recover(folder):
    """Only called for explicit resume. Never deletes failed checkpoint evidence."""
    candidates, invalid = checkpoint_candidates(folder)
    for _, _, path in candidates:
        try:
            state, meta = verified_load(path)
            return state, meta, invalid
        except Exception as exc:
            invalid.append({"path": str(path), "error": repr(exc)})
    raise ValueError(f"No verified resumable checkpoint; failures={invalid}")


def resolve_checkpoint(folder, which="last"):
    folder = Path(folder)
    pointer = read_json(folder / f"{which}.json")
    meta = folder / "checkpoints" / pointer["metadata"]
    _, header = verified_load(meta)
    return meta.parent / header["file"]


def publish_checkpoint(folder, state, milestone_every=5000):
    folder = Path(folder)
    directory = folder / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step-{state['step']:06d}-{uuid.uuid4().hex[:12]}"
    target = directory / f"{name}.pt"
    temporary = directory / f"{name}.pt.tmp"
    with temporary.open("xb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    check = torch.load(temporary, map_location="cpu", weights_only=True)
    if check["step"] != state["step"] or check["run_id"] != state["run_id"]:
        raise ValueError("Checkpoint failed readability verification")
    # Detect serialization damage to every model tensor, including calibration buffers.
    for key, value in state["model"].items():
        if not torch.equal(value.detach().cpu(), check["model"][key]):
            raise ValueError(f"Checkpoint tensor verification failed: {key}")
    del check
    checksum = sha256(temporary)
    os.replace(temporary, target)
    meta = {
        "file": target.name,
        "sha256": checksum,
        "step": state["step"],
        "run_id": state["run_id"],
        "created_unix": time.time(),
        "milestone": state["step"] > 0 and state["step"] % milestone_every == 0,
        "best_step": state["best"]["step"] if state.get("best") else None,
    }
    metadata = directory / f"{name}.json"
    atomic_json(metadata, meta)
    pointer = {"metadata": metadata.name, **meta}
    if (folder / "last.json").exists():
        old = read_json(folder / "last.json")
        # Do not promote a damaged last checkpoint into the previous-valid slot.
        try:
            verified_load(directory / old["metadata"])
        except Exception as exc:
            journal(folder / "integrity-events.jsonl", {"time": time.time(), "error": repr(exc)})
            raise
        atomic_json(folder / "previous.json", old)
    atomic_json(folder / "last.json", pointer)
    if meta["best_step"] == meta["step"]:
        atomic_json(folder / "best.json", pointer)
    journal(folder / "checkpoint-events.jsonl", pointer)
    # Keep two recent generations, selected best, every immutable milestone, and invalid evidence.
    keep = {name + ".json"}
    for ref in ("last", "previous", "best"):
        if (folder / f"{ref}.json").exists():
            keep.add(read_json(folder / f"{ref}.json")["metadata"])
    for path in directory.glob("step-*.json"):
        try:
            header = read_json(path)
            if path.name in keep or header["milestone"] or header["step"] == meta["best_step"]:
                continue
            payload = directory / header["file"]
            if payload.exists() and sha256(payload) == header["sha256"]:
                payload.unlink()
                path.unlink()
        except (OSError, ValueError, KeyError):
            pass
    return pointer
