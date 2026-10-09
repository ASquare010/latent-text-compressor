"""Fresh comparison worker, resource probes, frozen configuration and final verification."""

import argparse
import gc
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .codec import Codec
from .comparison_data import (
    RowStore,
    SharedSampler,
    length_group,
    load_tokenizer,
    prepare_comparison,
    verify_data,
)
from .data import batch
from .durable import (
    OwnedLock,
    atomic_json,
    identity,
    journal,
    publish_checkpoint,
    read_json,
    recover,
    sha256,
    verified_load,
)
from .model import Config, Model


def setup(config):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This frozen comparison requires CUDA with BF16 support")
    torch.cuda.manual_seed_all(config["seed"])


def environment():
    props = torch.cuda.get_device_properties(0)
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": props.name,
        "vram_mib": props.total_memory / 2**20,
        "platform": platform.platform(),
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32": False,
        "precision": "bf16",
        "device_capability": list(torch.cuda.get_device_capability()),
    }


def validate_config(c):
    if c["effective_batch"] != 32:
        raise ValueError("Comparison effective batch is fixed at 32")
    for name in (
        "updates",
        "warmup_updates",
        "checkpoint_every",
        "milestone_every",
        "evaluate_every",
        "threads",
    ):
        if type(c[name]) is not int or c[name] < 1:
            raise ValueError(f"Invalid {name}")
    if c["updates"] <= c["warmup_updates"] or c["precision"] != "bf16":
        raise ValueError("Require a cosine phase after warmup and BF16 precision")
    if not 0 < c["minimum_lr"] <= c["learning_rate"] or c["clip"] <= 0:
        raise ValueError("Invalid optimizer settings")
    if set(c["models"]) - {"A", "B", "C"} or not c["models"]:
        raise ValueError("Select A, B and/or C")
    for value in c["models"].values():
        cfg = Config(**(c["model"] | value))
        if cfg.positional != "rope" or cfg.ffn != "branch_sigmoid_v1":
            raise ValueError("Comparison requires RoPE and Branch Sigmoid")
    output = Path(c["output_dir"]).resolve()
    if not output.is_relative_to((Path.cwd() / "artifacts").resolve()):
        raise ValueError("Output must be inside repository artifacts")


def lr_at(step, c):
    if not 1 <= step <= c["updates"]:
        raise ValueError("Update out of schedule range")
    if step <= c["warmup_updates"]:
        return c["learning_rate"] * step / c["warmup_updates"]
    progress = (step - c["warmup_updates"]) / (c["updates"] - c["warmup_updates"])
    return c["minimum_lr"] + 0.5 * (c["learning_rate"] - c["minimum_lr"]) * (
        1 + math.cos(math.pi * progress)
    )


def optimizer_for(model, c):
    return torch.optim.AdamW(
        model.parameters(),
        lr=c["learning_rate"],
        betas=tuple(c["betas"]),
        eps=c["adam_eps"],
        weight_decay=c["weight_decay"],
        fused=True,
    )


def accumulated_update(model, optimizer, rows, microbatch, c, step, device="cuda"):
    total = sum(len(r["ids"]) for r in rows)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    lr = lr_at(step, c)
    for group in optimizer.param_groups:
        group["lr"] = lr
    nll = 0.0
    for start in range(0, len(rows), microbatch):
        tokens, mask = batch(rows[start : start + microbatch], device)
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(tokens, mask)
            loss = (
                F.cross_entropy(
                    logits.float().flatten(0, 1),
                    tokens.masked_fill(~mask, -100).flatten(),
                    reduction="sum",
                )
                / total
            )
        loss.backward()
        nll += float(loss.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["clip"], error_if_nonfinite=True)
    optimizer.step()
    return {"nll": nll, "tokens": total, "lr": lr, "gradient_norm": float(norm)}


def memory():
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def probe(config_path, name, microbatch, output):
    c = read_json(config_path)
    validate_config(c)
    setup(c)
    if c["effective_batch"] % microbatch:
        raise ValueError("Microbatch must divide effective batch")
    cfg = Config(**(c["model"] | c["models"][name]))
    report = {
        "model": name,
        "microbatch": microbatch,
        "accumulation": 32 // microbatch,
        "environment": environment(),
        "discarded_probe_updates": 0,
    }
    try:
        before = torch.cuda.mem_get_info()[0]
        model = Model(cfg, c["seed"]).cuda()
        optimizer = optimizer_for(model, c)
        rng = random.Random(1799)
        rows = [
            {"ids": [rng.randrange(3, cfg.vocab_size) for _ in range(cfg.max_tokens)]}
            for _ in range(32)
        ]
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        stats = accumulated_update(model, optimizer, rows, microbatch, c, 1)
        report["discarded_probe_updates"] = 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tokens, mask = batch(rows[:microbatch], "cuda")
            z, lengths = model.encode(tokens, mask)
            logits = model.decode(z, lengths)
            assert z.shape == (microbatch, cfg.max_tokens // cfg.span, cfg.width)
            assert logits.shape == (microbatch, cfg.max_tokens, cfg.vocab_size)
            assert torch.isfinite(logits).all()
        torch.cuda.synchronize()
        report.update(
            {
                "status": "passed",
                "parameters": sum(p.numel() for p in model.parameters()),
                "latent_shape": list(z.shape),
                "update_seconds": elapsed,
                "tokens_per_vector": cfg.span,
                "stats": stats,
                **memory(),
                "device_free_change_mib": (before - torch.cuda.mem_get_info()[0]) / 2**20,
            }
        )
    except torch.cuda.OutOfMemoryError:
        report.update({"status": "oom", "error": traceback.format_exc(), **memory()})
    except BaseException:
        report.update({"status": "failed", "error": traceback.format_exc()})
        atomic_json(output, report)
        raise
    atomic_json(output, report)
    print(json.dumps(report), flush=True)
    return report


def bounded_checks(config_path):
    c = read_json(config_path)
    validate_config(c)
    root = Path(c["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if (root / "frozen.json").exists():
        raise ValueError("Already frozen; probes may not be repeated")
    results = {}
    with OwnedLock(Path("artifacts/comparison/resource-probe.lock")):
        for name in c["models"]:
            for micro in c["microbatch_candidates"]:
                path = root / "probes" / f"{name}-micro{micro}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    report = read_json(path)
                else:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "latent_text.comparison",
                            "probe",
                            "--config",
                            str(config_path),
                            "--model",
                            name,
                            "--microbatch",
                            str(micro),
                            "--output",
                            str(path),
                        ],
                        check=True,
                    )
                    report = read_json(path)
                if report["status"] == "passed":
                    results[name] = report
                    break
                if report["status"] != "oom":
                    raise RuntimeError("Probe failure requires inspection; no automatic retry")
            else:
                raise RuntimeError(f"{name} does not fit microbatch 1; configuration preserved")
    atomic_json(root / "resources.json", results)
    return results


def freeze(config_path):
    c = read_json(config_path)
    validate_config(c)
    root = Path(c["output_dir"])
    with OwnedLock(root / "freeze.lock"):
        path = root / "frozen.json"
        if path.exists():
            existing = read_json(path)
            if existing["config"] != c:
                raise ValueError("Configuration is frozen; choose a different run folder")
            return path
        manifest = verify_data(c["data_dir"])
        resources = read_json(root / "resources.json")
        setup(c)
        source = root / "source" / "latent_text"
        source.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for item in Path(__file__).parent.glob("*.py"):
            shutil.copy2(item, source / item.name)
            hashes[item.name] = sha256(source / item.name)
        protocol = {
            "config": c,
            "dataset_manifest": manifest,
            "source_hashes": hashes,
            "resources": resources,
            "environment": environment(),
            "created_unix": time.time(),
        }
        protocol["run_id"] = identity(protocol)
        shutil.copy2("docs/fresh_comparison.md", root / "protocol.md")
        atomic_json(path, protocol)
        atomic_json(root / "frozen.sha256.json", {"sha256": sha256(path)})
        return path


def load_frozen(path):
    path = Path(path)
    if sha256(path) != read_json(path.with_name("frozen.sha256.json"))["sha256"]:
        raise ValueError("Frozen configuration checksum failed")
    protocol = read_json(path)
    for name, expected in protocol["source_hashes"].items():
        if sha256(Path(__file__).with_name(name)) != expected:
            raise ValueError("Worker must run from its frozen source snapshot")
    if verify_data(protocol["config"]["data_dir"]) != protocol["dataset_manifest"]:
        raise ValueError("Frozen dataset changed")
    return protocol


@torch.no_grad()
def evaluate_store(model, store, micro, boundaries):
    model.eval()
    groups = defaultdict(
        lambda: {
            "sequences": 0,
            "exact": 0,
            "correct_tokens": 0,
            "tokens": 0,
            "nll_sum": 0.0,
            "vectors": 0,
        }
    )
    prediction_hash = __import__("hashlib").sha256()
    for start in range(0, len(store), micro):
        rows = [store[i] for i in range(start, min(start + micro, len(store)))]
        tokens, mask = batch(rows, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z, lengths = model.encode(tokens, mask)
            logits = model.decode(z, lengths)
        loss = F.cross_entropy(
            logits.float().transpose(1, 2), tokens.masked_fill(~mask, -100), reduction="none"
        )
        logits[..., :3] = -torch.inf
        predictions = logits.argmax(-1)
        for i, row in enumerate(rows):
            n = len(row["ids"])
            pred = predictions[i, :n]
            correct = int((pred == tokens[i, :n]).sum())
            prediction_hash.update(pred.cpu().numpy().tobytes())
            nll = float(loss[i, :n].sum())
            for group in (
                "all",
                "source:" + row["source"],
                f"length:<={length_group(n, boundaries)}",
            ):
                g = groups[group]
                g["sequences"] += 1
                g["exact"] += int(correct == n)
                g["correct_tokens"] += correct
                g["tokens"] += n
                g["nll_sum"] += nll
                g["vectors"] += math.ceil(n / model.config.span)
    for g in groups.values():
        g.update(
            {
                "exact_recovery": g["exact"] / g["sequences"],
                "sequence_errors": g["sequences"] - g["exact"],
                "token_errors": g["tokens"] - g["correct_tokens"],
                "token_accuracy": g["correct_tokens"] / g["tokens"],
                "nll": g["nll_sum"] / g["tokens"],
                "actual_tokens_per_vector": g["tokens"] / g["vectors"],
            }
        )
    return {"groups": dict(groups), "prediction_sha256": prediction_hash.hexdigest()}


def evaluate_pair(model, c, micro, split="valid"):
    folder = c["data_dir"]
    return {
        "shared_short": evaluate_store(
            model, RowStore(folder, f"{split}-short"), micro, c["sampling"]["length_boundaries"]
        ),
        "packed_task": evaluate_store(
            model,
            RowStore(folder, f"{split}-{model.config.max_tokens}"),
            micro,
            c["sampling"]["length_boundaries"],
        ),
    }


def rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def final_verification(folder, c, protocol, name, micro):
    checks = {}
    for which in ("last", "best"):
        pointer = read_json(folder / f"{which}.json")
        state, meta = verified_load(folder / "checkpoints" / pointer["metadata"])
        if state["run_id"] != protocol["run_id"]:
            raise ValueError("Wrong run identity")
        model = Model(Config(**state["protocol"]["model"]), c["seed"]).cuda()
        model.load_state_dict(state["model"])
        actual = evaluate_pair(model, c, micro)
        recorded = state["validations"][-1]["metrics"]
        if actual != recorded:
            raise ValueError(f"Full validation replay mismatch: {which}")
        if which == "last":
            if state["step"] != c["updates"] or [r["step"] for r in state["history"]] != list(
                range(1, c["updates"] + 1)
            ):
                raise ValueError("Incomplete optimizer-update history")
            if sum(r["tokens"] for r in state["history"]) != state["training_tokens"]:
                raise ValueError("Training token accounting mismatch")
            # Replay every sampler batch without performing optimizer updates.
            store = RowStore(c["data_dir"], f"train-{model.config.max_tokens}")
            sampler = SharedSampler(store.meta, c["sampling"], c["sampling_seed"])
            for row in state["history"]:
                indices = sampler.take(c["effective_batch"])
                if identity([store.meta[i]["id"] for i in indices]) != row["batch_sha256"]:
                    raise ValueError("Sampler/update history mismatch")
        tokenizer = load_tokenizer(c["data_dir"])
        codec = Codec(model, tokenizer, "cuda", "bf16")
        export_path = codec.export_encoder(folder / f"encoder-{which}.pt")
        with export_path.open("rb+") as f:
            os.fsync(f.fileno())
        exported = Codec.load_encoder(export_path, "cuda", "bf16")
        # Full validation parity covers short and full context inputs, including padding.
        for suite in ("short", str(model.config.max_tokens)):
            store = RowStore(c["data_dir"], f"valid-{suite}")
            for start in range(0, len(store), micro):
                tokens, mask = batch(
                    [store[i] for i in range(start, min(start + micro, len(store)))], "cuda"
                )
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    z, n = model.encode(tokens, mask)
                    ez, en = exported.model(tokens, mask)
                if not torch.equal(z, ez) or not torch.equal(n, en):
                    raise ValueError("Encoder export parity failed")
        checks[which] = {
            "checkpoint_sha256": meta["sha256"],
            "step": state["step"],
            "validation": actual,
            "encoder_sha256": sha256(export_path),
            "encoder_full_validation_parity": True,
        }
        # The test split is first evaluated only after selection and validation replay.
        if which == "best":
            checks[which]["test"] = evaluate_pair(model, c, micro, "test")
        del state, model, codec, exported
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(folder / "final-verification.json", checks)
    return checks


def worker(frozen_path, name, resume=False):
    protocol = load_frozen(frozen_path)
    c = protocol["config"]
    setup(c)
    if environment() != protocol["environment"]:
        raise ValueError("Environment changed since freeze")
    folder = Path(c["output_dir"]) / name
    folder.mkdir(parents=True, exist_ok=True)
    with OwnedLock(folder / "worker.lock"):
        return _worker(folder, c, protocol, name, resume)


def _worker(folder, c, protocol, name, resume):
    attempt = uuid.uuid4().hex
    started = time.perf_counter()
    micro = protocol["resources"][name]["microbatch"]
    cfg = Config(**(c["model"] | c["models"][name]))
    store = RowStore(c["data_dir"], f"train-{cfg.max_tokens}")
    sampler = SharedSampler(store.meta, c["sampling"], c["sampling_seed"])
    model = Model(cfg, c["seed"]).cuda()
    optimizer = optimizer_for(model, c)
    history, validations, step, training_tokens, elapsed_before, best = [], [], 0, 0, 0.0, None
    evidence = folder / "events.jsonl"
    try:
        if resume:
            state, meta, invalid = recover(folder)
            if (
                state["run_id"] != protocol["run_id"]
                or state["model_name"] != name
                or state["recipe"] != c
            ):
                raise ValueError("Resume identity mismatch")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            sampler.load_state_dict(state["sampler"])
            restore_rng(state["rng"])
            step, training_tokens = state["step"], state["training_tokens"]
            history, validations, best = state["history"], state["validations"], state["best"]
            elapsed_before = state["elapsed_seconds"]
            events, truncated = [], 0
            if evidence.exists():
                for line in evidence.read_text(encoding="utf-8").splitlines():
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        truncated += 1
            beyond = [e for e in events if e.get("step", 0) > step]
            atomic_json(
                folder / f"resume-evidence-{attempt}.json",
                {
                    "checkpoint": meta,
                    "invalid_checkpoints": invalid,
                    "truncated_log_lines": truncated,
                    "discarded_completed_updates": [
                        e for e in beyond if e.get("event") == "completed"
                    ],
                    "possibly_inflight_updates": [e for e in beyond if e.get("event") == "intent"],
                    "note": "These attempts remain in events.jsonl; updates after this checkpoint will be repeated.",
                },
            )
            # Save broken pointer evidence, then repair from self-contained verified generation metadata.
            if (folder / "last.json").exists():
                shutil.copy2(folder / "last.json", folder / f"last-before-resume-{attempt}.json")
            metadata_name = Path(meta["file"]).with_suffix(".json").name
            atomic_json(folder / "last.json", {"metadata": metadata_name, **meta})
            if best:
                from .durable import checkpoint_candidates

                for saved_step, _, candidate in checkpoint_candidates(folder)[0]:
                    if saved_step == best["step"]:
                        _, best_meta = verified_load(candidate)
                        atomic_json(folder / "best.json", {"metadata": candidate.name, **best_meta})
                        break
                else:
                    raise ValueError(
                        "Selected best checkpoint missing; preserve evidence and inspect"
                    )
            del state
        elif (folder / "checkpoints").exists() or evidence.exists():
            raise ValueError(
                "Existing run evidence; explicit resume is required, never silently restart"
            )
        atomic_json(
            folder / "status.json",
            {"state": "running", "pid": os.getpid(), "step": step, "target": c["updates"]},
        )
        torch.cuda.reset_peak_memory_stats()

        def save():
            state = {
                "format": "fresh-comparison-v1",
                "run_id": protocol["run_id"],
                "model_name": name,
                "protocol": {"model": asdict(cfg)},
                "recipe": c,
                "dataset": protocol["dataset_manifest"],
                "environment": protocol["environment"],
                "source_hashes": protocol["source_hashes"],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "tokenizer": load_tokenizer(c["data_dir"]).to_str(),
                "step": step,
                "schedule": {
                    "completed_updates": step,
                    "next_lr": lr_at(step + 1, c) if step < c["updates"] else None,
                },
                "sampler": sampler.state_dict(),
                "rng": rng_state(),
                "history": history,
                "validations": validations,
                "best": best,
                "training_tokens": training_tokens,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                "memory": memory(),
                "microbatch": micro,
                "accumulation": c["effective_batch"] // micro,
            }
            return publish_checkpoint(folder, state, c["milestone_every"])

        if not resume:
            save()
        while step < c["updates"]:
            if (folder / "PAUSE").exists():
                save()
                atomic_json(
                    folder / "status.json",
                    {"state": "paused", "step": step, "target": c["updates"]},
                )
                return
            begin = time.perf_counter()
            indices = sampler.take(c["effective_batch"])
            batch_hash = identity([store.meta[i]["id"] for i in indices])
            journal(
                evidence,
                {
                    "event": "intent",
                    "attempt": attempt,
                    "step": step + 1,
                    "batch_sha256": batch_hash,
                    "time": time.time(),
                },
            )
            rows = [store[i] for i in indices]
            stats = accumulated_update(model, optimizer, rows, micro, c, step + 1)
            torch.cuda.synchronize()
            step += 1
            training_tokens += stats["tokens"]
            stats.update(
                {
                    "step": step,
                    "batch_sha256": batch_hash,
                    "seconds": time.perf_counter() - begin,
                    "training_tokens": training_tokens,
                    "source_sequences": dict(Counter(r["source"] for r in rows)),
                    "source_tokens": {
                        s: sum(len(r["ids"]) for r in rows if r["source"] == s)
                        for s in c["sampling"]["cycle_counts"]
                    },
                    **memory(),
                }
            )
            history.append(stats)
            journal(evidence, {"event": "completed", "attempt": attempt, **stats})
            if step % c["evaluate_every"] == 0 or step == c["updates"]:
                metrics = evaluate_pair(model, c, micro)
                validations.append({"step": step, "metrics": metrics})
                score = metrics["shared_short"]["groups"]["all"]
                key = [-score["exact"], score["nll"], step]
                if best is None or key < best["key"]:
                    best = {"step": step, "key": key}
                atomic_json(folder / "validation" / f"step-{step:06d}.json", validations[-1])
                print(name, "validation", step, score, flush=True)
            if (
                step % c["checkpoint_every"] == 0
                or step % c["evaluate_every"] == 0
                or step == c["updates"]
            ):
                save()
            if step % c["log_every"] == 0 or step == c["updates"]:
                status = {
                    "state": "running",
                    "model": name,
                    "pid": os.getpid(),
                    "step": step,
                    "target": c["updates"],
                    "microbatch": micro,
                    "effective_batch": 32,
                    "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                    **stats,
                }
                atomic_json(folder / "status.json", status)
                print(json.dumps(status), flush=True)
        model = optimizer = sampler = store = None
        gc.collect()
        torch.cuda.empty_cache()
        final_verification(folder, c, protocol, name, micro)
        atomic_json(
            folder / "status.json",
            {
                "state": "complete",
                "step": step,
                "training_tokens": training_tokens,
                "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            },
        )
    except BaseException:
        failure = {
            "state": "failed",
            "model": name,
            "step": step,
            "attempt": attempt,
            "error": traceback.format_exc(),
            "automatic_retry": False,
            "time": time.time(),
        }
        atomic_json(folder / f"failure-{attempt}.json", failure)
        atomic_json(folder / "status.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare",
            "checks",
            "probe",
            "freeze",
            "worker",
            "launch",
            "controller",
            "status",
        ],
    )
    parser.add_argument("--config", default="config/comparison.json")
    parser.add_argument("--frozen")
    parser.add_argument("--model", choices=["A", "B", "C"])
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare_comparison(read_json(args.config))["summaries"], flush=True)
    elif args.action == "checks":
        bounded_checks(args.config)
    elif args.action == "probe":
        probe(args.config, args.model, args.microbatch, args.output)
    elif args.action == "freeze":
        print(freeze(args.config))
    elif args.action == "worker":
        worker(args.frozen, args.model, args.resume)
    else:
        from .comparison_controller import controller, launch, status

        if args.action == "controller":
            controller()
        elif args.action == "launch":
            print(launch(args.frozen or freeze(args.config), args.resume))
        else:
            print(json.dumps(status(read_json(args.config)["output_dir"]), indent=2))


if __name__ == "__main__":
    main()
