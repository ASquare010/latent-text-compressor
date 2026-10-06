"""Five explicit training stages: learn recovery first, then tighten the bottleneck."""

import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from .codec import Codec
from .data import batch, load_data
from .evaluate import evaluate
from .model import Config, Model
from .runtime import (
    artifact_path,
    autocast,
    choose_device,
    precision_for,
    save_checkpoint,
    sources,
    write_json,
)


def synthetic_rows(count, seed, vocab_size, max_tokens, patterns=False):
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        if patterns:
            alphabet = rng.sample(range(3, vocab_size), rng.randint(1, min(64, vocab_size - 3)))
            length = rng.randint(1, max_tokens)
            if rng.random() < 0.5:
                ids = rng.choices(alphabet, k=length)
            else:
                motif = rng.choices(alphabet, k=rng.randint(1, 16))
                ids = (motif * ((length + len(motif) - 1) // len(motif)))[:length]
        else:
            length = rng.randint(min(32, max_tokens), min(128, max_tokens))
            ids = [rng.randrange(3, vocab_size) for _ in range(length)]
        rows.append({"ids": ids})
    return rows


def make_optimizer(model, recipe, device):
    return torch.optim.AdamW(
        model.parameters(),
        lr=recipe["learning_rate"],
        weight_decay=recipe["weight_decay"],
        fused=recipe.get("fused_optimizer", True) and device == "cuda",
    )


def validate_recipe(recipe):
    Config(**recipe["model"])
    for key in ("threads", "microbatch", "accumulation", "checkpoint_every", "log_every"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if recipe["learning_rate"] <= 0 or not 0 <= recipe["minimum_lr"] <= recipe["learning_rate"]:
        raise ValueError("Invalid learning-rate range")
    names = set()
    if not recipe["stages"]:
        raise ValueError("At least one training stage is required")
    for stage in recipe["stages"]:
        name = stage["name"]
        if not name.replace("_", "").replace("-", "").isalnum() or name in names:
            raise ValueError("Stage names must be unique simple folder names")
        names.add(name)
        Config(**{**recipe["model"], "span": stage["span"]})
        if stage["updates"] < 1 or stage["schedule"] not in ("cosine", "warmup_constant"):
            raise ValueError("Invalid stage schedule or update count")
        if min(stage.get("uniform", 0), stage.get("patterns", 0)) < 0:
            raise ValueError("Synthetic row counts cannot be negative")


def train(recipe, resume=None, stop_after=None):
    """stop_after pauses after this many additional updates, without changing the recipe."""
    validate_recipe(recipe)
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    torch.set_num_threads(recipe["threads"])
    device = choose_device(recipe["device"])
    tokenizer, splits, manifest = load_data(recipe["data_dir"])
    if not splits["train"] or not splits["valid"]:
        raise ValueError("Training and validation splits must both be nonempty")
    vocab = tokenizer.get_vocab_size()
    if vocab > recipe["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary exceeds model size")
    if any(
        not r["ids"] or len(r["ids"]) > recipe["model"]["max_tokens"]
        for rows in splits.values()
        for r in rows
    ):
        raise ValueError("Data lengths do not fit the configured model")
    output = artifact_path(recipe["output_dir"])
    source_hashes = sources()
    stage_index = stage_step = total_steps = 0
    rng = random.Random(recipe["seed"])
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if state["recipe"] != recipe or state["dataset"] != manifest:
            raise ValueError("Resume requires the same configuration and dataset")
        if state["sources"] != source_hashes:
            raise ValueError("Source changed. Resume with the frozen source snapshot in the run.")
        if state["environment"]["device"] != device or state["environment"]["torch"] != str(
            torch.__version__
        ):
            raise ValueError("Exact resume requires the same device type and PyTorch build")
        model = Model(Config(**state["protocol"]["model"]), recipe["seed"]).to(device)
        model.load_state_dict(state["model"])
        optimizer = make_optimizer(model, recipe, device)
        optimizer.load_state_dict(state["optimizer"])
        stage_index, stage_step, total_steps = (
            state["stage_index"],
            state["stage_step"],
            state["total_steps"],
        )
        rng.setstate(state["rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Run directory already exists; use --resume or a new output_dir")
        config = Config(**{**recipe["model"], "span": recipe["stages"][0]["span"]})
        model = Model(config, recipe["seed"]).to(device)
        optimizer = make_optimizer(model, recipe, device)
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "recipe.json", recipe)
        for name in source_hashes:
            frozen = output / "source" / name
            frozen.parent.mkdir(exist_ok=True)
            frozen.write_bytes(Path(__file__).with_name(name).read_bytes())
    environment = {
        "torch": str(torch.__version__),
        "device": device,
        "precision": precision_for(device),
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "fused_optimizer": recipe.get("fused_optimizer", True) and device == "cuda",
    }
    write_json(output / "environment.json", environment)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    added = 0
    for index in range(stage_index, len(recipe["stages"])):
        stage = recipe["stages"][index]
        step = stage_step if index == stage_index else 0
        if model.config.span != stage["span"]:
            # Preserve learned blocks; only the differently shaped span maps start fresh.
            config = Config(**{**asdict(model.config), "span": stage["span"]})
            new_model = Model(config, recipe["seed"]).to(device)
            shared = {
                k: v
                for k, v in model.state_dict().items()
                if k not in ("compress.weight", "expand.weight")
            }
            missing = new_model.load_state_dict(shared, strict=False)
            assert set(missing.missing_keys) == {"compress.weight", "expand.weight"}
            assert not missing.unexpected_keys
            model = new_model
            optimizer = make_optimizer(model, recipe, device)
            rng = random.Random(recipe["seed"])
        rows = (
            splits["train"]
            + synthetic_rows(
                stage.get("uniform", 0),
                stage.get("uniform_seed", 219),
                vocab,
                model.config.max_tokens,
            )
            + synthetic_rows(
                stage.get("patterns", 0),
                stage.get("pattern_seed", 217),
                vocab,
                model.config.max_tokens,
                True,
            )
        )
        folder = output / stage["name"]
        folder.mkdir(exist_ok=True)
        checkpoint = folder / "last.pt"
        print(
            f"Stage {stage['name']}: {step}/{stage['updates']}; "
            f"{sum(p.numel() for p in model.parameters()):,} parameters",
            flush=True,
        )
        while step < stage["updates"]:
            started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if stage["schedule"] == "warmup_constant":
                lr = recipe["learning_rate"] * min(1, (step + 1) / max(1, stage.get("warmup", 100)))
            else:
                progress = step / max(1, stage["updates"] - 1)
                lr = recipe["minimum_lr"] + 0.5 * (
                    recipe["learning_rate"] - recipe["minimum_lr"]
                ) * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
            micros = [
                rng.choices(rows, k=recipe["microbatch"]) for _ in range(recipe["accumulation"])
            ]
            token_count = sum(len(r["ids"]) for micro in micros for r in micro)
            nll = 0.0
            for micro in micros:
                tokens, mask = batch(micro, device)
                with autocast(device):
                    logits = model(tokens, mask)
                    loss = (
                        F.cross_entropy(
                            logits.float().flatten(0, 1),
                            tokens.masked_fill(~mask, -100).flatten(),
                            reduction="sum",
                        )
                        / token_count
                    )
                loss.backward()
                nll += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), recipe["clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if device == "cuda":
                torch.cuda.synchronize()
            step += 1
            total_steps += 1
            added += 1
            peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
            stats = {
                "stage": stage["name"],
                "update": step,
                "total_updates": total_steps,
                "nll": nll,
                "lr": lr,
                "seconds": time.perf_counter() - started,
                "peak_allocated_mib": peak,
            }
            with (output / "training.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(stats) + "\n")
            if step % recipe["log_every"] == 0 or step == stage["updates"]:
                print(stats, flush=True)
            pause = stop_after is not None and added >= stop_after
            if step % recipe["checkpoint_every"] == 0 or step == stage["updates"] or pause:
                save_checkpoint(
                    checkpoint,
                    {
                        "format": "latent-text-training-v1",
                        "protocol": {"model": asdict(model.config)},
                        "recipe": recipe,
                        "dataset": manifest,
                        "sources": source_hashes,
                        "environment": environment,
                        "model": model.state_dict(),
                        "tokenizer": tokenizer.to_str(),
                        "optimizer": optimizer.state_dict(),
                        "stage_index": index,
                        "stage_step": step,
                        "total_steps": total_steps,
                        "rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                    },
                )
            if pause:
                print(f"Paused. Resume from {checkpoint}", flush=True)
                return checkpoint
        report = evaluate(model, splits["valid"], tokenizer, device)
        write_json(folder / "validation.json", report)
        print("Validation:", report["summary"], flush=True)
    Codec(model, tokenizer, device, precision_for(device)).export_encoder(output / "encoder.pt")
    print(f"Finished. Inference checkpoint: {checkpoint}", flush=True)
    return checkpoint
