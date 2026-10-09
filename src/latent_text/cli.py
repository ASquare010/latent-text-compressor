"""One command line for preparing data, training and using the compressor."""

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import Timeout

from .runtime import artifact_path, read_config, write_json


def read_text(args):
    if args.text is not None:
        return args.text
    with Path(args.input).open(encoding="utf-8", newline="") as stream:
        return stream.read()


def text_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="A literal input string")
    group.add_argument("--input", help="UTF-8 text file; preserves line endings")
    return group


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare a pinned corpus or local documents")
    prep.add_argument("--config", default="config/train.json")
    prep.add_argument("--input", help="Local UTF-8 file; blank lines separate source documents")
    prep.add_argument("--smoke", action="store_true", help="Generated tiny dataset; no download")
    fit = commands.add_parser("train", help="Train a fresh model or resume the same run")
    fit.add_argument("--steps", type=int, default=50000)
    fit.add_argument("--encoder", type=int, required=True, help="Encoder layers")
    fit.add_argument("--decoder", type=int, required=True, help="Decoder layers")
    fit.add_argument("--context", type=int, required=True, help="Input context; 512 or 1024")
    fit.add_argument("--span", type=int, required=True, help="Tokens per latent vector")
    fit.add_argument("--microbatch", type=int, default=32)
    fit.add_argument("--lr", type=float, default=3e-4)
    fit.add_argument("--warmup", type=int, default=1000)
    fit.add_argument(
        "--name", required=True, help="Output folder; archives an old run on fresh start"
    )
    fit.add_argument("--resume", action="store_true", help="Resume this run's verified checkpoint")
    fit.add_argument("--inspect", action="store_true", help="Print settings without training")
    infer = commands.add_parser("infer", help="Reconstruct text or a saved latent file")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    group = text_arguments(infer)
    group.add_argument("--from-memory", help="Decode saved vectors without source text")
    infer.add_argument("--save-memory", help="Save vectors under artifacts/")
    infer.add_argument("--output", help="Save the JSON result under artifacts/")
    encode = commands.add_parser("encode", help="Use an encoder-only export to save vectors")
    encode.add_argument("--encoder", required=True)
    encode.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    text_arguments(encode)
    encode.add_argument("--output", required=True)
    export = commands.add_parser("export", help="Export only the trained encoder")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", default="artifacts/encoder.pt")
    evaluate = commands.add_parser("evaluate", help="Evaluate a complete held-out split")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", default="artifacts/data/fineweb")
    evaluate.add_argument("--split", choices=("valid", "test"), default="valid")
    evaluate.add_argument("--control", choices=("correct", "zero", "shuffle"), default="correct")
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate.add_argument("--output", default="artifacts/evaluation.json")
    app = commands.add_parser("app", help="Launch the local Gradio app")
    app.add_argument("--config", default="config/app.json")
    app.add_argument("--checkpoint", help="Override the app configuration checkpoint")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    except Timeout:
        parser.exit(2, "Error: This run is already active; wait for it or use another --name.\n")


def run(args):
    if args.command == "prepare":
        from .data import prepare

        config = read_config(args.config)
        if args.smoke and args.input:
            raise ValueError("Choose --smoke or --input")
        if args.smoke and config["dataset"]["counts"]["train"] > 100:
            raise ValueError("Use config/smoke.json with --smoke")
        documents = (
            [
                f"Document {i}: Mira ordered {i % 97 + 1} blue parts. "
                "The delivery is not ready. Keep the original order and numbers."
                for i in range(5000)
            ]
            if args.smoke
            else args.input
        )
        path = prepare(config["data_dir"], documents, **config["dataset"])
        print(f"Prepared: {path}")
    elif args.command == "train":
        run_training(args)
    elif args.command == "app":
        from .app import launch

        config = read_config(args.config)
        if args.checkpoint:
            config["checkpoint"] = args.checkpoint
        launch(config)
    else:
        from .codec import Codec

        if args.command == "encode":
            from .runtime import choose_device

            codec = Codec.load_encoder(args.encoder, choose_device(args.device))
            text = read_text(args)
            path = codec.save_memory(text, args.output)
            result = codec.measurements(text, codec.encode(text))
            result["memory_file"] = str(path)
        elif args.command == "export":
            print(Codec.load(args.checkpoint).export_encoder(args.output))
            return
        else:
            codec = Codec.load(args.checkpoint, args.device)
            if args.command == "infer":
                if args.from_memory:
                    if args.save_memory:
                        raise ValueError("--save-memory requires text input")
                    result = {
                        "output": codec.recover_file(args.from_memory),
                        "note": "Decoded from saved vectors and lengths alone.",
                    }
                else:
                    text = read_text(args)
                    result = codec.reconstruct(text)
                    if args.save_memory:
                        result["memory_file"] = str(codec.save_memory(text, args.save_memory))
            else:
                from .data import load_data
                from .evaluate import evaluate
                from .runtime import digest

                tokenizer, splits, manifest = load_data(args.data)
                tokenizer.encode_special_tokens = True
                if tokenizer.to_str() != codec.tokenizer.to_str():
                    raise ValueError("Evaluation dataset uses a different tokenizer")
                result = evaluate(
                    codec.model, splits[args.split], tokenizer, codec.device, args.control
                )
                result.update(
                    checkpoint_sha256=digest(args.checkpoint),
                    dataset=manifest,
                    split=args.split,
                    precision=codec.precision,
                )
        if args.output and args.command != "encode":
            write_json(artifact_path(args.output), result)
        print(json.dumps(result, indent=2, ensure_ascii=True))


def prepare_run_output(output, resume=False):
    """Replace a stopped run with an empty folder, preserving all old evidence."""
    from .durable import locked

    artifacts = (Path.cwd() / "artifacts").resolve()
    training = (artifacts / "training").resolve()
    output = Path(output).resolve()
    if training.parent != artifacts or output.parent != training:
        raise ValueError("Run output must be a direct child of repository artifacts/training")
    # Also recognize locks made by the previous nested run layout.
    locks = [output / "worker.lock", output / "launch.lock", *output.glob("*/worker.lock")]
    if any(path.exists() and locked(path) for path in locks):
        raise ValueError("This run already has an active worker; use another --name")
    if resume:
        if not (output / "frozen.json").is_file():
            raise ValueError("No saved run exists under this --name")
        return None
    archived = None
    if output.exists() and any(output.iterdir()):
        history = (artifacts / "training-history").resolve()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archived = (history / f"{output.name}-{stamp}-{uuid.uuid4().hex[:8]}").resolve()
        if history.parent != artifacts or archived.parent != history:
            raise ValueError("Archive must stay inside repository artifacts/training-history")
        history.mkdir(parents=True, exist_ok=True)
        output.rename(archived)
        print(f"Previous run archived: {archived}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    return archived


def run_training(args):
    root = Path(__file__).resolve().parents[2]
    os.chdir(root)
    os.environ.setdefault("HF_HOME", str(root / "artifacts/cache/huggingface"))

    import torch

    from latent_text.comparison import freeze, validate_config
    from latent_text.comparison_data import ensure_training_data
    from latent_text.durable import OwnedLock, atomic_json, read_json
    from latent_text.residual import Config, Model

    recipe = read_json(root / "config/comparison.json")
    name = args.name
    if not name or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in name
    ):
        raise ValueError("--name may contain only letters, numbers, '-' and '_'")
    if args.microbatch < 1 or 32 % args.microbatch:
        raise ValueError("--microbatch must divide 32")
    if args.steps <= args.warmup:
        raise ValueError("--steps must exceed --warmup; set --warmup lower for a short run")
    if Path(name).is_reserved():
        raise ValueError("--name must not be a reserved Windows device name")
    selected = {
        "encoder_layers": args.encoder,
        "decoder_layers": args.decoder,
        "max_tokens": args.context,
        "span": args.span,
    }
    if selected["max_tokens"] not in (512, 1024):
        raise ValueError("Prepared data currently supports --context 512 or 1024")
    recipe.update(
        name=name,
        output_dir=f"artifacts/training/{name}",
        updates=args.steps,
        warmup_updates=args.warmup,
        learning_rate=args.lr,
        models={name: selected},
        flat_output=True,
        microbatch_candidates=[args.microbatch],
    )
    validate_config(recipe)
    torch.set_num_threads(1)
    model = Model(Config(**(recipe["model"] | selected)), seed=recipe["seed"])
    parameters = sum(p.numel() for p in model.parameters())
    del model
    print(
        f"{name}: {selected['encoder_layers']} encoder / {selected['decoder_layers']} decoder layers"
    )
    print(
        f"Context {selected['max_tokens']} | {selected['span']} tokens/vector | width 256 | {parameters:,} parameters"
    )
    print(
        f"{args.steps:,} updates | microbatch {args.microbatch} x {32 // args.microbatch} accumulation = batch 32"
    )
    print(f"Output: {recipe['output_dir']}", flush=True)
    output = Path(recipe["output_dir"])
    folder = output
    print(f"Checkpoints: {(folder / 'checkpoints').resolve()}", flush=True)
    if args.inspect:
        return
    if not torch.cuda.is_available():
        raise ValueError(
            "CUDA is unavailable in this Python environment; select the repository .venv"
        )
    # This lock stays outside the replaceable folder and is held until the worker exits.
    with OwnedLock(root / "artifacts/.training-locks" / f"{name}.lock"):
        if args.resume:
            prepare_run_output(output, resume=True)
            if read_json(output / "frozen.json")["config"] != recipe:
                raise ValueError("Resume requires the saved settings; omit --resume to start fresh")
        ensure_training_data(recipe, allow_prepare=not args.resume)
        if not args.resume:
            prepare_run_output(output)
        frozen = output / "frozen.json"
        if not args.resume:
            atomic_json(
                output / "resources.json",
                {
                    name: {
                        "microbatch": args.microbatch,
                        "accumulation": 32 // args.microbatch,
                        "parameters": parameters,
                        "note": "Direct training; no controller or extra probe updates",
                    }
                },
            )
            requested = output / "requested.json"
            atomic_json(requested, recipe)
            freeze(requested)
        pause = folder / "PAUSE"
        if args.resume and pause.exists():
            pause.unlink()
        env = os.environ.copy()
        env["PYTHONPATH"] = str((output / "source").resolve())
        env["PYTHONUNBUFFERED"] = "1"
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        command = [
            sys.executable,
            "-m",
            "latent_text.comparison",
            "worker",
            "--frozen",
            str(frozen.resolve()),
            "--model",
            name,
        ]
        if args.resume:
            command.append("--resume")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        process = subprocess.Popen(command, env=env, creationflags=flags)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            pause.write_text("User requested a graceful pause.\n", encoding="utf-8")
            print(
                "Saving after the current update. Wait for the worker to stop; use --resume later.",
                flush=True,
            )
            code = process.wait()
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
