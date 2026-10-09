"""One command line for preparing data, training and using the compressor."""

import argparse
import json
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare a pinned corpus or local documents")
    prep.add_argument("--config", default="config/train.json")
    prep.add_argument("--input", help="Local UTF-8 file; blank lines separate source documents")
    prep.add_argument("--smoke", action="store_true", help="Generated tiny dataset; no download")
    fit = commands.add_parser("train", help="Run the full staged recipe or resume it")
    fit.add_argument("--config", default="config/train.json")
    fit.add_argument("--resume")
    fit.add_argument("--stop-after", type=int, help="Pause after N additional updates")
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
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"Error: {exc}\n")


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
        recipe = read_config(args.config)
        if recipe.get("mode") in ("residual64_continuation", "residual64_fresh"):
            if args.resume or args.stop_after:
                raise ValueError("The continuation checkpoint and budget are set in config/train.json")
            import subprocess, sys
            runner = Path(__file__).resolve().parents[2] / "notebooks/train_residual64.py"
            subprocess.run([sys.executable, str(runner), "--config", str(Path(args.config).resolve())], check=True)
        else:
            from .train import train
            train(recipe, args.resume, args.stop_after)
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


if __name__ == "__main__":
    main()
