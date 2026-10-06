"""A local playground. Checkpoint choice belongs to the operator, not remote visitors."""

import difflib
from pathlib import Path

import gradio as gr

from .codec import Codec


def compare(codec, text, limit):
    if len(text) > limit:
        raise gr.Error(f"Keep the input under {limit:,} characters for this interactive app.")
    result = codec.reconstruct(text)
    output = result.pop("output")
    result.pop("input")
    status = "Exact reconstruction" if result["exact_match"] else "Reconstruction differs"
    # repr makes tabs, newlines and trailing spaces visible in the comparison.
    difference = (
        "Identical, including whitespace."
        if text == output
        else "\n".join(difflib.ndiff([repr(text)], [repr(output)]))
    )
    return output, status, result, difference


def build_app(config):
    path = Path(config["checkpoint"])
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train with 'latent-text train --config config/train.json', "
            "or set checkpoint in config/app.json. No weights are downloaded automatically."
        )
    codec = Codec.load(path, config.get("device", "auto"))
    with gr.Blocks(title="Latent Text Compressor", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Latent Text Compressor\n"
            "Turn text into fewer learned memory vectors, then reconstruct every position "
            "in parallel. Try names, numbers, negation and whitespace."
        )
        gr.Markdown(
            f"**{codec.model.config.span} tokens per vector** at full windows · "
            f"{codec.model.config.width} features per vector · {codec.device.upper()}\n\n"
            "This is an experimental model and can make mistakes. Long inputs use independent "
            f"{codec.model.config.max_tokens}-token windows. Fewer vectors do not necessarily "
            "mean fewer bytes than the original text."
        )
        with gr.Row():
            text = gr.Textbox(label="Original text", lines=9, placeholder="Enter a paragraph…")
            output = gr.Textbox(label="Reconstruction", lines=9, interactive=False)
        button = gr.Button("Compress and reconstruct", variant="primary")
        status = gr.Textbox(label="Recovery", interactive=False)
        with gr.Row():
            stats = gr.JSON(label="Positions and storage measurements")
            differences = gr.Textbox(label="Exact differences (escaped whitespace)", lines=7)
        gr.Examples(
            [
                ["Invoice 17019: 17.09, not 19.07."],
                ["The car is not ready. Do not ship it."],
                ["Hello! Bonjour! مرحبا! 你好! 🙂"],
                ["First line.\n\tIndented second line.  "],
            ],
            inputs=text,
        )
        button.click(
            lambda value: compare(codec, value, config.get("max_characters", 16000)),
            inputs=text,
            outputs=[output, status, stats, differences],
            api_name="reconstruct",
            concurrency_limit=1,
        )
    return demo


def launch(config):
    demo = build_app(config)
    demo.queue(default_concurrency_limit=1, max_size=16).launch(
        server_name=config.get("host", "127.0.0.1"),
        server_port=config.get("port", 7860),
        share=False,
        inbrowser=False,
    )
