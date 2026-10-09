"""A local playground with a dropdown of operator-approved saved models."""

import difflib
import gc
from pathlib import Path
from threading import RLock

import gradio as gr

from .codec import Codec
from .durable import read_json, resolve_checkpoint


def discover_models(config):
    """List configured weights and published run pointers; never arbitrary visitor paths."""
    choices = {}
    configured = config.get("checkpoint")
    if configured == "hf-default":
        choices["hf-default"] = ("Default · 64 tokens/vector · step 31,000", Path("hf-default"))
    if configured and Path(configured).is_file():
        path = Path(configured).resolve()
        choices[str(path)] = ("Configured model", path)
    root = Path(config.get("models_dir", "artifacts/training"))
    for folder in sorted([*root.glob("*"), *root.glob("*/*")]):
        if not folder.is_dir():
            continue
        for which in ("best", "last"):
            pointer = folder / f"{which}.json"
            if not pointer.is_file():
                continue
            run_label = (
                folder.name
                if folder.parent == root
                else f"Model {folder.name} Â· {folder.parent.name}"
            )
            label = f"{run_label} Â· {'Best' if which == 'best' else 'Latest'}"
            try:
                step = read_json(pointer).get("step")
                if isinstance(step, int):
                    label += f" Â· update {step:,}"
            except (OSError, ValueError):
                # Keep the choice visible; loading will report damaged metadata.
                pass
            path = pointer.resolve()
            choices.setdefault(str(path), (label, path))
    return choices


class InferenceModels:
    """Keep one model in memory and use each request's explicit selection."""

    def __init__(self, config):
        self.config = config
        self.choices = discover_models(config)
        self.codec = None
        self.cache_key = None
        self.lock = RLock()

    def options(self):
        return [(label, key) for key, (label, _) in self.choices.items()]

    def release(self):
        on_cuda = self.codec is not None and self.codec.device == "cuda"
        self.codec = None
        self.cache_key = None
        gc.collect()
        if on_cuda:
            import torch

            torch.cuda.empty_cache()

    def load(self, selected):
        # Callers hold the lock across both loading and inference. Another browser
        # session cannot switch the shared cache halfway through reconstruction.
        if selected not in self.choices:
            raise gr.Error(
                "Select a saved model from the dropdown. Use Refresh models after training saves a checkpoint."
            )
        _, source = self.choices[selected]
        try:
            if str(source) == "hf-default":
                key = ("hf-default",)
                if key == self.cache_key:
                    return self.codec
                path = "hf-default"
            elif source.suffix == ".json":
                # The pointer changes as training progresses. Include its checksum
                # in the cache key so Latest never silently serves stale weights.
                reference = read_json(source)
                key = (str(source), reference["metadata"], reference["sha256"])
                if key == self.cache_key:
                    return self.codec
                path = resolve_checkpoint(source.parent, source.stem)
            else:
                stat = source.stat()
                key = (str(source), stat.st_mtime_ns, stat.st_size)
                if key == self.cache_key:
                    return self.codec
                path = source
            self.release()
            self.codec = Codec.load(path, self.config.get("device", "auto"))
            self.cache_key = key
            return self.codec
        except Exception as exc:
            raise gr.Error(f"Could not load this model: {exc}") from exc

    def select(self, selected):
        with self.lock:
            if not selected:
                self.release()
                message = "No saved models yet. Train a model, then click **Refresh models**."
            else:
                codec = self.load(selected)
                c = codec.model.config
                message = (
                    f"**{c.encoder_layers} encoder / {c.decoder_layers} decoder layers** Â· "
                    f"**{c.max_tokens}-token context** Â· **{c.span} tokens per vector** Â· "
                    f"{c.width} features per vector Â· {codec.device.upper()}"
                )
            # Clear previous results so they cannot be mistaken for this model's output.
            return message, "", "", None, "", gr.Button(interactive=bool(selected))

    def refresh(self, selected):
        with self.lock:
            self.choices = discover_models(self.config)
            if selected not in self.choices:
                selected = next(iter(self.choices), None)
            dropdown = gr.Dropdown(choices=self.options(), value=selected, interactive=True)
            return dropdown, *self.select(selected)

    def reconstruct(self, text, selected):
        with self.lock:
            codec = self.load(selected)
            output, status, stats, differences = compare(
                codec, text, self.config.get("max_characters", 16000)
            )
            stats["model"] = self.choices[selected][0]
            return output, status, stats, differences


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
    models = InferenceModels(config)
    selected = next(iter(models.choices), None)
    with gr.Blocks(title="Latent Text Compressor", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Latent Text Compressor\n"
            "Turn text into fewer learned memory vectors, then reconstruct every position "
            "in parallel. Try names, numbers, negation and whitespace."
        )
        with gr.Row():
            model = gr.Dropdown(
                choices=models.options(),
                value=selected,
                label="Inference model",
                info="Choose a saved model and checkpoint.",
                allow_custom_value=False,
                interactive=True,
                scale=4,
            )
            refresh = gr.Button("Refresh models", scale=1)
        details = gr.Markdown(
            "Select a model to load its saved weights."
            if selected
            else "No saved models yet. Train a model, then click **Refresh models**."
        )
        gr.Markdown(
            "Reconstruction can contain mistakes. Long inputs use independent windows. "
            "Fewer vectors do not necessarily mean fewer storage bytes."
        )
        with gr.Row():
            text = gr.Textbox(label="Original text", lines=9, placeholder="Enter a paragraphâ€¦")
            output = gr.Textbox(label="Reconstruction", lines=9, interactive=False)
        button = gr.Button(
            "Compress and reconstruct", variant="primary", interactive=bool(selected)
        )
        status = gr.Textbox(label="Recovery", interactive=False)
        with gr.Row():
            stats = gr.JSON(label="Positions and storage measurements")
            differences = gr.Textbox(label="Exact differences (escaped whitespace)", lines=7)
        gr.Examples(
            [
                ["Invoice 17019: 17.09, not 19.07."],
                ["The car is not ready. Do not ship it."],
                ["Hello! Bonjour! Ù…Ø±Ø­Ø¨Ø§! ä½ å¥½! ðŸ™‚"],
                ["First line.\n\tIndented second line.  "],
            ],
            inputs=text,
        )
        selection_outputs = [details, output, status, stats, differences, button]
        model.change(
            models.select,
            inputs=model,
            outputs=selection_outputs,
            concurrency_id="inference",
            concurrency_limit=1,
            api_name=False,
        )
        refresh.click(
            models.refresh,
            inputs=model,
            outputs=[model, *selection_outputs],
            concurrency_id="inference",
            concurrency_limit=1,
            api_name=False,
        )
        demo.load(
            models.select,
            inputs=model,
            outputs=selection_outputs,
            concurrency_id="inference",
            concurrency_limit=1,
            api_name=False,
        )
        button.click(
            models.reconstruct,
            inputs=[text, model],
            outputs=[output, status, stats, differences],
            api_name="reconstruct",
            concurrency_id="inference",
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
