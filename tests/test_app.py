"""Model discovery and inference selection, without training or GPU allocation."""

from dataclasses import asdict

import gradio as gr
import pytest
import torch

from latent_text.app import InferenceModels, build_app, discover_models
from latent_text.app_ui import result_summary
from latent_text.codec import Codec
from latent_text.data import tokenizer_for
from latent_text.durable import publish_checkpoint
from latent_text.residual import Config, Model


@pytest.fixture
def saved_models(tmp_path):
    torch.set_num_threads(1)
    tokenizer = tokenizer_for(["A small saved model. Another example text."], 512)
    root = tmp_path / "training"
    folders = {}
    for name, span in (("A", 4), ("C", 8)):
        folder = root / f"run-{name}" / name
        config = Config(
            vocab_size=512,
            width=8,
            heads=2,
            encoder_layers=1,
            decoder_layers=1,
            max_tokens=16,
            span=span,
        )
        model = Model(config, seed=17)
        state = {
            "step": 1000,
            "run_id": name,
            "model": model.state_dict(),
            "protocol": {"model": asdict(config)},
            "tokenizer": tokenizer.to_str(),
            "best": {"step": 1000},
        }
        publish_checkpoint(folder, state)
        folders[name] = (folder, state)
    return {"models_dir": str(root), "device": "cpu"}, folders


def test_discovery_best_latest_and_stale_configured_path(saved_models):
    config, _ = saved_models
    choices = discover_models(config | {"checkpoint": "missing-historical.pt"})
    labels = [label for label, _ in choices.values()]
    assert len(labels) == 4
    assert any("Model A" in label and "Best" in label for label in labels)
    assert any("Model C" in label and "Latest" in label for label in labels)


def test_flat_named_runs_are_discovered_and_loadable(saved_models):
    config, folders = saved_models
    folder = folders["A"][0].parent.parent / "6enc-512-ctx"
    publish_checkpoint(folder, folders["A"][1])
    models = InferenceModels(config)
    selected = str((folder / "best.json").resolve())
    assert "6enc-512-ctx · Best" in models.choices[selected][0]
    assert "4 tokens per vector" in models.select(selected)[0]


def test_switching_uses_selected_weights_and_clears_previous_result(saved_models):
    config, folders = saved_models
    models = InferenceModels(config)
    a = str((folders["A"][0] / "best.json").resolve())
    c = str((folders["C"][0] / "best.json").resolve())
    selected = models.select(a)
    assert "4 tokens per vector" in selected[0]
    assert selected[1:5] == ("", result_summary(), None, "")
    result_a = models.reconstruct("A small saved model.", a)
    assert "Model A" in result_a[2]["model"]
    assert "8 tokens per vector" in models.select(c)[0]
    result_c = models.reconstruct("A small saved model.", c)
    assert result_c[2]["output_vectors"] <= result_a[2]["output_vectors"]
    assert "Model C" in result_c[2]["model"]
    # A request from a second tab still uses A even though C was selected last.
    assert models.reconstruct("A small saved model.", a) == result_a


def test_latest_pointer_reloads_changed_generation(saved_models, monkeypatch):
    config, folders = saved_models
    models = InferenceModels(config)
    folder, state = folders["A"]
    selected = str((folder / "last.json").resolve())
    loaded = []
    original_load = Codec.load

    def tracking_load(path, device):
        loaded.append(path)
        return original_load(path, device)

    monkeypatch.setattr(Codec, "load", tracking_load)
    models.select(selected)
    models.select(selected)
    assert len(loaded) == 1
    publish_checkpoint(folder, state | {"step": 1250})
    models.select(selected)
    assert len(loaded) == 2 and loaded[0] != loaded[1]


def test_unknown_model_cannot_load_arbitrary_path(saved_models):
    config, _ = saved_models
    models = InferenceModels(config)
    with pytest.raises(gr.Error, match="Select a saved model"):
        models.reconstruct("hello", "C:/unapproved.pt")


def test_refresh_discovers_first_checkpoint(tmp_path, saved_models):
    _, folders = saved_models
    root = tmp_path / "new-runs"
    models = InferenceModels({"models_dir": str(root), "device": "cpu"})
    assert not models.choices
    assert "No saved models yet" in models.select(None)[0]
    publish_checkpoint(root / "run-A" / "A", folders["A"][1])
    refreshed = models.refresh(None)
    assert len(models.choices) == 2
    assert "4 tokens per vector" in refreshed[1]


def test_gradio_reconstruct_receives_text_and_dropdown(saved_models):
    config, _ = saved_models
    demo = build_app(config)
    try:
        ui = demo.get_config_file()
        picker = next(c for c in ui["components"] if c["type"] == "dropdown")
        assert picker["props"]["label"] == "Inference model"
        assert len(picker["props"]["choices"]) == 4
        event = next(e for e in ui["dependencies"] if e["api_name"] == "reconstruct")
        assert len(event["inputs"]) == 2 and picker["id"] in event["inputs"]
    finally:
        demo.close()
