"""Small real workflows, independent of private data or pretrained checkpoints."""

import copy
from dataclasses import asdict

import pytest
import torch

from latent_text.codec import Codec
from latent_text.data import load_data, prepare, token_ids, tokenizer_for
from latent_text.model import Config, Model
from latent_text.runtime import read_config, save_checkpoint
from latent_text.train import train


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_tokenizer_unicode_whitespace_and_control_literals():
    texts = [
        "Invoice 17019: 17.09, not 19.07.",
        "Hello مرحبا 你好 🙂",
        "A\r\n\tB  ",
        "<eos><pad><bos>",
    ]
    tokenizer = tokenizer_for(texts, 512)
    for text in texts:
        ids = token_ids(tokenizer, text)
        assert not any(i < 3 for i in ids)
        assert tokenizer.decode(ids, skip_special_tokens=False) == text


def test_padding_bottleneck_and_gradients():
    model = Model(
        Config(
            vocab_size=512,
            width=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            max_tokens=64,
            span=8,
        )
    )
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 11]])
    mask = torch.ones_like(tokens, dtype=torch.bool)
    z, lengths = model.encode(tokens, mask)
    padded = torch.nn.functional.pad(tokens, (0, 8))
    padded[0, 9:] = 91  # Masked nonzero IDs must not affect real tokens.
    valid = torch.arange(17)[None] < 9
    z_pad, lengths_pad = model.encode(padded, valid)
    torch.testing.assert_close(z, z_pad[:, :2], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        model.decode(z, lengths), model.decode(z_pad, lengths_pad), atol=2e-6, rtol=2e-5
    )
    loss = model.decode(z, lengths).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="right padding"):
        model.encode(
            tokens, torch.tensor([[True, False, True, True, True, True, True, True, True]])
        )


def test_latent_only_save_load_and_identity(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tokenizer = tokenizer_for(["A small original text."] * 2, 512)
    model = Model(
        Config(
            vocab_size=512,
            width=32,
            heads=4,
            encoder_layers=1,
            decoder_layers=1,
            max_tokens=16,
            span=4,
        )
    )
    codec = Codec(model, tokenizer)
    text = "Hello! مرحبا! 你好! 🙂\n" * 8
    payload = codec.encode(text)
    assert len(payload["chunks"]) > 1
    assert set(payload) == {"format", "decoder_id", "tokenizer_id", "chunks"}
    for chunk in payload["chunks"]:
        assert set(chunk) == {"vectors", "lengths"}
    expected = codec.decode(payload)
    path = codec.save_memory(text, "artifacts/memory.pt")
    assert codec.recover_file(path) == expected
    checkpoint = "artifacts/model.pt"
    save_checkpoint(
        checkpoint,
        {
            "protocol": {"model": asdict(model.config)},
            "model": model.state_dict(),
            "tokenizer": tokenizer.to_str(),
        },
    )
    loaded = Codec.load(checkpoint, "cpu")
    assert loaded.recover_file(path) == expected
    encoder_path = codec.export_encoder("artifacts/encoder.pt")
    encoder = Codec.load_encoder(encoder_path)
    torch.testing.assert_close(
        encoder.encode(text)["chunks"][0]["vectors"], payload["chunks"][0]["vectors"]
    )
    assert loaded.decode(encoder.encode(text)) == expected
    assert loaded.reconstruct("")["exact_match"]
    corrupt = {**payload, "decoder_id": "wrong model"}
    with pytest.raises(ValueError, match="identity mismatch"):
        loaded.decode(corrupt)


def test_training_resume_and_stage_transfer(tmp_path, monkeypatch):
    from pathlib import Path

    recipe = read_config(Path(__file__).parents[1] / "config/smoke.json")
    monkeypatch.chdir(tmp_path)
    documents = [
        f"Document {i}: Mira ordered {i % 97} blue parts. "
        "The delivery is not ready. Preserve the numbers and the order."
        for i in range(5000)
    ]
    prepare(recipe["data_dir"], documents, **recipe["dataset"])
    tokenizer, splits, manifest = load_data(recipe["data_dir"])
    assert manifest["complete"]
    assert manifest["revision"] is None
    docs = [{r["document_id"] for r in splits[s]} for s in ("train", "valid", "test")]
    assert not (docs[0] & docs[1] or docs[0] & docs[2] or docs[1] & docs[2])
    recipe["output_dir"] = "artifacts/full"
    full_path = train(recipe)
    interrupted = copy.deepcopy(recipe)
    interrupted["output_dir"] = "artifacts/resumed"
    pause = train(interrupted, stop_after=2)
    resumed_path = train(interrupted, resume=pause)
    full = torch.load(full_path, weights_only=True)
    resumed = torch.load(resumed_path, weights_only=True)
    assert full["total_steps"] == resumed["total_steps"] == 6
    assert full["protocol"]["model"]["span"] == 8
    for name, tensor in full["model"].items():
        torch.testing.assert_close(tensor, resumed["model"][name], rtol=0, atol=0)
    assert full["rng"] == resumed["rng"]
    changed = copy.deepcopy(interrupted)
    changed["learning_rate"] *= 2
    with pytest.raises(ValueError, match="same configuration"):
        train(changed, resume=resumed_path)


def test_missing_checkpoint_shows_empty_model_picker(tmp_path):
    from latent_text.app import build_app

    demo = build_app({"checkpoint": "nonexistent-model.pt", "models_dir": str(tmp_path)})
    try:
        components = demo.get_config_file()["components"]
        assert any(c["type"] == "dropdown" and not c["props"]["choices"] for c in components)
        assert any("No saved models yet" in str(c["props"].get("value", "")) for c in components)
    finally:
        demo.close()
