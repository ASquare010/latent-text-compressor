"""Bounded correctness checks, not additional research training runs."""

import copy
import subprocess
import sys
from collections import Counter
from dataclasses import asdict

import numpy as np
import pytest
import torch

from latent_text.comparison import accumulated_update, lr_at
from latent_text.comparison_data import SharedSampler, difficulty_weights, packed_rows
from latent_text.durable import locked, publish_checkpoint, recover, verified_load
from latent_text.model import Attention
from latent_text.residual import Config, Model


@pytest.fixture(autouse=True)
def small_threads():
    torch.set_num_threads(2)


def recipe():
    return {
        "updates": 8,
        "warmup_updates": 2,
        "learning_rate": 3e-4,
        "minimum_lr": 1e-5,
        "clip": 1.0,
        "weight_decay": 0.01,
    }


def tiny():
    return Config(
        vocab_size=32,
        width=8,
        heads=2,
        encoder_layers=1,
        decoder_layers=1,
        max_tokens=16,
        span=4,
        position_encoding="rope_v1",
    )


def test_rope_norm_padding_and_expected_counts():
    x = torch.randn(2, 4, 16, 8)
    rotated = Attention(Config(width=32, heads=4, max_tokens=16, span=4)).rotate(x)
    torch.testing.assert_close(rotated.square().sum(-1), x.square().sum(-1), rtol=1e-6, atol=2e-6)
    model = Model(tiny())
    tokens = torch.tensor([[3, 4, 5, 6, 7]])
    z, n = model.encode(tokens, tokens.ne(0))
    padded = F_pad(tokens, (0, 3), value=23)
    zp, npad = model.encode(padded, torch.arange(8)[None] < 5)
    torch.testing.assert_close(z, zp, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(n, npad)
    assert not hasattr(model, "position")
    configs = [(4, 1, 512, 64, 10456576), (6, 1, 512, 64, 12555776), (8, 2, 1024, 128, 19800064)]
    for enc, dec, context, span, expected in configs:
        model = Model(
            Config(
                encoder_layers=enc,
                decoder_layers=dec,
                max_tokens=context,
                span=span,
                position_encoding="rope_v1",
            )
        )
        assert sum(p.numel() for p in model.parameters()) == expected


def F_pad(*args, **kwargs):
    return torch.nn.functional.pad(*args, **kwargs)


def test_repeatable_initialization_and_token_normalized_accumulation():
    c = tiny()
    a = Model(c)
    b = Model(Config(**asdict(c)))
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
    b = Model(c)
    rows = [{"ids": [3 + i] * n} for i, n in enumerate([1, 3, 5, 9])]
    oa, ob = [torch.optim.SGD(m.parameters(), lr=3e-4) for m in (a, b)]
    sa = accumulated_update(a, oa, rows, 1, recipe(), 1, "cpu")
    sb = accumulated_update(b, ob, rows, 4, recipe(), 1, "cpu")
    assert sa["tokens"] == sb["tokens"] == 18
    assert sa["nll"] == pytest.approx(sb["nll"], rel=1e-6)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=1e-5, atol=1e-7)


def policy():
    return {
        "cycle_counts": {"a": 16, "b": 16},
        "uniform_component": 0.75,
        "difficulty_weight_min": 0.5,
        "difficulty_weight_max": 2.0,
        "length_boundaries": [4, 8, 16],
    }


def test_sampler_quotas_no_replacement_bounds_and_resume():
    rows = [{"ids": [i + 3] * 3, "source": source} for source in ("a", "b") for i in range(20)]
    difficulty_weights(rows, np.arange(32), policy())
    assert all(0.875 <= r["weight"] <= 1.25 for r in rows)
    a, b = [SharedSampler(rows, policy(), 1701) for _ in range(2)]
    first = a.take(32)
    assert first == b.take(32)
    assert Counter(rows[i]["source"] for i in first) == {"a": 16, "b": 16}
    state = copy.deepcopy(a.state_dict())
    rest = a.take(200)
    b.load_state_dict(state)
    assert rest == b.take(200)
    for source in ("a", "b"):
        by_source = [i for i in first + rest if rows[i]["source"] == source]
        for begin in range(0, len(by_source) - 19, 20):
            assert len(set(by_source[begin : begin + 20])) == 20


def test_packing_preserves_exact_same_stream_at_different_lengths():
    rows = [{"ids": [i + 3] * n, "paragraph_id": str(i)} for i, n in enumerate([4, 9, 3, 8, 2])]
    short = list(packed_rows(rows, 4, [30, 31], 17))
    long = list(packed_rows(rows, 8, [30, 31], 17))
    assert [t for r in short for t in r["ids"]] == [t for r in long for t in r["ids"]]
    assert sum(len(r["ids"]) for r in short) == sum(len(r["ids"]) for r in rows) + 8


def test_schedule_boundaries():
    c = recipe() | {"updates": 50000, "warmup_updates": 1000}
    assert lr_at(1, c) == pytest.approx(3e-7)
    assert lr_at(1000, c) == 3e-4
    assert lr_at(50000, c) == 1e-5
    assert lr_at(1001, c) < lr_at(1000, c)


def test_verified_checkpoint_fallback_and_immutable_milestone(tmp_path):
    state = {"step": 0, "run_id": "fixture", "model": {"x": torch.tensor([1.0])}, "best": None}
    publish_checkpoint(tmp_path, state, 2)
    publish_checkpoint(tmp_path, state | {"step": 1}, 2)
    milestone = publish_checkpoint(tmp_path, state | {"step": 2}, 2)
    newest = publish_checkpoint(tmp_path, state | {"step": 3}, 2)
    bad = tmp_path / "checkpoints" / newest["file"]
    with bad.open("ab") as f:
        f.write(b"corruption")
    recovered, meta, invalid = recover(tmp_path)
    assert recovered["step"] == 2 and invalid and bad.exists()
    assert meta["sha256"] == milestone["sha256"]
    with pytest.raises(ValueError, match="checksum"):
        verified_load(tmp_path / "checkpoints" / newest["metadata"])


def test_cpu_resume_optimizer_sampler_and_rng(tmp_path):
    rows = [{"ids": [i + 3] * n, "source": "a", "weight": 1.0} for i, n in enumerate([1, 3, 5, 9])]
    pol = {"cycle_counts": {"a": 4}}
    a = Model(tiny())
    oa = torch.optim.AdamW(a.parameters(), lr=3e-4, weight_decay=0.01)
    sa = SharedSampler(rows, pol, 1701)
    accumulated_update(a, oa, [rows[i] for i in sa.take(4)], 2, recipe(), 1, "cpu")
    saved = {
        "step": 1,
        "run_id": "fixture",
        "model": a.state_dict(),
        "best": None,
        "optimizer": oa.state_dict(),
        "sampler": sa.state_dict(),
        "rng": torch.get_rng_state(),
    }
    meta = publish_checkpoint(tmp_path, saved)
    accumulated_update(a, oa, [rows[i] for i in sa.take(4)], 2, recipe(), 2, "cpu")
    state, _ = verified_load(tmp_path / "checkpoints" / meta["metadata"])
    b = Model(tiny(), seed=98)
    b.load_state_dict(state["model"])
    ob = torch.optim.AdamW(b.parameters(), lr=3e-4, weight_decay=0.01)
    ob.load_state_dict(state["optimizer"])
    sb = SharedSampler(rows, pol, 99)
    sb.load_state_dict(state["sampler"])
    torch.set_rng_state(state["rng"])
    accumulated_update(b, ob, [rows[i] for i in sb.take(4)], 2, recipe(), 2, "cpu")
    for key, tensor in a.state_dict().items():
        torch.testing.assert_close(tensor, b.state_dict()[key], rtol=0, atol=0)
    assert sa.state_dict() == sb.state_dict()


def test_process_owned_lock_releases_after_exit(tmp_path):
    path = tmp_path / "owned.lock"
    code = "from filelock import FileLock; import sys; lock=FileLock(sys.argv[1]); lock.acquire(); print('ready',flush=True); input()"
    child = subprocess.Popen(
        [sys.executable, "-c", code, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        assert locked(path)
        child.communicate("\n", timeout=10)
        assert not locked(path)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)
