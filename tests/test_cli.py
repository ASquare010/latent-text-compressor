"""The unified CLI exposes the simple training workflow without starting workers."""

from pathlib import Path

import pytest
import torch

from latent_text import cli, comparison, comparison_data
from latent_text.durable import OwnedLock, atomic_json, read_json, run_folder


def train_args(encoder=6, decoder=1, context=512, span=64, name="my-run"):
    return [
        "train",
        "--encoder",
        str(encoder),
        "--decoder",
        str(decoder),
        "--context",
        str(context),
        "--span",
        str(span),
        "--steps",
        "50000",
        "--name",
        name,
    ]


@pytest.mark.parametrize(
    "encoder,decoder,parameters,context,span",
    [(4, 1, "10,456,576", 512, 64), (6, 1, "12,555,776", 512, 64), (8, 2, "19,800,064", 1024, 128)],
)
def test_explicit_train_inspect_without_preparation_or_training(
    encoder, decoder, parameters, context, span, monkeypatch, capsys
):
    def unexpected(*args, **kwargs):
        pytest.fail("Inspect must not prepare data or start a training process")

    monkeypatch.setattr(comparison_data, "ensure_training_data", unexpected)
    monkeypatch.setattr(cli.subprocess, "Popen", unexpected)
    cli.main(train_args(encoder, decoder, context, span) + ["--inspect"])
    output = capsys.readouterr().out
    assert f"{parameters} parameters" in output
    assert f"Context {context} | {span} tokens/vector" in output
    assert "microbatch 32 x 1 accumulation = batch 32" in output
    assert "my-run" in output


def test_train_custom_flags_and_boolean_resume_share_cli(monkeypatch):
    called = []
    monkeypatch.setattr(cli, "run_training", called.append)
    cli.main(train_args() + ["--resume"])
    assert len(called) == 1
    args = called[0]
    assert (args.encoder, args.decoder, args.context, args.span, args.steps) == (
        6,
        1,
        512,
        64,
        50000,
    )
    assert args.name == "my-run" and args.resume is True


def test_train_invalid_microbatch_is_actionable(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(train_args() + ["--microbatch", "3", "--inspect"])
    assert exc.value.code == 2
    assert "--microbatch must divide 32" in capsys.readouterr().err


def test_no_implicit_architecture_or_model_selector():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["train", "--name", "my-run"])
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(train_args() + ["--model", "A"])
    assert not cli.build_parser().parse_args(train_args()).resume


def test_same_name_launches_fresh_and_resume_is_explicit(tmp_path, monkeypatch):
    recipe = read_json(Path(cli.__file__).resolve().parents[2] / "config/comparison.json")
    assert "models" not in recipe and "worker_cpu_affinity" not in recipe
    atomic_json(tmp_path / "config/comparison.json", recipe)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "__file__", str(tmp_path / "src/latent_text/cli.py"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    prepared, frozen, launched = [], [], []
    monkeypatch.setattr(
        comparison_data,
        "ensure_training_data",
        lambda c, allow_prepare: prepared.append(allow_prepare),
    )

    def freeze(path):
        config = read_json(path)
        frozen.append(config)
        atomic_json(path.parent / "frozen.json", {"config": config, "run_id": len(frozen)})

    class Process:
        def __init__(self, command, **kwargs):
            launched.append(command)

        def wait(self):
            return 0

    monkeypatch.setattr(comparison, "freeze", freeze)
    monkeypatch.setattr(cli.subprocess, "Popen", Process)
    cli.main(train_args())
    output = tmp_path / "artifacts/training/my-run"
    old_weights = output / "checkpoints/old.pt"
    old_weights.parent.mkdir()
    old_weights.write_bytes(b"preserve old weights")
    cli.main(train_args())
    archives = list((tmp_path / "artifacts/training-history").iterdir())
    assert len(archives) == 1
    assert (archives[0] / "checkpoints/old.pt").read_bytes() == b"preserve old weights"
    assert not old_weights.exists()
    assert read_json(archives[0] / "frozen.json")["run_id"] == 1
    assert read_json(output / "frozen.json")["run_id"] == 2
    assert len(frozen) == 2 and all("--resume" not in command for command in launched)
    assert set(frozen[0]["models"]) == {"my-run"}
    assert run_folder(frozen[0], "my-run") == Path("artifacts/training/my-run")
    cli.main(train_args() + ["--resume"])
    assert launched[-1][-1] == "--resume" and len(frozen) == 2
    assert prepared == [True, True, False]
    with pytest.raises(SystemExit):
        cli.main(train_args(encoder=4) + ["--resume"])
    assert len(launched) == 3


@pytest.mark.parametrize("lock_name", ["worker.lock", "launch.lock", "A/worker.lock"])
def test_active_run_cannot_be_archived(tmp_path, monkeypatch, lock_name):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "artifacts/training/my-run"
    with OwnedLock(output / lock_name):
        with pytest.raises(ValueError, match="active worker"):
            cli.prepare_run_output(output)
    assert output.exists()
    assert not (tmp_path / "artifacts/training-history").exists()


def test_replacement_rejects_paths_outside_training(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="direct child"):
        cli.prepare_run_output(tmp_path / "unrelated")
    with pytest.raises(ValueError, match="No saved run"):
        cli.prepare_run_output(tmp_path / "artifacts/training/missing", resume=True)
