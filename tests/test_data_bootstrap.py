"""Fresh-run preparation must not replace missing resume data or damaged data."""

import pytest

from latent_text import comparison_data
from latent_text.durable import atomic_json


def test_fresh_missing_data_prepares_once(tmp_path, monkeypatch):
    config = {"data_dir": str(tmp_path)}
    calls = []

    def prepare(received):
        calls.append(received)
        manifest = {"files": {}}
        atomic_json(tmp_path / "manifest.json", manifest)
        return manifest

    monkeypatch.setattr(comparison_data, "prepare_comparison", prepare)
    assert comparison_data.ensure_training_data(config) == {"files": {}}
    assert comparison_data.ensure_training_data(config) == {"files": {}}
    assert calls == [config]


def test_missing_resume_data_is_not_regenerated(tmp_path, monkeypatch):
    def unexpected_prepare(config):
        pytest.fail("A saved run's data must not be regenerated")

    monkeypatch.setattr(comparison_data, "prepare_comparison", unexpected_prepare)
    with pytest.raises(FileNotFoundError, match="Restore its original dataset"):
        comparison_data.ensure_training_data({"data_dir": str(tmp_path)}, allow_prepare=False)


def test_corrupt_dataset_is_not_regenerated(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("damaged", encoding="utf-8")

    def unexpected_prepare(config):
        pytest.fail("Damaged data must remain available for inspection")

    monkeypatch.setattr(comparison_data, "prepare_comparison", unexpected_prepare)
    with pytest.raises(ValueError):
        comparison_data.ensure_training_data({"data_dir": str(tmp_path)})
