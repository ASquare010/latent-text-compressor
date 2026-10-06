"""Device selection, hashes and small local artifact writes."""

import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_path(path):
    path = Path(path).resolve()
    root = (Path.cwd() / "artifacts").resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Write generated files inside artifacts/ (run from the repository root)")
    return path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def choose_device(device="auto"):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ValueError("Use auto, cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Install the cuda extra and check your NVIDIA driver.")
    return device


def precision_for(device):
    return "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "fp32"


def autocast(device):
    return torch.autocast(device, dtype=torch.bfloat16, enabled=precision_for(device) == "bf16")


def sources():
    root = Path(__file__).parent
    return {p.name: digest(p) for p in sorted(root.glob("*.py"))}


def save_checkpoint(path, state):
    path = artifact_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)
