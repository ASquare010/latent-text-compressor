# Fresh11.02M model replaces deleted checkpoints

User requested random-initialized11M,10000 total updates. Old compressor weights,
including the999/1000 winner,have been deleted;Step1/data/logs/results retained.
The new model has11022336 parameters,64tokens/vector,context512,code_features142.
No previous accuracy claim applies. config/train.json now describes the fresh run,
not15000-step continuation. Training notebooks are opt-in;do not duplicate the live
run. Existing output guard rejects duplicate launches. No ready encoder-only export
until the new training/evaluation/export completes. See docs/residual64_fresh11m.md.
# Latent Text Compressor

The selected model is **plain residual attention + Branch Sigmoid + RoPE**:
**64 tokens per vector,context512**,7.33M encoder/10.46M total parameters.
Final validation: **999/1000 short paragraphs and202/206 packed sequences exact**.
Training memory:426.78MiB allocated/474MiB reserved. Not universally lossless;
sequence reduction is not byte compression. See [results](docs/residual64_long.md).

Steps1 and2 are finished for the selected-model handoff;[scope and limitations](docs/selected_handoff.md).

## Use saved weights

Run `uv run --no-sync python -m latent_text.cli app` or open
`notebooks/playground.ipynb`. The configured saved weights are local,ignored and
not included in a fresh clone. The notebook loads them;it does not retrain.

## Optional continuation

`config/train.json` specifies15000 additional updates from the saved44000-update
checkpoint,keeping64:1,context512 and model size. Optimizer and sampling RNG are
restored. Constant lr3e-5. This is prepared only;100% recovery is not guaranteed.

```powershell
uv run --no-sync python notebooks/train_residual64.py --inspect
# Explicitly starts training:
uv run --no-sync python notebooks/train_residual64.py
```

Or open `notebooks/residual64_training.ipynb` and set `RUN_TRAINING=True`.
Full validation every500;final/best replay,encoder parity,hashes and memory guards
remain. An existing output directory is rejected. Never blindly retry native/data
failures. CPU affinity16 is this machine's tested mitigation,not a root-cause fix.
Edit data/weight/output paths in the recipe when moving machines;the original
matched dataset is required. Generic smoke training remains underconfig/smoke.json;
the old curriculum is archived underdocs/history,not the active default.

## Environment and data

Preserve the installed CUDA environment with `uv run --no-sync`. For a fresh install
only,use `uv sync --locked --extra cuda` or `--extra cpu`. Do not replace an active
environment during training. The model uses ordinary PyTorch,not custom kernels.
FineWeb-Edu data retains its dataset terms;see the pinned revision in the manifest.
No downstream LLM reasoning or memory-integration benefit has been demonstrated.
