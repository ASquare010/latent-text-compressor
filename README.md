# Latent Text Compressor

The selected model uses plain residual depth attention, Branch Sigmoid FFNs,
RoPE and factorized packing with 125 internal code features. The model code is
unchanged. Training and inference share [one CLI](src/latent_text/cli.py).

## Train

Run from the repository folder. Install the NVIDIA extra before training:

```powershell
uv sync --locked --extra cuda
```

Plain `uv sync` omits the optional CUDA/CPU extras and can remove PyTorch.
After installation use the `.venv` commands below or `uv run --no-sync`.
Use a separate terminal for each model:

```powershell
.\.venv\Scripts\python.exe -m latent_text.cli train --encoder 4 --decoder 1 --context 512 --span 64 --steps 50000 --name 4enc-512-ctx-64-comp-50000-steps
.\.venv\Scripts\python.exe -m latent_text.cli train --encoder 6 --decoder 1 --context 512 --span 64 --steps 50000 --name 6enc-512-ctx-64-comp-50000-steps
.\.venv\Scripts\python.exe -m latent_text.cli train --encoder 8 --decoder 2 --context 1024 --span 128 --steps 50000 --name 8enc-1024-ctx-128-comp-50000-steps
```

| Model | Encoder / decoder | Context | Tokens/vector | Parameters |
| --- | --- | ---: | ---: | ---: |
| A | 4 / 1 | 512 | 64 | 10,456,576 |
| B | 6 / 1 | 512 | 64 | 12,555,776 |
| C | 8 / 2 | 1024 | 128 | 19,800,064 |

These are example configurations, not presets. Encoder layers, decoder layers,
context, span and name are required flags; there is no `--model` selector.

Microbatch and effective batch both default to **32** (one accumulation step).
`--span` is tokens per vector. `--inspect` prints settings without training;
`--name` only labels the output folder. Reusing it starts from random weights
with a fresh optimizer, moving the previous run to `artifacts/training-history/`.
Only explicitly adding `--resume` continues the same run's verified checkpoint
and requires matching settings. An active run cannot be replaced.
Interrupting with Ctrl+C requests a saved pause.

The first fresh run downloads and prepares the shared dataset automatically;
parallel commands wait and reuse it. A missing or damaged dataset during resume
must be restored, not silently regenerated. Data and weights are local artifacts,
not included in Git.

Checkpoints save every 250 updates under
`artifacts/training/<name>/checkpoints/`. The parent folder's `last.json`
and `best.json` identify the latest and selected best checkpoint. Full immutable
backups are retained every 5,000 updates. See the [training protocol](docs/fresh_comparison.md).

## Inference

```powershell
.\.venv\Scripts\python.exe -m latent_text.cli infer --checkpoint artifacts/path/to/checkpoint.pt --text "Your text here."
.\.venv\Scripts\python.exe -m latent_text.cli app
```

The app's **Inference model** dropdown finds saved named runs under
`artifacts/training`, with separate **Best** and **Latest** choices. Click
**Refresh models** to discover new checkpoints. It also offers the checkpoint
in `config/app.json` when that file exists. No weights are downloaded.
Historical 999/1000 and 1000/1000 scores involved inherited weights/curriculum;
they are not fresh-training baselines. Historical evidence remains in
[results](docs/residual64_long.md) and [handoff notes](docs/selected_handoff.md).

## Environment and data

Preserve the installed CUDA environment with `uv run --no-sync`. For a fresh install
only,use `uv sync --locked --extra cuda` or `--extra cpu`. Do not replace an active
environment during training. The model uses ordinary PyTorch,not custom kernels.
FineWeb-Edu data retains its dataset terms; see the pinned revision in the manifest.
No downstream LLM reasoning or memory-integration benefit has been demonstrated.
