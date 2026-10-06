# Latent Text Compressor

A small PyTorch encoder that turns **up to 256 text tokens into eight ordered memory vectors**, with a parallel decoder that reconstructs the text. The selected model uses **32 tokens per vector**, 256 features per vector, and Curve-Wide feed-forward layers.

**Research result:** the selected checkpoint reconstructed 1,000/1,000 development paragraphs and a separate 1,000/1,000 fresh confirmation paragraphs exactly. This is a measured result on bounded tests, not a guarantee for arbitrary text. [Results and limits](docs/results.md).

**No pretrained weights are downloaded or published by this project.** Train locally with the complete recipe below. A fresh clone's app needs a checkpoint before it can run.

## What it is useful for

- Studying how much text information can survive in fewer neural memory positions.
- Building reproducible encoder/decoder and bottleneck experiments on a small GPU.
- Saving ordered latent bundles and reconstructing them with the matching model later.
- Prototyping external-memory interfaces for a future model trained to consume these vectors.

It is not a ZIP replacement, a semantic search embedding model, or a drop-in context extender for an existing LLM. Direct reasoning over these vectors and end-to-end LLM savings have not been demonstrated.

## Install with UV

Use Python 3.12 and [UV](https://docs.astral.sh/uv/getting-started/installation/). Run commands from this repository's root. Choose **one** hardware extra:

```sh
# NVIDIA GPU (CUDA 13.2 wheel; requires a compatible NVIDIA driver)
uv sync --locked --extra cuda

# Or CPU-only, including macOS CPU use
uv sync --locked --extra cpu
```

The lockfile includes the dependencies. No separate CUDA toolkit, C++ compiler or kernel package is required. The measured device was an 8 GB RTX 4070 Laptop GPU. The code uses ordinary PyTorch SDPA, BF16 on supported CUDA devices, and fused AdamW on CUDA. CPU and GPUs without BF16 support use FP32. No custom kernels or mandatory compilation.

Subsequent commands use `--no-sync` to preserve your chosen hardware installation. Re-run the matching `uv sync --locked --extra ...` only when you intend to update the environment. The `python -m` spelling also works on Windows installations that block generated console launchers.

```sh
uv run --no-sync python -m latent_text.cli --help
```

## Train from scratch

Settings live in [config/train.json](config/train.json). The first command streams the pinned FineWeb-Edu sample, prepares 50,000 training paragraphs, 1,000 validation paragraphs and 1,000 reserved test paragraphs, and trains a byte-level BPE tokenizer on training candidates only.

```sh
uv run --no-sync python -m latent_text.cli prepare --config config/train.json
uv run --no-sync python -m latent_text.cli train --config config/train.json
```

Preparation requires internet access and may take longer than training. Streaming avoids downloading the entire 10-billion-token subset, but still transfers source data and uses the Hugging Face cache. Prepared files live in `artifacts/data/fineweb/`. This repository does not contain the corpus.

The complete training lineage matters:

| Stage | Tokens/vector | Updates | Training examples |
| --- | ---: | ---: | --- |
| `base8` | 8 | 2,000 | 50k natural paragraphs |
| `natural8` | 8 | 1,000 | Same natural paragraphs |
| `coverage8` | 8 | 1,000 | Natural + 50k uniform token sequences |
| `patterns8` | 8 | 1,000 | Natural + 25k uniform + 25k repeated-pattern sequences |
| `span32` | 32 | 2,000 | Same natural/uniform/pattern mixture |

Total: **7,000 optimizer updates**, effective batch 16. The last stage keeps learned encoder/decoder blocks, reinitializes the two span maps and starts a fresh optimizer. Earlier continuations retain optimizer and sampling state. Synthetic IDs teach recovery coverage; they are not additional natural-language reasoning data.

Training prints loss and measured GPU memory, saves every 250 updates, evaluates the complete validation split at each stage end, and exports the final encoder. It never evaluates the reserved test split automatically. The final inference checkpoint is:

```text
artifacts/runs/paragraph/span32/last.pt
```

The historical final 2,000-update transfer took about 413 seconds on the measured GPU; **that is not the time for the full recipe**, preparation, or evaluation. A new run may differ with tokenizer builds, data preparation and hardware. Measure its actual recovery rather than assuming the historical scores.

### Resume or pause cleanly

```sh
# A bounded run saves before returning; the recipe itself stays unchanged.
uv run --no-sync python -m latent_text.cli train --config config/train.json --stop-after 250

# Resume whichever stage checkpoint you reached.
uv run --no-sync python -m latent_text.cli train --config config/train.json --resume artifacts/runs/paragraph/base8/last.pt
```

Resume checks source hashes, configuration, data hashes and PyTorch/device identity. Checkpoints include optimizer and RNG state. Each run stores a source snapshot. New experiments should use a new `output_dir`; existing runs are never silently replaced. If memory is tight, before starting a new run reduce `microbatch` from 4 to 2 and increase `accumulation` from 4 to 8.

### Your own text

Copy `config/train.json` to another configuration and change `data_dir`, `output_dir`, split counts, and length limits as needed. Put your UTF-8 file under ignored `artifacts/`:

```sh
uv run --no-sync python -m latent_text.cli prepare --config config/my-data.json --input artifacts/documents.txt
uv run --no-sync python -m latent_text.cli train --config config/my-data.json
```

Blank lines separate source documents in local files. Within each document, each nonempty line is a candidate paragraph. Original characters within a selected paragraph are preserved; line separators are not included as part of that paragraph. The API also accepts a list of whole document strings. Document hashing assigns splits before extraction; exact paragraph duplicates are removed across splits. A 1%/1% validation/test allocation needs enough documents. Preparation reports incomplete counts; training requires nonempty train/validation splits.

The default sampler keeps paragraphs with 32â€“256 tokens and candidate character lengths 40â€“3,072. It excludes longer text rather than silently truncating. This filtered sample does not represent all languages, code or all text distributions. The manifest records the resolved revision, exclusions, document/paragraph identities in split files, and tokenizer/data hashes.

FineWeb-Edu is published by HuggingFaceFW under ODC-By; the dataset card also specifies Common Crawl terms. See the [pinned dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md). The code license does not replace the dataset's terms.

## Inference

```sh
uv run --no-sync python -m latent_text.cli infer --checkpoint artifacts/runs/paragraph/span32/last.pt --text "Invoice 17019: 17.09, not 19.07."
```

The output includes the exact input/reconstruction, exact-match status, token/vector counts and byte measurements. Use `--input artifacts/example.txt` to preserve multiline input and whitespace. Longer inputs use consecutive independent 256-token windows. They are not a single long-context reasoning window.

### Save vectors, then decode without the original text

```sh
uv run --no-sync python -m latent_text.cli infer --checkpoint artifacts/runs/paragraph/span32/last.pt --text "The car is not ready. Do not ship it." --save-memory artifacts/car.pt
uv run --no-sync python -m latent_text.cli infer --checkpoint artifacts/runs/paragraph/span32/last.pt --from-memory artifacts/car.pt
```

A memory file contains ordered vector chunks, exact token lengths and model/tokenizer identity hashes. It contains no source text, source token IDs or residual correction stream. The matching checkpoint still provides the decoder and tokenizer. Never discard irreplaceable source data based on this experimental reconstruction.

### Encoder-only use

```sh
uv run --no-sync python -m latent_text.cli export --checkpoint artifacts/runs/paragraph/span32/last.pt --output artifacts/encoder.pt
uv run --no-sync python -m latent_text.cli encode --encoder artifacts/encoder.pt --text "Amina ordered 17 batteries." --output artifacts/amina.pt
uv run --no-sync python -m latent_text.cli infer --checkpoint artifacts/runs/paragraph/span32/last.pt --from-memory artifacts/amina.pt
```

The exported encoder excludes the decoder. A latent bundle must keep its order and length metadata; individual vectors are not independently meaningful facts. Load checkpoints you generated or trust. Tensor loading uses `weights_only=True`.

## Gradio app

Configure checkpoint/device/port in [config/app.json](config/app.json), then:

```sh
uv run --no-sync python -m latent_text.cli app --config config/app.json
```

Open **http://127.0.0.1:7860**. The app compares input and reconstruction, exposes whitespace differences, and displays vector counts and byte costs. It binds to loopback, creates no public share link and does not upload text or model weights. You can override the checkpoint with `--checkpoint artifacts/path/to/model.pt`.

## Notebook

One short [playground notebook](notebooks/playground.ipynb) covers configuration, optional preparation/training, checkpoint loading, custom inputs and latent-only recovery.

```sh
uv sync --locked --extra cuda --extra notebook
uv run --no-sync python -m jupyterlab notebooks/playground.ipynb
```

Use `--extra cpu` instead of `--extra cuda` on a CPU machine. Training cells require explicitly enabling their switches. The notebook is saved without outputs or private data.

## Evaluate and verify

```sh
uv run --no-sync python -m latent_text.cli evaluate --checkpoint artifacts/runs/paragraph/span32/last.pt --split valid --output artifacts/validation.json
uv run --no-sync python -m latent_text.cli evaluate --checkpoint artifacts/runs/paragraph/span32/last.pt --split valid --control zero --output artifacts/zero-memory.json
uv run --no-sync python -m latent_text.cli evaluate --checkpoint artifacts/runs/paragraph/span32/last.pt --split valid --control shuffle --output artifacts/shuffled-memory.json
uv run --no-sync python -m pytest -q --basetemp artifacts/test-temp
```

Evaluation reports reconstruction NLL, free parallel-generation exact-text match, token accuracy, edit distances and payload bytes. It uses the entire requested split. A low training loss alone is not evidence of exact recovery.

Quick **workflow-only** check without downloading a corpus:

```sh
uv run --no-sync python -m latent_text.cli prepare --config config/smoke.json --smoke
uv run --no-sync python -m latent_text.cli train --config config/smoke.json
uv run --no-sync python -m latent_text.cli infer --checkpoint artifacts/runs/smoke/transfer/last.pt --text "The car is not ready."
```

That six-update tiny model is expected to reconstruct poorly. It tests installation and plumbing, not model quality. The smoke data/model are deliberately different from the research model.

## How it works

```text
text -> reversible BPE -> token + position embeddings
     -> 4 bidirectional encoder blocks
     -> group consecutive 32 token features -> learned linear compression
     -> ordered latent vectors + original token lengths
     -> learned expansion -> 1 bidirectional decoder block
     -> all original token positions predicted at once -> text
```

All architecture code, including Curve-Wide, is in [model.py](src/latent_text/model.py). There is no VAE sampling, pretrained embedding model, teacher-forced decoder input, or encoder-to-decoder bypass. The decoder expands the memory back to token length and attends over those expanded positions. Future LLM consumption of only the short memory sequence is a separate experiment.

## Structure

```text
config/                 train, smoke and app settings
src/latent_text/
  model.py              complete architecture
  data.py               tokenizer, preparation and batching
  train.py              readable staged training loop and resume
  evaluate.py           reconstruction metrics and memory controls
  codec.py              text/latent API and encoder export
  app.py                Gradio playground
  cli.py                command entry points
  runtime.py            device selection and artifact/provenance helpers
notebooks/playground.ipynb
tests/test_workflow.py
docs/results.md         measured evidence and limitations
docs/notes.md           decisions and research board
artifacts/              generated locally, ignored by Git
```

`uv.lock` is tracked. Data, weights, latent files, logs, caches, notebook outputs and environments should stay local; the notebook in Git is output-free. There are no generated datasets or checkpoints in the source distribution. The code is MIT licensed; see [LICENSE](LICENSE).
