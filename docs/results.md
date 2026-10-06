# Reconstruction results

These are bounded reconstruction experiments, not language-model benchmarks or proof of universal lossless compression. Scores below were measured in the original research workspace. This standalone port also rechecked the selected trained checkpoint; it has not yet rerun the full 7,000-update recipe from scratch.

## Compression sweep

All candidates inherited the same 5,000-update span-8 encoder/decoder weights, then trained fresh span maps for 2,000 additional updates. Their parameter counts and total lineage matter when comparing ratios.

| Tokens/vector | Development exact (1,000) | Uniform exact (256) | Pattern exact (512) | Full parameters | Peak allocated MiB | Final 2k update seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **32 â€” selected** | **1,000** | **256** | **512** | **7,292,928** | **296.62** | **412.81** |
| 16 | 1,000 | 256 | 512 | 5,195,776 | 260.17 | 388.20 |
| 64 | 31 | 0 | 39 | 11,487,232 | 367.61 | 451.36 |

Measured hardware: RTX 4070 Laptop GPU, 8 GB; CUDA BF16, microbatch four, four-step accumulation. The memory number is PyTorch peak allocated memory during the transfer run, not total device use. Timings exclude the parent training, data preparation and evaluation. The 64-token candidate failed this budget; that does not establish a mathematical compression limit.

The selected encoder alone has **4,798,720 parameters**. Both passing candidates also recovered all 256 Unicode/formatting stress strings exactly. Selected-model controls on the same 64 examples: correct memory 64/64; zeroed and shuffled memories 0/64. The output depends on the supplied memories.

## Fresh confirmation

With the span-32 checkpoint frozen, a newly selected, document-disjoint set gave:

- 1,000/1,000 exact paragraphs; 100% token accuracy.
- Reconstruction NLL: 0.00311618298.
- 92,016 input tokens -> 3,387 vectors, or **27.17 tokens per vector** after short-paragraph rounding.
- BF16 vectors plus lengths: 1,742,144 bytes. Original UTF-8: 328,357 bytes.

It preserved information on that set while using fewer neural positions. The vectors used **more storage bytes** than the text. Identity and file-container overhead increase stored size further.

## What a 32:1 ratio means

For 256 tokens at width 256:

| Representation | Positions | Features | BF16 payload bytes |
| --- | ---: | ---: | ---: |
| Token feature matrix | 256 | 65,536 | 131,072 |
| Learned memory | 8 | 2,048 | 4,096 |

Length metadata adds eight bytes per window. Original token IDs would use 1,024 bytes as int32; UTF-8 size depends on the text. A serialized `.pt` file also includes container and identity overhead. Checkpoint size is a separate one-time model cost.

If a future consumer actually applies full self-attention to 8 memory positions instead of 256 token positions, the attention-pair count drops from 65,536 to 64 (1,024x). This is arithmetic, **not a measured whole-model speedup**. Our reconstruction decoder expands back to 256 positions. A receiving LLM must learn to read the memories; reconstructability does not establish direct reasoning ability.

An isolated attention-kernel test at 4,096 versus 256 positions measured about 24.4x faster kernel execution, versus 256x fewer pairs. It excluded encoders, projections, FFNs and generation and is not an LLM throughput result. Encoder timing differences across spans were inconclusive.

## Standalone packaging verification

The extracted model has bit-identical initial weights and FP32 forward results compared with the original implementation. Using the original selected weights in this repository's own locked CUDA environment reproduced:

- Full development set: **1,000/1,000 exact**, 100% token accuracy.
- Reconstruction NLL: **0.0030713155624070587**.
- Invoice numbers, negation, multilingual text and whitespace examples: four/four exact.
- Tests cover tokenizer round trips, right-padding isolation, finite gradients, source-free saved-latent recovery, encoder export, and bit-identical interrupted/resumed CPU training across a span change.

No checkpoint or source text from the research corpus is redistributed here. Reproduction requires preparing the corpus and training. Numerical identity across unrelated library versions or devices is not promised.

The Gradio HTTP inference endpoint and the notebook were executed with the local
trained checkpoint, including exact saved-vector recovery. A separate bounded
CUDA check ran all five stages (two updates per stage) with the real model
sizes, BF16 and fused AdamW. This was a plumbing check, not a quality experiment.
The full from-scratch reproduction remains a separate run.
