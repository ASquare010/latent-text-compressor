> Historical CurveFFN documentation. Active v0.2.0 uses an untrained Branch Sigmoid architecture; old scores and checkpoint compatibility do not apply.

# Research and development board

## Current milestone

A learned encoder can pack ordered token information into fewer vectors and a small parallel decoder can recover it exactly on the measured bounded tests. The selected setting is 32 tokens per vector at width 256. Its vectors may distribute information across slots; we do not assign separate fact/concept roles.

## Decisions to preserve

- Keep the architecture readable in one file and use ordinary PyTorch fast paths.
- Keep the complete five-stage training lineage; never describe the last transfer as the entire training cost.
- Keep lengths and model/tokenizer identities with saved memories. The memory format contains no source text, token IDs or correction stream.
- Keep data, checkpoints and raw evidence under ignored artifacts/. Keep configurations and compact result summaries tracked.
- No downloadable inference weights are required or published. Fresh users train locally.
- Use full held-out free reconstruction, memory controls and byte accounting when reporting results.
- Keep the working 32-token recipe stable while researching alternatives in separate configurations and run directories.

## Next experiments

| Status | Question | Evidence needed |
| --- | --- | --- |
| Open | Can a small consumer answer directly from these vectors? | Matched text-versus-memory QA, including names, quantities and negation; accuracy, latency and VRAM |
| Open | Can 64 tokens/vector recover reliably? | Controlled capacity/training sweep; same held-out suites; retain failures |
| Open | Can the span maps be smaller? | Factorized or shared maps; matched recovery, parameters and measured runtime |
| Open | Can precision be reduced? | Quantization/noise curves, exact recovery, actual payload and container bytes |
| Future | Can external memory support repeated retrieval? | Separate search keys, ordered payload bundles, provenance and retrieval recall |
| Future | Can a streaming encoder share context between windows? | Explicit state budget, cross-window facts, long-range recovery and forgetting tests |

A vector database is a possible future memory store. Exact float equality is not a reliable semantic deduplication rule. Identical text can be keyed by a content hash; semantically similar facts may differ in crucial numbers, dates or negation. A reconstruction code is not automatically a good nearest-neighbor search key.

## Boundaries

The analogy with image latents is useful: expensive processing can potentially operate on fewer positions. This implementation is a deterministic autoencoder, not a variational model. It does not demonstrate semantic concept discovery, infinite context, universal lossless string compression, or that a tiny model can replace a much larger reasoning model.

Related ideas already exist in latent representations and retrieval-assisted models. See [latent diffusion](https://arxiv.org/abs/2112.10752) and [RETRO](https://arxiv.org/abs/2112.04426) for context. This repository reports a specific implementation and experiments; it does not establish novelty over all prior work.

## Maintaining the public repository

Run the tests and a small command-line workflow after edits. Architecture changes require checking saved memories and source compatibility. Commit only source, configs, tests, the output-free notebook, docs and uv.lock. Publish updated scores only after evaluating the complete named split, with parameter counts, seed, training lineage, hardware, precision and actual memory use. Record failures alongside wins.
