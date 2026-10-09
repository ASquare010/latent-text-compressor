# Fresh RoPE comparison protocol

This protocol replaces no historical result. Historical 999/1000 and 1000/1000
scores used inherited weights/curricula (and older FFNs), and are not baselines.
One initialization seed is a screening experiment, not evidence of general superiority.

| Model | Encoder / decoder | Context | Tokens/vector | Full input memory | Parameters |
| --- | --- | ---: | ---: | --- | ---: |
| A | 4 / 1 | 512 | 64 | 8 x 256 | 14,683,136 |
| B | 6 / 1 | 512 | 64 | 8 x 256 | 16,781,312 |
| C | 8 / 2 | 1024 | 128 | 8 x 256 | 28,317,184 |

The requested approximate A/B budgets do not match the existing architecture.
Each Branch Sigmoid block has 1,049,088 parameters. The two un-factorized span
maps alone cost 8,388,608 parameters for A/B and 16,777,216 for C. We preserve
those maps, width 256, four heads, 4,096 vocabulary, tied output embedding, and
the existing 1,534-feature/four-gate FFN. RoPE rotates query/key pairs in every
encoder and decoder attention block, base 10,000; no learned position table.
Plain residual attention means x + Attention(RMSNorm(x)), not residual logits.
Shared parameters use name-keyed seed 17 initialization. The FFN residual scale
reference is fixed at four layers for all models, avoiding a depth-dependent
initialization change between A and B. No encoder-to-decoder bypass exists.

A/B isolate encoder depth/capacity; C changes context, ratio and depth together.
C is a separate scaling experiment, not a controlled architecture win.

## Frozen training and sampling specification

- Exactly 50,000 fresh AdamW updates per model; effective batch 32 sequences.
  Probe microbatch 2, then 1 only if necessary; accumulation = 32 / microbatch.
  BF16 activations, FP32 model/optimizer and cross entropy, no dropout, no margin
  objective. Sum nonpadding-token losses across the entire accumulated batch,
  divide by its total actual tokens, then clip global gradient norm to 1.
- AdamW betas (0.9,0.999), epsilon 1e-8, decay 0.01 on all parameters. Update
  u=1..1000 uses 3e-4*u/1000; u=1001..50000 uses cosine from 3e-4 to 1e-5,
  progress=(u-1000)/49000. No validation-driven adjustments.
- Pinned FineWeb-Edu revision from config, training-only byte BPE vocabulary.
  Document hashes assign train/valid/test before paragraph extraction. Exact
  and whitespace/case-normalized duplicate paragraphs are removed globally;
  audit source domains, length quantiles, split intersections and repeated IDs.
  This is broad English educational/web text, not balanced multilingual data.
  Near-duplicate paraphrases are not guaranteed absent; audit discloses this.
- Source quotas per 320 sampled sequences (ten optimizer updates): 128 natural
  short (32..256 tokens), 128 natural packed, 32 patterned, 32 random synthetic.
  Thus 80% natural / 20% synthetic by sequence, not necessarily by token count.
  Natural packed rows concatenate split-local paragraphs with a tokenized double
  newline separator and cut contiguous 512/1024-token windows without dropping
  the final tail. Provenance records every contributing paragraph. Packing
  uses seed 1702; A/B share the exact same 512 manifest. C uses the same base
  train/valid/test documents at 1024. Short texts are identical for all three.
- Sampling seed 1701 shuffles the quota cycle. Each source uses a complete
  weighted random permutation before any row repeats; deterministic source
  seeds use SHA-256, not Python's randomized hash. Persist permutations, cursor,
  epoch and all PRNG state. A/B have identical batches, targets and batch hashes,
  even when their update speeds differ. Weighting changes order, not total
  exposure within a source epoch.
- Difficulty is a fixed proxy: add-one training-natural-token unigram surprisal
  averaged per token. Scores are normalized by midrank separately inside each
  source and length bucket (<=128, <=256, <=512, <=1024). Base weights are
  0.5+1.5*rank in [0.5,2]. Sampling rate = 0.75 + 0.25*base weight in
  [0.875,1.25]. Weighted permutation uses exponential race keys. This gives a
  substantial uniform component and bounded priority; synthetic/noisy rows
  cannot increase their fixed source share. Scores are measured once on
  training data before freeze, with no model-scored updates or held-out errors.
- Synthetic seed 1703 generates a frozen set of 4,096 unique rows per source
  at each context. Synthetic data are training-only. Reject collisions with
  any held-out token sequence. Random lengths span 32..context; patterns use
  bounded alphabets/motifs. Synthetic text decoding is not claimed meaningful.

## Execution, durability and evaluation

One GPU controller admits separate processes using measured peak reserved VRAM,
512 MiB per-process allowance, 20% measurement margin and 1,536 MiB free
headroom. Measure full-length backward/AdamW and evaluation at microbatch size.
All model/context/effective-batch settings remain fixed if workers must queue.
One OS-owned controller lock and one OS-owned lock per model prevent duplicates;
locks release on process death. No unrelated process is stopped. Failure is
recorded and not automatically retried. Explicit resume is required.

Freeze config, data manifests, environment, source snapshot and resource results
before launch. Probe updates use fresh disposable models and are recorded
separately; no probe weights or optimizer states enter training. Checkpoints
include the entire history, model, optimizer, schedule, sampler, tokenizer,
configuration, manifest hashes and CPU/CUDA/Python/NumPy RNG state. Flush/fsync,
reload verification, SHA-256 and atomic publication precede pointer changes.
Keep last, previous valid, best, and immutable full backups every 5,000 updates.
Save every 250 updates plus update zero and graceful pause boundaries.
Filesystem durability depends on the underlying disk honoring flush requests;
same-disk backups are not protection against disk loss.

Validate every 1,000 updates on the frozen shared short texts and ratio-specific
packed validation. Select highest exact token-sequence recovery on shared short,
then lowest token NLL, then earliest update. Store separate last/best references.
Record token accuracy, NLL, exact sequences, source/length errors, actual tokens
per vector, parameter counts, actual training tokens, elapsed time and allocated/
reserved VRAM. At completion reload last and best, verify full histories/hashes,
replay full validation, verify encoder-export parity, and only then evaluate the
untouched test split for selected best. No final quality claim before completion.

Resume reads verified checkpoint metadata rather than status.json. Evidence
after its completed update is retained as discarded/repeated work; intent logs
also expose a possibly interrupted in-flight update. Never silently initialize
a run that already has artifacts. Native/data-integrity failures require review;
an explicit resume acknowledges that review, and no automatic retry occurs.

References: [RoPE paper](https://arxiv.org/abs/2104.09864) and
[pinned FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md).
