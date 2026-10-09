# Fresh RoPE comparison protocol

This protocol replaces no historical result. Historical 999/1000 and 1000/1000
scores used inherited weights/curricula, and are not baselines.
One initialization seed is a screening experiment, not evidence of general superiority.

The table describes three example CLI invocations, not built-in model presets.
Each invocation supplies encoder, decoder, context, span and name explicitly.
`config/comparison.json` contains shared training/data settings only; the resolved
per-run configuration is saved in `artifacts/training/<name>/frozen.json`.

| Model | Encoder / decoder | Context | Tokens/vector | Full input memory | Parameters |
| --- | --- | ---: | ---: | --- | ---: |
| A | 4 / 1 | 512 | 64 | 8 x 256 | 10,456,576 |
| B | 6 / 1 | 512 | 64 | 8 x 256 | 12,555,776 |
| C | 8 / 2 | 1024 | 128 | 8 x 256 | 19,800,064 |

Use the latest pulled `residual.Model` unchanged: plain depth-route residual
attention across previous encoder sublayer outputs, factorized packing with
125 code features, width 256, four heads, 4,096 vocabulary, tied output
embedding, and the existing 1,534-feature/four-gate Branch Sigmoid FFN. RoPE
rotates query/key pairs in every encoder and decoder attention block, base
10,000; no learned position table. The pair of factorized maps costs 4,160,000
parameters for A/B and 8,256,000 for C. Each transformer block adds 1,049,088
parameters; depth queries add 512 parameters per encoder layer.

Call the existing model constructor with seed 17, without changing model code
or loading any weights. Its built-in map seeds 481/482 and FFN calibration seed
9127 remain unchanged. Its FFN initialization scale depends on encoder depth,
and sequential initialization means shared-shaped tensors need not be identical
between A/B. These are existing initialization tradeoffs, disclosed rather than
silently changing the selected model. No encoder-to-decoder bypass exists.

A/B isolate encoder depth/capacity; C changes context, ratio and depth together.
C is a separate scaling experiment, not a controlled architecture win.

## Frozen training and sampling specification

- Exactly 50,000 fresh AdamW updates per model; effective batch 32 sequences.
  The simple CLI defaults to microbatch 32 and one accumulation step. An explicit
  `--microbatch` divisor of 32 sets accumulation = 32 / microbatch; no automatic
  batch reduction or probe updates occur in this path.
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

Workers retain process-local CPU pinning and one CPU thread each. A null
`cpu_affinity` selects the lowest available core on the current host; the CLI
resolves and validates it before data preparation and freezes the actual core
number in the run configuration. Explicit core numbers must exist on the host.
The simple CLI launches one worker per command, without
resource probes or automatic GPU admission/queuing. Concurrent commands must fit
the available VRAM; an out-of-memory failure is recorded without reducing the
requested model, context or batch. Run commands sequentially if they do not fit.
An OS-owned launch lock outside each replaceable output folder and a worker lock
prevent duplicate runs with the same name. No unrelated process is stopped.
Failure is recorded and not automatically retried.

Freeze config, data manifests, environment, source snapshot and resource results
before launch. Direct CLI runs record the requested microbatch and actual model
parameter count, not a measured pre-launch memory estimate. Checkpoints
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
a run from old weights without `--resume`. A fresh invocation with the same name
archives the entire previous folder under `artifacts/training-history/` before
creating a new source snapshot, run identity, random model and optimizer. Active
workers cannot be replaced. Native/data-integrity failures require review; an
explicit resume acknowledges that review, and no automatic retry occurs.

References: [RoPE paper](https://arxiv.org/abs/2104.09864) and
[pinned FineWeb-Edu card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md).
