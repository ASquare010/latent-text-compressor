# Completed result (2026-10-07)

Both candidates completed10,000 fresh updates. Best and final checkpoints are
the10,000-update endpoints. Neither is a usable exact compressor on these sets.

| Model | Packed97 exact | Packed token recovery | Packed NLL | Short1000 exact | Short token recovery | Short NLL | Training allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
|Plain residual|0|28.4054%|3.394535|0|10.2761%|7.385770|847.34 /990|
|Residual + margin|0|25.7271%|3.550404|0|17.9936%|4.992787|847.34 /988|

Plain wins the primary packed-token metric; margin does better on short rows.
Each model has19,998,208 total /13,672,704 encoder parameters and received the same
30,714,208 training tokens. All update/validation sequences, source/data/target
hashes, checkpoints, exports, records, final/best replays and encoder parity
passed. Independent saved-evidence rehash passed against the frozen source
snapshot; active residual-package integration changes were disclosed and were
not used to reevaluate historical checkpoints. Receipt:
`dump/compression128-scale-v2/independent_completion_verification.json`.

No new native worker failure occurred in this completed batch. The zero-update
v1 UTF-8 serialization failure and two discarded probe updates remain recorded.
One fresh seed and reused validation; packed and short views share the same text.
Depth, size, compression, context and training history changed together: this
does not isolate a context or parameter bottleneck. No additional training is
authorized by completion. The protocol below is retained as launch history.

---

# Seed41 scaling pair:20M parameters, context1024, span128

**Historical startup observation:** both v2 workers have passed complete packed-data identities, the
matching initial-state hash and full initial97-sequence validation, then advanced
through10 updates each. No new worker failure observed. Controller PID24492;
total GPU2148/8188MiB. Early training peaks832.97MiB allocated /914–916MiB reserved.
These are startup observations, not final quality/resource results. Sources are
frozen. Progress command:
`powershell -NoProfile -File D:\Git\complex_fnn\dump\compression128-scale-v2\progress.ps1`.

## Corrected launch

The initial launch passed each worker's complete packed-data identities and full
initial validation, then both stopped before update1: a checkpoint metadata read
used Windows cp1252 for UTF-8 tokenizer JSON. This was a deterministic Python
encoding error, not a native crash or a data mismatch. **Zero research updates**
were made, and all v1 failure evidence is retained.

V2 fixes only the checkpoint/export tokenizer reads to explicit UTF-8, bounds
error text, and uses a new output directory. Computational source files, data,
initial model, schedules and budget are unchanged. The successful model/memory
probes remain applicable and are linked by hash; a new UTF-8 metadata torch-save/
load roundtrip passed, with zero additional optimizer updates. Old process owners
were confirmed absent before this diagnosed repair; there is no automatic retry
loop or environment change.

Active sources: `dump/compression128-scale-v2/`.
Active output: `D:/Git/latent-text-compressor/artifacts/runs/compression128-scale20m-s41-10000-v2/`.
The original v1 output remains a failed zero-update attempt. The total authorized
budget remains two models x10,000 fresh updates, with two discarded probe updates.

The user authorized two candidates, about20M total parameters, a different seed,
1024-token context,128 tokens per vector and10,000 training updates each in
parallel. The selected pair is **plain residual attention** (best final endpoint
in the preceding comparison) and **residual attention + mild margin** (best saved
exact-recovery checkpoint). No additional candidates, retries or budget extensions
are included.

## Architecture and comparison

Both models have exactly **19,998,208 total /13,672,704 encoder parameters**:
width256, four heads, **8 encoder blocks /2 decoder blocks**, Branch Sigmoid FFNs,
RoPE, learned residual attention across encoder depth and factorized ordered
packing. The token projection has128 features; each128-token group maps to one
256-dimensional BF16 vector. Decoder reconstruction uses only vectors and explicit
token lengths, with no encoder-to-decoder feature skip or supplied target tokens.

This raises depth from4/1 to8/2 while keeping vector width256 unchanged.
1024 input tokens produce8 vectors. Vector payload is4096 bytes for1024 tokens,
plus8 length bytes per sequence and container metadata. This is **128 tokens per
vector**, not128:1 byte compression or a guarantee of exact recovery.

Both are **fresh initializations at seed41**, verified identical by state-tensor
hash, with fresh AdamW state. No previous trained weights or optimizer are
inherited. Margin subtracts0.25 from the target logit during training, fading
linearly to0 over the final25% of the budget. Plain uses ordinary cross-entropy.
Inference architecture is identical between the pair.

Each receives10,000 updates with the same sampled rows, seeds and learning-rate
schedule: warmup250 updates to3e-4, cosine decay to3e-5, weight decay0.01, gradient
clip1, BF16, threads1. Microbatch1 x accumulation4 gives four sequences per update,
up to4096 tokens. The prior short-context runs used microbatch4 x accumulation4;
this is a changed sequence count, with the same maximum4096 input tokens/update.

Changing size, depth, context, span, fresh training and seed together is a scaling
screen. It does **not** isolate a parameter bottleneck or constitute a second-seed
replication at the original architecture. The old high scores inherited26,000
updates; these runs get10,000 fresh updates, so weaker results would not establish
a capacity limit. No comparison is presented as equal to that training history.

## Long-context data

The source remains the pinned50,000/1000/1000 paragraph dataset and its existing
4096-token vocabulary. Whole cached paragraphs are packed greedily in source
order with newline separators, without crossing split boundaries or truncating
paragraphs. A packed sequence can contain several source documents; this is
packed text, not a newly collected coherent1024-token document dataset.

| Split | Source paragraphs | Packed sequences | Tokens | Length range |
|---|---:|---:|---:|---:|
|Train|50,000|4831|4,653,227|373–1024|
|Validation|1000|97|93,419|782–1024|
|Test, not evaluated|1000|93|89,636|466–1023|

Source file hashes, split/document separation, every paragraph's membership and
complete decoded identities of all new packed sequences were checked. Workers
independently repeat the new packed-data byte/identity checks before updates.
Raw source data and tokenizer are unchanged. Prepared data is under
`dump/data/paragraph-packed1024-v1`; the original dataset is preserved.

Training samples50% packed natural text,25% random token sequences,25% repeated
patterns. Synthetic sequences now span128–1024 tokens (5000 cached rows per source,
seeds241/243); both candidates receive the same draws. No validation/test content
is used for training.

The primary metric is full recovery over **97 packed validation sequences**.
Report it separately from the original **1000 short validation paragraphs**,
which are also evaluated. These are two views of the same held-out text, not two
independent validations. Historical validation has already been reused for model
selection. Test content receives no model evaluation.

## Resource checks and launch

Both new models passed finite-gradient, parameter count,1024→8-vector shape,
padding exclusion and encoder-only parity checks. One synthetic optimizer update
per arm was discarded (two discarded updates total, zero research updates).

| Probe | Training allocated/reserved MiB | Validation allocated/reserved MiB |
|---|---:|---:|
|Plain|679.14 /772|332.55 /426|
|Margin|684.48 /752|331.11 /422|

The previous474.6MiB cap belongs to the small-model comparison. The newly
authorized larger model/context uses a per-worker ceiling3000MiB allocated /
3300MiB reserved, allowing two workers within the8GiB device with driver/context
headroom. Actual memory and GPU total are recorded; the probes are not a promise
of unchanged VRAM. Phase-boundary cache release and completed evaluation-batch
tensor release are applied equally. Any new worker/native/data failure stops
that worker; there is no automatic retry.

An initial probe invocation used the main repository's environment and failed on
missing rapidfuzz before any optimizer work. It was corrected to the existing
standalone compressor environment with `uv run --no-sync`; no packages were
installed or changed. That zero-update failure log is preserved. No native
failure occurred in the successful probes; earlier unrelated native failures
remain unresolved and preserved.

Sources: `dump/compression128-scale-v2/`. Sources freeze and are archived at launch.
Status: `D:/Git/latent-text-compressor/artifacts/runs/compression128-scale20m-s41-10000-v2/status.json`.
Persistent ignored records: `records/compressor128-scale20m-s41-{plain,margin}-v1.json`.
The selected span32 package and all prior weights/results remain unchanged.

Both workers perform an initial full packed validation, then full packed and
short validation every500 updates. Final and best checkpoints are saved and
replayed independently, with encoder export parity and hashes. Completion
requires both complete10,000-update sequences, equal initial states/target
schedules, source/data/checkpoint/export/record hashes and memory measurements.
Report selected and final scores; select by exact packed-sequence recovery,
then token accuracy, then NLL. No quality result is claimed before training.
