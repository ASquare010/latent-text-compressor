# Smaller residual-attention comparison

## Completed results

All four models completed **4,000 additional updates each**, with no training
worker failures. Best and final checkpoints coincide at4,000. The controller's
completion audit passed full final/best validation replays, encoder export parity,
source/checkpoint/record hashes, matched targets, pair initialization and complete
update/validation sequences. No additional training has been launched.

| Model | Long exact /206 | Long token recovery | Long NLL | Short exact /1000 | Short token recovery | Short NLL |
|---|---:|---:|---:|---:|---:|---:|
|64:1 plain residual|103|99.8521%|0.011889|904|99.8876%|0.010493|
|**64:1 residual + margin**|**107**|**99.8650%**|**0.011393**|**904**|**99.8887%**|**0.010094**|
|128:1 plain residual|0|22.8261%|4.854762|93|50.0465%|3.208889|
|128:1 residual + margin|0|23.1240%|4.826250|122|50.7458%|3.158122|

The64:1 models improved from the parent's781 to904 exact short paragraphs,
at the same10,456,576 total /7,327,232 encoder parameters. They used423.61MiB
allocated /472MiB reserved training memory versus the parent's423.39 /474MiB.
Margin wins the long-set comparison by four examples; the short exact scores tie.
Plain residual remains a reasonable simple choice, as requested. The selected
inference checkpoint is not automatically changed by this status report.

Both128 models have10,457,600 total /7,327,744 encoder parameters. Training peaks
were422.88MiB allocated,472MiB reserved for plain and474MiB for margin. They
failed this compression/transfer/training recipe badly. This does not establish
the theoretical limit of128:1 compression or prove that more parameters help.

All models used512 context. Long metrics are for206 packed examples; short
metrics use the original1,000 paragraphs. These are two views of the same text.
Improvement over the saved parent includes4,000 extra updates, a changed context,
mixed-length sampling and an optimizer reset. It cannot be attributed to context
or architecture alone. One inherited initialization, reused validation, no fresh
test evaluation and no100% recovery; sequence shortening is not byte compression.

Completion receipt:
`D:/Git/latent-text-compressor/artifacts/runs/residual512-screen-4000-v5/completion_verification.json`.
Four persistent records remain in `records/residual512-screen-*-v1.json`.

**Post-training audit limitation:** an extra independent PowerShell check hit a
native CLR crash (`0x80131506`,process exit`0xc0000005`) while serializing its final
JSON report. No independent completion receipt was saved, so that extra audit is
not certified. It did no model or optimizer work and did not change trained
checkpoints. The controller's successful final audits are retained. No automatic
retry of this new native failure. The runtime workaround held throughout this
training batch but does not establish that all machine/runtime issues are fixed.

The launch observations and earlier failure history below are historical.

## Running after a tested runtime workaround

**2026-10-07: all four candidates are now training**, observed at175–200 of4,000
additional updates each. No new worker failures at this observation. The user
explicitly authorized diagnosing/fixing the repeated failures and launching.

The workaround pins each worker to a separate logical CPU (16,17,18,19), with
controller/preparation on16. It changes only process scheduling. Four concurrent
processes using the real worker imports each passed three complete52,000-row
hash and decoded-identity checks:624,000 row checks, zero optimizer updates.
The intermittent runtime failure's root cause remains unproven; this is not a
diagnosis of defective hardware or a proven tokenizer-library bug. Python,
tokenizers, PyTorch/CUDA, the parser and input data remain unchanged.

The first unpinned diagnostic controls were mixed: plain Python and import-only
passed, while tokenizer use sometimes produced invalid IDs or wrong decoded text.
Some tokenizer controls also passed. A diagnostic reporting bug briefly left
`passed:true` alongside a later error in `tokenizer_decode_only.json`; that run
counts as **failed**, and the reporting code was corrected. All raw diagnostics
and previous failures are retained under `dump/residual512-runtime-fix/`.

Preparation now completed10,303 training /206 validation /199 test512-token packs.
All four workers independently passed complete decoded checks for their packed
data and original train/validation rows, parent/source hashes and matching initial
states within each ratio. Test data receives identity checking only, no model
evaluation. Separate model/resource probes and initial full validation remain
explicitly waived; passing data diagnostics does not mean those gates passed.

| Model | Parameters | Encoder parameters | Early allocated /reserved MiB |
|---|---:|---:|---:|
|64:1 plain residual|10,456,576|7,327,232|421.94 /468|
|64:1 residual + margin|10,456,576|7,327,232|421.94 /468|
|128:1 plain residual|10,457,600|7,327,744|421.78 /466|
|128:1 residual + margin|10,457,600|7,327,744|421.78 /466|

Total GPU use observed2,503/8,188MiB. These are early peaks, not final memory or
quality results. The same16,000-update aggregate budget remains fixed. All arms
use the saved plain residual64 endpoint,26,000 inherited updates, fresh AdamW,
lr3e-5,microbatch2 x accumulation4,sampler seed41 and max context512. The128 map
transfer retains53.41%/63.52% of the encoder/decoder token-projection singular
energy; information preservation is not claimed.

Frozen sources: `dump/residual512-affinity-v5/`. Live status:
`D:/Git/latent-text-compressor/artifacts/runs/residual512-screen-4000-v5/status.json`.
Verified launch and repair receipts are in the source directory. To check:

```powershell
powershell -NoProfile -File D:\Git\complex_fnn\dump\residual512-affinity-v5\progress.ps1
```

Full206-pack and1,000-short-row validation every500 updates, final/best checkpoint
replays, encoder parity, hashes and target schedules remain required. No automatic
new-failure retries. The plain residual256 product checkpoint remains selected;
no new quality win is claimed. Historical preparation/failure entries follow.
**Latest user-requested direct attempt v4:** hidden uv launcher6360 started the
controller, but preparation exited1 when Python's JSON parser rejected integer
`376`. Zero training workers or optimizer updates. Original split hashes still
match; environment unchanged. Failure evidence/source snapshots retained under
`dump/residual512-direct-v4` and standalone `artifacts/runs/residual512-screen-4000-v4`.
No automatic retry. The same four x4,000 budget remains unused. This occurred in
required data loading with the separate preflight already skipped.


**Latest retry,2026-10-07 21:36PDT:** after the user disabled Smart App Control
and said "now try", the Windows block cleared and Python/controller started.
The preparation subprocess then exited1 with `ValueError: invalid literal for
int() with base 10: '264'` inside Python's JSON parser. No model workers or
optimizer/probe/research updates ran. All source train/validation/test hashes
still match, and independent System.Text.Json parsed all50,000/1,000/1,000 rows
successfully. This is an unresolved runtime/parsing failure, not evidence that
the input files changed. It is a Python exception, not a new native crash.

Attempt sources and diagnostic: `dump/residual512-direct-v3/`; failed output:
`D:/Git/latent-text-compressor/artifacts/runs/residual512-screen-4000-v3/`.
The partial packed-v3 directory is retained. No automatic retry or parser/identity
bypass; the same four x4,000 budget remains entirely unused. No Python/CUDA
environment changes were made. The policy-block report below is historical.

**Latest direct attempt,2026-10-07 21:31PDT:** the user again requested direct
training. The hidden `uv run --no-sync` launcher started, but Windows Application
Control blocked `D:/Git/latent-text-compressor/.venv/Scripts/python.exe` before
Python or the controller could start (error4551; CodeIntegrity event3077).
Policy ID: `0283ac0f-fff1-49ae-ada1-8a933130cad6`. No workers, data preparation
or optimizer updates occurred. This policy block is separate from the earlier
native crashes; those remain unresolved. No security bypass or environment
changes were attempted. The existing interpreter needs approval through the
device's application-control policy before this launch can proceed.

New attempt sources/evidence: `dump/residual512-direct-retry/`; blocked output:
`D:/Git/latent-text-compressor/artifacts/runs/residual512-screen-4000-v2/`.
Budget remains four x4,000 additional updates with up to four workers. Separate
preflight/resource probes and initial validation remain explicitly waived,
not passed; independent decoded identities and final audits remain required.
Everything below records the earlier preparation and direct attempt history.

Status: **direct launch attempted; native failure before training**. The user chose
**4,000 additional updates per variant**. New research updates: **0**; discarded
optimizer probes: **0**.

The latest instruction, "strat traning directly", authorized one direct attempt
without separate preflight/resource probes or initial full validation. Those
checks were waived, not passed. Worker data identities, source/parent hashes,
matching initialization, finite gradients and final audits remained required.

The controller started through `uv run --no-sync`, but its data-preparation child
PID22944 crashed in **python312.dll / 0xc0000005** at21:11:04PDT on2026-10-07.
Exit code3221225477; no Python traceback. It created only the tokenizer copy in
the new `paragraph-packed512-direct-v2` directory. No workers or optimizer updates
started. Previous partial data and failures remain intact. No environment changes
or automatic second retry followed. GPU was idle at0/8188MiB after the failure.

Attempt sources: `dump/residual512-direct/`; archived under the run's `source/`.
Run evidence: `D:/Git/latent-text-compressor/artifacts/runs/residual512-screen-4000-v1/`.
Windows crash events: `dump/residual512-direct/failure_events.json`.
The full four x4,000 training budget remains unspent. The protocol and earlier
preparation history below are retained; its original preflight requirement was
superseded only for this expressly authorized attempt.

## Scope

| Candidate | Context | Tokens/vector | Full parameters (expected) | Encoder parameters (expected) |
|---|---:|---:|---:|---:|
| Plain residual64 |512|64|10,456,576|7,327,232|
| Residual64 + margin |512|64|10,456,576|7,327,232|
| Plain residual128 |512|128|10,457,600|7,327,744|
| Residual128 + margin |512|128|10,457,600|7,327,744|

All use width256, four encoder blocks, one decoder block, Branch Sigmoid FFNs and
RoPE. Four parallel workers, each4,000 additional updates. Total authorized new
budget16,000 updates. No other models, baseline reruns or automatic retries.

Parent is the **plain residual64 final checkpoint**,781/1000 exact paragraphs,
99.71897% tokens,NLL0.0189274. It has26,000 inherited updates. This chooses the
fixed endpoint rather than the best selected checkpoint782/1000. SHA256:
`1a9c57693b14258a2e090f4f27ecff8a2b69129cd4c2e12d3f19b8a882cecb35`.

Both repos now select this plain residual checkpoint for inference. It retains
its tested256-token window;512-context candidates are separate experiments.
The old span32 model remains available, with its stronger1000/1000 historical
validation result. Selecting residual64 is the user's architecture preference,
not a claim that it has already matched span32 recovery.

## Why this comparison

The failed20M/context1024 experiment changed depth, compression, context, seed
and training history together. It started from scratch, whereas the previous
729/1000 residual result inherited18,000 updates and a compression curriculum.
Longer examples also make exact-match harsher: one wrong token fails the whole
example. However,28.4% long-token recovery in the20M plain run indicates a much
larger learning problem than this metric effect alone. Context is not established
as the cause; we have not identified a theoretical128:1 capacity limit.

This screen retains learned encoder and decoder weights, restores the smaller
model, and compares two compression ratios at one context. It tests whether
transfer and a mixed-length training distribution recover a useful balance.
It cannot isolate context alone or demonstrate independent-seed replication.

For64:1 all learned weights are copied exactly. For128:1 the factorized token
maps shrink125 to63 features using their leading singular subspaces; the
encoder initially averages two64-position groups and the decoder repeats them.
This keeps parameters within0.01%. It is an **untested transfer hypothesis**, not
an information-preserving conversion. The blocks, embeddings and depth queries
stay exact copies. Plain/margin pairs must start with identical tensors.

Every arm gets a fresh AdamW optimizer,lr3e-5,weight_decay0.01,clip1. Resetting all
four avoids unequal optimizer histories after changing packing shapes.
Microbatch2 x accumulation4, sampler seed41. This is a new sampling seed, not an
independent model initialization. Margin0.25 fades to zero in the last25%.

Proposed training draws:25% original short paragraphs,25% whole-paragraph packs
up to512 tokens,25% uniform sequences,25% repeated patterns. Synthetic lengths
32–512. Shared targets/order across all arms. Packs stay within original splits
and may join documents; they are not a coherent long-document corpus.

Evaluate the complete packed validation and the original1000 short rows initially
and every500 updates, then replay final and selected checkpoints independently.
These are two views of the same text, not independent validation sets. No test-set
model evaluation. Rank packed exact recovery, then token accuracy, then NLL;
report short results, actual token/vector ratios, bytes and memory alongside it.
Full-span64/128 ratios are sequence shortening, not byte compression claims.

## Runtime blocker and retained evidence

Earlier512 preparation in `dump/compression512-balance/preparation.log` failed
while Python parsed valid JSON, with `ValueError: invalid literal for int() with
base 10: '290'`. Source hashes remained unchanged and independent System.Text.Json
parsed all50,000 training rows. Partial `paragraph-packed512-v1` contains only a
tokenizer copy and no completed manifest. It is retained. The current attempt
stopped at the existing-directory guard; no failed parsing was retried.

The default-selection command saved exports and verified sample encoder parity
in both repos, then its process crashed: Python PID18452,
`ucrtbase.dll / 0xc0000409`,2026-10-07 21:00:45PDT. An earlier wrong-working-directory
export was rejected by the output guard and corrected without optimizer work.
The successful parity receipt does **not** certify a clean process exit.

The native cause is unresolved. Existing instructions prohibit automatic native
or data-failure retries. No new training or CUDA optimizer probes were started.
No environment sync/replacement or data bypass was attempted. Windows events
and process snapshot are in `dump/residual512-screen/`; no training workers are
alive. The old512 draft and all prior failure evidence remain intact.

Draft controller: `dump/residual512-screen/run.py`. It requires a passed
preflight, complete independent decoded identities, saved-parent replay, source
and parent hashes, gradient/padding/export checks and matched initialization.
None of the new screen's preflight/resource gates is claimed passed. Proposed
per-worker ceiling1300MiB allocated/1500MiB reserved; measured peaks remain
unknown. No claim of unchanged memory versus the old474MiB run.

The generic standalone `config/train.json` now names the plain residual family,
but it is a fresh-start example, not this warm-start study. The old7,000-update
curriculum is retained as `config/branch_rope_curriculum.json`. Launching that
generic recipe would be additional, unauthorized training.
