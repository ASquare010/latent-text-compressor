# Combined and linear-attention variants

## Completed results

All five runs completed **8000 additional updates each**, with no new worker
failures. All used the same saved parent, target schedule, optimizer starting
state, context256/span64/width256, **7,327,232 encoder /10,456,576 full parameters**.
The winner by best exact-paragraph recovery is **combined at update7500**:
residual attention + coverage + difficulty0.25 + margin0.25, with both loss
add-ons fading out over the last25% of training. Encoder token attention remains
softmax. The linear variants do not win this screen.

| Variant | Best exact /1000 | Best token accuracy | Best NLL | Best update | Final exact /1000 | Final NLL | Logged allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
|Saved parent (fewer updates)|729|99.6606%|0.041006|Prior8000|729|0.041006|426.93 /452|
|Combined|**788**|99.7406%|0.018383|7500|**786**|**0.017736**|424.87 /574|
|Coverage + difficulty fade|786|99.7427%|0.018844|7500|769|0.019123|424.55 /574|
|Full linear attention + combined|784|99.7417%|**0.018020**|7500|773|0.018077|424.79 /572|
|Last-two-layer linear attention + combined|776|99.7211%|0.018784|7500|759|0.019216|424.83 /574|
|Coverage + margin fade|769|99.7071%|0.020117|7500|767|0.018240|424.87 /574|

Combined improves saved-parent exact recovery by59 paragraphs (+5.9 percentage
points), with240 wrong tokens versus314 on the same92,516-token validation. Its
final checkpoint has238 wrong tokens but two fewer completely correct paragraphs.
Hard-fade also has238 wrong tokens at its selected checkpoint; full linear has
lower selected-checkpoint NLL than combined. Thus combined leads exact paragraphs,
not every metric. Its two-paragraph lead over hard-fade is small.

**Training is a confound:** the parent had18,000 ancestral updates, selected new
checkpoints25,500, and final checkpoints26,000. No plain-continuation arm ran for
this latest budget. This demonstrates improvement after more training with the
new recipes; it does not isolate the add-ons' benefit over training alone.
All five new variants had matching8000-update budgets and identical target/source
schedules. One inherited seed and repeated validation/checkpoint selection limit
conclusions. The test split was not evaluated. Perfect recovery is still unmet;
the reliable selected span32 model remains unchanged.

**Memory:** these are recorded training-phase peaks, not a complete evaluation
lifecycle measurement. Allocated peaks are sampled before periodic validation;
the allocator retains validation cache, affecting later reserved readings.
Combined's reserved peak is122MiB above the saved parent's452MiB. We therefore
cannot claim the requested same-VRAM constraint passed, despite unchanged model
size and slightly lower logged training allocated memory. No additional resource
probe was run after the user's waiver.

Saved best combined stress recovery remains low: uniform256 **4/64** exact and
pattern256 **13/64**. Span64 means up to64 tokens per vector, not byte compression:
the full validation stores1897 width256 BF16 vectors (971,264 bytes), plus8000
length bytes, versus328,100 original UTF-8 bytes.

Completion verified source/parent/checkpoint/export/record hashes, all8000 updates
and16 full-validation points per arm, matched target schedules, full final and
selected checkpoint replays, and exact exported-encoder parity. An independent
standard-library-only saved-evidence rehash passed; it did not rerun training,
tokenization, or model evaluation. Preflight, repeated decoded identities and
initial validation remain waived, not passed. No new worker/native failures
occurred; earlier preparation failures below remain unresolved and preserved.

Artifacts: standalone `artifacts/runs/compression64-variations-8000-v1/`.
Winner checkpoint: `combined/best.pt`; encoder: `combined/encoder.pt`.
These use the layout-aware loaders in frozen `dump/compression64-variations/run.py`;
they are not silently installed over the selected span32 package weights.
Persistent records: ignored `records/compressor64-variation-{kind}-v1.json`.
Controller receipt: `completion_verification.json`; independent receipt:
`dump/compression64-variations/independent_completion_verification.json`.
All workers and locks have finished. No further training is authorized.

## Latest authorization: direct launch, checks waived

**Launch observation (historical):** all five workers had advanced beyond300 updates, with no new worker
failures at this observation. Budget is8000 additional updates each. GPU total
was3080/8188MiB at99% utilization. Training-only allocated peaks were422.04-422.11MiB,
reserved456-458MiB; later validation can raise these. Final memory compliance and
quality improvement were not yet established. Read saved progress with
`powershell -NoProfile -File D:\Git\complex_fnn\dump\compression64-variations\progress.ps1`.

The user explicitly said to skip checks and start training. The five x8000-update
budget remains unchanged. The direct runner is `dump/compression64-variations/run.py`;
output/status is standalone `artifacts/runs/compression64-variations-8000-v1`.
Separate preflight, repeated full re-tokenization and initial full-validation
replay are waived. This is not a passing result for those gates. Each worker uses
the existing cached token IDs and verifies their file hashes against the historical
dataset manifest, plus saved parent and source hashes. Training uses existing
thread limits and `uv run --no-sync`; no environment changes.

Actual full validation remains every500 updates. Both final and best checkpoints
are preserved and replayed, with explicit attention-layout metadata in checkpoints
and encoder exports. Parameter counts/budget and finite gradients remain checked.
Actual memory is reported rather than assumed to pass the waived resource gate.
No automatic retry if a worker fails again. All earlier failure records below are
retained. This latest direct user instruction supersedes the earlier launch block.

## Preparation history

The latest user explicitly requested variations and immediate parallel training,
including linear attention if compatible. Budget declared using the preceding
proposal: five variants x8000 additional updates each, at most five workers.
All start from the saved729/1000 ordinary-continuation winner. No controls are
retrained and no additional budget is authorized beyond this batch.

| Variant | Encoder token attention | Coverage | Difficulty /margin |
|---|---|---|---|
|combined|Existing softmax|Yes|0.25 /0.25|
|hard_fade|Existing softmax|Yes|0.25 /0|
|margin_fade|Existing softmax|Yes|0 /0.25|
|linear_tail|Softmax first2 blocks, linear last2|Yes|0.25 /0.25|
|linear_full|Linear all4 encoder blocks|Yes|0.25 /0.25|

All retain learned residual attention across encoder depth, Branch Sigmoid FFNs,
RoPE, factorized ordered packing, and the original softmax decoder. Coverage uses
the same shuffled queues across variants; losses fade to ordinary CE during the
final25% of the budget. Planned seed17, lr3e-5, microbatch4 x accumulation4,
context256/span64/width256. Parameters are exactly7327232 encoder/10456576 full
for every variant. No inference ensemble or checkpoint averaging.

Linear attention is mathematically compatible with the existing architecture:
residual attention mixes depth outputs, whereas this change mixes token features.
The implementation follows the RoFormer linear-attention construction: ELU+1
query/key features, RoPE after the feature map in the numerator, and a positive
unrotated normalization denominator. Associating K-transpose with V avoids the
token-by-token matrix. Existing Q/K/V/output projections and Parameter objects
are reused, preserving parameter count and optimizer mapping. This is a known
attention construction, not a newly discovered method. It changes the model's
function immediately despite transferring identical weights. Initial full
validation was waived; the first measurement is at500 updates, and quality may be
worse. Short-context speedups are not
assumed. Sources: [RoFormer section3.3](https://arxiv.org/html/2104.09864v5#S3.SS3)
and [Linear Transformers](https://proceedings.mlr.press/v119/katharopoulos20a.html).

Implementation and batch specification are under
`dump/compression64-variations/`; the combined loss remains under
`dump/compression64-combined/recipe.py`. Decoder checkpoints/encoder exports must
persist and enforce the attention layout in the current runner;
silently loading a linear variant as the old softmax architecture is invalid.

## Checks and failures

1. The user-authorized retry of the full decoded-identity diagnostic failed again
   before model evaluation: tokenizers.pyd0xc0000005, PID4064,
   2026-10-07 15:11:13PDT. No research updates. Evidence:
   `dump/compression64-combined/authorized-retry.log` and matching native events.
2. Independent CPU tensor checks passed for linear-attention outputs and gradients
   against an explicit quadratic reference, masked-key exclusion, same Parameter
   objects/state keys, and causal-call rejection. These checks never call a
   tokenizer or load data and do not replace the failed identity gate.
3. Independent combined-loss tests and synthetic full-model forward/backward
   checks passed for all five, with identical counts and finite gradients. No
   optimizer steps. Uncheckpointed allocated/reserved MiB:
   combined419.65/450, hard419.85/462, margin419.85/462,
   linear_tail431.39/474, linear_full442.93/484. Linear variants exceed the saved
   winner's426.93MiB allocated peak; do not call the memory constraint satisfied.
   These synthetic values also omit an optimizer step and final validation.
4. Added ordinary PyTorch non-reentrant activation recomputation inside linear
   attention to reduce stored intermediates. A separate synthetic-only check
   then failed natively before producing test output: ucrtbase.dll0xc0000409,
   PID9076,2026-10-07 15:16:52PDT. This did not rerun the failed tokenizer and took
   zero optimizer steps. The revised implementation's memory/gradient checks are
   unverified. Earlier passing source/receipts are preserved separately.

Training/validation/tokenizer file SHA256 values still match the original saved
manifest. This rules out an on-disk dataset change in those files, but does not
explain the native failures. Older CPU hardware-error events are recorded but
are not established as the cause. No environment, dependencies, system settings,
datasets, selected package sources or saved weights were changed. No automatic
retry occurred before the latest direct-launch authorization. All check processes
exited; the subsequently authorized training workers are described above.

**Historical status before the latest waiver: zero research updates; training not launched.** The new budget
is authorized, but runtime stability and complete independent decoded checks
remain blockers. Correctness of the final memory revision, full-model memory,
initial full validation, checkpoint/export roundtrip and frozen-source gates must
pass before launch under the earlier instructions. The latest user waived the
separate prelaunch checks as documented above. No pass is inferred from the
successful historical batch.
