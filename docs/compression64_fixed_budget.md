# Fixed-size compression64 study

User requested architecture variations that compress more tokens without growing
the model. Five declared candidates, all span64 with width256, four encoder blocks,
one decoder block, RoPE, Branch Sigmoid, and BF16 latent vectors. Parameter caps:
7,342,336 encoder /10,488,832 full, the measured span32 parent's counts.
Context stays256. A longer context alone does not increase each vector's capacity;
changing it would also change training inputs/compute and obscure this comparison.

## Hypotheses

| Candidate | Change | Expected full /encoder parameters |
|---|---|---:|
|compact_linear|Shared256-to125 feature projection per token, ordered64-token packing; reversed factorization in decoder|10,454,528 /7,325,184|
|depth_route|Compact bottleneck plus learned input-dependent attention over encoder branch outputs|10,456,576 /7,327,232|
|nonlinear_pack|Compact bottleneck with h+0.5*tanh(h) on both sides of latent mapping|10,454,528 /7,325,184|
|depth_nonlinear|Combine residual attention with nonlinear packing|10,456,576 /7,327,232|
|iterative_decode|Compact bottleneck; reuse the same decoder block for three scaled refinement passes|10,454,528 /7,325,184|

These are expected analytic counts, to be measured by preflight. No unused padding
parameters. Same new bottleneck matrices/initialization across candidates. All
retained backbone weights come from the same verified7000-update span32 parent.
The decoder sees only vectors and length metadata; no encoder skip path or tokens.
Latent count is ceil(tokens/64), width256 and BF16 storage, unchanged across arms.

Depth attention is inspired by Moonshot/Kimi's Attention Residuals:
https://arxiv.org/abs/2603.15031
https://github.com/MoonshotAI/Attention-Residuals
Our adaptation multiplies attention-weighted branch sums by the number of branches
so zero queries recover ordinary residual summation, easing parent transfer. It is
not an exact reproduction or a proven improvement for compression. Uses queries
on normalized branch values, not extra attention across the token sequence.

Perceiver IO offers a broader precedent for latent bottlenecks and queried decoding:
https://arxiv.org/abs/2107.14795
The current candidates use ordered learned packing rather than Perceiver pooling.
VAE-style stochastic sampling or a KL penalty is not added: exact retention is the
objective, and an additional information constraint would require separate evidence.

## Fixed budget and measures

Five x3000 additional updates, seed17, same targets/mixture/AdamW/cosine schedule,
five concurrent workers once preflight passes. 50k natural +25k uniform +25k pattern
training rows. No saved model retraining and no test-split selection. The compact
linear model is a new budget-matched architecture control; the earlier span64 model
had larger projections and2000 updates, so it is only a historical reference.
Shared decoder reuse costs additional compute despite constant parameters; report
time and allocated/reserved VRAM rather than infer equal resource use.

Full1000-paragraph validation, token accuracy/NLL, length bins, fixed256-token
uniform/pattern stress; actual vectors and bytes, including lengths. Success needs
1000/1000 exact AND100% token accuracy, with stress results reported separately.
No universal-capacity or lossless-storage claim from one seed and selected validation.
All exports/checkpoints/source/parent/target hashes, final full validation replay,
encoder-only parity and persistent records are required. No budget extensions.

## Current status: blocked before training

The first correctness/parameter/memory preflight crashed before producing results.
Windows Event1000, 2026-10-07 12:54:24 PDT: python312.dll access violation0xc0000005,
PID25644. No Python traceback, no trained candidate, zero research updates.
Existing thread limits were enabled. This new failure is not bypassed/retried.
Earlier CPU parity errors and unrelated native crashes remain unexplained.

Prepared implementation: D:/Git/complex_fnn/dump/compression64-study/model.py
Prepared guarded runner: D:/Git/complex_fnn/dump/compression64-study/train.py
Preflight source/failure snapshot and event evidence are retained in that directory.
Future output: D:/Git/latent-text-compressor/artifacts/runs/compression64-fixed-budget-v1.
The controller requires a passing preflight tied to the exact model-source hash;
it has NOT been launched. Existing selected model/package sources remain unchanged.

## Authorized launch retry: preflight passed, five workers launched

The user explicitly requested training again. No workers/output from an earlier
attempt existed. Preserved the original failed preflight and native failure evidence.
The new check exposed3/532480 FP32 logits outside the old1e-5 absolute tolerance
(max absolute error1.90e-5) when uniform depth attention changes summation order.
Added an independent sum-identity check against FP64 accumulation, and changed
only the end-to-end absolute tolerance to3e-5. No model/training change. Exact
checkpoint recovery, exact parent transfer, encoder-only parity, padding, gradients,
parameter caps and CUDA optimizer smoke tests all passed for all five models.

Measured encoder/full counts:7,325,184/10,454,528 for compact_linear,
nonlinear_pack and iterative_decode;7,327,232/10,456,576 for depth_route and
 depth_nonlinear. All remain below the selected span32 model's counts.
Single-worker allocated/reserved probe MiB: linear210.45/236, depth286.40/334,
nonlinear218.67/236, combined286.89/334, iterative254.26/274. These are preflight
probes, not actual training peaks. Sources are frozen at launch. Five x3000
updates, seed17, five workers, each retaining complete decoded data checks.
Earlier native failures are not claimed fixed; no automatic retry on a new failure.

Live status: D:/Git/latent-text-compressor/artifacts/runs/compression64-fixed-budget-v1/status.json
Per-candidate status/training.jsonl are in named child directories. Completion
requires full final validation replays, source/target/checkpoint/export checks and
persistent records; no further training is authorized by this run.


## Completed result

All five completed3000 transfer updates. All final full-validation replays,
checkpoint/export/record/parent/source hashes, matched target schedules and complete
update sequences verified. Both parameter caps passed for every model.
Completion receipt: artifacts/runs/compression64-fixed-budget-v1/completion_verification.json.

| Model | Exact paragraphs /1000 | Token accuracy | Reconstruction NLL | Encoder/full parameters | Allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|
|depth_route|557|99.2769%|0.166150|7,327,232 /10,456,576|425.54 /486|
|compact_linear|533|99.2434%|0.172380|7,325,184 /10,454,528|354.31 /412|
|iterative_decode|503|98.8099%|0.193441|7,325,184 /10,454,528|390.72 /452|
|depth_nonlinear|493|99.0985%|0.187458|7,327,232 /10,456,576|426.02 /486|
|nonlinear_pack|472|99.1266%|0.192875|7,325,184 /10,454,528|354.80 /412|

Depth routing is the best candidate:557/1000 exact,99.2769% token accuracy,
0.166150 NLL. Its gain over the matched compact-linear control is modest:
24 additional exact paragraphs and31 fewer erroneous tokens across92516 tokens.
All64-token candidates fail the exact-recovery goal. Retain the selected32-token
model (1000/1000,100% token accuracy). The attention-residual winner is smaller
in parameter count but uses425.54/486MiB versus saved32's379.44/470MiB; no claim
of equal VRAM. Nonlinear packing and decoder reuse did not beat the linear control
on exact paragraphs, token accuracy or NLL. Combined routing/nonlinearity also loses.

Winner stress token accuracy:97.9126% uniform /97.1985% patterned256-token samples;
exact0/64 uniform and3/64 patterns, versus64/64 for both with saved32.
1897 vectors versus saved32's3422 on validation (44.6% fewer); vector+length
storage979264 bytes exceeds328100 UTF-8 bytes, excluding model/container costs.
This is sequence compression, not a smaller lossless archive.

One seed, previously examined validation, and3000-update transfer budget limit
conclusions. Earlier span64 used larger projections and2000 updates, so its lower
accuracy is not a clean architecture comparison. Reused-decoder compute differs.
No reserved test selection or new training after this batch. The final run had no
new worker failures; earlier native crashes and tolerance correction remain recorded.
