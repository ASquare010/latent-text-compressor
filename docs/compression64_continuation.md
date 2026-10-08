# Residual-attention compression64: bounded continuation

The user authorized 8000 longer training updates and parallel improvements using
the residual-attention encoder as the current span64 base. This means **8000
additional updates per arm**, starting from the completed 3000-update depth_route
checkpoint (which itself inherited 7000 span32 curriculum updates). Five arms,
at most five concurrent workers; no further budget extension or automatic retry.

All five have exactly **7,327,232 encoder /10,456,576 full parameters**. Context256,
span64, width256 BF16 vectors, RoPE, Branch Sigmoid, four encoder blocks and one
decoder block remain unchanged. The original successful span32 model remains the
selected reliable compressor; this batch does not replace it automatically.

## Hypotheses and controlled differences

| Arm | Change from continued depth_route | Hypothesis and risk |
|---|---|---|
|continue|None; ordinary cross-entropy and original sampling stream|The new bottleneck needs more training. Validation may still plateau or regress.|
|coverage|Shuffle each training source without replacement|Expose every training row before repeating within that source; fewer redundant draws may help. Changed sample order/lengths can also hurt.|
|hard_tokens|Bounded, detached difficulty weighting of cross-entropy|Give uncertain training tokens more influence while retaining easy-token learning. May worsen probability calibration or amplify outliers.|
|coverage_hard|Coverage plus difficulty weighting|Test whether broad exposure and mistake emphasis complement each other; gains may fail to combine.|
|margin|Subtract1 from the correct token's logit during training before cross-entropy|Sparse reconstruction mistakes may need larger classification margins. This can distort probabilities or waste effort on already correct tokens.|

These are **training variations on the same architecture**, not five new encoder
architectures. Their inference compute, parameter count and vector format are
identical. Training memory and speed must be measured; parameter equality does not
imply equal training VRAM or identical token counts.

Difficulty weighting is inspired by [Focal Loss](https://arxiv.org/abs/1708.02002),
which addresses easy-example dominance in object detection. Here the deliberate
adaptation is w=1+0.5*(1-p_correct)^2, detached from gradients and normalized to
mean1 over real tokens in each microbatch. It does not suppress ordinary CE and
is not standard focal loss or established compression evidence.

[Random reshuffling](https://proceedings.mlr.press/v119/rajput20a.html) studies
sampling without replacement; its theoretical results do not guarantee gains for
this Transformer with AdamW. The coverage arms preserve the exact original
natural/uniform/pattern source-choice sequence while replacing within-source
choices by shuffled queues. The margin arm computes CE after subtracting1 from
each correct target's logit, making that target harder to classify during training.
Inference logits and validation CE remain ordinary. All three add-ons are
unverified hypotheses here.

## Fixed protocol

- Same verified depth_route parent weights, AdamW moments and step3000 state for
  all arms; no restart from random weights or from the span32 parent.
- Constant learning rate3e-5, weight decay0.01, gradient norm limit1, BF16,
  microbatch4 x accumulation4, seed17 sampling continuation. Hard/margin arms
  deliberately change the training objective; coverage arms deliberately change
  target rows. Ordinary unweighted training NLL is logged separately in every arm.
- Same frozen 50k natural,25k uniform,25k patterned training pool. Baseline/hard/
  margin share exact targets; coverage/coverage_hard share exact targets. All five
  share exact source-type schedules, but differing lengths mean token totals and
  compute are not exactly matched across sampler groups.
- Full1000-paragraph validation before training and every500 updates. Every worker
  independently checks all decoded train/validation identities before any update.
- Preserve last and best-validation checkpoints and their optimizer/sampler/RNG
  states. Best is ranked by exact paragraphs, then token accuracy, then NLL. Final
  and best results are both reported, with full replay and stress diagnostics.
- Training source/model/data hashes are pinned. Completed historical sources and
  weights are untouched. Correctness/optimizer restoration and maximum-length
  CUDA probes precede launch. Discarded smoke steps are not research updates.
- Existing native failures remain unexplained. A new native/data-integrity failure
  stops that worker and is recorded; no automatic retry or decoded-check bypass.

Status and artifacts:
`D:/Git/latent-text-compressor/artifacts/runs/compression64-continuation-8000-v1`.
Sources: `D:/Git/complex_fnn/dump/compression64-continuation`.
Preflight: standalone `artifacts/runs/compression64-continuation-preflight-v2`.
Persistent records: ignored `records/compressor64-continuation-{arm}-v1.json`.

Completion requires all five complete8000-update sequences, target/source/parent/
checkpoint/export/record hashes, independent encoder parity, saved parent replay,
full final and selected-validation replay, and a completion verification receipt.
One inherited seed and repeated validation selection do not establish universal
losslessness. The reserved test is not used for selection. Longer training is a
training-budget gain, not an architectural gain. No claim that100% is guaranteed.

## Completed results

All five finished8000 additional updates, for11000 span64 updates after the shared
7000-update parent curriculum. All17 validation checkpoints per arm were recorded.
The controller verified complete update sequences, target/source schedules,
parent/source/checkpoint/export/record hashes, full final and best validation
replays, and encoder-only parity. An independent completion check rehashed all
saved artifacts and frozen/current sources and confirmed the full update and
validation sequences. No workers or process-owned locks remain; GPU usage is zero.

| Arm | Best exact /1000 | Best additional update | Final exact /1000 | Final token accuracy | Final NLL | Training allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
|continue|729|8000|729|99.6606%|0.041006|426.93/452|
|coverage|723|6000|707|99.6217%|0.043568|422.71/452|
|hard_tokens|715|5000|690|99.5936%|0.045444|426.93/452|
|coverage_hard|729|6000|728|99.6638%|0.044175|422.71/452|
|margin|708|8000|708|99.6314%|0.038694|426.93/452|

The predeclared ranking (exact paragraphs, then token accuracy, then NLL) selects
**continue at8000**. Its729 exact paragraphs tie coverage_hard's selected6000
checkpoint, but its token accuracy is higher (99.6606% versus99.6563%) and NLL is
lower (0.041006 versus0.059339). At the common8000 endpoint, coverage_hard has one
fewer exact paragraph but three fewer wrong tokens (311 versus314); margin has
the lowest NLL. Thus there is no winner on every metric, and the add-ons do not
establish a clear exact-recovery improvement over longer ordinary training.

Compared with the starting depth_route model, ordinary continuation improves
557 to729 exact paragraphs,99.2769% to99.6606% token accuracy, and0.166150 to
0.041006 NLL. Wrong tokens fall from669 to314 across92516 validation tokens.
Parameters remain exactly7327232 encoder/10456576 total, context256/span64/width256.
Measured allocated memory increases slightly from425.54 to426.93MiB (+1.39MiB),
while reserved memory decreases from486 to452MiB. The original strict allocated
envelope passed in the probe but is slightly exceeded by actual full training;
do not claim the final run used identical or lower allocated VRAM.

The ordinary continuation's final validation NLL keeps falling:0.056672,0.051459,
0.049802,0.045703,0.041006 at updates6000,6500,7000,7500,8000. Exact counts fluctuate
690,705,690,693,729, so lower NLL is not a reliable substitute for complete recovery.
The coverage arms visited all100000 training rows but did not clearly win. All
arms had the same source draw counts64187/31763/32050; other arms visited36074/
18070/18057 distinct rows. These are continuation-stage counts only.

Winner stress results at256 tokens: uniform2/64 exact,99.0051% token accuracy;
patterned9/64 exact,99.0906% tokens. None meets100% recovery. The saved span32
model remains1000/1000 on the same natural validation and64/64 on both stress
sets. This remains one inherited seed and previously used validation, with
checkpoint selection; the reserved test was not scored. Sequence reduction is
not byte compression. No new worker/native/data failures occurred; prior native
failures and the discarded memory-rejected preflight remain preserved. No extra
training or automatic model replacement is authorized by completion.

Completion receipt: standalone
`artifacts/runs/compression64-continuation-8000-v1/completion_verification.json`.
Selected checkpoint: `continue/best.pt` (same update as `continue/last.pt`), with
the corresponding `continue/encoder.pt` export. Other arms' best and final
checkpoints and all persistent records are retained.

## Launch history

The guarded controller has launched all five workers. Each worker must complete
its independent decoded-data checks and replay the saved557/1000 parent before
its first update. Source files are now frozen. No additional batch is authorized.
Launch verified: all five independently replayed557/1000 exact and NLL0.166150,
then advanced beyond100 updates without failure. Early training allocated peaks
422.16-425.50MiB and reserved450MiB per worker; GPU-wide usage about3044MiB.

The first probe passed correctness and optimizer restoration but rejected the
original separate hinge-margin objective:436.65MiB allocated versus the previous
winner's425.54MiB. The other four were419.65-419.85MiB. Sources and failure evidence
are preserved. Before launch, replaced the fifth hypothesis with an additive
target-logit margin using one CE gradient path. This is a documented objective
change, not a relaxed memory bound. Two probes used five discarded optimizer
steps each (ten total); all candidate checkpoints remain unchanged.

Final probe measurements: continue419.65/450MiB allocated/reserved; other four
419.85/462MiB. All fit the previous winner's425.54/486MiB training envelope in this
probe. Actual training peaks are logged separately; these are not promises about
the final measured peak. No new native or data-integrity failure in preflight.

To check progress from the main repository:

```powershell
& D:\Git\complex_fnn\dump\compression64-continuation\progress.ps1
```
