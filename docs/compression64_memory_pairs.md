# Residual-attention combinations under a memory ceiling

## Reviewed outcome: four completed, coverage rejected

**Residual attention + margin has the best saved exact-recovery checkpoint:
806/1000 at update7500.** Four arms completed8000 updates. Coverage was stopped
by the declared memory guard and is excluded from the completed comparison.
The controller's failed status accurately preserves that incomplete fifth arm.

| Arm | Best exact /1000 | Best update | Best token accuracy | Best NLL | Final exact /1000 | Final NLL | Training allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
|Plain residual attention|782|7000|99.7309%|0.019919|**781**|0.018927|423.39 /474|
|+ Difficulty|799|7000|**99.7557%**|0.019997|777|0.018541|423.39 /474|
|+ Margin|**806**|7500|99.7536%|**0.018572**|773|**0.018324**|423.39 /474|
|+ Difficulty and margin|787|7500|99.7384%|0.019333|771|0.018759|423.39 /474|
|+ Coverage, incomplete|763*|4500*|99.7049%*|0.024620*|—|—|423.85 /476: rejected|

*Coverage values are saved intermediate validation, without final/best replay or
encoder export. Coverage received only4869 optimizer updates, so its quality is
not a matched8000-update comparison. Do not interpret it as a completed loss.

All four completed models satisfy the declared5% cap: allocated448.2772MiB and
reserved474.6MiB. Their measured training peak is423.39/474MiB, validation
238.95/378MiB and export parity152.04/172MiB. Compared with the original saved
426.93/452MiB training result, reserved memory is22MiB higher (+4.87%), within
the declared ceiling. Compared with the previous574MiB batch, it is100MiB lower.
No new parameters or inference activation types were added: encoder7,327,232 /
total10,456,576, context256/span64/width256 throughout.

Margin's selected checkpoint gains24 exact paragraphs over this matched plain
control and77 over the older729/1000 parent. The older parent has fewer training
updates. **At the fixed8000 endpoint, plain is best (781), and margin drops to773.**
Thus the result supports retaining margin's selected checkpoint, not claiming
a stable endpoint improvement. Difficulty has two fewer wrong tokens than margin
at its selected checkpoint (226 versus228), despite seven fewer exact paragraphs.
One inherited seed, reused validation and checkpoint selection limit the ranking.
No fresh test was evaluated, and100% recovery is still unmet.

Margin's selected stress results remain weak: uniform2561/64 exact and
pattern25613/64. The reliable selected span32 package is unchanged. Winner artifacts:
`D:/Git/latent-text-compressor/artifacts/runs/compression64-memory-pairs-8000-v1/margin/best.pt`
and `margin/encoder.pt`, using this study's layout-aware checkpoint/export loaders.

Coverage's guard stopped at reserved476MiB, **1.4MiB above the ceiling**, with
allocated423.85MiB still below its limit. This was an intentional resource stop,
not a native crash. Update4869 had been applied before the guard raised;4868
updates were logged. Its partial checkpoints and failure evidence are preserved,
and it was not restarted. Total research updates applied:36,869, plus five
discarded synthetic probe updates. No further training is authorized.

The four completed runs passed saved source/parent/checkpoint/export/record hashes,
all8000 updates and16 validation points, identical target/source schedules, full
final/best replays, encoder parity, and every measured memory phase. Independent
standard-library saved-evidence verification also passed; it performed no new
model evaluation or training. Receipt: `dump/compression64-memory-pairs/reviewed_results.json`;
registration: ignored `records/compressor64-memory-pairs-summary-v1.json`.
The partial-result receipt does not mark the full five-run batch complete.
Old decoded-identity/initial-validation waivers remain disclosed below. No new
native training failure occurred. Failed controller/coverage owners are no longer
alive; their stale locks are retained as failure evidence. GPU is idle.

## Original authorization and protocol

The user requested simpler paired runs with difficulty weighting, margin, and
explicitly residual attention + coverage, seeking the best score at nearly the
original memory use. This authorizes five parallel continuations, each capped at
8000 additional updates. No old run is overwritten; no automatic worker retry or
budget extension is allowed.

## Controlled comparison

| Arm | Sampling | Difficulty | Target-logit margin |
|---|---|---:|---:|
|plain|Original parent stream|0|0|
|difficulty|Same exact stream as plain|0.25|0|
|margin|Same exact stream as plain|0|0.25|
|combined|Same exact stream as plain|0.25|0.25|
|coverage|Shuffle within each source without replacement|0|0|

Every arm retains residual/depth attention, Branch Sigmoid FFNs, RoPE, softmax
token attention and factorized ordered packing. Encoder parameters **7,327,232**,
total **10,456,576**, context256, span64 and width256 are unchanged. No linear
attention arms or model size increases. Difficulty/margin fade to zero over the
last25% of training using the previously verified recipe. Coverage changes row
order; it preserves the source-category schedule but can change token totals.
The other four arms have identical row and source schedules.

All start from the same saved729/1000 `continue/last.pt`, SHA256
`d3f9570dab29a989d6242bd3c19c004f801ccb681c20200dcc198c3a1dbae4e4`,
including its AdamW moments/step11000 and RNG. Starting ancestry is18,000 updates;
final ancestry26,000. Learning rate3e-5, weight decay0.01, gradient clipping1,
BF16, microbatch4 x accumulation4, inherited seed17 and thread limits1 remain.
The plain arm is a newly authorized matched continuation control; it resolves
the previous batch's missing training-duration comparison. The new combined arm
does not include coverage, unlike the previous788/1000 recipe.

## Memory finding and change

Prior combined logs show458MiB reserved at update500 and574MiB immediately after
the first validation (update501), while allocated training peak stays422.11MiB.
The evaluator retained the previous batch's logits/vectors while starting the
next batch, and cached allocations persisted into training. This evidence points
to evaluation/cache behavior rather than added model parameters.

A study-local evaluator now releases completed batch tensors before processing
the next batch. Evaluation order, microbatch8, arithmetic, data and metrics stay
unchanged. Every arm clears unused allocator cache at training/evaluation phase
boundaries after gradient release. No selected package or historical source file
was changed. This can affect speed; no equal-speed claim is made.

"Nearly the same" is predeclared as at most5% above the saved parent's
426.9307MiB allocated /452MiB reserved: **448.2772 /474.6MiB**. Training and
validation peaks are measured separately, including full validation. Crossing
either limit stops that worker; a failed or memory-rejected arm is not silently
restarted or presented as successful. Final replay/export memory is also checked.
These per-process PyTorch figures exclude driver/context overhead; they are not
the five-worker total shown by nvidia-smi.

Five fresh-parent probes passed, with one discarded synthetic optimizer update
per arm (five discarded updates total, zero research updates). Training allocated
419.65–419.85MiB / reserved450–462MiB; validation allocated238.83–238.95MiB /
reserved338MiB. Plain-model original versus cleaned evaluation on the same first16
validation rows was identical. Probes use maximum-length synthetic batches and
do not guarantee complete-run peaks; the runtime ceiling remains enforced.

Existing cached token IDs and current dataset/parent/source byte hashes are used;
the earlier repeated full decoded-identity and initial-validation waiver is
retained for this follow-on workflow. Those gates are not claimed as passed.
Native failures from earlier work remain unexplained. A read-only PowerShell
inspection also exited0xc0000374 during preparation; it performed no model or
optimizer work. No native failure occurred during these five CUDA probes.

## Status and completion requirements

**Historical launch observation:** all five workers advanced through their first optimizer update,
with no launch failures. Controller PID24932 (uv launcher24856); total GPU
3106/8188MiB at the initial observation. These are launch facts, not final resource
or quality results. Live status is recorded in
`D:/Git/latent-text-compressor/artifacts/runs/compression64-memory-pairs-8000-v1/status.json`.
Study sources/receipts: `dump/compression64-memory-pairs/`.
Persistent records: ignored `records/compressor64-memory-pair-{arm}-v1.json`.
Frozen sources are copied into the run directory at launch. No source changes
are permitted during training.

Full1000-paragraph validation every500 updates; retain final and best checkpoints.
Completion requires all update/validation sequences, source/parent/checkpoint/
encoder/record hashes, source schedules, target schedule equality except coverage,
full final/best replays, exported encoder parity and all measured memory phases
within the ceiling. Rank eligible models by exact paragraphs, then token accuracy,
then NLL; report final separately and include losses. One inherited seed and
reused selected validation remain limitations. Perfect recovery and byte
compression are not established by this study. The selected span32 model stays.
