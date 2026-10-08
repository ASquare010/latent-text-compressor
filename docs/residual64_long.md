# Plain residual64: 10,000 additional updates

User authorized only the10.46M model for10,000 additional updates. Resume final
957/1000 short,151/206 packed checkpoint from residual64-growth-4000-v1/continue_plain.
Parent SHA256:7c72c9e2a69b361225e998c36feb122aa59a4591103eadffcd3970bc9bd2b3e3.
34,000 cumulative updates before this run;44,000 if completed. No larger models.

Same10,456,576total/7,327,232encoder params,4encoder/1decoder,plain residual depth
attention,Branch Sigmoid,RoPE,width256,context512,span64,code125. Exact original
computational sources and every parent tensor verified before training. Restore
AdamW moments and step counters,Python sampler and torch CPU/CUDA RNG. Constant
lr3e-5,weight decay.01,clip1,micro2xaccum4. Existing25%short/25%packed/25%uniform/
25%pattern schedule continues from its saved sampler state, not reset seed43.
No new capacity,context,data or margin modification. More training does not prove
that encoder information capacity was the limiting factor.

One worker pinned CPU16;uv run --no-sync;no environment changes. Existing decoded
identity/source/parent/finite-gradient/memory guards remain,1300/1500MiB caps.
Previous separate resource probes and initial full validation remain waived,not
passed. No additional optimizer probes. Prior native failures remain unresolved;
no automatic retries of new native/data failures. Bounded metadata rename retries
handle only Windows PermissionError,not model updates.

Every500 updates evaluate all206 packed and1000 short examples;save last every250,
best by packed exact then token recovery/NLL. Full final/best replay,encoder parity,
source/checkpoint/record hashes and complete10000-step sequence required. Optimizer
step counters must end14000 (4000 inherited since its last reset),not10000. Report
both final and selected results against saved parent. Same reused validation,
one inherited initialization;no untouched test evaluation,100% not guaranteed.

Sources dump/residual64-long-v1 freeze at launch. Output standalone artifacts/runs/
residual64-long-10000-v1. Progress: dump/residual64-long-v1/progress.ps1.
Prior checkpoints retained;product default unchanged. No automation.
Status: prepared for launch.

## Running

Worker7312 verified at100/10000 additional updates. Resume receipt confirms full
AdamW state(step4000),sampler and torch CPU/CUDA RNG restored. Complete decoded
identities and exact model/source/parent checks passed before updates. Early peaks
423.08/470MiB allocated/reserved;total GPU629/8188MiB. No launch failure. These are
not final memory or quality results. Training sources frozen;no automatic retry.

## Complete: 2026-10-08

All10,000 additional updates completed;44,000 cumulative. Final and packed-selected
best both occur at10,000. No worker failures.

| Metric | Before | Final |
|---|---:|---:|
| Short exact paragraphs |957/1000|999/1000|
| Packed exact sequences |151/206|202/206|
| Short token recovery |see parent record|99.9989191%|
| Packed token recovery |see parent record|99.9957132%|
| Short NLL |0.005691557|0.001107957|
| Packed NLL |see parent record|0.001151862|

Unchanged10,456,576total/7,327,232encoder parameters;context512,span64,width256.
Training peak426.7817MiB allocated/474MiB reserved. Summed training-update time
50.22minutes excludes validation/preparation. GPU idle after completion.

Controller verified full final/best validation replay,export parity and hashes.
Independent pinned read-only saved-evidence rehash verified result,final/best/
encoder,persistent record,frozen source archives and10,000 consecutive updates.
No new model reevaluation or retry of the old failed audit script was performed.

One short paragraph and four packed sequences still fail exact recovery. This is
not100% or universal losslessness. One inherited initialization and reused selected
validation;short/packed share source text. No untouched test evaluation. More
training clearly improved this run;it does not prove an encoder-capacity mechanism.
No further training authorized by completion and no inference checkpoint promotion.
