# One combined residual-attention compressor recipe

The user requested one model combining the useful lessons of the completed runs,
within the existing parameter, context, compression and memory constraints.

## Selection and rationale

Use the ordinary-continuation winner's final checkpoint as the starting weights:
729/1000 exact paragraphs,99.6606% token accuracy,0.041006 NLL. All five previous
continuations already used residual attention. Coverage, difficult-token emphasis
and logit margin are training rules, not additional inference branches. Their
benefits cannot be imported just by enabling switches on already trained weights,
and this proposal does not average checkpoints or run an inference ensemble.

The final coverage_hard model has311 wrong tokens versus314 for ordinary
continuation, but728 exact paragraphs versus729. Margin reduces NLL to0.038694
but has708 exact paragraphs. These mixed outcomes do not prove the ingredients
will combine beneficially. Keep the current729/1000 checkpoint as the reference.

The single prepared recipe is:

1. Retain the winning residual-attention encoder, Branch Sigmoid FFNs, RoPE and
   factorized ordered span packing. Keep exact parameter counts7327232 encoder /
   10456576 full, context256, span64, width256 BF16 vectors.
2. Resume that checkpoint's optimizer and source-selection RNG. Use shuffled
   within-source queues to cover training examples without replacement, preserving
   the existing natural/uniform/pattern source-choice stream.
3. Apply bounded difficulty weighting1+0.25*(1-p_correct)^2, normalized to mean1
   over real tokens within a microbatch. Weights are detached and computed from
   ordinary unshifted logits. This is milder than the previous0.5 coefficient.
4. Subtract0.25 from the training target logit before CE, versus the previous1.0
   margin. Combine both additions in one retained CE gradient path. Ordinary NLL
   is logged separately; inference and validation use unmodified logits.
5. Fade both additions linearly to zero over the final25% of a future declared
   budget. The final objective exactly recovers ordinary cross-entropy. This
   finishing schedule is a new hypothesis, not an established improvement.

The lower strengths and fade are chosen because the stronger separate objectives
did not improve complete reconstruction. They aim to reduce conflicting emphasis
while preserving the strongest model's ordinary CE behavior. Parameters and
inference work are unchanged; training memory still needs measurement. There is
no evidence yet that this combined recipe beats its parent.

Implementation: `dump/compression64-combined/recipe.py` (loss, scheduling, verified
parent/optimizer loading, coverage sampler). Prepared tests:
`dump/compression64-combined/test_recipe.py`. Preparation receipt:
`dump/compression64-combined/prepared.json`. No training controller was launched,
no new trained checkpoint exists, and no additional update budget is authorized.

## Diagnostic failure and current status

A read-only per-token error-overlap diagnostic crashed while verifying decoded
dataset identities, before evaluation of any model and before any update. Windows
Application Event1000 at2026-10-07 14:55:42 PDT reports tokenizers.pyd access
violation0xc0000005, PID11492. Existing thread limits were enabled. This is a new
native failure, not evidence that the combined loss is wrong or that saved data
changed. Root cause remains unresolved. No automatic retry or data-check bypass.

Evidence: `dump/compression64-combined/inspect.log` and `native-events.json`.
The winner checkpoint was independently SHA256-checked after the failure and
still matches its audited record:
`d3f9570dab29a989d6242bd3c19c004f801ccb681c20200dcc198c3a1dbae4e4`.
The diagnostic has exited. Existing model/package sources, all saved weights,
and data were not modified.

The recipe and tests were prepared after that failure but have not executed.
Correctness, gradients, decoded identities, initial full-validation replay,
optimizer/encoder parity and allocated/reserved GPU memory gates remain pending.
Do not claim these gates passed or that a combined model has trained. A controlled
runtime investigation/retry and an explicit bounded training budget are needed
before a future launch. Full final and best replay and provenance checks remain
required; no universal recovery claim from the reused one-seed validation.
