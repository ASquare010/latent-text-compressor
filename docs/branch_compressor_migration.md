# Branch Sigmoid compressor migration

The encoder and reconstruction decoder now both use the selected Branch Sigmoid
FFN: 1,534 squared-ReLU features in four groups (384,384,383,383), controlled by
four independent 2*sigmoid gates, followed by a learned output projection.
Every group and gate reads the full input. The fixed feature count matches the
Step 1 winner; input/output width remains the compressor's configured width.
`hidden=1024` specifies the original variance-calibration reference, not the
number of intermediate features. Output calibration is saved in checkpoints.

This is a breaking architecture revision, not a conversion of trained weights.
Legacy CurveFFN checkpoints and encoder exports are rejected explicitly. Existing
weights and historical evidence remain untouched; use their original archived
sources if old inference is needed. New encoder exports fingerprint the FFN
source as well as the model source and use position-encoder-branch-v2.

No new compressor training or quality/VRAM benchmark was performed for this
migration. Old exact-reconstruction scores describe only the CurveFFN model.
Branch Sigmoid's language-model win is not evidence of better compression.
The wider FFN also increases compressor parameters and may increase memory.
A new trained encoder and matching decoder are required for real inference.

Validation: the two repository implementations produce bit-exact CPU outputs and
states at the same seed. CPU forward/backward, padding, cross-seed checkpoint
roundtrip, encoder export and legacy/source-tamper rejection pass. CUDA BF16
forward/backward passes with finite gradients. The standalone workflow tests also
pass exact interrupted/resumed tiny training and stage transfer (test fixtures,
not a new research run). Existing Step 1 checkpoint output parity still passes.

At width256/span32 with four encoder and one decoder blocks: 7,407,872 encoder
parameters; 10,554,368 full parameters. Historical CurveFFN had 4,798,720 and
7,292,928 respectively. No equal-parameter or equal-VRAM claim is made.
