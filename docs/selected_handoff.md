# Selected model handoff

Steps1 and2 are finished for this project handoff. This closes the selected-model
implementation work, not every original research target or a universal-losslessness claim.

- Step1: Branch Sigmoid LM,8,654,208 parameters;webNLL3.820069/chat2.411506.
- Step2: plain residual attention + Branch Sigmoid + RoPE,64tokens/vector,
  context512;7,327,232encoder/10,456,576total parameters;44000 cumulative updates.
  Final validation999/1000 short and202/206 packed exact,99.9989191% and99.9957132%
  token recovery. The old span32 result was1000/1000 on its historical validation.
  We doubled nominal tokens per vector and the earlier256-token context, but64:1
  is not yet perfect reconstruction and is not a byte-compression ratio.

Results use one inherited initialization and reused validation;no fresh test claim.
The product default loads the saved final64:1 checkpoint. Weights/data stay local
and ignored;Git does not contain them. Historical research evidence is retained.

Optional next training is15000 ADDITIONAL updates,44000to59000,from the saved
model AND optimizer/sampler state,constant learning rate3e-5. It is configured,
not launched, and cannot guarantee100%. No model-size growth or margin loss.
Notebook inference loads saved weights;training is opt-in in a separate notebook.
