# Selected residual64 handoff

The product default is the saved plain residual attention + Branch Sigmoid + RoPE
64:1/context512 checkpoint (44,000 cumulative updates). Validation999/1000short,
202/206packed exact;not universal losslessness. Steps1/2 are complete for the
selected-model handoff,not every historical research target.

config/train.json prepares15,000 ADDITIONAL updates from that checkpoint,including
optimizer/sampler state. It is not launched by editing or opening a notebook.
Inference notebook loads saved weights;training notebook defaults RUN_TRAINING=False.
Use uv run --no-sync;preserve CUDA environment. Retain process-local CPU16 pinning
and source/data/parent/finite-gradient/memory/final-audit checks. No blind retries
of native/data failures;earlier runtime root cause remains unproven.

Keep checkpoints,data,records and historical source snapshots local/ignored.
Do not delete historical evidence or restore retired architectures into active code.
The archived curriculum is underdocs/history. Base model/FFN modules are required
by the selected residual model. See docs/selected_handoff.md and residual64_long.md.
