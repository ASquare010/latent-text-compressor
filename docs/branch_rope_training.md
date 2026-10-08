# Branch Sigmoid + RoPE compressor

User authorized fresh training and removal of old compressor weights on 2026-10-07.
One seed17 run, 7,000 optimizer updates total, the existing five-stage curriculum:
base8 2000, natural8 1000, coverage8 1000, patterns8 1000, span32 2000.
Four encoder blocks, one decoder block, width256, four heads, Branch Sigmoid.
RoPE on query/key pairs in encoder and decoder replaces learned absolute embeddings.
Order is also retained by ordered span concatenation and decoder expansion.
No claim of improved recovery until full validation is available. No language-model
quality claim or test-split selection. Existing train/validation dataset is reused,
with independent decoded identities checked before launch. No automatic retry of
native/data-integrity failures and no budget extensions. One worker; no controls retrained.
Run: D:/Git/latent-text-compressor/artifacts/runs/branch-rope-v1.

Prelaunch tests: all seven workflow/migration tests plus the RoPE relative-position
and norm-preservation test passed. CUDA BF16 forward/backward passed, and both
repository models have identical CPU states/outputs. An initial RoPE test fixture
used span32 with max_tokens16; corrected to span4 before the passing run.
New counts: encoder 7,342,336; full model 10,488,832.
Old compressor cleanup removed 37 weight files, 0.646 GiB. Exact paths/hashes are
in D:/Git/complex_fnn/dump/rope-compressor-launch/deleted.json. Step 1 weights,
source snapshots, datasets and result records are preserved. Old weight replay
is no longer available. No weights are restored or controls retrained.

Live status and logs: D:/Git/complex_fnn/dump/rope-compressor-launch/.
Training updates: D:/Git/latent-text-compressor/artifacts/runs/branch-rope-v1/training.jsonl.
Completion requires 7000 logged updates, frozen-source checks, full final validation
replay and encoder-export parity; controller writes completion.json only after these pass.

## Completed result

All 7,000 updates completed. Full final validation replay passed: 1,000/1,000
paragraphs reconstructed exactly; token accuracy 100%; reconstruction NLL
0.00227359. Encoder: 7,342,336 parameters; full model: 10,488,832.
Peak training memory: 379.44 MiB allocated / 470 MiB reserved.
Checkpoint, encoder, frozen-source hashes and persistent result record verified.
Encoder exports exist in both repositories; no additional training was started.

This is one seed on the existing validation split, not proof of general lossless
compression or improvement over the historical encoder, which also reached
1,000/1,000. There is no matched RoPE ablation. Target is 32 tokens per vector;
92,516 validation tokens occupied 3,422 vectors after paragraph padding.
Vectors plus length metadata occupied 1,760,064 bytes versus 328,100 UTF-8 bytes,
excluding model/tokenizer and file overhead: this reduces sequence length, not
storage bytes. Reserved test data has not been scored for this run.

Receipt: D:/Git/complex_fnn/dump/rope-compressor-launch/completion.json.
