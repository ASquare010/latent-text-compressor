# Fresh residual64, approximately11M parameters

User requests a random-initialized11M model,10000 total updates and deletion of
old compressor models. This replaces the previous continuation. Its process tree
was stopped at7105 logged recovery updates;historical logs/records retained.

New exact model:11,022,336 total /7,610,112 encoder parameters. Same width256,
4encoder/1decoder,plain residual depth attention,Branch Sigmoid,RoPE,64tokens per
256-feature vector,max context512. Increase code_features125to142,adding565760
parameters to internal packing/unpacking projections without widening stored vectors.
Fresh seed47,zero inherited updates,zero loaded weights,zero inherited optimizer.
AdamW warmup250 updates to3e-4 then cosine decay to3e-5 at10000,weight decay.01,
clip1,micro2xaccum4. Same25%short/25%packed/25%uniform/25%pattern source mixture.

One pinnedCPU16 worker;existing uv run --no-sync environment. Complete independent
decoded data identities,source hashes,parameter checks,finite gradients and memory
guards1300/1500MiB allocated/reserved retained. Prior standalone probe/initial full
validation waivers remain,not passed. Save initial checkpoint before updates,last
every250,best on full validation every500. Final/best full replays,encoder parity,
source/checkpoint/record hashes and exact10000-step sequence required.

Sources dump/residual64-fresh11m-v1;output standalone artifacts/runs/
residual64-fresh11m-10000-s47-v1. No guarantee of100% recovery. This is a fresh10k
run,not a matched-history comparison against the old44k model. Test untouched.

Cleanup: broad deletion was rejected by automatic approval review because it did
not prove every file excluded Step1. Read-only tensor/config classification is
being used to construct a narrow manifest of positively verified compressor files.
Do not bypass the rejection or delete unclassified files. Step1,data,logs,records
and source snapshots retained. Historical metrics do not apply to this fresh model.
Status: prepared;not yet launched.

## Launched and cleanup complete

Worker11280 verified at50/10000 updates. Fresh initialization receipt confirms
11,022,336 total /7,610,112 encoder parameters,seed47,no parent/learned weights,
zero inherited updates and fresh AdamW. Initial checkpoint saved. Complete data/
source/parameter checks passed. Early432.62/476MiB allocated/reserved,totalGPU635MiB;
not final peaks. Sources frozen,no worker failure. No accuracy result yet.

Metadata classification proved113 files were compressor weights and excluded all
Step1 model structures. Narrow exact-path/hash-verified deletion approved and
completed:11,635,507,119bytes. Includes the old999/1000 winner and encoder exports.
Three non-checkpoint files retained. Datasets,logs,records and Step1 weights retained.
The earlier broad deletion was rejected;none was deleted through that rejected
script. Verified manifest and deletion log are underdump/residual64-fresh11m-v1.
