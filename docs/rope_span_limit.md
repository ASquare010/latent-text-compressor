# RoPE compressor span limit screen

Six independent transfers from the same verified span32 / 7000-update checkpoint:
spans48,64,96,128,192,256; 2000 additional updates each; seed17; two workers.
Identical targets, data mixture and optimizer schedule. Shared encoder/decoder
weights copied exactly; only differently shaped compression/expansion projections
start fresh. Larger spans increase projection parameters; this is not an equal-size
architecture comparison. Saved span32 baseline is evaluated, never retrained.

Each worker verifies full decoded train/validation identities before updates.
Full 1000-paragraph validation, four length bins, and 64 uniform plus 64 repeating
256-token stress examples are measured. Stress tests are reported separately.
Exact natural recovery requires 1000/1000 exact paragraphs AND 100% token accuracy.
Report actual tokens/vector and vector+length bytes; compare to UTF-8 and token IDs.
Highest passing tested span is a bounded result, not a theoretical limit; one seed,
fixed transfer budget and previously used validation. Context ceiling is256.
No test-split selection, control retraining, extra candidates or automatic retries.
Failure stops new launches; already-running workers finish their bounded budgets.

Status: D:/Git/latent-text-compressor/artifacts/runs/rope-span-limit-v1/status.json
Per-arm checkpoints, audit records, source snapshot, full validation replays and
encoder exports remain local. Completion checks matched target hashes and weights.

## Blocked before updates

Span48 failed independent decoded-data identity verification. Span64 exited during
verification; Windows Event1000 identifies tokenizers.pyd access violation
0xc0000005, PID23632. Neither worker reached training; no new updates/checkpoints.
Queue stopped; spans96/128/192/256 were not launched. No workers remain.
Train/validation/tokenizer files still match recorded SHA256 hashes. Root cause
is unresolved; do not bypass checks or automatically retry native/data failures.
Saved span32 full validation remains1000/1000. Additional baseline stress checks
passed64/64 uniform and64/64 patterned256-token examples, both100% token accuracy.
These are baseline evaluations, not higher-span evidence. Higher-span limit is
unmeasured. Logs and frozen sources remain in artifacts/runs/rope-span-limit-v1.

## Tested runtime mitigation and authorized retry

User asked to investigate/fix the failure. Applied process-local limits:
TOKENIZERS_PARALLELISM=false; RAYON_NUM_THREADS, OMP_NUM_THREADS,
MKL_NUM_THREADS, OPENBLAS_NUM_THREADS=1; PyTorch intra/inter-op threads=1.
No packages, CUDA installation, data or OS power/firmware settings changed.
Five concurrent isolated tokenizer checks each verified all51000 train/valid rows.
Five further concurrent checks plus span256 CUDA BF16 forward/backward and one
synthetic optimizer smoke update each passed (not research-training updates).
Probe peak per worker: about748MiB allocated/920MiB reserved; all five remained
resident together at a completion barrier. Evidence: dump/tokenizer-repair and
 dump/tokenizer-gpu-repair in complex_fnn. Tests support this mitigation, not a
proven root cause or permanent fix; earlier unrelated native crashes and historical
CPU parity errors remain unresolved.

Same six spans x2000 updates, same parent/targets/optimizer; FIVE workers, new
output artifacts/runs/rope-span-limit-v2. Prior failures and v1 frozen sources remain.
Only v2 runner execution/threading/diagnostic handling changed; selected model and
training package sources remain unchanged. Baseline diagnostics reused, not retrained.
Each worker still verifies complete decoded identities before updates. No automatic
retry on new failures. New controller records native exit codes and faulthandler logs.


## Completed six-span screen

All six finished2000 transfer updates, same parent and matched target schedule.
All final replays, checkpoint/encoder/source hashes and persistent records verified.
No new worker failure occurred with the tested thread limits; prior failures remain.

| Tokens/vector | Exact paragraphs /1000 | Token accuracy | NLL | Encoder parameters | Allocated/reserved MiB |
|---|---:|---:|---:|---:|---:|
|32 (saved)|1000|100%|0.002274|7,342,336|379.44 /470|
|48|972|99.9697%|0.024477|8,390,912|391.30 /444|
|64|139|80.7471%|1.092528|9,439,488|427.43 /480|
|96|131|64.0343%|2.444137|11,536,640|496.81 /556|
|128|38|56.5351%|2.901396|13,633,792|571.27 /624|
|192|6|47.5529%|3.344563|17,828,096|711.62 /800|
|256|12|47.6447%|3.468620|22,022,400|859.09 /992|

32 remains the highest tested span with perfect natural validation recovery.
48 is near-perfect:28 wrong tokens across92516 tokens;972/1000 exact paragraphs.
It uses2402 vectors versus3422 atspan32 (29.8% fewer vectors on these paragraphs).
64 and above lose substantial information. This fixed2000-update transfer screen
cannot distinguish optimization/training-budget limits from representational limits;
it does not prove that48+ is impossible. No additional training is authorized.

The 48-token stress results are99.707% uniform and99.677% patterned token accuracy,
versus100% for saved32. All larger candidates fail perfect stress recovery too.
Length-bin evaluations use separate batches; BF16 rounding can slightly change
near-tied predictions, so their exact counts need not sum to the full-split run.
Single seed and repeated validation selection apply; reserved test was not scored.
Vector storage still exceeds raw text: span48 vectors+lengths1,237,824 bytes;
span256 520,000 bytes versus328,100 UTF-8 bytes, excluding model/container costs.
Final receipt: artifacts/runs/rope-span-limit-v2/completion.json.
