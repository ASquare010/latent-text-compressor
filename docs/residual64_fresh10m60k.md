# Fresh 10.46M residual64 training

The authorized run has started: one model, **60,000 fresh optimizer updates**,
seed47. This is not a continuation of the historical999/1000 model.

| Setting | Value |
|---|---|
| Total / encoder parameters | 10,456,576 / 7,327,232 |
| Tokens per vector / context | 64 / 512 |
| Width / encoder / decoder layers | 256 / 4 / 1 |
| Model | Plain residual attention, Branch Sigmoid FFN, RoPE |
| Internal code features | 125 |
| Learning rate | 1,000-update warmup to0.0006; cosine decay to0.00001 |
| Training mixture | 25% each short,packed,uniform,pattern |
| Microbatch / accumulation | 2 / 4 |

The worker independently passed decoded data identities, frozen source checks,
and exact initial FP32 logits/vector parity between the standalone implementation
and `src/models/position_compressor/residual.py`. That implementation uses learned
depth queries to weight previous sublayer outputs. No legacy CurveFFN is selected.
Initial measured training memory336.04MiB allocated/376MiB reserved is not the final
peak. Separate probes and initial full validation remain waived, not passed.

Worker12192 passed update1; controller22044 was launched through hidden uv24860.
Status: `D:/Git/latent-text-compressor/artifacts/runs/residual64-fresh10m-60000-s47-v1/continue_plain/status.json`.
Checkpoints are saved every250 updates and validation runs every500. Full final
and best validation replays, encoder parity and hashes are required at completion.
No new validation accuracy is available yet. More training or a higher peak rate
does not guarantee100% exact recovery.

The four completed11M checkpoint files were SHA-verified and classified by their
actual tensors before deletion. Logs, results, datasets and source archives remain.
See `dump/residual64-fresh10m60k-v1/deleted_11m_weights.json`.

Two process-guard attempts refused unrelated Python processes before any deletion
or training. Elevated inspection identified VSCode formatters and Jupyter processes;
GPU was idle before launch. PowerShell startup failed before any command executed;
the existing CPU-pinned Python runtime via CMD was used, without security or
environment changes. Prior native/runtime failures remain unresolved. Do not
automatically retry a new native/data failure or modify the frozen training sources.
No monitoring automation or additional experiments were created.

## User-confirmed restart

User explicitly confirmed they stopped the first attempt and requested restart.
This explains missing controller/worker; do not label it an unexplained native
training crash. First attempt preserved in artifacts/runs/
residual64-fresh10m-60000-s47-v1-user-stopped-26:26 logged discarded updates,
possible unlogged in-flight work,no trained checkpoint (initial.pt retained).
Same frozen sources/recipe restarted fresh60000 updates,launcher28032,
worker22900 verified at update1. All in-worker identities and both-repo
fresh output parity passed again. No further weights deleted, no environment
changes or additional experiments. Previous launch PIDs below are historical.
Restart receipt:dump/residual64-fresh10m60k-v1/restart_receipt.json.


# Fresh60k reboot recovery TRAINING (2026-10-08)

User reported PC reboot and requested continuation. No surviving Python/uv
workers. Original last.pt is incomplete (missing ZIP central directory); original
training log has a null tail,with250 consecutive durable rows after10000. Saved
status says10300,so at least300 updates were executed after the recovery point;
exact additional unlogged work unknown. Preserve original files and stale locks.
Verified best.pt at10000 is readable,finite,and matches exact model/source/data;
SHA ad6bdbfbf413c135afd66b5dc7d684630c3b856ea6049a965798e65e5f609370.
Resume only50000 remaining to60000,not a new60000 budget. Restore AdamW10000,
sampler,CPU/CUDA RNG; reconstruct and verify original target-schedule hash.
Original model/data/LR schedule unchanged. Frozen recovery sources:
dump/residual64-fresh10m60k-recovery-v2. Output standalone artifacts/runs/
residual64-fresh10m-60000-s47-recovery-v2. Original10k log/validation prefix copied
for full cumulative audits; previous26 discarded first-attempt updates retained.
Worker25864 verified update10025; independent complete data
identities and resume checks passed. V1 recovery stopped before updates on damaged memory.json metadata. V2 reconstructs
training memory from retained logs; prior validation memory peaks unavailable.
No decoded-data/native failure. Checkpoint
and null-tail damage diagnosed after reboot; no parser/data-identity bypass.
OneCPU16worker,uv run --no-sync,no environment changes,extra experiments,automation
or deletion. Do not alter frozen sources or blindly retry new failures. Full
final/best validation/export/hash audits retained. No inference promotion.


# Fresh60k second PC-crash recovery TRAINING (2026-10-08)

User again reported PC crash and explicitly requested continuing. No prior workers
survived. Recovery-v2 durable logs reached46500 but last.pt and best.pt have missing
payload/zero tails (not merely missing ZIP directory); unrecoverable as complete
optimizer checkpoints. Status,best_validation,memory JSON zero-filled. All retained.
Original best.pt at10000 still matches SHA
ad6bdbfbf413c135afd66b5dc7d684630c3b856ea6049a965798e65e5f609370 and loads correctly.
Resume that checkpoint:50000 remaining to60000.36500 logged v2 updates discarded;
unlogged work unknown,plus previously documented crash/first-attempt discards.
This is not resumption from46500 and must not be reported as such.
New sources dump/residual64-fresh10m60k-recovery-v3 and standalone output
artifacts/runs/residual64-fresh10m-60000-s47-recovery-v3. Restored AdamW10000,
sampler/CPU/CUDA RNG and reconstructed/verified target prefix. Full independent
decoded data identities retained. Same model/data/LR schedule and final target.
Launcher4492;worker17216 verified at10050. Checkpoint saving now flushes/fsyncs,
verifies ZIP CRC,then Windows write-through replacement; retains previous and
5000-step milestone backups. This reduces risk,not proof against disk/power loss.
Original v2 sources/evidence untouched. Prior validation memory peaks unavailable.
All final/best replay/export/hash checks retained;oneCPU16worker,no env changes,
extra experiments,automatic retry or automation. Sources frozen at launch.

