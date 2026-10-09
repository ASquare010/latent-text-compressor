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

# Fresh60k user-confirmed restart TRAINING (2026-10-08)

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

# Fresh selected 10.46M / 60,000 updates TRAINING (2026-10-08)

Latest user explicitly requested fresh 10M,60000 updates,delete current11M,
same64:1/context512,selected plain residual attention,and higher then decaying LR.
One fresh seed47 run:10456576 total/7327232 encoder parameters,code125,width256,
4encoder/1decoder,Branch Sigmoid/RoPE/depth_route. Zero inherited weights or
optimizer updates; fresh AdamW. Same25/25/25/25 short/packed/uniform/pattern mix,
micro2 x accumulation4. Warmup1000 to6e-4 then cosine to1e-5 at60000.
Launcher24860/controller22044/worker12192 verified past update1. Complete data
identities,source checks,and exact fresh logits/vector parity across both repos
passed inside worker. Initial336.04/376MiB allocated/reserved,not final peaks.
Only four positively verified completed11M checkpoints were deleted:initial,last,
best,encoder. Result/data/source history and Step1 retained; deletion manifest in
dump/residual64-fresh10m60k-v1/deleted_11m_weights.json. Earlier113 deletions remain.
Two conservative process-guard refusals preceded launch; no updates/deletions
in those attempts. Elevated inspection identified VSCode formatter/Jupyter
processes and GPU was idle. PowerShell startup failed before commands with
BadImageFormatException/security configuration read error; unchanged environment,
existing pinned Python via CMD used instead. Earlier runtime failures unresolved.
Sources and recipe now frozen. Do not modify or duplicate workers or automatically
retry new native/data failures. One worker CPU16,uv run --no-sync,no extra probes,
no additional training or automation. Full final/best replays and hashes retained.
Output standalone artifacts/runs/residual64-fresh10m-60000-s47-v1. Config points
to this fresh run; no trained-quality claim or100% guarantee. See
docs/residual64_fresh10m60k.md. Old999/1000 weights were deleted previously.

# Fresh11.02M model replaces deleted checkpoints

User requested random-initialized11M,10000 total updates. Old compressor weights,
including the999/1000 winner,have been deleted;Step1/data/logs/results retained.
The new model has11022336 parameters,64tokens/vector,context512,code_features142.
No previous accuracy claim applies. config/train.json now describes the fresh run,
not15000-step continuation. Training notebooks are opt-in;do not duplicate the live
run. Existing output guard rejects duplicate launches. No ready encoder-only export
until the new training/evaluation/export completes. See docs/residual64_fresh11m.md.
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

Fresh run11280 verified50/10000;113 metadata-verified old compressor checkpoints deleted,including previous winner;Step1/data/results retained. Main cleanup manifest records exact hashes. No prior accuracy applies to current fresh model. Sources frozen during training.
