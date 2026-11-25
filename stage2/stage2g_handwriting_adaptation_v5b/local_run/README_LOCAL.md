# V5b local execution

Generated-only training has been stopped. **Update 750 is selected**, even though update 1000 is also preserved. No full overnight folds or final fit have been launched. The earlier 20-update fold/final smoke tests passed and their temporary outputs were deleted.

## PowerShell setup

```powershell
Set-Location 'C:\NAQSHKASH\stage2\stage2g_handwriting_adaptation_v5b\local_run'
& 'C:\Users\hp\AppData\Local\Programs\Python\Python38\python.exe' -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -B -c "import torch, cv2, scipy, PIL; print(torch.__version__)"
```

This uses the already tested packages on this machine; no download is required. If activation is unavailable, replace `python` below with `& '.\.venv\Scripts\python.exe'`.

## 1. Run the selected checkpoint’s gates FIRST

```powershell
python -B run_gates_local.py
```

This is inference only: no training or optimizer updates. It compares update 750 with frozen E1 on clean historical, clean validation, positive-edge and long-positive synthetic cohorts, reporting raw and role-masked results, with and without whitespace. All views use the existing gate tolerances.

Before opening W05, it fixes the gate decision and updates `config.json`: update 750 if all synthetic gates pass, otherwise frozen E1. It then evaluates W05 once, sharing each image’s posterior between the two decoders, and reports both 14 and 13 tiers. W05 never chooses initialization or settings. It writes `../final_report_v5b.md` with all existing 105-tier checkpoint results, gates, PASS/FAIL, and informational W05 numbers.

The report’s overall PASS additionally retains the original requirement of at least 3 percentage points lower raw macro-writer CER on the 105 tiers. Update 750 already exceeds that CER threshold; the full synthetic gates remain untested until you run this command.

If interrupted, or if you need to complete an existing gate run:

```powershell
python -B run_gates_local.py --resume
```

Completed synthetic and W05 posterior traces are reused, so already completed W05 forwards are not repeated. Do not delete `gate_outputs/update_00750` when resuming. The script refuses a duplicate full run without `--resume`.

A tiny synthetic-only dry run was actually executed (four tiers per model, zero W05, zero training updates). Its outputs were deleted; verification is in `run_gates_local_dry_run_verification.json`. The dry run never changes initialization or supplies a full PASS/FAIL.

## 2. Run the overnight folds yourself

```powershell
python -B run_folds.py
```

Two configurations: CNN + BiLSTM with classifier frozen, and all layers. Two held-out writers: W01, then W06. Each update accumulates 16 tiers: 6 safe4 synthetic, 3 real originals, 3 real augmented crops, 4 clean synthetic replay. Real gradient updates exclude every descendant of the held-out writer. The full allowed pool remains 105 originals and 2,413 crops. KL uses frozen E1 and the configured clipping rule. Limits are 2,000–6,000 updates, with held-out evaluation every 250 and patience 4. ETA prints every 50 updates.

Every setting is in `config.json`. The gates script sets `initialization` to `step2` or `frozen_e1` automatically. You may explicitly choose `frozen_e1`. Do not modify settings while resuming a run; use a new run for changed settings. All run outputs use new timestamped subfolders under `outputs/`.

Keep the V5/V2 data folders and existing project/reference code in place: this package uses their audited manifests and original image paths. The global safe4 calibration used all four training writers; the Step 2 checkpoint was also selected on all 105 development tiers. Thus Step 2 initialization is not fully nested writer validation, even though held-out real gradient updates are excluded.

## 3. Final fitting after the folds

```powershell
python -B run_final.py
```

The latest completed fold set supplies the chosen configuration and update count automatically. Final fitting uses all four training writers, then runs one final-model W05 evaluation and all synthetic gate views. Frozen E1 remains the default if gates fail; no operational model file is changed automatically.

For an explicit fold set:

```powershell
python -B run_final.py --fold-run '.\outputs\folds_REPLACE_WITH_RUN_ID'
```

## Resume after interruption or sleep

```powershell
python -B run_folds.py --resume
python -B run_final.py --resume
```

Or select a particular run:

```powershell
python -B run_folds.py --resume '.\outputs\folds_REPLACE_WITH_RUN_ID'
python -B run_final.py --resume '.\outputs\final_REPLACE_WITH_RUN_ID'
```

Checkpoints every 250 updates include model, optimizer, Python/NumPy/Torch RNG states, clipping history and early-stopping state. A crash may lose up to 249 uncheckpointed updates. Resume appends a new `attempt_*` log segment and preserves abandoned tails. Before the first checkpoint, a fresh run folder is required. If Python remains open during sleep, waking continues the same process. Never run two copies against the same output folder.

## Runtime and files to return

Generated-only training averaged approximately **7.1 seconds/update through update 1000**, including early CPU contention with smoke tests. At that speed, 8,000 minimum fold updates plus a 2,000-update final fit would take roughly **20 hours**; the combined ceilings would take about **59 hours**. Real handwriting crops are wider, so these are estimates rather than a one-night guarantee. Gates perform inference only and are much shorter than training.

Send back the selected gate folder (`synthetic_gates.json`, `decision_before_W05.json`, `W05_once.json`, prediction CSVs), `final_report_v5b.md`, and the updated config. After the overnight runs, also send each fold/final run’s selection/results CSVs, `completion.json`, `run_config.json`, `progress.json`, `last_checkpoint.json`, all JSONL logs and evaluation predictions/metrics, plus the selected final checkpoint. Preserve other checkpoints locally for resume.
