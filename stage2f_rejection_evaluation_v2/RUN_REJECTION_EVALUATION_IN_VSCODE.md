# Manual rejection evaluation — development only

Open C:\NAQSHKASH in VS Code. Use a PowerShell terminal. Stop on any failed
command. Codex implemented/tested the scripts with mocks; no real inference or
full R0/R1 evaluation has been executed.

## 1. Existing environment

```powershell
Set-Location 'C:\NAQSHKASH'
$R = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2'
$Dataset = 'C:\NAQSHKASH\stage2\stage2f_negative_dataset_v2\dataset\full_20261106_194834_776'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH="$R\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$Run = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Raw = "$R\metrics\${Run}_raw"
& $Python312 -B -c "import sys,torch,cv2,numpy,PIL; print(sys.executable); print('torch',torch.__version__,'CUDA',torch.cuda.is_available()); print('numpy',numpy.__version__,'Pillow',PIL.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Existing Python environment unavailable; resolve dependencies before inference' }
```

Use the same Python3.12 interpreter/dependency directory as prior stages. The
verified historical environment used CPU torch2.2.2, numpy1.26.4, Pillow10.2.0
and OpenCV4.13.0.92. No new package is required in that environment. Select the
same executable in VS Code's Python interpreter menu. CPU is default and fully
supported; no CUDA installation is needed. Optional --device auto/cuda reuses
the old loader and records the actual device; CUDA explicitly fails if unavailable.

If an import fails, first correct the interpreter and PYTHONPATH above and use
`& $Python312 -m pip show torch numpy Pillow opencv-python` to inspect versions.
Restore the prior working environment rather than upgrading it mid-run. If torch
is truly absent, a manual isolated CPU restoration is possible:

```powershell
# Run ONLY if needed; this downloads dependencies, not model weights/training.
# & $Python312 -m pip install --target "$R\dependencies" torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
# $env:PYTHONPATH="$R\dependencies;$env:PYTHONPATH"
```

Use the pinned previous-stage versions for other missing modules. Do not edit
the E1 checkpoint/model/vocabulary to solve an environment problem. Device or
library changes cannot be mixed into an existing resumed inference run.

## 2. Dataset preflight and lightweight tests

```powershell
& $Python312 -B "$R\src\verify_artifacts.py" --dataset $Dataset --split development_design --check-dataset
if ($LASTEXITCODE -ne 0) { throw 'Development dataset or historical integrity failed' }
& $Python312 -B -m unittest discover -s "$R\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Lightweight tests failed' }
```

Expected selection:2800 images/700 parents. Only these image hashes and annotation
files are checked. Other split arguments are unavailable, not silent defaults.
No CRNN inference occurs in this preflight. Full dataset metadata is read for
lineage/hash ownership checks; validation/protected image bytes are not opened.

## 3. Expensive D1/E1 inference — execute once

```powershell
& $Python312 -B "$R\src\run_ocr_baseline.py" --dataset $Dataset --split development_design --device cpu --output-dir $Raw --progress-every 25
if ($LASTEXITCODE -ne 0) { throw 'Inference interrupted/failed; preserve outputs and use resume instructions' }
```

This is the expensive step. It reads pixels and calls the EXISTING D1/E1 pipeline
one image/tier at a time. No labels/CTC training targets, optimizer or model changes.
It records original metadata/predictions/statuses/confidence and runtime rendering
provenance, with atomic immutable per-record checkpoints under records/.

Success: completion.json with complete=true and records=2800, plus plan.json and
observations.jsonl. These sealed raw observations are used unchanged for BOTH
candidate evaluations. No tensors or dataset image copies are saved.

### Resume or deliberately run bounded chunks

Keep the same $Raw, device, code, environment and dataset. Resume skips committed
records, including recorded pipeline failures; it does not silently retry errors.

```powershell
& $Python312 -B "$R\src\run_ocr_baseline.py" --dataset $Dataset --split development_design --device cpu --output-dir $Raw --resume --progress-every 25
if ($LASTEXITCODE -ne 0) { throw 'Resume failed; inspect lineage/environment or corruption message' }
```

For manual chunks add --max-new-records 200 to the first command, then to each
resume command. A partial run returns complete=false and has no completion.json;
R0/R1 refuse it. Per-record checkpoints are the durable progress, not console
percentages. An uncommitted image at a hard interruption may run again; committed
ones never need a second inference pass. Harmless leftover journal_*.tmp files
are ignored; never edit committed JSON files or observations.jsonl.

If resuming in another terminal, restore the setup and explicitly restore the
original $Raw/$Run values; do not accidentally create a new timestamped run.
Persistent run.lock is an OS lock file; process exit releases it automatically.

## 4. R0 baseline decisions from saved predictions (no model execution)

```powershell
& $Python312 -B "$R\src\evaluate_rejection.py" --dataset $Dataset --split development_design --raw-dir $Raw --config "$R\configs\r0_baseline.yaml" --output-dir "$R\metrics\${Run}_R0"
if ($LASTEXITCODE -ne 0) { throw 'R0 saved-output evaluation failed' }
```

## 5. R1 tier-aware decisions from the SAME raw folder

```powershell
& $Python312 -B "$R\src\evaluate_rejection.py" --dataset $Dataset --split development_design --raw-dir $Raw --config "$R\configs\r1_tier_aware.yaml" --output-dir "$R\metrics\${Run}_R1"
if ($LASTEXITCODE -ne 0) { throw 'R1 saved-output evaluation failed' }
```

Each candidate writes results.json, summary.json, by_parent.json, by_group.json,
per_image.csv and completion.json. Invalid inputs have null CER/reference values;
valid outputs retain exact/CER even if segmentation or decisions fail. T1 raw
reasons/text are preserved alongside interpreted results and correction assessments.

## 6. Paired comparison

```powershell
& $Python312 -B "$R\src\compare_rejection.py" --dataset $Dataset --split development_design --r0-dir "$R\metrics\${Run}_R0" --r1-dir "$R\metrics\${Run}_R1" --output-dir "$R\reports\${Run}_comparison"
if ($LASTEXITCODE -ne 0) { throw 'Candidate input pairing or comparison failed' }
```

Outputs: comparison.json, comparison.csv, parent_comparison.csv, comparison.md,
completion.json. No winner or threshold change is automatic. A reduced review
rate can hide increased invalid acceptance or accepted transcription errors.

## 7. Failure tables, reports and visual galleries

```powershell
& $Python312 -B "$R\src\generate_error_gallery.py" --dataset $Dataset --split development_design --results-dir "$R\metrics\${Run}_R0" --output-dir "$R\reports\${Run}_R0_errors" --limit 24
if ($LASTEXITCODE -ne 0) { throw 'R0 reporting failed' }
& $Python312 -B "$R\src\generate_error_gallery.py" --dataset $Dataset --split development_design --results-dir "$R\metrics\${Run}_R1" --output-dir "$R\reports\${Run}_R1_errors" --limit 24
if ($LASTEXITCODE -ne 0) { throw 'R1 reporting failed' }
```

Each error folder contains report.json/md, error_gallery.html, completion.json
and CSV/JSON lists for invalid accepted/review/rejected, valid rejected/review/
accepted-wrong-text, preprocessing/D1/OCR failures, unlocalized exceptions,
wrong-tier predictions, T1 edits and ambiguous/unscorable interventions.
Lists overlap and must not be added as independent image totals. Galleries
show selected development inputs and matched controls only, with original-image
links for dense long pages; they never open protected/validation images.

## 8. Final integrity and sharing results

```powershell
& $Python312 -B "$R\src\verify_artifacts.py"
if ($LASTEXITCODE -ne 0) { throw 'Final historical or experiment integrity failed' }
```

All output directories must be new except deliberate --resume of the same raw
journal. Never delete/overwrite historical outputs to make a command succeed.
Do not regenerate locks after observing candidate results. A new code/policy
experiment needs a separately reviewed version and compatible raw provenance.

Share candidate summary.json, by_parent.json, by_group.json, per_image.csv,
raw completion.json/plan.json, comparison folder and representative failure
reports. Keep original datasets unchanged. No validation/protected evaluation,
training, dataset regeneration, grammar changes or calibration is authorized by
this guide. The canonical synthetic labels still lack human Talim expert review.
