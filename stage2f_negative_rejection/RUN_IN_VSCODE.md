# Run Stage 2F manually in VS Code (Windows PowerShell)

Implementation stops after lightweight tests. Run expensive commands yourself.
No command below trains or changes E1. F3 is unavailable because a classifier is
not justified by this one-parent fixture collection.

## 1. Open project and select Python

In VS Code choose **File → Open Folder → C:\NAQSHKASH** and **Terminal → New Terminal**.
Use PowerShell. Paste the following block in that terminal once per session:

```powershell
Set-Location 'C:\NAQSHKASH'
$F = 'C:\NAQSHKASH\stage2\stage2f_negative_rejection'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
if (!(Test-Path -LiteralPath $Python312)) { throw 'Python 3.12 interpreter not found' }
$env:PYTHONPATH="$F\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$Run = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
& $Python312 -B -c "import sys,torch,cv2,numpy,PIL,scipy; print(sys.executable); print('torch',torch.__version__); print('CUDA available:',torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies are unavailable' }
```

This is the actual Python 3.12 interpreter and dependency directory used by
Stage2E. No named virtual environment was assumed. In VS Code's **Python: Select
Interpreter**, select that same executable. If you already use a project venv,
activate it with `& 'C:\path\to\your\venv\Scripts\Activate.ps1'` and set
`$Python312` to that environment's actual `Scripts\python.exe` instead. That path
is an example, not a required project artifact. Do not activate an unrelated env.

Existing environment: torch 2.2.2, numpy 1.26.4, opencv-python 4.13.0.92,
Pillow 10.2.0, scipy 1.17.1, per Stage2E environment record. Standard library
unittest runs the tests. No package installation is required in the verified
environment. Keep the existing environment stable; do not upgrade it mid-run.

The verified machine is CPU-only. Use `--device cpu` as shown. `--device auto`
selects CUDA if available; `--device cuda` fails clearly if unsupported. Never
install a different torch build solely because this guide mentions CUDA.

## 2. Lightweight validation and tests

```powershell
& $Python312 -B "$F\src\prepare.py" --validate-only
if ($LASTEXITCODE -ne 0) { throw 'Dataset/protocol validation failed' }
& $Python312 -B -m unittest discover -s "$F\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Stage2F tests failed' }
```

Success: parent-disjoint=true, frozen-test-excluded=true, historical hash checks
pass, followed by unittest `OK`. Validation checks archive member existence
without running OCR. Do not rerun `prepare.py` without `--validate-only`: the
already generated protocol is frozen and preparation refuses to overwrite it.

Optional tiny smoke (exactly these three clean fixture IDs, never more than 3):

```powershell
& $Python312 -B "$F\src\evaluate_baseline.py" --manifest "$F\manifests\evaluation_manifest.csv" --output-dir "$F\metrics\${Run}_smoke" --device cpu --smoke --image-id negative_count_zero_2_clean --image-id negative_blank_png_2_clean --image-id negative_corrupt_png_2_clean
if ($LASTEXITCODE -ne 0) { throw 'Smoke failed' }
```

Smoke outputs are explicitly labeled and refused by full comparison/calibration.

## 3. Execute F0 baseline and capture confidence signals

These are the expensive inference jobs. They are **not run by Codex**.
No negative label is used as a CTC target. The two manifests are different,
parent-disjoint partitions; both contain historically exposed controls.

```powershell
& $Python312 -B "$F\src\evaluate_baseline.py" --manifest "$F\manifests\development_manifest.csv" --output-dir "$F\metrics\${Run}_dev_baseline" --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Development baseline failed' }
& $Python312 -B "$F\src\evaluate_baseline.py" --manifest "$F\manifests\evaluation_manifest.csv" --output-dir "$F\metrics\${Run}_eval_baseline" --device cpu
if ($LASTEXITCODE -ne 0) { throw 'Evaluation baseline failed' }
& $Python312 -B "$F\src\evaluate_rejection.py" --baseline-dir "$F\metrics\${Run}_eval_baseline" --candidate F0 --output-dir "$F\metrics\${Run}_F0"
if ($LASTEXITCODE -ne 0) { throw 'F0 metrics failed' }
```

Success: each baseline directory has `observations.jsonl` and `completion.json`.
Each observation records preprocessing/segmentation, OCR per tier, syntax-ready
rows, raw confidence signals, F0 action/reasons and offline truth agreement.
The completion marker records hashes and coverage. F0 metrics are in `summary.json`.

## 4. Evaluate F1 using saved baseline outputs (no second model run)

```powershell
& $Python312 -B "$F\src\evaluate_rejection.py" --baseline-dir "$F\metrics\${Run}_eval_baseline" --candidate F1 --output-dir "$F\metrics\${Run}_F1"
if ($LASTEXITCODE -ne 0) { throw 'F1 evaluation failed' }
```

## 5. Inspect calibration feasibility and optionally select F2 quality gate

The current fixture suite supplies **no invalid-content development parents**.
The default command records this limitation; it does not invent a threshold:

```powershell
& $Python312 -B "$F\src\calibrate.py" --baseline-dir "$F\metrics\${Run}_dev_baseline" --output-dir "$F\metrics\${Run}_calibration_feasibility"
if ($LASTEXITCODE -ne 0) { throw 'Calibration feasibility check failed' }
```

Expected output: `calibration_report.json`, status `NOT_CALIBRATED`. This is
intentional. No policy is created. You can compare F2 with disabled thresholds
(F2 equals F1), using the supplied policy in step 6.

If you explicitly want a **positive-only OCR quality gate**, run:

```powershell
& $Python312 -B "$F\src\calibrate.py" --baseline-dir "$F\metrics\${Run}_dev_baseline" --output-dir "$F\metrics\${Run}_quality_gate" --allow-positive-only --max-review-fraction 0.05
if ($LASTEXITCODE -ne 0) { throw 'Quality gate selection failed' }
```

It selects the lower 5% nonblank-score quantile among F1-accepted valid development
pages (at least 20 required). This bounds additional review on scored development
pages, not population/test behavior. Policy scope clearly says quality-only,
not calibrated invalidity or optimized negative rejection. No evaluation/test
parent enters selection. If fewer than 20 eligible pages remain, selection fails;
use disabled thresholds and report that limitation. Do not tune on evaluation.

## 6. Evaluate F2

Choose **one** policy assignment, then execute the evaluation:

```powershell
# Default: thresholds disabled; this F2 comparison should match F1.
$Policy = "$F\configs\confidence_policy.yaml"
# If and only if the optional quality-gate command succeeded, use this instead:
# $Policy = "$F\metrics\${Run}_quality_gate\confidence_policy.yaml"
& $Python312 -B "$F\src\evaluate_rejection.py" --baseline-dir "$F\metrics\${Run}_eval_baseline" --candidate F2 --policy $Policy --output-dir "$F\metrics\${Run}_F2"
if ($LASTEXITCODE -ne 0) { throw 'F2 evaluation failed' }
```

Low quality/missing required signal → REVIEW_UNCERTAIN. REVIEW remains separate
from invalid rejection. No raw CTC score is presented as calibrated probability.

## 7. Classifier/F3

Skip training and F3. There is no justified independent negative training set,
so no classifier script or resume command exists. E1 never trains in Stage2F.
If a future classifier is authorized after new data collection, it needs a new
parent-disjoint frozen protocol and separate implementation/checkpoints.

## 8. Compare actual outputs and generate report

```powershell
& $Python312 -B "$F\src\compare_candidates.py" --candidate-dir "$F\metrics\${Run}_F0" --candidate-dir "$F\metrics\${Run}_F1" --candidate-dir "$F\metrics\${Run}_F2" --output-dir "$F\reports\${Run}_comparison"
if ($LASTEXITCODE -ne 0) { throw 'Candidate comparison failed' }
& $Python312 -B "$F\src\report_generator.py" --comparison "$F\reports\${Run}_comparison\comparison.json" --output "$F\reports\${Run}_comparison\final_report.md"
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
& $Python312 -B "$F\src\prepare.py" --validate-only
if ($LASTEXITCODE -ne 0) { throw 'Final integrity check failed' }
```

Success: `comparison.csv`, `comparison.json`, `final_report.md`. Per candidate,
`per_image.csv`, `results.json`, `summary.json`, `groups.json` and completion marker
provide the actual decisions, denominators, category and parent breakdowns.

## Interrupted commands and preserving results

Every output directory must be new; scripts refuse overwrites. Keep `$Run` to
reuse saved baselines. Record it with `Write-Output $Run`. In a new terminal,
restore it manually, for example `$Run = 'YOUR_PREVIOUS_RUN_VALUE'`.
If baseline inference is interrupted, its absent `completion.json` makes it
ineligible for evaluation. There is no inference resume: use a new `$Run` and
rerun that partition. Already completed baseline outputs may be reused for F1/F2
and comparison. There is no training to resume in this implementation.

## What to share for review

Share the complete `${Run}_comparison` folder, F0/F1/F2 `summary.json`,
`groups.json` and `per_image.csv`, calibration report plus selected policy if any,
and both baseline `completion.json` files. For detailed error analysis share
`observations.jsonl` (or selected failing records) too. Also include the console
error if any command failed. Never describe these fixtures or exposed controls
as an independent unseen test. Collect new image-adjudicated invalid Talim
parents before claiming content-rejection efficacy or training a classifier.
