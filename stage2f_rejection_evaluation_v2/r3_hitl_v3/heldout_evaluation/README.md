# Final frozen R3 held-out evaluation

Selected policy: **R3_HITL_V3**, approved development run `20261107_192326_500_058300e0`. R3.1 work is excluded and preserved. No policy tuning, retraining or Git/GitHub operations.

Validation: `validation_calibration`, 600 images / 150 parents, 300 VALID / 300 INVALID. Protected test: `protected_test`, same counts. Manifest sample IDs, image hashes, parent IDs and semantic parent keys are disjoint across development/validation/test. The adapter checks frozen AND actual metadata checksums and protected seal identity, but preflight never reads image bytes. The old development loader is copied unchanged; its held-out prohibition is not bypassed.

## Release and preflight

```powershell
Set-Location 'C:\NAQSHKASH'
$F = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_hitl_v3\frozen_release'
$H = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_hitl_v3\heldout_evaluation'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$H\src"

& $Python312 -B "$F\verify_freeze.py" --replay-approved-development
if ($LASTEXITCODE -ne 0) { throw 'Final freeze verification failed; stop.' }

& $Python312 -B "$H\src\heldout_cli.py" preflight --split validation_calibration
if ($LASTEXITCODE -ne 0) { throw 'Validation preflight failed; stop.' }
```

This integrity preflight reproduces the approved saved R3 decisions and checks lazy inference imports/environment without constructing a model, reading pixels or running D1/E1. A fresh Python process is required; working-project imports are deliberately blocked. The release is self-contained for local project code/canonical references/checkpoint/metadata; it uses the recorded installed third-party packages.

## Validation: manual inference, then scoring

No compatible v2 held-out OCR cache exists at release. Old fixture caches and development predictions are not substitutes. The infer command generates exactly the selected 600 images with frozen CPU D1/E1, original per-record seed/batching/greedy decode and checkpoint. It performs no training. Only image bytes enter OCR. Rendering provenance is whitelisted before T1; labels and mutation records enter scoring only after final decisions and region planning.

```powershell
$Stamp = "$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')_$([guid]::NewGuid().ToString('N').Substring(0,8))"
$ValRaw = "$H\runs\validation_raw_$Stamp"
$ValOut = "$H\runs\validation_evaluation_$Stamp"

& $Python312 -B "$H\src\heldout_cli.py" infer --split validation_calibration --authorization I_AUTHORIZE_VALIDATION --output-dir $ValRaw
if ($LASTEXITCODE -ne 0) { throw 'Validation inference failed; preserve output and resume only that claim.' }

& $Python312 -B "$H\src\heldout_cli.py" evaluate --split validation_calibration --authorization I_AUTHORIZE_VALIDATION --raw-dir $ValRaw --output-dir $ValOut
if ($LASTEXITCODE -ne 0) { throw 'Validation evaluation failed; stop and preserve output.' }

Get-Content "$ValOut\report.md"
$ValidationMetrics = Get-Content "$ValOut\metrics.json" -Raw | ConvertFrom-Json
$ValidationMetrics.summary | ConvertTo-Json -Depth 15
$ValidationMetrics.R2_to_R3_transitions | Format-Table
$ValidationMetrics.localization | ConvertTo-Json -Depth 10
```

STOP and review validation before authorizing protected test. Do not tune the policy based on validation outcomes. Check per-category/parent outcomes, reviews versus definitive rejects, invalid ACCEPT, false definitive rejects, reasons, localization eligibility and declines from development.

## Protected test: separate authorization AFTER validation review

Only execute this block when the user separately authorizes protected test after reviewing validation. A completed registered validation result, its immutable completion hash, an explicit review acknowledgment and the different protected-test authorization string are mandatory.

```powershell
# Restore the exact registered validation output if this is a new PowerShell session.
$ValOut = (Get-Content "$H\state\validation_calibration\evaluation_completed.json" -Raw | ConvertFrom-Json).run_directory
& $Python312 -B "$H\src\heldout_cli.py" preflight --split protected_test
if ($LASTEXITCODE -ne 0) { throw 'Protected-test preflight failed; stop.' }

$Stamp = "$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')_$([guid]::NewGuid().ToString('N').Substring(0,8))"
$TestRaw = "$H\runs\protected_raw_$Stamp"
$TestOut = "$H\runs\protected_evaluation_$Stamp"

& $Python312 -B "$H\src\heldout_cli.py" infer --split protected_test --authorization I_AUTHORIZE_PROTECTED_TEST_AFTER_VALIDATION_REVIEW --validation-run $ValOut --ack-validation-reviewed --output-dir $TestRaw
if ($LASTEXITCODE -ne 0) { throw 'Protected inference failed; preserve output and resume only that claim.' }

& $Python312 -B "$H\src\heldout_cli.py" evaluate --split protected_test --authorization I_AUTHORIZE_PROTECTED_TEST_AFTER_VALIDATION_REVIEW --validation-run $ValOut --ack-validation-reviewed --raw-dir $TestRaw --output-dir $TestOut
if ($LASTEXITCODE -ne 0) { throw 'Protected evaluation failed; stop and preserve output.' }

Get-Content "$TestOut\report.md"
```

## Final comparison report

After both one-time evaluations complete:

```powershell
$ValOut = (Get-Content "$H\state\validation_calibration\evaluation_completed.json" -Raw | ConvertFrom-Json).run_directory
$TestOut = (Get-Content "$H\state\protected_test\evaluation_completed.json" -Raw | ConvertFrom-Json).run_directory
$Comparison = "$H\runs\final_comparison_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')_$([guid]::NewGuid().ToString('N').Substring(0,8))"
& $Python312 -B "$H\src\heldout_cli.py" compare --validation-run $ValOut --protected-run $TestOut --output-dir $Comparison
if ($LASTEXITCODE -ne 0) { throw 'Final comparison failed.' }
Get-Content "$Comparison\report.md"
```

The final comparison reports the approved development and actual held-out results without rerunning decisions. Full outcome/rate summaries are retained in comparison.json. Detailed localization, categories and parent evidence remains in each sealed per-split metrics.json. REVIEW is never counted as definitive REJECT. Synthetic fixed-grid coverage, three mutation categories, OCR normalization, lack of domain-expert verification and unresolved physical carpet alignment remain limitations.

## Resume and one-time safeguards

Inference claims are exclusive and bound to split, all IDs/hashes, selection/dataset/release hashes, software, seed and raw output directory. Existing records cannot be overwritten. An interrupted inference may resume ONLY its same claim/output with `--resume`; no new inference trial is permitted. Example for validation:

```powershell
$ValRaw = (Get-Content "$H\state\validation_calibration\inference_claim.json" -Raw | ConvertFrom-Json).raw_directory
& $Python312 -B "$H\src\heldout_cli.py" infer --split validation_calibration --authorization I_AUTHORIZE_VALIDATION --output-dir $ValRaw --resume
```

Protected resume requires the same protected authorization/review arguments as its original command plus `--resume`. Completed evaluation markers prevent repeat evaluation/inference. Do not delete claims/markers. If scoring itself fails, stop and report the implementation defect; completed/raw artifacts stay preserved. Output directories are collision-safe and exclusive; completion.json is written last. Partial results are not complete evaluations.

## Output contract and localization

Each raw run has plan.json, immutable hashed records, observations.jsonl and completion.json. Each evaluation has run_manifest.json, metrics.json, per_image.jsonl, outcomes.csv, report.md and completion.json. Manifests include frozen source/config/dependency/dataset/checkpoint hashes, exact split/parents, raw lineage, authorization and software. Metrics reuse unchanged R3 evaluation helpers: three outcomes and rates by validity/category/parent/variant, invalid acceptance, valid/invalid reviews, false definitive rejection, R2-to-R3 transitions, reasons and localization coverage.

The optional D1-return hook observes the unchanged original process_image result to retain verified matrices/boxes; it neither changes model crops nor inference. Broad row/tier regions are planned after decisions and before ground-truth scoring. Mutation metadata enters only the existing offline localization scorer. Text-column eligibility and broad row overlap remain distinct from precise glyph localization. No gallery is generated by this adapter.

R3 ACCEPT does not imply a uniquely resolved physical carpet reading. The reporting wrapper explicitly sets carpet_generation_ready=false and selected_carpet_reading=null. Ambiguous accepted readings must not go to carpet generation without separate authorized alignment resolution. No R3.1 alignment is selected or applied.

## Lightweight tests

```powershell
& $Python312 -B -m unittest discover -s "$H\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Release adapter tests failed.' }
```

Tests use synthetic observations/temp journals and development metadata only; no held-out inference or image reads. See reports/test_results.json and the frozen preflight report for executed counts/status.
