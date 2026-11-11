# Run T0 and T1 manually from saved development predictions

Open C:\NAQSHKASH in VS Code, then Terminal → New Terminal → PowerShell.
These scripts do not load the CRNN, train, tune thresholds or run OCR. Full
candidate postprocessing/evaluation is left for you to execute.

## 1. Environment

```powershell
Set-Location 'C:\NAQSHKASH'
$T = 'C:\NAQSHKASH\stage2\stage2f_tier_aware_decoding'
$B = 'C:\NAQSHKASH\stage2\stage2f_negative_rejection\metrics\20261106_154728_812_dev_baseline'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH="$T\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$Run = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
& $Python312 -B -c "import sys,PIL; print(sys.executable); print('Pillow',PIL.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Verified Python environment is unavailable' }
```

This is the actual Python3.12 environment used by previous stages. No new package
installation is needed. The implementation uses Python standard library and
existing Pillow for font/image galleries. It does not require CUDA, GPU selection,
torch imports or a model checkpoint load. The local font stays in its existing
location and is only read for internal comparisons; it is not bundled here.

## 2. Verify frozen baseline and integrity; optionally run lightweight tests

```powershell
& $Python312 -B "$T\src\verify_artifacts.py" --baseline
if ($LASTEXITCODE -ne 0) { throw 'Baseline or frozen-input verification failed' }
& $Python312 -B -m unittest discover -s "$T\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Tier-aware unit tests failed' }
```

Verification reads saved coverage and hashes, not candidate outcomes. Expected
baseline_rows=776, unchanged=true. A hash mismatch is a stop condition: inspect
the changed input rather than regenerating locks. Do not rerun audit_glyphs.py
or replace mapping/configuration after experiment results are known.

## 3. T0 saved-prediction evaluation

```powershell
& $Python312 -B "$T\src\evaluate_tier_aware.py" --config "$T\configs\t0_baseline.yaml" --baseline-dir $B --output-dir "$T\metrics\${Run}_T0"
if ($LASTEXITCODE -ne 0) { throw 'T0 evaluation failed' }
```

T0 uses unchanged raw predictions and F1 v2. No source image is reprocessed.

## 4. T1 saved-prediction evaluation

```powershell
& $Python312 -B "$T\src\evaluate_tier_aware.py" --config "$T\configs\t1_tier_aware.yaml" --baseline-dir $B --output-dir "$T\metrics\${Run}_T1"
if ($LASTEXITCODE -ne 0) { throw 'T1 evaluation failed' }
```

T1 checks original source rendering metadata, saved D1 accepted path/roles and
the verified mapping. It neither uses ground-truth characters to choose edits
nor forces a correction for unknown rendering/role. Both candidates write
results.json, summary.json, by_parent.json, by_cohort.json, per_image.csv and
completion.json. The completion file indicates a complete, hash-checked run.

## 5. Paired comparison, metrics and error gallery

```powershell
& $Python312 -B "$T\src\compare_tier_aware.py" --t0-dir "$T\metrics\${Run}_T0" --t1-dir "$T\metrics\${Run}_T1" --output-dir "$T\reports\${Run}_comparison" --gallery-limit 24
if ($LASTEXITCODE -ne 0) { throw 'Paired comparison failed' }
```

This writes comparison.json, comparison.md, summary.csv, parent_comparison.csv,
error_gallery.html and completion.json. The gallery embeds up to24 original
images with changed text/decisions; it does not regenerate crops or OCR. All
changed cases remain in comparison.json regardless of the display limit.
Open error_gallery.html or galleries/glyph_comparison.html in your browser.

## 6. Final historical/experiment integrity check

```powershell
& $Python312 -B "$T\src\verify_artifacts.py"
if ($LASTEXITCODE -ne 0) { throw 'Final integrity verification failed' }
```

All results live in new timestamped folders. Existing folders are never
overwritten. If a run is interrupted, it has no completion marker and cannot
enter comparison. Use a new $Run for a new attempt; there is no model training
or training-resume command. To reuse completed runs in a new terminal, restore
the old $Run value after the environment block.

Share summary.json, by_parent.json, by_cohort.json and per_image.csv for each
candidate, the comparison directory and any failure message. Check exact
accuracy, CER, incorrect/unscorable corrections, accepted-output correctness and
long/multi-row cohorts together. Review-to-ACCEPT conversion is not enough.

No command here evaluates the exposed434 partition. This development evidence
is also not independent unseen testing. Unknown real fonts and genuine image-
level negatives require separate data/protocol design before broader deployment.
