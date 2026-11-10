# Run dataset-v2 manually after preflight review

Open C:\NAQSHKASH in VS Code. Start a PowerShell terminal. Codex generated only
the small preflight; do not run full generation until you approve its scope.

## 1. Environment and paths

```powershell
Set-Location 'C:\NAQSHKASH'
$N = 'C:\NAQSHKASH\stage2\stage2f_negative_dataset_v2'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH="$N\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$Run = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
& $Python312 -B -c "import sys,PIL,numpy; print(sys.executable); print('Pillow',PIL.__version__); print('numpy',numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Existing Python dependencies unavailable' }
```

This reuses the actual Python3.12 and dependency directory from previous stages.
No new installation or GPU is needed. Pillow handles glyphs/images; the existing
scan engine uses NumPy. No torch/model import, CUDA, training or resume is needed.
The local Talamat font and canonical tools are read in place, never copied.

## 2. Inspect preflight and run lightweight checks

Open `$N\galleries\preflight.html` and read reports/canonical_rule_audit.md,
invalidity_categories.md, preflight_validation.md and manual_review_checklist.md.
The full-resolution zooms make dense examples readable. No human review is claimed.

```powershell
& $Python312 -B -m unittest discover -s "$N\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Lightweight tests failed' }
& $Python312 -B "$N\src\validate_dataset.py" --dataset "$N\dataset\preflight" --report-dir "$N\reports\${Run}_preflight_check"
if ($LASTEXITCODE -ne 0) { throw 'Preflight validation failed' }
```

Success requires complete=true in preflight completion.json, no validation
exception, no duplicates/cross-split parents, and all semantic admissions proven.
Do not rerun generation into dataset/preflight: existing outputs are preserved.
Do not regenerate design/historical locks to bypass a failure.

## 3. Build complete historical overlap index (MANUAL metadata sweep)

This reads JSON from the known100,000 Stage1 parents,10,000 general parents,
326 positive-edge parents and two old negative content templates. It also
fingerprints the original archives. It does not read/run model predictions or
extract images. It may take time; Codex has NOT run this full historical sweep.

```powershell
$History = "$N\manifests\history_full_${Run}.json"
& $Python312 -B "$N\src\build_history_index.py" --mode full --output $History
if ($LASTEXITCODE -ne 0) { throw 'Historical index creation failed' }
```

Expected: full_historical_coverage=true, errors=[], source counts100000/10000/326.
If counts differ or source errors exist, investigate the source index; do not
claim the resulting protected split is independently unseen. The earlier bounded
preflight index is intentionally insufficient for full generation.

## 4. Generate full dataset (MANUAL, only after approval)

```powershell
$Dataset = "$N\dataset\full_${Run}"
& $Python312 -B "$N\src\generate_negative_v2.py" --config "$N\configs\generation_v2.yaml" --mode full --history-index $History --output-dir $Dataset
if ($LASTEXITCODE -ne 0) { throw 'Full generation failed: preserve incomplete folder and inspect failure; do not bypass verifier' }
```

Target:1,000 independent canonical families, one clean negative and matched
control per family, and one illumination variant of each (approximately4,000
files). New semantic collisions are excluded rather than weakly filling quotas.
All variants inherit their source's split; preflight sources cannot enter full.

Source splits are70% development_design,15% validation_calibration and15%
protected_test. Full construction verification runs before sealing. The gallery
excludes protected examples. After protected_test_seal.json exists, automated
validation checks only their hashes/ownership, never re-parses their notation.
Do not inspect those examples to modify rules, augmentations or thresholds.

Expected run outputs: valid_controls/, invalid_talim/, scanned_variants/,
annotations/, parents/, manifest.json, manifest.csv, parents.json, reports/,
gallery.html, protected_test_seal.json, completion.json. Completion is written
only after verification and integrity checks. Invalids have no fake CTC targets.

## 5. Validate full output and create a fresh non-test gallery

```powershell
& $Python312 -B "$N\src\validate_dataset.py" --dataset $Dataset --report-dir "$N\reports\${Run}_full_validation"
if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed' }
& $Python312 -B "$N\src\gallery.py" --dataset $Dataset --output "$N\galleries\${Run}_full_non_test.html" --limit 40
if ($LASTEXITCODE -ne 0) { throw 'Non-test gallery failed' }
```

The validator also verifies old E1/D1/T1/F1/canonical inputs and the frozen
generation design. Protected test files remain hash-only. No CRNN evaluation,
training or calibration is invoked by any command.

## Preservation, interruption and review

All output paths must be new; scripts refuse overwrites. Keep `$Run` and the
printed paths. An incomplete run lacks completion.json and cannot be treated as
a valid dataset. There is no automatic resume/backfill: use a new timestamp after
reviewing the failure. Preserve failed outputs as evidence. Identical seed/config
reproduces the same parents; separate copies are not new independent evidence.

Share preflight review decisions first. After an authorized full run, share
generation_summary.json, construction_validation.json, excluded_sources.json,
validation.json, completion.json, history-index coverage counts and non-test
gallery findings. Do not share protected-test image content for design/debugging.
No classifier training, rejection evaluation or new experiments are authorized
by these dataset-generation commands.
