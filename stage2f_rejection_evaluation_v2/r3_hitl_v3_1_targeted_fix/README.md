# R3.1 final targeted fix

Revision `targeted_pitch_fix_1`, isolated in `r3_hitl_v3_1_targeted_fix`. Original R3.1 code and run `20261107_203523_866_d442571d`, verified R3, D1, E1, Stage 1 and all historical results remain unchanged. Read-only locks cover 80 historical inputs. No new rejection/selection architecture is introduced: the post-R3 policy, candidate filtering, 0.5-column tolerance, monotonic invariants and carpet-readiness safeguards are unchanged.

## Demonstrated correction

The former estimator required EVERY occurrence of one OCR character to have an identical binarized silhouette. OCR labels do not guarantee identical source glyphs/pixels; antialiasing/cleaning/context can also change the silhouette. Treating hash differences as failure is unsupported. Moreover, first-ink distance between DIFFERENT glyphs includes their different left bearings, so it need not equal cell pitch.

The existing silhouette hashes now provide positive translation anchors instead of a global equality veto. For consecutive occurrences of an identical observed silhouette WITHIN one count tier:

`pitch = (second_first_ink_x - first_first_ink_x) / (second_decoded_column - first_decoded_column)`

The same shape's bearing and the row origin cancel. The fixed estimator uses the median of these bearing-invariant observations. Decoded column separation includes preserved ASCII spaces; no source offsets/font/grid dimensions are consulted. Different silhouettes do not form an anchor pair and do not by themselves invalidate all other measurements.

Still required: exact group/visible-OCR-character cardinality, supported whitespace, at least THREE translation observations, compatible one-pixel observation intervals, all repeated-shape anchors supporting one lattice within one pixel, group width below pitch, exact D1 box equality and complete verified coordinates. The original three-observation minimum, one-pixel raster allowance, D1 threshold and 0.5-column candidate tolerance have NOT been relaxed. For an anchor spanning `d` decoded columns, one pixel means pitch uncertainty `1/d` pixel per column. All observation intervals must intersect and contain the median; otherwise measurement stays unavailable. No fallback or fitted offset is introduced. These support checks remain conditional on correct grouping and saved OCR spacing; they do not recognize glyph identities.

Three cached diagnostic cases are small regression fixtures, not a new full evaluation. Two provide five matching-shape translations all yielding 50 pixels per column. The third provides only two observations and correctly remains unavailable. Those 50-pixel results came from cached ink coordinates and decoded-column differences, not source pitch. No claim is made about final dataset accuracy/coverage or target outcome counts.

## Selected-carpet scoring correction

Verified selected-reading correctness is evaluated ONLY when a valid ACCEPT has one physically verified, carpet-generation-ready selected reading. Text-only selections without physical verification and absent selections are UNSCORABLE (`selected_reading_correct=null`), excluded from the incorrect-carpet denominator, and reported separately. If no reading is eligible, correctness/error rates are null, not zero or 100 percent. Coverage and unavailable valid-ACCEPT counts remain explicit.

Compare the intended ordered `symbol`/`count` pairs; source `direction` and `encoded_count` fields are annotation metadata, not unequal carpet content. No glyph-alias normalization is added. A genuinely different verified symbol/count reading is still counted incorrect. Invalid references remain unscorable for canonical carpet correctness, while unsafe ACCEPT is reported independently.

## One final manual development evaluation

```powershell
Set-Location 'C:\NAQSHKASH'
$Fix = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_hitl_v3_1_targeted_fix'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$Fix\src;C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"

# Metadata-only preflight; no image bytes or D1 execution.
& $Python312 -B "$Fix\src\evaluate_v3_1.py" --preflight --measure-scope all-development
if ($LASTEXITCODE -ne 0) { throw 'Targeted-fix preflight failed.' }

# Execute manually: all development images, frozen D1 only, no E1 OCR.
$OutFix = "$Fix\runs\$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')_$([guid]::NewGuid().ToString('N').Substring(0,8))"
& $Python312 -B "$Fix\src\evaluate_v3_1.py" --measure-scope all-development --output-dir $OutFix
if ($LASTEXITCODE -ne 0) { throw 'Final development evaluation failed; do not use incomplete output.' }

Get-Content "$OutFix\report.md"
Get-Content "$OutFix\R3_differences.csv" -TotalCount 20
$MetricsFix = Get-Content "$OutFix\metrics.json" -Raw | ConvertFrom-Json
$MetricsFix.candidates.R3_1.summary | ConvertTo-Json -Depth 15
$MetricsFix.scoring | ConvertTo-Json -Depth 8
$MetricsFix.unavailable_reasons | ConvertTo-Json -Depth 8
Write-Host "Detailed measurements: $OutFix\per_image.jsonl"
Write-Host "Candidate traces: $OutFix\measurement_diagnostics.json"
```

Lightweight tests, separately:

```powershell
& $Python312 -B -m unittest discover -s "$Fix\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Targeted-fix tests failed.' }
```

The unchanged evaluator checks the exact completed R3 run, raw/Stage 1 lineage and 2,800 development IDs / 700 parents, uses the unchanged protected development loader and original D1-only adapter, writes an exclusive new run, and seals completion/output hashes last. No validation/protected-image access, OCR, training, threshold sweep or Git/GitHub operation is supported. Source notation enters offline scoring only after decisions. No full development run or automatic freeze was executed during implementation.

Every ACCEPT retains its explicit selected-reading status. ACCEPT without `carpet_generation_ready=true` must not generate a carpet. Existing REVIEW/REJECT never becomes ACCEPT; no new REJECT is created. Old findings, region records and localization results remain preserved. Raw first-ink positions still have glyph-bearing uncertainty; no arbitrary bearing correction is fitted.

If the final manual run still lacks sufficient reliable image positioning or shows unsafe selected carpets, STOP and retain verified R3_HITL_V3. Do not tune thresholds or build another architecture from the exposed run.
