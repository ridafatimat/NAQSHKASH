# R3_HITL_V3_1: measured ink-offset disambiguation

Isolated post-R3 candidate. R3 source/configuration and its completed development run remain read-only. No R3 acceptance logic is changed. Existing R3 REVIEW/REJECT decisions, explanations, findings, localization scores and review-region records are preserved. Only R3 ACCEPT can become REVIEW. No new definitive rejection is introduced.

## Runtime and carpet-generation contract

`policy_v3_1.decide(r3_decision, decoded_rows, analysis, measurement=None)` accepts only the existing decision, interpreted OCR text, saved Stage 1 candidates and image-measurement evidence. No source annotation, labels, mutation, expected offsets, renderer font/geometry or matched controls enter this API. Canonical candidates are rechecked with the existing `validate_aligned_row`; saved ordered runs and totals must match the check exactly. Incomplete evidence or missing run data is a recorded blocker, never an invented reading.

Distinct readings comprise the actual ordered symbol/count runs of every row with a common page row total. Offset alternatives yielding identical runs are grouped. The page candidate space is counted exactly in factored form without exponentially enumerating products; selected readings retain every supporting offset per row. No first-candidate selection occurs.

- One distinct complete reading supported by verified image offsets: ACCEPT, selected reading recorded, `carpet_generation_ready=true`.
- No complete reading matches positions/common totals: REVIEW, `R3_1_MEASURED_POSITION_MISMATCH`.
- Several distinct readings remain: REVIEW, `R3_1_AMBIGUOUS_AFTER_MEASUREMENT`.
- Positioning unavailable and text has one complete reading: preserve ACCEPT with `unique_text_only_position_unavailable`; **generation remains blocked**.
- Positioning unavailable and several readings remain: REVIEW, `R3_1_POSITION_UNAVAILABLE`.
- Existing REVIEW/REJECT: unchanged, no selected reading, generation blocked.

Consumers must require BOTH ACCEPT and `carpet_generation_ready=true` and use the explicitly selected complete reading. ACCEPT alone is insufficient. A future authorized human-confirmation path is not implemented here.

## D1-only measurement

The adapter imports the original `pipeline_d1.py` and calls `process_image` directly; it does not instantiate `FrozenPipeline`, load an E1 state dictionary or run OCR. Importing original D1 dependencies can import PyTorch utilities; this is not model construction/inference. Source-image bytes are read only with the unchanged development loader/hash check.

New D1 segmentation rows (roles, row indices, coordinates and order) must exactly equal saved R2 boxes. Mismatch diagnostics preserve BOTH versions and make measurement unavailable. All measurement occurs in processed-image coordinates. Source/processed dimensions, source SHA, processed-pixel SHA, D1 hashes and matrix are recorded. No old geometry is replaced. Binarization exactly reuses the frozen D1 engine's whole-page inverse Otsu method. The documented 128 fallback is unused because D1 binarization is available.

First ink is the leftmost occupied processed-image column within each bounded tier. Projection groups merge disconnected components only when x projections overlap/touch; component counts are stored. Separated fragments are not fused using assumed font size or cell width.

Pitch is the median distance between successive grouped count-glyph starts whose corresponding saved OCR characters are adjacent (no intervening decoded space). Before use:

1. Group cardinality must match visible decoded count-character cardinality, with supported whitespace only.
2. Repeated decoded characters must have identical binarized group silhouettes across contributing count tiers.
3. At least three usable adjacent-start distances must support a finite positive pitch.
4. Distances must agree within the fixed one-pixel raster allowance; all contributing groups must agree with the same decoded count-grid lattice within one pixel.
5. Groups spanning a pitch or more are unsupported. No renderer pitch, annotation offsets or numerical fallback is used.

The minimum support and one-pixel raster allowance are fixed engineering checks, selected before real-image evaluation, not fitted thresholds. Pitch observations, rejected grouping cases, residuals, interval and MAD are saved. The checks constrain fragmentation/merging but do not prove glyph identities: coincident compensating split/merge or OCR errors remain a measurement limitation. Real handwriting is outside this fixed-grid feasibility scope.

For each row, origin is minimum first ink across its observed tiers. Offset is `(tier_first_ink_x - row_origin) / page_pitch`. Nominal candidates use the requested **inclusive 0.5-column tolerance**, unchanged. The recorded pitch interval and one-pixel anchor allowance also propagate to offset intervals. If a potentially viable page candidate straddles the fixed tolerance boundary, evidence is treated as unavailable rather than resolving an ambiguity using uncertain rounding. No arbitrary offsets or bearing correction are fitted. Short/single-count pages, fragmented/touching ink and nonuniform bearings can be unscorable.

This is intentionally conservative. The reference 16,434/16,442 exact offsets, 138 valid ambiguities resolved and approximately 32 invalid ACCEPT are **unverified reference observations**, not promised results or parameter targets.

## Manual VS Code PowerShell commands

```powershell
Set-Location 'C:\NAQSHKASH'
$R31 = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_hitl_v3_1'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$R31\src;C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"

# Lightweight metadata, identity, saved-evidence and source integrity only.
# No image bytes, D1 execution or OCR in preflight.
& $Python312 -B "$R31\src\evaluate_v3_1.py" --preflight --measure-scope all-development
if ($LASTEXITCODE -ne 0) { throw 'R3.1 preflight failed.' }

# Run manually: all 2,800 development records, D1 only, including measurement
# diagnostics for prior REVIEW images so tier-offset coverage is independently audited.
$Out31 = "$R31\runs\$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')_$([guid]::NewGuid().ToString('N').Substring(0,8))"
& $Python312 -B "$R31\src\evaluate_v3_1.py" --measure-scope all-development --output-dir $Out31
if ($LASTEXITCODE -ne 0) { throw 'R3.1 development run failed; partial output is not complete.' }

Get-Content "$Out31\report.md"
Get-Content "$Out31\R3_differences.csv" -TotalCount 20
Get-Content "$Out31\outcomes.csv" -TotalCount 12
$Metrics31 = Get-Content "$Out31\metrics.json" -Raw | ConvertFrom-Json
$Metrics31.candidates.R3_1.summary | ConvertTo-Json -Depth 15
$Metrics31.scoring | ConvertTo-Json -Depth 8
Write-Host "Full evidence: $Out31\per_image.jsonl"
Write-Host "Measurement traces: $Out31\measurement_diagnostics.json"
```

Optional focused mode: replace `--measure-scope all-development` with `--measure-scope accepted` in BOTH commands. It still evaluates every saved development decision, but runs D1 only on prior R3 ACCEPT images. Coverage denominators explicitly retain all saved tiers and the attempted subset. The focused mode cannot verify the full reference tier-offset claim.

Lightweight tests only:

```powershell
& $Python312 -B -m unittest discover -s "$R31\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'R3.1 lightweight tests failed.' }
```

## Outputs, integrity and scoring

Each run uses an exclusive new directory within `r3_hitl_v3_1/runs`. `run_manifest.json` records source/config/protocol/test hashes, read-only R3/D1/Stage 1 dependencies, dataset manifest/raw lineage, fixed parameters, versions, split and measurement scope. Existing R3 completion/output hashes and exact 2,800 IDs/700 parents are checked. Held-out CLI choices do not exist; the original protected loader is unchanged. Global ownership metadata is checked by that loader, but excluded-split images/annotation files are never opened.

`per_image.jsonl` records decisions, complete measurement/provenance, factorized selection/support, every examined candidate's tier tests and page-total eligibility, unavailable evidence, preserved R3 diagnostics and separate offline scores. `metrics.json` records three outcomes and rates, transition matrix, parent/category/variant results, geometry equality, measurement coverage, offset errors/denominators, distinct reading counts, resolved/remaining ambiguities, unavailable positioning, generation readiness and selected-reading correctness. `R3_differences.csv` lists EVERY changed image; no counts are forced. `measurement_diagnostics.json`, `outcomes.csv`, and `report.md` provide compact inspection. `completion.json` is written LAST and seals output hashes; partial runs must not be used. Optional later overlays are not generated by this experiment; historical region records are retained unchanged.

Source raw display (including MUTATED notation), offsets, labels and source structured runs enter only `offline_scoring.py` after policy decisions. Row/tier correspondence must be established for offset scoring. Integer original tier offsets are compared to measured offsets relative to the observed row origin; exact and half-column accuracy remain separate. These are text-cell-offset scores, not glyph pixel-location accuracy. Correct selected reading means exact ordered symbol/count equality on valid ACCEPT images, with no alias normalization. Invalid images have no canonical true carpet reading: their selections are explicitly unscorable for correctness, and unsafe acceptance is reported separately. Full/eligible denominators and per-parent results remain visible.

No full real-image evaluation or freeze is executed by implementation/tests. Review the manual results, every changed valid case and every incorrect generation-ready selected reading before deciding on freeze. Results may differ materially from earlier reference observations because unsupported measurements remain unavailable. No threshold sweep, automatic rule change or protected-split adapter is provided.
