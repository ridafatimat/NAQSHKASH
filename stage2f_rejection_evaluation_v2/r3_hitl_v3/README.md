# R3_HITL_V3 — final Stage 2F development candidate

**At implementation delivery: unit-tested, full development evaluation NOT RUN, not frozen. No held-out evaluation is authorized or implemented by this entry point.**

R3 runs after existing R2 and imports the unchanged Stage 1 text-space engine. It preserves every R2 REVIEW/REJECT and its evidence. It never creates a text-only definitive REJECT. No model, preprocessing, canonical rule or image-based rejection method is added.

## Selected policy semantics

For an R2 ACCEPT, retain ACCEPT when usable OCR has **at least one adequately supported complete canonical page interpretation** with a common candidate-derived row total. Multiple complete candidates do not require REVIEW. Contradiction, resource cap, unsupported representation, validator uncertainty or unavailable OCR still yields REVIEW. Text consistency does not establish true image geometry or image validity.

The extra uniqueness requirement has been removed. Incomplete evidence still receives fail-safe REVIEW, so historical Stage 1 counts are comparisons rather than forced targets. The evaluator lists every differing image and reason. No confidence threshold is used.

Search budgets remain the existing 128 offset columns / 258 trials per row. Existing leading-blank and unsupported-space treatment, canonical uncertainty and conditional completeness are retained. No missing character, count, offset, expected total or source glyph is supplied from annotations.

## Exact manual VS Code PowerShell commands

```powershell
Set-Location 'C:\NAQSHKASH'
$R = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_hitl_v3'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$R\src;C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$Out = "$R\runs\$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')_$([guid]::NewGuid().ToString('N').Substring(0,8))"

# Metadata, lineage and code/config checks only; no policy evaluation or pixels.
& $Python312 -B "$R\src\evaluate_v3.py" evaluate --preflight
if ($LASTEXITCODE -ne 0) { throw 'R3 preflight failed.' }

# Saved development evidence only; no D1/E1 inference or source-image reading.
& $Python312 -B "$R\src\evaluate_v3.py" evaluate --output-dir $Out
if ($LASTEXITCODE -ne 0) { throw 'R3 development evaluation failed.' }

Get-Content "$Out\report.md"
Get-Content "$Out\outcomes.csv" -TotalCount 12
Get-Content "$Out\Stage1_differences.csv" -TotalCount 12
Write-Host "Detailed metrics: $Out\metrics.json"
Write-Host "Finding/localization evidence: $Out\per_image.jsonl"
```

Completed R2 and Stage 1 runs are located from existing artifacts, with source/completion hashes checked. If several R2 runs exist, explicitly append --r2-dir 'ACTUAL_DIRECTORY' and optionally --stage1-dir 'ACTUAL_COMPLETED_STAGE1_DIRECTORY'. The accepted source is exactly development_design, 2,800 IDs / 700 parents. The existing loader is not modified. No held-out split CLI exists.

### Optional small review gallery, AFTER the completed decision run

```powershell
& $Python312 -B "$R\src\evaluate_v3.py" overlays --run-dir $Out --limit 6
if ($LASTEXITCODE -ne 0) { throw 'Review gallery generation failed.' }

$Gallery = Get-ChildItem "$Out\overlays" -Directory | Sort-Object Name | Select-Object -Last 1
Get-Content "$($Gallery.FullName)\overlay_manifest.json"
Start-Process "$($Gallery.FullName)\index.html"
```

Optional --image-id IMAGE_ID may be repeated up to the selected limit (maximum 12). Only development REVIEW records are allowed. Selection favors new reviews with verified geometry. If none can be highlighted, the gallery contains structured explanations without reading image bytes or fabricating boxes. No 2,800-image overlay pass is offered.

The original development raw observations save processed D1 boxes but omit transforms. A perspective=False flag is insufficient to draw those boxes on original images. The generator uses only identity-matched, checksum-verified D1 transforms from the separate 48-image extraction cache, when available. Missing geometry stays unavailable. Source annotations, mutation locations, generator cell widths and candidate text columns are **never used to draw overlays**. Broad row/tier highlights say “REVIEW REGION — precise invalid glyph location not verified.” Annotated thumbnails are the only image copies made.

### Lightweight tests

```powershell
& $Python312 -B -m unittest discover -s "$R\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'R3 tests failed.' }
```

The supplied verification record reports 47 passed / 0 failed, including a two-image synthetic saved-evaluation round trip. No E1 model was initialized. Tests read no real image or held-out content.

## Outputs and separation

- run_manifest.json: timestamp/run ID, policy/config, input and dependency hashes, software, scope and initial status; completion.json is the authoritative final status.
- metrics.json: R2/Stage1/R3 outcome counts/rates, category/parent/variant/transcription results, transitions, new/preserved reviews, invariant checks, search coverage, all REVIEW reasons and localization coverage.
- per_image.jsonl: source identity, saved interpreted text, retained R2 decision, Stage 1 outcome, R3 decision/findings, alignment evidence, planned broad regions and separate offline localization scores. No mutation record/raw annotation is copied into runtime evidence.
- Stage1_differences.csv: every differing image and the policy-semantic explanation; no forced regression counts.
- outcomes.csv, report.md and completion.json: compact comparison and immutable output hashes.
- geometry_provenance.json: independently available D1 mapping evidence and unsupported mappings.
- overlays/<unique_gallery>/...: optional, separately hashed visualization artifacts; never policy inputs.

Review findings use zero-based internal row/column indices and one-based human text. Failure columns come only from supported canonical failure evidence on sampled candidate reconstructions. They are not a detected glyph position and are not an exhaustive candidate-location set. A decoded whole count tier is labeled as such rather than invented token boundaries. Associated symbols remain null when unsupported. Global row-total conflicts do not identify a guilty row.

Localization uses mutation records **only after the decision and region plan**. Row/tier metrics, conditional candidate-text-column hits, broad source-region/cell overlap and exact pixel localization are separate. Column scoring requires a demonstrably comparable count-origin string; otherwise it is unscorable. Broad row overlap is not precise mutation localization. Valid reviews have explanation/region availability metrics and no invented invalid mutation. Exact glyph/pixel localization has zero coverage in this implementation.

## What remains before formal closure

1. Run and review development results, all invariant checks, every Stage 1 difference, valid-review burden, remaining unsafe accepts and localization coverage.
2. Fix any genuine implementation defects before selecting the final version; preserve earlier runs.
3. Follow reports/evaluation_protocol.md to generate a **candidate** freeze manifest after explicit development review. It is not an automatic approved freeze.
4. Authorize and implement/audit a separate held-out reader/cache adapter without changing the protected development loader. Freeze that evaluation adapter and obtain validation authorization.
5. Run the frozen validation split once; review honestly; separately authorize and run the protected test once without policy tuning.
6. Complete the FYP template with actual results and limitations. Domain-expert validation and handwriting adaptation remain distinct future work.

No Git/GitHub work, OCR, training, threshold tuning, new image rejection, held-out image access or historical-file modification occurred during implementation.
