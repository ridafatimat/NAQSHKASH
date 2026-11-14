# Minimal CTC alignment feasibility pilot

**Ready for manual execution. The real experiment has NOT been run.**

This isolated pilot selects 12 development parents, four per category, with each parent's clean/illumination valid and invalid images: exactly 48 images. Only the affected logical row is forwarded, at most three tiers/image (144 tier forwards). Source pages have 6–23 columns: relatively short mutation-bearing cases within the frozen 5–64 medium length group. No long-page conclusion is possible.

Selection is purposeful, based on exposed saved predictions: four distinct wrong-tier glyph errors (D/U/F/T); four preserved count-order mutations spanning available lengths; two lost internal gaps (one/two cells); one leading and one trailing absent count. See configs/selection.json for all IDs, hashes, row indices, categories and rationales. Illumination controls are selected from the same parent quartet, rather than comparing an illuminated negative with the clean anchor.

## Exact VS Code PowerShell commands

No new packages are required beyond the existing Python 3.12 CPU evaluation environment. Stop if any command fails.

```powershell
Set-Location 'C:\NAQSHKASH'
$P = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$P\src;C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$Tag = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$Raw = "$P\runs\${Tag}_extraction"
$Scored = "$P\runs\${Tag}_evaluation"

& $Python312 -B "$P\src\experiment.py" preflight
if ($LASTEXITCODE -ne 0) { throw 'Pilot preflight failed.' }

# The only command that loads E1 and performs real OCR: CPU, selected 48 images.
& $Python312 -B "$P\src\experiment.py" extract --output-dir $Raw
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed; inspect the error before resuming.' }

# Cached-data scoring and gallery only. No model or source-image loading.
& $Python312 -B "$P\src\experiment.py" evaluate --run-dir $Raw --output-dir $Scored
if ($LASTEXITCODE -ne 0) { throw 'Alignment evaluation failed.' }

Get-Content "$Scored\report.md"
Write-Host "Metrics: $Scored\metrics.json"
Start-Process "$Scored\gallery\index.html"
```

Resume an interrupted extraction with the **same $Raw**, code, config and environment:

```powershell
& $Python312 -B "$P\src\experiment.py" extract --output-dir $Raw --resume
if ($LASTEXITCODE -ne 0) { throw 'Resume failed.' }
```

For deliberate chunks, append `--max-new-images 8`; repeat with `--resume` until 48 records are sealed. Evaluation rejects partial runs. Committed images are skipped; abandoned uncommitted asset directories are preserved and ignored. Do not restart under a new $Raw unless intentionally starting another experiment. There is no automatic retry, threshold search, alternate checkpoint or GPU option.

Optional lightweight tests (synthetic tensors/mocked pipeline only):

```powershell
& $Python312 -B -m unittest discover -s "$P\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Pilot tests failed.' }
```

## Reuse and isolation

The original FrozenPipeline.observe performs byte decoding, D1, grayscale normalization, the existing resize, greedy decoding and confidence computation. A temporary D1 wrapper retains its full matrix/dimensions and filters its return to the selected row; a forward hook copies logits. Both hooks are removed in finally. No D1/E1/T1/F1/R2 file is modified. The adapter receives **only image bytes and the selected row index**, not labels, layout, source glyphs, mutation type, focus column or desired outcomes. Row selection is an offline sampling operation, not a decision feature. No T1 or rejection policy is run.

The existing DevelopmentDataset, image_bytes, saved-raw loader, deterministic per-image seed, CPU model loader and immutable ObservationStore are reused. The selected configuration and code are frozen by a new local lock; existing historical locks remain unchanged. Only selected image bytes are opened by extract. The loader's development manifest metadata and saved development observations are read for checks; validation/protected images are never opened. Evaluation reads cached crops/tensors and source-renderer metadata, not dataset image bytes.

## Evidence and coordinate conventions

Each extraction record retains original/processed dimensions, full D1 row geometry, source-to-processed 3x3 matrix, target crop boxes/dimensions, exact resized dimensions and zero padding. Per-tier compressed NPZ files contain float32 logits [T,66] and the full greedy class path. JSON contains every constant-class path run, including blanks and repeat boundaries, plus emitted character spans. Processed crops are saved losslessly.

Width is floor-divided twice: T=floor(W/4). Nominal lattice centers use 2+4t in resized pixel-edge coordinates. Inverse bilinear resizing, actual rounded width, crop offsets and inverse D1 homography map spans to source intervals/polygons. Mapping math uses CV2 pixel-center coordinates, first pixel=0; interval edges can be -.5. The saved source point uses the crop's vertical midline. **Only horizontal/source-x accuracy is scored**; this is not vertical glyph localization.

Both stride-cell spans and nominal 42px CNN support are saved. They are not calibrated confidence intervals; the BiLSTM uses the entire tier. A blank class (0) does not mean visible whitespace. Literal space is class 1 and remains distinct. No blank run is converted into a count absence or a new space.

Before scoring, all target-tier strings, roles and full D1 geometry must match saved raw observations. Mismatches are recorded, receive zero coverage, and are excluded from conditional positional/association accuracy. They are never overwritten or corrected to match the baseline.

## Offline measurements and limits

Reference ink boxes are reconstructed from **each image's own raw_display**, its verified layout and the source font. Mutated images never inherit a valid parent's transcription. Renderer metadata and the focus column enter only offline scoring. No CTC targets are created or forced.

All minimum-edit text alignments are considered; only uniquely corresponding exact/B-X-equivalent visible glyphs receive position scores. Unrecognized or ambiguous glyphs remain uncovered in the denominator. Metrics include source-x error in pixels/glyph widths, coverage, ambiguous/wrong/extra/unmapped glyphs, whole-row and focused-token order, conditional symbol/count association accuracy, association scoring coverage, valid-control error/incomplete-evidence flags, parent/category/variant results, and paired clean/illumination source-position shifts.

The **diagnostic association algorithm** takes predicted glyph positions alone. It partitions the count sequence into nonempty contiguous groups in predicted symbol-x order, minimizing first-count/symbol-center distance. Equal-cost partitions are ambiguous; fewer counts than symbols is unscorable. It does not parse count values or make Talim decisions. Empty tokens are excluded from order accuracy and explicitly reported as having no visible count to align. Association errors can reflect this deliberately simple candidate algorithm as well as localization; they are not proof that all future alignment methods will fail.

The focused-token metric prevents unaffected header/neighbor tokens from hiding a mutation-local failure. Valid controls are scored at the same source column as their invalid pair, only for offline comparison. A zero false-risk flag with incomplete coverage does not establish safety. No actual false rejection rate exists because no rejection rules are executed.

Evaluation produces report.md, metrics.json, per_image.json, completion.json and a compact gallery (clean valid/invalid controls; illumination crops remain in the extraction cache). Blue boxes show offline reference ink; red lines show nominal CTC span centers. The root gallery/index.html is a clearly labeled synthetic preview until a real run is available.

The generated feasibility verdict uses preregistered necessary checks (reproduction, control errors/coverage, token assignment coverage). No numerical localization threshold is calibrated or claimed. Review upper-tail position error against association boundaries and glyph widths. Even a promising pilot does not authorize R3 rejection; missing-count absence and wrong glyph identity still require independent image support.
