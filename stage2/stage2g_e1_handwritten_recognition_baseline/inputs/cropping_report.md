# Handwritten tier cropping: NAQSHKASH Stage 2G (authoritative split applied)

Update, 2026-11-09 (later): all_003191 row 1 re-made from a re-photograph of that row; all 14 validation tiers are now validation-ready. See **all_003191 row 1 from re-photograph** below.

Update, 2026-11-09: four validation tiers corrected and one validation count crop re-checked; see **Validation corrections (2026-11-09)** below.

Update, 2026-11-08: the authoritative Stage 2G source split has been applied. No cropping was rerun. All 394 existing crops, labels and preview sheets were moved into split-isolated folders, byte-identical (SHA-256 verified), and all 163 CVAT polygons were preserved.

## Split

| Split | Sources | Tiers | Ready | Not ready |
|---|---:|---:|---:|---:|
| train | 24 | 169 | **88 training-ready** | 81 (58 crop correction + 23 full manual annotation) |
| validation | 3 (all_003191, all_003556, all_009197) | 14 | **14 validation-ready** | 0 |
| protected_test | 3 (all_002266, all_009421, all_009894) | 18 | 6 held out | 12 (all_009894 needs full manual annotation) |
| **Total** | 30 | 201 | 108 | 93 |

Rules enforced in the manifests:
- `training_ready = True` **only** for CONFIDENT tiers from **train** sources (88). Every unresolved tier is `False`, whatever its split.
- `usage` says exactly what each tier may be used for: TRAIN_READY, VALIDATION_READY, PROTECTED_TEST_HOLDOUT, or NOT_READY_<reason>.
- **Protected test is isolated.** Its crops, labels, preview sheets, CVAT polygons and manifest live only under `protected_test/`. The development `manifest.csv`, `ambiguous_tiers_prioritized.csv`, `cvat_tier_polygons_development.xml` and `crop_preview_sheets/` contain no protected-test source, which was checked automatically.
- **all_003191 row 1 upper** (validation) stays flagged: `review_type = LABEL_VERIFICATION_BY_TALIM_READER`, `talim_reader_confirmation_required = True`. The label has 5 marks (o v = .|. ||), but the photo shows only `.|.` and `||`. A Talim reader must confirm the physical writing before this tier is used.

## Prioritised ambiguous list (`ambiguous_tiers_prioritized.csv`)

81 development tiers, in this order:
- **P1, validation**: none remaining. All validation tiers were resolved on 2026-11-09.
- **P2, train crop corrections** (58 tiers): sources with the fewest flagged tiers first, so the most pages become fully usable soonest.
- **P3, train full manual annotation** (23 tiers): pages all_004254 and all_007011.


The 12 unresolved protected-test tiers (all_009894) are listed separately in `protected_test/ambiguous_protected_test_tiers.csv`. They should be annotated by someone not involved in model development, without looking at any model output.

## Per source

| Source | Split | Tiers | Confident | Needs correction | Manual annotation |
|---|---|---:|---:|---:|---:|
| all_003191 | validation | 6 | 6 | 0 | 0 |
| all_003556 | validation | 3 | 3 | 0 | 0 |
| all_009197 | validation | 5 | 5 | 0 | 0 |
| all_000128 | train | 6 | 2 | 4 | 0 |
| all_000500 | train | 6 | 4 | 2 | 0 |
| all_000742 | train | 3 | 3 | 0 | 0 |
| all_000917 | train | 5 | 5 | 0 | 0 |
| all_001009 | train | 3 | 2 | 1 | 0 |
| all_001486 | train | 2 | 2 | 0 | 0 |
| all_001623 | train | 12 | 1 | 11 | 0 |
| all_001958 | train | 6 | 4 | 2 | 0 |
| all_002028 | train | 12 | 3 | 9 | 0 |
| all_002430 | train | 12 | 4 | 8 | 0 |
| all_002851 | train | 3 | 1 | 2 | 0 |
| all_003801 | train | 11 | 9 | 2 | 0 |
| all_004233 | train | 12 | 10 | 2 | 0 |
| all_004254 | train | 12 | 0 | 0 | 12 |
| all_004383 | train | 12 | 6 | 6 | 0 |
| all_005114 | train | 5 | 5 | 0 | 0 |
| all_005932 | train | 3 | 0 | 3 | 0 |
| all_006214 | train | 3 | 3 | 0 | 0 |
| all_006767 | train | 6 | 2 | 4 | 0 |
| all_007011 | train | 11 | 0 | 0 | 11 |
| all_007516 | train | 3 | 3 | 0 | 0 |
| all_008342 | train | 6 | 6 | 0 | 0 |
| all_008907 | train | 12 | 12 | 0 | 0 |
| all_009637 | train | 3 | 1 | 2 | 0 |
| all_002266 | protected_test | 3 | 3 | 0 | 0 |
| all_009421 | protected_test | 3 | 3 | 0 | 0 |
| all_009894 | protected_test | 12 | 0 | 0 | 12 |

## Method

1. **Ink extraction.** Darkest colour channel divided by a smooth paper estimate, then hysteresis thresholding so faint strokes stay connected. Notebook ruling and margin lines are removed (long thin structures, plus Hough lines that must be at least 75% continuous ink, so a row of aligned glyph tops is not mistaken for ruling). Page-edge and binding strips are removed. One per-page override: all_004254, threshold 40, because dark spiral binding hid the faint pencil.
2. **Rows.** Lines are fitted through count-sized glyphs, which handles slanted rows. The label is used only for the number of rows and the expected glyphs per row, to choose among candidate lines. It never moves a boundary.
3. **Tiers.** Every remaining stroke is assigned to the upper, count or lower tier of the nearest row by its position relative to that row's fitted count band. Between two rows, the nearer band wins, unless the label says that tier is empty.
4. **Crops.** Each non-empty tier is rendered on a white background using only its own strokes; other tiers and ruling are whitened. The row is rotated so its baseline is horizontal, then cropped to full row width with padding. This matches the Stage 1 / Stage 2E convention of full-width tier bands.
5. **Labels.** `tier_labels/<tier_id>.txt` holds the raw tier string exactly as in the JSON (leading and internal spaces kept, no newline). The Stage 2E loader strips only leading spaces, and `manifest.csv` also lists `ocr_text` (leading spaces removed) and `leading_offset`.
6. **Checks.** Automatic: glyph and mark counts vs the label, missing ink, ink in an empty-label tier, contact with the photo border. Then every page's preview sheet was checked visually.

## Who verified what

- Status values come from **Claude's visual QA** of `crop_preview_sheets/`. Claude is not a human Talim reader.
- **CONFIDENT** means the crop contains all of that tier's handwritten marks and nothing from other tiers, and the mark/glyph count is consistent with the label.
- A full symbol-by-symbol check that each person copied the reference correctly was not done. One clear discrepancy was found: **all_003191 row 1 upper**. The label has 5 marks (o v = .|. ||), but the photo shows only `.|.` and `||`. It is excluded from the confident set. Before using validation numbers in the report, a Talim reader should confirm the validation pages.

## Who verified what

- Status values come from **Claude's visual QA** of the crop preview sheets. Claude is not a human Talim reader.
- **CONFIDENT** means the crop contains all of that tier's handwritten marks and nothing from other tiers, and the mark/glyph count is consistent with the label.
- A full symbol-by-symbol check that each person copied the reference correctly was not done. Have a Talim reader confirm the validation pages before reporting results on them.
- All 30 reference TXT files agree exactly with their JSON `logical_rows`.

## Modelling caveats for fine-tuning

- Handwritten spacing does not follow the synthetic one-character grid, but the labels keep internal spaces from the synthetic layout. Report CER with and without spaces, or decide on a space policy before training.
- Crops are grayscale with a white background and varied stroke widths (ballpoint, pencil, marker), unlike the clean black Stage 1 renders.
- With 88 training-ready tiers, fixing the P2/P3 items roughly doubles the training data and is worth doing before fine-tuning.

## Files

| Path | Contents |
|---|---|
| `train/tier_crops/`, `train/tier_labels/` | 88 training-ready pairs (the only folder to train from) |
| `validation/tier_crops/`, `validation/tier_labels/` | 14 validation-ready pairs (4 corrected on 2026-11-09; all_003191 row 1 re-made from a re-photograph) |
| `validation/source_photos/` | The re-photograph of all_003191 row 1 as supplied |
| `validation/superseded_crops_v1/` | The 7 crops replaced on 2026-11-09, kept unchanged |
| `validation_manifest.csv` | Validation tiers only (14) |
| `review/train/`, `review/validation/` | Unresolved crops and labels (not for training or scoring) |
| `protected_test/` | All protected-test material, its own manifest, ambiguous list, CVAT polygons, preview sheets and `DO_NOT_USE_FOR_DEVELOPMENT.md` |
| `manifest.csv` | Train + validation tiers (183) with split, usage and training_ready |
| `ambiguous_tiers_prioritized.csv` | 81 unresolved development tiers (all training) |
| `split_assignment.csv` | Authoritative source → split mapping (30 sources) |
| `cvat_tier_polygons_development.xml` | Tier polygons for the 27 development images |
| `crop_preview_sheets/` | 27 development pages |
| `scripts/` | Pipeline plus `apply_split.py` (this update) |

Original ZIPs, the E1 checkpoint and Stage 2E/2F/2G results were not modified. No OCR, training or Git operations were run.

## Validation corrections (2026-11-09)

Pixel-level reassignment of existing pen ink from the original handwritten photographs. No synthetic reference images were used, nothing was drawn, and no label was changed. Details, before/after previews and checks are in `validation_corrections_20261109.zip`.

| Tier | Change |
|---|---|
| all_009197_row_01_upper | Restored the second dash of `≐` (1,778 ink px), which had been merged into the count oval |
| all_009197_row_01_count | That dash removed from the count crop |
| all_003556_row_00_lower | The two upper dots of `∵` reassigned from the count glyph. The faint bottom dot was always in the ink mask but had been whitened by the renderer, so this crop keeps original pixel values for pen ink. Two notebook-ruling fragments removed |
| all_003556_row_00_count | Two lower-tier dots removed. **This crop had been marked CONFIDENT on 2026-11-08; that QA call was wrong** |
| all_003191_row_01_upper | Unchanged; still awaiting Talim-reader confirmation |

**Package-wide finding (not changed):** the standard renderer whitens any pixel lighter than 200, including pixels inside the pen-ink mask. That thins faint strokes in every crop; in 14 training-ready and 7 validation-ready crops, at least 30% of their own ink pixels are whitened. Fixing it means re-rendering all crops consistently, which was not done here because training and protected-test crops were to be preserved. Decide on that re-render before fine-tuning, so that training and validation crops are rendered the same way.

## all_003191 row 1 from re-photograph (2026-11-09)

The user supplied a new photograph of all_003191 row 1, on a ruled spiral notebook in dark ballpoint, rotated 90°. Unlike the original page photo, it contains all five upper marks (o v = .|. ||). That resolves the label/handwriting mismatch with real handwriting; the label was not changed.

- **Processing:** the photo was rotated 90° counter-clockwise to upright and run through the same ink, row, tier and crop pipeline, with the label used only for the row and checks.
- **Pen ink restored on the ruled line:** a ruled line runs along the count glyphs, and this pen is the same grey as the ruling, so removing the line took pen pixels too. Removed pixels were restored only where the column's dark run is longer than the bare line (3 px + 1), only inside boxes of handwriting found by the standard pass, and only when attached to existing pen strokes. 2,951 px were restored and 40 unattached px dropped. Nothing was drawn.
- **Result:** upper 5/5 marks, count 15/15 glyphs, lower 6/6 marks (visual check by Claude). Ink is at least 11 px from every crop edge.
- **Provenance:** all three row 1 tiers now come from the re-photograph, so the row comes from one photo. Row 0 still comes from the original page photo, so a page-level end-to-end evaluation of all_003191 would need both photos. The earlier crops are in `validation/superseded_crops_v1/`, and the photo is in `validation/source_photos/`.
