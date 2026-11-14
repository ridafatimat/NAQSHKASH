# Frozen E1 handwritten recognition baseline — PROVISIONAL

Recognition-only evaluation of the supplied corrected crops, not end-to-end page OCR. No page preprocessing, segmentation, tier detection, training or test-set evaluation was run.

## Actual results
| Metric | Result |
|---|---:|
| Sources scored | 3 |
| Tiers scored | 14/14 |
| CER including spaces | 0.742268 (74.23%) |
| CER excluding all whitespace | 0.887324 (88.73%) |
| Exact-tier accuracy (Stage 2E) | 0.00% (0/14) |
| Substitutions / deletions / insertions (spaces included) | 48 / 22 / 2 |
| Reference characters, with / without whitespace | 97 / 71 |
| Input exclusions / inference failures | 0 / 0 |

CER is micro-averaged edit distance/reference characters and may exceed 100%. Exact-tier equality is space-sensitive under the original E1 convention. Leading layout offsets are stripped with lstrip(' '); internal spaces are preserved. The no-space metric removes all whitespace consistently on both sides; it is not Stage 2E's separate collapse-spaces diagnostic. All per-tier predictions/references are in predictions.csv; no samples silently omitted.

## Per source
| Source | Tiers | CER with spaces | CER without whitespace | Exact tiers |
|---|---:|---:|---:|---:|
| all_003191 | 6 | 73.21% | 89.47% | 0/6 |
| all_003556 | 3 | 76.47% | 92.86% | 0/3 |
| all_009197 | 5 | 75.00% | 84.21% | 0/5 |

## Common edit alignments
- substitution: 'a' -> 'i', 5 times
- substitution: 'b' -> 'i', 3 times
- substitution: 'a' -> 'c', 2 times
- deletion: ' ' -> '', 2 times
- substitution: 'd' -> 'i', 2 times
- substitution: 'j' -> 'c', 2 times
- deletion: 'd' -> '', 2 times
- substitution: ' ' -> 'B', 2 times
- deletion: 'f' -> '', 2 times
- deletion: 'w' -> '', 2 times

These are minimum-edit alignments using Stage 2E tie preference (substitution, deletion, insertion), not independent semantic diagnoses. Per-role metrics separate upper/lower symbols from count strings in metrics.json.

## Input validation and provenance
The exact Downloads archive handwritten_tier_crops (5).zip contains an internal folder named handwritten_tier_crops (4); the outer version is authoritative, and current files/hashes were checked. Fourteen current validation crop/TXT pairs decode, dimensions match the manifest and all supplied crop hashes match. Labels match manifest source/zero-based row/tier and the existing source JSON. All inputs belong only to all_003191, all_003556 and all_009197. A visual inspection of the 14 current crops shows separated handwritten tiers, not synthetic renders or whole pages. This validates visible tier format, not every Talim symbol's semantic correctness. Faint strokes and residual ruling remain quality limitations.

Only validation crops, their TXT labels, the validation re-photograph, development manifest metadata, report and relevant rendering code were extracted under inputs/. Protected-test image/label members were never opened or extracted. No training images were loaded. Development metadata confirms 88 training-ready and 81 pending training tiers; these statuses were not independently approved by this evaluation.

## all_003191 row 1 correction
The current row-1 upper/count/lower crops have 2026-11-09 correction dates and hashes matching the latest manifest, different from previous hashes. Their source is the supplied real re-photograph, whose SHA-256 begins 7af54bc69e92490f. The current upper crop visibly contains five marks. Older report statements that only two marks are present, auto_flags MARK_GROUPS_2_VS_LABEL_5 and observed_ink_groups=2 are stale metadata; they describe the superseded original-page crop. They do not override the current re-photograph and corrected files. All three row-1 tiers are scored provisionally. Row 0 and row 1 now come from different photographs of the same source; this cannot be presented as one-photo end-to-end OCR.

## Faint-ink rendering limitation
The archived scripts/crop.py render_tier explicitly sets out[(out > 200) & keep] = 255. This can erase actual faint pen ink as well as paper texture. The current all_003556 lower crop uses the documented keep-faint exception; other current crops retain standard rendering lineage. Inspection shows faint/fragmented strokes, especially on all_003556, so quality is a significant confounder of recognition performance. Current pixel statistics are in metrics.json, but counts of surviving bright pixels do not measure erased strokes. The report's earlier 7-validation-crop loss estimate predates the newest row-1 re-photograph and is not claimed as a fresh measurement. Exact current loss requires original tier masks/normalization outputs that are not supplied as reusable masks. No re-segmentation, silent regeneration, threshold change or diagnostic rerender was performed. The primary baseline uses the supplied PNG bytes unchanged.

## Frozen model and validation
E1 checkpoint: C:\NAQSHKASH\stage2\stage2e_finetuning\checkpoints\selected_model\best.pt

SHA-256: 27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1 (matches original selected release metadata before/after inference). Vocabulary hash: 3a6a370c6940b737301025ac5182bf1862490b2e93d55307488c86c86c674bb5. Existing Stage 2E CRNN architecture, strict weights, tokenizer, grayscale /255 normalization, aspect-preserving height-64 resize, collator, true CTC lengths, greedy decoder and edit metrics were reused through context.py and evaluation.evaluate(output=None). No E0/E2 selection. 7 validation tests passed, 0 failed.

## Interpretation and readiness
All results remain PROVISIONAL: prior visual QA was performed by an AI assistant, not an independent Talim reader, and file-level agreement does not establish that the writer copied every symbol/count correctly. Crop preparation also used reference row/glyph information, so this is an oracle-assisted manually corrected crop benchmark, never a claim of independent segmentation performance.

The frozen model/input adapter is operational and this validation set can serve as a provisional recognition baseline. The full dataset is not yet ready for an unqualified fine-tuning release: 81 training tiers remain unresolved, faint-ink rendering is inconsistent, internal label spaces follow the synthetic grid rather than handwritten spacing, and semantic verification is incomplete. A separately authorized, controlled pilot using only the 88 currently marked training-ready tiers is technically possible after deciding the space policy, verifying those pairs and rendering train/validation consistently. Do not automatically approve those labels. Preserve this baseline and the protected test split.

Next: independent Talim-reader QA; correct/verify pending training pairs; recover faint ink through a separately versioned, consistent rendering pass; freeze a training-only protocol and compare against this unchanged provisional baseline. No fine-tuning was performed here. Original ZIP, source annotations, E1, Stage 2E/F and earlier Stage 2G results remain unchanged. No Git operations or dashboards.
