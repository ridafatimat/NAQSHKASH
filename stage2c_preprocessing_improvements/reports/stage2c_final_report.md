# NAQSHKASH Stage 2C — robust preprocessing and segmentation

Completed locally on 3 November 2026. All new work is under C:\NAQSHKASH\stage2\preprocessing_improvements. Stage 1, Stage 2A and Stage 2B remain unchanged. No source images/labels were changed or regenerated. No neural model, count classifier or OCR model was trained. No Git operations or cloud services were used.

## Outcome and recommendation

**The predeclared tuning rule selected C4 before validation was opened. This is an experimental selection, not a deployment endorsement.**

Do not promote the selected configuration into the main Stage 1 workflow. It improves recall/coverage but still wrongly accepts 56 internal-validation scanned pages. Keep it isolated. Prioritise role-confidence calibration, difficult row grouping and crop validation on development data; preserve the current checkpoint and do not start CRNN fine-tuning.

The experiment deliberately exposes both improvements and wrong acceptances. Higher proposal coverage or lower conditional OCR error alone is not sufficient evidence of a reliable blind page pipeline. No existing application/preprocessing entry point was switched to a candidate.

## Frozen development design

| Pool | Tuning parents | Internal validation parents | Separate clean regression parents |
|---|---:|---:|---:|
| Positive Stage 2A training partition | 182 | 46 | 0 |
| Unexposed general supplementary development | 100 | 50 | 0 |
| Confirmed original Stage 1 training | 0 | 0 | 100 |

There are **478 source parents and 1,612 existing images**: 228 positive parents with one clean and three scanned versions (912 pages), 150 general parents similarly represented (600 pages), and 100 normal clean training pages. Six configurations produce **9,672 page-candidate outcomes**. All variants remain with their source parent. Complete source-image hashes and full logical identities were checked for cross-partition duplication. Shared isolated glyphs are not treated as identical source parents.

All 150 parents exposed in Stage 2B (49 positive, 101 general) were excluded, including the general content-only match. Positive sources come only from the frozen 228-parent training pool; the 49 original positive validation parents and 49 positive test parents were not used for optimization or comparison. The general development pool is drawn from previously unexamined, unmapped parents, and remains **exploratory/development-only**. It is not relabelled as an independent Stage 1 test. Normal clean parents are selected from the actual 150,000-tier Stage 1 training manifest, which contains 79,918 distinct parent IDs.

Data and evaluation thresholds were frozen before experiments. All 478 selected source TXT files agree with JSON tiers and existing canonical Talim rules. Original clean/scanned ZIPs are read in place. The protocol locks source archive hashes, manifests, old split assignments, model, vocabulary and decoder; a software lock preserves the ten files used by the inference implementation. The final audit verified 6,565 protected historical project-file hashes. The reproduction launcher additionally verifies the full recorded history before inference. All old Stage 2B reports/manifests are preserved.

Files: `../protocol/design.json`, `../protocol/lock.json`, `../protocol/software_lock.json`, `../protocol/protected_before.json`, `../manifests/parents.csv`, `../manifests/pages.csv`.

## C0–C5 candidates

| Candidate | Change from reference |
|---|---|
| C0 | Exact Stage 2B image-only baseline proposals, original Stage 1 preprocessing |
| C1 | C0 plus image-derived illumination/background normalization |
| C2 | Component-aware tier bands, joint image-only row/role grammar, adaptive whitespace-limited crop padding |
| C3 | Original rotation correction plus guarded baseline-convergence projective correction; original blind grouping |
| C4 | Combined C1/C2/C3 processing |
| C5 | C4 plus gated edge-preserving noise handling and stricter ambiguity rejection |

The robust grouper reconnects short vertical glyph fragments at a scale estimated from connected components. It scores band morphology and the Talim grammar of optional upper, mandatory count, optional lower without an expected number of rows. Count-role evidence uses component height/width and loop structure; it is a hand-designed heuristic, not a trained role classifier. Alternative assignments with close scores can be rejected. Crop padding is clamped to observed whitespace separators rather than allowed to overlap the neighbouring band.

Normalization estimates local paper illumination and divides it out; already-white clean pages bypass this change. Geometry reuses the unchanged Stage 1 rotation estimator and adds a projective attempt only when image baseline support, transform extent and image-only projection-score guards pass. It does not use known augmentation angles/corners. The current projective model is limited to supported baseline convergence; it is not a general page-curl or camera-calibration solution. When both rotational and projective corrections occur, two resamplings are possible, which is a remaining image-quality limitation. C5's denoising is gated by image texture; it does not blindly erase small connected components in the OCR image.

No sample IDs, source recipes, expected counts, label strings, tier annotations or OCR predictions enter `process_image(image, configuration)`. No OCR hypothesis selects or repairs a crop. Ground truth is consulted only by the separate offline evaluator after proposals are fixed.

## How segmentation correctness is measured

The selected source renderer uses 24-pixel margins, eight-pixel line spacing and 20-pixel inter-row spacing. Offline reference tier masks come from those source slots, verified against every clean source pixel and annotated nonempty tier. Only the reference masks are transformed using the existing recorded rotation/perspective recipe; no degraded image dataset is regenerated. These annotations are never available to deployment.

A page is reference-correct only when row count, ordered tier roles and every crop pass: **at least 98% of its intended ideal tier ink retained, and at most 2% neighbouring-tier contamination**. Reference points are mapped forward through the candidate transform, so clipping outside its canvas cannot disappear from the denominator. CTC checks use the original vocabulary and repeat-aware required sequence lengths. OCR is evaluated only for complete pages with correct reference geometry and feasible targets; incomplete/wrong proposals are not artificially scored as character deletions.

- Proposal coverage: algorithm accepts / all input pages.
- Correct segmentation coverage: reference-correct proposals / all input pages.
- Wrong acceptances: algorithm accepts, but offline reference rejects.
- Safe rejections: algorithm abstains; this means no unverified crop is used, **not** proof that a valid page is inherently unreadable.
- Accepted segmentation accuracy: reference-correct / algorithm-accepted.
- Conditional CER and exact tier accuracy: only correctly segmented, CTC-feasible pages.
- Page-exact/all: complete correct segmentation and exact OCR for every tier / all pages, including rejections as unsuccessful.

Recorded geometry is exact at the documented operation level but pixel masks are nearest-neighbour ideal-ink references. Blur halos, JPEG interpolation and subpixel effects are approximations; masks are not manually verified pixel polygons for all scans. The separate component diagnostic and visual panels help expose preservation issues that aggregate scores can miss.

## Candidate selection and internal-validation results

The tuning-only pilot used 20 tuning parents (80 pages); C0 correctly segmented 36/80, robust candidates 60/80, while C5 reduced pilot wrong acceptances from 14 to four. Pilot results are preserved and were not used as validation results.

The complete selection rule was: highest correct scanned tuning-page count, then fewer wrong acceptances, then lower CER on the shared correctly segmented tuning subset. Selection was written after **6,768 tuning page-candidate evaluations** and before any validation page was evaluated. Its saved winner remains C4, regardless of subsequent validation findings. This lexicographic rule can favour a small coverage gain despite a large wrong-acceptance increase; the result demonstrates why a precision/safety gate is needed before future promotion. We did not retrospectively relabel another candidate as the winner.

### positive internal validation — scanned pages

| Candidate | Proposals | Correct segmentation | Wrong acceptances | Rejections | Correct among accepted | OCR tiers | Conditional CER | Exact tiers | Page exact/all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 54/138 | 42/138 | 12 | 84 | 77.778% | 84 | 2.381% | 96.429% | 28.261% |
| C1 | 54/138 | 42/138 | 12 | 84 | 77.778% | 84 | 2.381% | 96.429% | 28.261% |
| C2 | 135/138 | 94/138 | 41 | 3 | 69.630% | 226 | 27.837% | 88.938% | 56.522% |
| C3 | 54/138 | 42/138 | 12 | 84 | 77.778% | 84 | 2.381% | 96.429% | 28.261% |
| C4 | 135/138 | 94/138 | 41 | 3 | 69.630% | 226 | 27.532% | 90.708% | 57.246% |
| C5 | 107/138 | 94/138 | 13 | 31 | 87.850% | 226 | 27.522% | 90.708% | 57.246% |

| Candidate | Common pages | Common tiers | Common CER | Common exact accuracy |
|---|---:|---:|---:|---:|
| C0 | 32 | 64 | 3.125% | 95.312% |
| C1 | 32 | 64 | 3.125% | 95.312% |
| C2 | 32 | 64 | 3.125% | 95.312% |
| C3 | 32 | 64 | 3.125% | 95.312% |
| C4 | 32 | 64 | 3.125% | 95.312% |
| C5 | 32 | 64 | 3.125% | 95.312% |
### general internal validation — scanned pages

| Candidate | Proposals | Correct segmentation | Wrong acceptances | Rejections | Correct among accepted | OCR tiers | Conditional CER | Exact tiers | Page exact/all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 112/150 | 111/150 | 1 | 38 | 99.107% | 697 | 6.411% | 86.227% | 48.667% |
| C1 | 112/150 | 111/150 | 1 | 38 | 99.107% | 697 | 1.812% | 92.970% | 56.000% |
| C2 | 150/150 | 133/150 | 17 | 0 | 88.667% | 880 | 6.264% | 86.591% | 58.000% |
| C3 | 113/150 | 112/150 | 1 | 37 | 99.115% | 709 | 6.333% | 86.319% | 48.667% |
| C4 | 150/150 | 135/150 | 15 | 0 | 90.000% | 904 | 1.919% | 92.699% | 65.333% |
| C5 | 147/150 | 134/150 | 13 | 3 | 91.156% | 898 | 1.855% | 93.096% | 67.333% |

| Candidate | Common pages | Common tiers | Common CER | Common exact accuracy |
|---|---:|---:|---:|---:|
| C0 | 111 | 697 | 6.411% | 86.227% |
| C1 | 111 | 697 | 1.812% | 92.970% |
| C2 | 111 | 697 | 6.411% | 86.227% |
| C3 | 111 | 697 | 6.411% | 86.227% |
| C4 | 111 | 697 | 1.835% | 92.826% |
| C5 | 111 | 697 | 1.765% | 93.257% |

### Newly covered long sequences reveal a separate OCR limitation

Do not hide the increased positive conditional CER: C4 scores **27.532% CER** on its expanded scanned positive subset, versus C0's 2.381% on a much smaller subset. The common-subset CER is **3.125% for both**. One newly covered parent, **edge_00323**, contributes 710 of 723 C4 errors (98.20%) and 2274 of 2626 characters (86.60%). C4's conditional mean of per-parent CER is 4.494% over 32 scored parents; this diagnostic does not replace character-weighted CER.

The same parent has **112 deletions on its clean image**, all verified to be missing internal spaces: the non-space sequences match. Across C4's newly covered positive clean-validation subset, CER is **12.785%**, so this is not merely a scanned-background effect. On the long perspective variant, actual count/symbol substitutions and substantial deletions also remain. This exposes limitations of spacing/long-sequence decoding and unresolved distortion even with correct crop structure. C0 avoids the difficult parent by rejecting it; its smaller conditional CER cannot establish superior recognition on that parent.

No spaces were stripped from the targets or headline metrics, and no parent was removed. `long_sequence_diagnostic.json` records the concentration and clean-space-only diagnostic. Next-stage recognition analysis should separate exact spacing from glyph errors while preserving the original evaluation target, and should not automatically interpret this as permission to fine-tune.

On the general validation cohort, C1 isolates a clearer photometric improvement: the same 111/150 correctly segmented pages have CER **6.411% with C0 versus 1.812% with C1**. C4 scores 1.919% on its larger accepted subset and 1.835% on the all-candidate common subset. These comparisons support normalization while distinguishing its OCR benefit from the extra risks introduced by grouping/geometry.

Common-subset rows use the intersection of correctly segmented, OCR-feasible pages across **all six** configurations within each cohort. They compare identical page/tier labels, avoiding the claim that a lower conditional CER caused only by rejecting harder pages is an OCR improvement. Counts and all tuning/clean/validation rows are in `experiments_v1/summary.csv` and JSON. Per-augmentation metrics are in `augmentation_metrics.csv`; no new augmentation categories were invented.

Parent-bootstrap diagnostic intervals for the selected candidate versus C0:

- positive: correct scanned segmentation coverage change +37.68 percentage points; parent-bootstrap 95% interval [+19.57, +54.35], 46 parents.
- general: correct scanned segmentation coverage change +16.00 percentage points; parent-bootstrap 95% interval [+8.00, +24.67], 50 parents.

These intervals resample source parents (2,000 replicates, seed42), keeping their three correlated variants together. They describe internal development variation, not independent-test or handwriting generalisation.

## Clean-image regression

The table uses 100 confirmed Stage 1 training parents, not the original held-out test. They were not involved in the saved tuning selection. Candidate effects are measured with the same checkpoint on the same clean images.

| Candidate | Correct pages | Wrong acceptances | Rejections | Conditional CER | Exact tiers | Common pages | Common CER |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 91/100 | 0 | 9 | 0.185% | 98.985% | 88 | 0.191% |
| C1 | 91/100 | 0 | 9 | 0.185% | 98.985% | 88 | 0.191% |
| C2 | 97/100 | 3 | 0 | 0.176% | 99.055% | 88 | 0.191% |
| C3 | 90/100 | 0 | 10 | 0.186% | 98.974% | 88 | 0.191% |
| C4 | 96/100 | 4 | 0 | 0.177% | 99.046% | 88 | 0.191% |
| C5 | 96/100 | 3 | 1 | 0.177% | 99.046% | 88 | 0.191% |

Exact sequence and common-subset results must be read together with page coverage. Also see the positive/general clean-source rows in the machine-readable summary; no full Stage 1 held-out evaluation was rerun or overwritten.

## Failure patterns and limitations

First recorded failure reasons on internal-validation scanned pages:

- **C0:** blind_ambiguous_band_count:4: 65; blind_overlapping_padded_bands: 54; row_count_mismatch: 12; blind_ambiguous_band_count:7: 1; tier_role_or_grouping_mismatch: 1; blind_ambiguous_band_count:8: 1; blind_ambiguous_band_count:5: 1.
- **C1:** blind_ambiguous_band_count:4: 65; blind_overlapping_padded_bands: 54; row_count_mismatch: 12; blind_ambiguous_band_count:7: 1; tier_role_or_grouping_mismatch: 1; blind_ambiguous_band_count:8: 1; blind_ambiguous_band_count:5: 1.
- **C2:** tier_role_or_grouping_mismatch: 35; row_count_mismatch: 23; no_valid_row_grammar: 3.
- **C3:** blind_ambiguous_band_count:4: 66; blind_overlapping_padded_bands: 52; row_count_mismatch: 12; blind_ambiguous_band_count:8: 2; tier_role_or_grouping_mismatch: 1; blind_ambiguous_band_count:5: 1.
- **C4:** tier_role_or_grouping_mismatch: 33; row_count_mismatch: 23; no_valid_row_grammar: 3.
- **C5:** ambiguous_role_or_row_assignment: 31; tier_role_or_grouping_mismatch: 24; no_valid_row_grammar: 3; row_count_mismatch: 2.

Visual review confirms both improvement and regression. For `edge_00242__aug_03_jpeg`, adaptive grouping recovers the upper/count/lower blocks that C0 rejects. For `edge_00270__clean`, a three-dot upper symbol is still split into multiple bands and misassigned. For `edge_00049__aug_03_noise`, apparently reasonable rectangles have incorrect tier roles despite C0 assigning them correctly. Inferred roles are printed on the panels; good-looking boxes alone do not prove correct segmentation.

These reasons are not mutually exclusive underlying causes: an early band failure can hide a later role error. Wrong role/grouping and row assignments can remain despite improved band joining. Guarded perspective correction cannot reliably recover every long, sparse or converging layout. Small confidence differences and similar count/symbol morphology remain difficult. Fine-tuning a CRNN cannot directly repair these image-only grouping mistakes.

The original Stage 2B conditional OCR figures were verified independently from 3,737 saved predictions before this work. Its blind pipeline already failed on 35/49 clean positive parents. Stage 2C's new cohorts and stricter reference-ink checks mean these new results should be compared against **C0 on the same cohort**, not directly against an old Stage 2B headline number.

## Individual dot/component preservation

Before validation began, a separate fixed 90-page diagnostic was registered: ten positive validation parents, ten general validation parents (all clean/scanned versions), and ten normal clean regression parents. It compares C0 with the **already selected** candidate. It cannot select a new winner.

Every ideal transformed ink component of at least three reference pixels is checked individually; less than 95% retention is flagged. Components smaller than three pixels are outside this diagnostic. This is stricter for tiny dots than an aggregate 98% tier-ink threshold.

| Candidate | Pages | Structurally aligned pages | Components checked | Components with loss | Pages preserving all components | Aggregate-pass pages with component loss |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 90 | 50 | 2562 | 0 | 50 | 0 |
| C4 | 90 | 70 | 6254 | 0 | 70 | 0 |

Component-level counts are conditional on matching structure; rejected/misgrouped pages remain failures of coverage, not automatic preservation successes. Full records and flagged component examples are in `component_audit_pages.csv` and `component_loss_examples.csv`.

## Visual evidence and tests

Before/after panels include successful recovery, wrong acceptance, safe rejection, regression and very wide cases where available. Full-resolution images and left/right details are preserved so tiny dots/strokes can be inspected without relying on a downscaled thumbnail. The selected-candidate gallery is `../examples/selected_candidate/index.md`; earlier C0/C2/C5 experiment panels remain under `experiments_v1/examples/`. Expected labels and evaluated predictions are offline sidecars; no incorrect crop is passed to OCR solely to create an attractive example.

**69 tests passed; 0 failed, 0 errors, 0 skipped.** This includes new contract/safety/audit tests and relevant unchanged Stage 1 preprocessing, CRNN and CTC regression tests. Evidence is in `tests.json`, `tests.xml` and `tests.txt`. Tests cover annotation-free APIs, source/hash drift rejection, parent isolation, identical candidate cohorts, C0 equivalence, normalization, blank rejection, dot/noise handling, adaptive boundaries, clipped/contaminated crop rejection, reference transformations, source-label offsets, deterministic replay, pre-validation selection, checkpoint/vocabulary/decoder compatibility and historical-file preservation.

Inference used CPU with four PyTorch threads, batch size1, deterministic algorithms, float32 original decoding and no optimizer. Model parameters/buffers were checked equal to the loaded checkpoint after experiments. Independent uncached replay checks passed for all six configurations. The inference loop elapsed 34.21 minutes. Candidate `seconds` values are incremental timings in a shared page processor with exact-pixel prediction caching, **not standalone deployment latency comparisons**.

## Files and next action

All source, manifests, examples, locks and reports are inside this module. No Stage 1/2A/2B files were edited. The original checkpoint SHA256 is `22450cbddd42968149739ea0293a6f4ad105f593e82387d707ecb6418a6bac0f`. Software/source hashes, input checks, tests and outcome counts are recorded in `final_verification.json`.

Do not promote the selected configuration into the main Stage 1 workflow. It improves recall/coverage but still wrongly accepts 56 internal-validation scanned pages. Keep it isolated. Prioritise role-confidence calibration, difficult row grouping and crop validation on development data; preserve the current checkpoint and do not start CRNN fine-tuning.

The next iteration should add explicit wrong-acceptance limits to a newly registered selection rule and strengthen ambiguous-role rejection using development parents. Illumination normalization should be evaluated as its own low-risk component; combined geometry/grouping changes must not inherit its OCR benefit without meeting segmentation safety gates. Validation has now been observed: do not tune against these reported validation scores and then call them untouched test results. Keep the general cohort exploratory, keep old positive tests protected, and seek a prospective independently defined evaluation before claiming production readiness, camera-scan robustness or real-handwriting performance. No fine-tuning is recommended as the immediate next action.
