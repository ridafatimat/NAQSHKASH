# Stage 2B scanned-style robustness evaluation

Measured locally on 3 November 2026 in `C:\NAQSHKASH\stage2\scanned_robustness`. No training, optimizer updates, model/vocabulary modifications, Git operations or cloud execution occurred. All 6451 protected project files outside Stage 2B retained their hashes; Stage 1 and Stage 2A are unchanged.

## 1. Archive discovery and integrity

The named general ZIP `talim_scanned_augmented_v1.zip` was not found in the proposed Final dataset folder. The actual matching general dataset is **C:\FYP Dataset Generation\datasets\talim_10000_scanned.zip**, verified by its 30,000-row augmentation manifest and source metadata. The positive archive is **C:\FYP Dataset Generation\datasets\talim_edge_cases_positive_scanned_v1.zip**. No older workspace archive was substituted.

| Dataset | Augmented dataset images | Clean source parents | Variants per parent |
|---|---:|---:|---:|
| General | 30,000 | 10,000 | 3 |
| Positive edge cases | 978 | 326 | 3 |

The newly downloading Final dataset copies were not present at the last check; inference used the already available, fully verified datasets-folder copies above. `download_copy_check.json` records this explicitly, and `check_downloads.py` can compare the downloaded copies against the audited SHA256 values when present. No GitHub access or download was needed for this evaluation.

Both archives also contain one QA montage outside their images directory; montages are not counted as samples. They contain **no embedded TXT labels or per-page label JSON**; references resolve into their clean source ZIPs (`talim_10000.zip` and `talim_edge_cases_positive_v1.zip` in the same datasets folder). Every source inventory hash was checked against those ZIPs. All 30,978 augmented image bytes matched manifest SHA256 hashes, decoded successfully and matched declared dimensions. All clean source labels matched their JSON tiers and existing canonical Talim rules. Clean archive hashes match the recorded official source hashes. No full dataset extraction was needed.

Actual operations: brightness, contrast, gaussian_blur, jpeg, motion_blur, noise, perspective, resolution, rotation, shadow. There are 15 actual single/mixed recipes and mild/medium/strong metadata severities. JPEG output encoding also occurs as a container step for all images; augmentation category jpeg denotes the explicit recipe operation. The recorded coarse readability screen is not an OCR guarantee. No stains/folds/handwriting or other unrecorded category is assumed.

Full counts, recipes, severities, hashes and provenance: `dataset_audit.json`. Every parent and variant, including exclusions: `../manifests/parents.csv` and `all_variants.csv`.

## 2. Source-parent eligibility and leakage

The full original Stage 1 source ZIP was scanned: **100,000 image hashes and 100,000 complete logical-run structures**, against all 10,326 source parents. **No general source image is byte-identical to a Stage 1 source image.** One general parent, all_003613, has exactly the logical structure of Stage 1 test_009409; it lacks byte identity and is reported separately as a test-content-match exploratory case, not as confirmed source provenance.

The other 9,999 general parents have no exact complete-page image or logical-run match in the Stage 1 archive. Their clean-parent mapping is valid within the 10k source dataset, but membership in a previously frozen Stage 1 evaluation split cannot be established. They are **not an independent held-out Stage 1 test**. No random image-level split or new test split was invented.

Positive source identities match the original Stage 2A clean archive and inherit its 326 frozen parent assignments. All 228 source training parents and 49 validation parents (831 variants total) are excluded from held-out inference. All 49 positive test parents, including the previously evaluated/recovered test parents, are retained together with all their 147 variants. This is a robustness follow-up on exposed frozen test parents, not a newly blind benchmark.

| Parent classification | Parents | All variants | Variants evaluated |
|---|---:|---:|---:|
| general_unassigned_exploratory | 9999 | 29997 | 300 |
| general_test_structure_match_exploratory | 1 | 3 | 3 |
| excluded_stage2a_train | 228 | 684 | 0 |
| excluded_stage2a_validation | 49 | 147 | 0 |
| positive_frozen_test | 49 | 147 | 147 |

Before inference, a deterministic **100-parent / 300-variant general exploratory sample** was frozen using lowest SHA256('42:'+parent_id), plus the separate three-variant content-match case. All 147 eligible positive test variants were evaluated. Thus **450 scanned pages** were inferred in two modes; **the entire 30,978-image collection was integrity-audited, not exhaustively inferred**. This distinction limits broad category estimates for the general set. The remaining general images have no declared frozen test eligibility; a full exploratory run or new prospective benchmark requires a separately stated scope.

There are no repeated augmented image hashes, no repeated clean image hashes, and no parent whose variants were classified across splits. Exact source relations and complete logical content were checked; partial-row, near-duplicate and font-level similarity are not ruled out. Manifest: `../manifests/evaluation_pages.csv`; SHA256 `b5d38b2f393665a832a4ef00d7034107f4958dcb9e5681693bb986a3ccaf0262`.

## 3. Frozen model and segmentation modes

Checkpoint: `C:\NAQSHKASH\stage2\positive_edge_cases\checkpoints\stage1\best.pt`. SHA256 `22450cbddd42968149739ea0293a6f4ad105f593e82387d707ecb6418a6bac0f`. Exact original 65-character vocabulary, blank 0, 66 classifier outputs, hidden size 256 and two bidirectional LSTM layers. Model state tensors were checked equal to the loaded checkpoint after inference. CPU, four PyTorch threads, batch size 1, eval/inference mode, no AMP or training, deterministic algorithms enabled. Exact input-crop prediction caching only reuses identical pixels; independent uncached replay checks passed for every cohort/mode with successful predictions.

Common preprocessing is the original Stage 1 pipeline: deskew enabled, cleaning, margin crop and logical row detection; ruling-line removal remains disabled. Original height-64 aspect-preserving bilinear resizing, grayscale /255 normalization, white padding and greedy CTC decoding are unchanged. Clean parent comparisons use the same mode and settings.

**Mode A (blind):** row proposals come from the original image-only detector. Tier bands use the original ink-mask/projection utilities without an expected tier count or label values. One observed band is count; three are upper/count/lower. For two bands, a fixed 1.2 height ratio or 1.15 ink-mass ratio locates the count band; ambiguity is rejected. More than three bands and overlapping padded bands are rejected. These are explicit, untuned pixel heuristics, not a learned count-glyph classifier. No expected rows, expected tier presence, symbol counts, text labels, sample IDs, augmentation parameters or OCR predictions are used to select/retry/rank proposals.

Only **after** blind proposals are fixed does ground truth validate row count and upper/count/lower presence/order. Mismatches reject the page; the proposal is never relabelled or repaired using ground truth. This validation defines the conditional OCR subset and must not be confused with a deployable blind acceptance test.

**Mode B (annotation-assisted):** ground-truth row count may invoke Stage 2A's existing strict-whitespace fallback, and known nonempty tier presence is passed to original tier-band extraction. Raw tier text and leading offsets remain unchanged. The same structural checks, non-overlap and repeat-aware CTC feasibility checks apply. Whole pages are withheld from OCR metrics if any row/tier fails. No ground-truth character values guide image cropping in either mode; only Mode B uses structure.

Alignment here is verified by ordered structural correspondence and crop checks, with representative visual examples available. The augmentation set does not provide pixel-level transformed tier polygons; subtle residual crop/warp contamination may remain even when structure agrees. Recognition differences can therefore include residual image-preprocessing effects, not just classifier errors. No claim of geometrically perfect alignment for every successful tier is made.

## 4. Measured page coverage and conditional OCR

CER is total Levenshtein edits / ground-truth characters, counting spaces. Exact accuracy is complete tier-sequence equality. Edit ties use deterministic substitution/deletion/insertion preference. Failed pages do not become artificial deletion errors and do not enter the OCR denominator. Conditional metric changes must be interpreted with coverage because different modes can accept different pages.

| Cohort | Mode | Pages processed/attempted | Evaluated tiers | Characters | CER | Exact accuracy | S/D/I |
|---|---|---:|---:|---:|---:|---:|---:|
| general_test_structure_match_exploratory | blind | 2/3 | 6 | 8 | 0.000% | 100.000% | 0/0/0 |
| general_test_structure_match_exploratory | assisted | 2/3 | 6 | 8 | 0.000% | 100.000% | 0/0/0 |
| general_unassigned_exploratory | blind | 224/300 | 1592 | 10405 | 3.345% | 91.771% | 107/240/1 |
| general_unassigned_exploratory | assisted | 244/300 | 1717 | 10982 | 3.342% | 91.497% | 120/246/1 |
| positive_frozen_test | blind | 41/147 | 82 | 108 | 0.926% | 98.780% | 1/0/0 |
| positive_frozen_test | assisted | 142/147 | 334 | 653 | 3.828% | 96.108% | 14/11/0 |

Rows labelled exploratory must remain separate from the positive frozen-test results. The one-parent content-match sample is too small to support any general claim. Zero evaluated samples would produce n/a metrics, not 0% CER.

### Per-tier OCR results

| Cohort | Mode | Tier | Sequences | CER | Exact accuracy |
|---|---|---|---:|---:|---:|
| general_test_structure_match_exploratory | blind | upper | 2 | 0.000% | 100.000% |
| general_test_structure_match_exploratory | blind | count | 2 | 0.000% | 100.000% |
| general_test_structure_match_exploratory | blind | lower | 2 | 0.000% | 100.000% |
| general_test_structure_match_exploratory | assisted | upper | 2 | 0.000% | 100.000% |
| general_test_structure_match_exploratory | assisted | count | 2 | 0.000% | 100.000% |
| general_test_structure_match_exploratory | assisted | lower | 2 | 0.000% | 100.000% |
| general_unassigned_exploratory | blind | upper | 516 | 4.013% | 88.953% |
| general_unassigned_exploratory | blind | count | 551 | 2.440% | 93.829% |
| general_unassigned_exploratory | blind | lower | 525 | 4.015% | 92.381% |
| general_unassigned_exploratory | assisted | upper | 558 | 3.990% | 88.710% |
| general_unassigned_exploratory | assisted | count | 593 | 2.390% | 93.761% |
| general_unassigned_exploratory | assisted | lower | 566 | 4.117% | 91.873% |
| positive_frozen_test | blind | upper | 9 | 11.111% | 88.889% |
| positive_frozen_test | blind | count | 41 | 0.000% | 100.000% |
| positive_frozen_test | blind | lower | 32 | 0.000% | 100.000% |
| positive_frozen_test | assisted | upper | 110 | 2.581% | 96.364% |
| positive_frozen_test | assisted | count | 142 | 4.509% | 94.366% |
| positive_frozen_test | assisted | lower | 82 | 3.306% | 98.780% |

Detailed reports: `evaluation_v1/metrics.json`, `metrics.csv`, `tier_metrics.csv`, `category_metrics.csv`, `predictions.csv`, `recognition_errors.csv`, `page_processing.csv`, `segmentation_failures.csv`. Clean parent predictions and clean processing failures are also retained. Segmentation/alignment/CTC failures are separate page records, not recognition-edit counts. Conditional OCR does not equal end-to-end page accuracy.

### Pre-existing blind limitations versus new degradation failures

| Cohort | Mode | Clean parents processed/attempted | Scanned variants processed/attempted | Scan failures whose clean parent passed | Failures on both clean and scan |
|---|---|---:|---:|---:|---:|
| general_test_structure_match_exploratory | blind | 1/1 | 2/3 | 1 | 0 |
| general_test_structure_match_exploratory | assisted | 1/1 | 2/3 | 1 | 0 |
| general_unassigned_exploratory | blind | 92/100 | 224/300 | 52 | 24 |
| general_unassigned_exploratory | assisted | 100/100 | 244/300 | 56 | 0 |
| positive_frozen_test | blind | 14/49 | 41/147 | 1 | 105 |
| positive_frozen_test | assisted | 49/49 | 142/147 | 5 | 0 |

**Most positive blind-mode failures already occur on clean parents:** the blind extractor handles only 14/49 clean positive parents (28.6%), versus 41/147 scanned variants (27.9%). Of its 106 scanned failures, 105 also fail on the corresponding clean parent; only one is a new failure on an otherwise accepted clean parent. Thus poor blind positive coverage cannot be attributed entirely to scan corruption or to the CRNN. The prior excellent clean OCR result was annotation-assisted and conditional on segmentation.

In the general exploratory sample, blind clean coverage is 92/100 parents and scan coverage 224/300 variants; 52 newly failing variants came from clean parents that passed, while 24 fail in both conditions. Assisted clean coverage is complete for these selected cohorts, exposing genuine additional scan-processing losses (56 general and five positive variants). Coverage transitions are in `evaluation_v1/clean_scan_coverage.csv`.

## 5. Comparison with clean Stage 1

The preserved full clean synthetic Stage 1 test report records **66,914 tiers**, CER **0.106%**, exact accuracy **99.474%**. That frozen report is context, not an identical sample distribution. Stage 2A's 97 unique positive clean test tiers scored 0% CER/100% exact, but Stage 2B includes all raw parent-tier instances and augmentation repetitions, so its denominators differ.

For a fairer comparison, the same exact clean source parents were run through the same preprocessing/mode/model. The table below pairs only scanned tiers with an aligned clean counterpart. Clean predictions are repeated per matching scanned tier instance so character/sequence denominators are identical within each pair; it does not count those repetitions as independent parents. Failed clean pages are reported and excluded from pairing.

| Cohort | Mode | Paired tiers | Clean CER | Scan CER | CER change | Clean exact | Scan exact | Exact change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| general_test_structure_match_exploratory | blind | 6 | 0.000% | 0.000% | +0.000 pp | 100.000% | 100.000% | +0.000 pp |
| general_test_structure_match_exploratory | assisted | 6 | 0.000% | 0.000% | +0.000 pp | 100.000% | 100.000% | +0.000 pp |
| general_unassigned_exploratory | blind | 1592 | 0.086% | 3.345% | +3.258 pp | 99.435% | 91.771% | -7.663 pp |
| general_unassigned_exploratory | assisted | 1717 | 0.109% | 3.342% | +3.233 pp | 99.301% | 91.497% | -7.804 pp |
| positive_frozen_test | blind | 82 | 0.000% | 0.926% | +0.926 pp | 100.000% | 98.780% | -1.220 pp |
| positive_frozen_test | assisted | 334 | 0.000% | 3.828% | +3.828 pp | 100.000% | 96.108% | -3.892 pp |

Do not interpret differences against the full historical clean baseline as solely caused by degradations; use paired results and coverage together. Neither source-parent label uncertainty nor segmentation failures can be cured merely by substituting a headline OCR number.

## 6. Actual degradation groups and examples

Operation groups overlap for mixed recipes, so rankings show associations in this dataset, not isolated causal effects. Many groups contain correlated variants from the same parent and modest denominators. Severity values are generator metadata, not a calibrated perceptual scale. Complete recipe, operation and severity metrics include coverage and character denominators in the JSON/CSV.

### positive_frozen_test, blind

Largest page-failure rates (operation groups overlap):

- gaussian_blur: 14/17 failed pages.

- rotation: 16/20 failed pages.

- shadow: 15/19 failed pages.

- brightness: 17/22 failed pages.

- contrast: 13/17 failed pages.


Highest conditional OCR CER among groups with evaluated characters:

- rotation: CER 10.000%, exact 87.500%; 8 tiers / 10 characters.

- brightness: CER 0.000%, exact 100.000%; 10 tiers / 13 characters.

- contrast: CER 0.000%, exact 100.000%; 8 tiers / 12 characters.

- gaussian_blur: CER 0.000%, exact 100.000%; 6 tiers / 8 characters.

- jpeg: CER 0.000%, exact 100.000%; 26 tiers / 34 characters.

### positive_frozen_test, assisted

Largest page-failure rates (operation groups overlap):

- rotation: 3/20 failed pages.

- shadow: 2/19 failed pages.

- perspective: 2/21 failed pages.

- jpeg: 1/32 failed pages.

- brightness: 0/22 failed pages.


Highest conditional OCR CER among groups with evaluated characters:

- perspective: CER 10.169%, exact 95.349%; 43 tiers / 118 characters.

- shadow: CER 7.647%, exact 92.683%; 41 tiers / 170 characters.

- rotation: CER 6.452%, exact 92.857%; 42 tiers / 62 characters.

- brightness: CER 3.101%, exact 93.878%; 49 tiers / 129 characters.

- gaussian_blur: CER 3.030%, exact 95.349%; 43 tiers / 66 characters.

### general_unassigned_exploratory, blind

Largest page-failure rates (operation groups overlap):

- perspective: 34/40 failed pages.

- rotation: 23/32 failed pages.

- shadow: 20/42 failed pages.

- jpeg: 17/55 failed pages.

- resolution: 5/37 failed pages.


Highest conditional OCR CER among groups with evaluated characters:

- brightness: CER 10.300%, exact 87.385%; 325 tiers / 2165 characters.

- noise: CER 10.052%, exact 88.235%; 272 tiers / 1920 characters.

- perspective: CER 2.963%, exact 87.500%; 24 tiers / 135 characters.

- shadow: CER 2.870%, exact 91.912%; 136 tiers / 871 characters.

- gaussian_blur: CER 2.841%, exact 86.164%; 318 tiers / 2077 characters.

### general_unassigned_exploratory, assisted

Largest page-failure rates (operation groups overlap):

- perspective: 34/40 failed pages.

- rotation: 22/32 failed pages.

- shadow: 18/42 failed pages.

- jpeg: 12/55 failed pages.

- gaussian_blur: 0/46 failed pages.


Highest conditional OCR CER among groups with evaluated characters:

- brightness: CER 10.205%, exact 87.311%; 331 tiers / 2195 characters.

- noise: CER 9.892%, exact 88.732%; 284 tiers / 1951 characters.

- gaussian_blur: CER 3.162%, exact 85.044%; 341 tiers / 2182 characters.

- shadow: CER 3.043%, exact 91.216%; 148 tiers / 920 characters.

- perspective: CER 2.963%, exact 87.500%; 24 tiers / 135 characters.


### Specific reviewed findings

Actual combined recipes matter: in the general exploratory assisted cohort, **brightness+noise** has 19.277% CER (19/19 pages structurally processed), whereas **noise alone** has 0.105% CER. Do not describe the mixed-recipe result as the isolated effect of noise or brightness. The positive assisted subset has 14.118% CER for perspective+shadow (9/11 pages processed) and 11.429% for rotation (9/11). Small groups, accepted-page selection and correlated variants limit these comparisons.

A visually reviewed, structurally aligned general upper-tier example expects `BH`: its clean prediction is `BH`, while the brightness+noise crop predicts `U`. The two symbol clusters remain visible. The full-crop median/p95 intensities are 255/255 in the clean reference and 193/199 in the scanned crop, consistent with a visibly grey grainy background. This suggests testing background/illumination normalization on development data; these full-crop statistics are not a background-only measurement or a causal ablation. The matched clean crop and `qualitative_review.json` are saved alongside the examples. A reviewed positive three-dot `A` crop is predicted as `U`, showing that recognition errors also remain even when the symbol is visibly present.

Edit attribution confirms that general assisted errors include 107 deleted spaces out of 246 total deletions, plus 120 substitutions and one insertion. These spaces are part of the fixed OCR target and were not stripped to improve results. The positive assisted set has 14 substitutions and 11 deletions, no insertions. Attribution is deterministic and checked against every saved per-sample edit count (`edit_attribution.json`). Residual recognizer weakness and preprocessing/background sensitivity both need to be distinguished from rejected-page segmentation failures.

Representative failures were saved without modifying originals. `evaluation_v1/failure_examples.csv` includes expected text, predictions where inference ran, source parent, recipe, mode and descriptions; `evaluation_v1/examples/index.md` previews selected images. Segmentation examples have no prediction because OCR was withheld. Recognition examples contain the exact evaluated tier crop, expected label and prediction. All errors, not only selected examples, are retained in their machine-readable CSVs.

## 7. Tests and preservation

**12 automated tests passed**, zero failures/errors/skips. Tests cover archive counts and source-image/label/JSON links; parent-level frozen eligibility and source-identity conservatism; deterministic exploratory selection; blind interface/ground-truth separation; rejection instead of relabelling; label offsets/CTC feasibility; CER/exact/no-data calculations; strict checkpoint/vocabulary compatibility; complete failure accounting; and uncached deterministic inference replay/model immutability. Test evidence: `tests.json` and `tests.txt`.

Every selected variant has exactly one page outcome per mode, and each successful page's tier count matches prediction rows. All dataset images were checked before inference. 6451 protected project-file hashes and all audited scanned/clean archive hashes were rechecked after inference; Stage 1/Stage 2A files and source archives remain unchanged. No model state tensor changed. `final_verification.json` records these assertions.

## 8. Recommendation

**B — Preprocessing improvements are required before declaring scanned-page readiness. Prioritise robust blind row/tier grouping, role identification, perspective/rotation handling, illumination/background normalization and alignment rejection. Keep the model frozen while testing improvements on a separate development cohort. Reassess residual OCR errors only after crop quality is verified; do not infer that training alone will fix segmentation failures.**

Regardless of the next development choice, preserve this manifest, checkpoint and all results. Future model/preprocessing choices must use development-eligible parents and a new prospective benchmark where appropriate; this test has now been observed. Obtain documented frozen Stage 1 source correspondence or establish a separately named benchmark before calling the general set an independent test. Synthetic scan-style transformations do not demonstrate real camera-scan coverage or handwriting recognition.
