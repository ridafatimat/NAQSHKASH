# Stage 2D failure diagnosis

Diagnosis completed before D1 implementation. Stage 2C remains unchanged.

## Evidence and counting
All 486 inventoried artifacts and source/software locks passed. Recomputed cohort summaries from 9,672 saved page outcomes and 31,367 predictions. Acceptance and reference correctness were reproduced for 3224 C4/C5 outcomes on 1612 pages. Character errors were independently checked for the long parent. General sources remain exploratory; all old validation scores are exposed development comparisons, never a new independent test.

| Partition | Dataset | Kind | C4 | C5 | Images | Unique parents |
|---|---|---|---|---|---|---|
| tuning | positive | scanned | correct | correct | 327 | 113 |
| tuning | positive | scanned | correct | rejected | 1 | 1 |
| tuning | positive | scanned | rejected | rejected | 20 | 10 |
| tuning | positive | scanned | wrong | rejected | 152 | 59 |
| tuning | positive | scanned | wrong | wrong | 46 | 21 |
| tuning | positive | clean | correct | correct | 109 | 109 |
| tuning | positive | clean | rejected | rejected | 9 | 9 |
| tuning | positive | clean | wrong | rejected | 50 | 50 |
| tuning | positive | clean | wrong | wrong | 14 | 14 |
| tuning | general | scanned | correct | correct | 270 | 100 |
| tuning | general | scanned | correct | wrong | 2 | 2 |
| tuning | general | scanned | wrong | correct | 1 | 1 |
| tuning | general | scanned | wrong | wrong | 27 | 26 |
| tuning | general | clean | correct | correct | 99 | 99 |
| tuning | general | clean | wrong | wrong | 1 | 1 |
| validation | positive | scanned | correct | correct | 93 | 32 |
| validation | positive | scanned | correct | rejected | 1 | 1 |
| validation | positive | scanned | rejected | rejected | 3 | 1 |
| validation | positive | scanned | wrong | correct | 1 | 1 |
| validation | positive | scanned | wrong | rejected | 27 | 11 |
| validation | positive | scanned | wrong | wrong | 13 | 6 |
| validation | positive | clean | correct | correct | 31 | 31 |
| validation | positive | clean | rejected | rejected | 1 | 1 |
| validation | positive | clean | wrong | rejected | 9 | 9 |
| validation | positive | clean | wrong | wrong | 5 | 5 |
| validation | general | scanned | correct | correct | 134 | 49 |
| validation | general | scanned | correct | wrong | 1 | 1 |
| validation | general | scanned | wrong | rejected | 3 | 1 |
| validation | general | scanned | wrong | wrong | 12 | 11 |
| validation | general | clean | correct | correct | 48 | 48 |
| validation | general | clean | wrong | rejected | 1 | 1 |
| validation | general | clean | wrong | wrong | 1 | 1 |
| clean_regression | normal | clean | correct | correct | 96 | 96 |
| clean_regression | normal | clean | wrong | rejected | 1 | 1 |
| clean_regression | normal | clean | wrong | wrong | 3 | 3 |

Counts are images and unique source parents within each cell. Parent counts across cells overlap; do not add them as independent observations. Every page is a valid positive Talim source; every rejection is a missed valid page, not a negative example.

## Ranked issues supported by tuning data
1. **Wrong scale and disconnected glyph grouping.** Component-height percentile can be 8–23 pixels while the true count-band extent is 38–42 pixels. In edge_00025, the 41-pixel count and 23-pixel lower tier are forced into separate rows because the inferred pitch is too short. In edge_00001, separated dot clusters of the upper A remain separate bands. Correct paths become infeasible before confidence is assessed. Fix the scale estimate using observed band extent and constrain merges by total extent. This has high recovery potential, but must be tested on sparse unusual layouts.
2. **Indiscriminate small-gap merges.** General rotation/perspective tuning examples (all_000655, all_004700) join a full count tier to its lower symbol tier into an 80+-pixel band at scale ~38. A 6-pixel gap rule alone cannot distinguish fragments from full tiers. Add an extent guard and reject unresolved over-tall bands; do not blindly split ink.
3. **Heuristic role evidence and misleading certainty.** A single legal path receives margin=99 even when morphology is poor or the true path was pruned by geometry. This sentinel is not calibrated confidence. C5 only tests the best/second-path margin and therefore still accepts confidently wrong layouts. Expose alternative count, score and structural checks; retain ambiguity rejection without describing scores as probabilities.
4. **Role morphology for disconnected count glyphs.** Count glyphs may consist of individually short components; component-height alone underestimates the whole symbol. Evaluate a bounded band-height contribution using tuning examples, keeping all original pixels.
5. **Long recognition and spacing.** edge_00323 is confirmed in the saved artifacts. Its clean crops reproduce the saved greedy predictions with 112 errors, all missing internal spaces; non-space sequences match. Its ground-truth labels round-trip through the unchanged vocabulary; independently computed edit distances agree, repeat-aware CTC lengths are feasible, and reference crop retention passes. The space tokens are already missing in the collapsed network argmax output. This supports a frozen recognizer/spacing limitation, not a CER normalization or encoding bug. Degraded variants also have genuine glyph errors. No space-stripping or arbitrary crop workaround is justified.

## Role search audit
The exact Stage 2C scoring uses optional upper, mandatory count, optional lower; row penalty -1.4; log-transformed logistic morphology evidence; geometric pitch and boundary constraints. C4 accepts the best legal path regardless of ambiguity; C5 rejects margin <0.2. Two paths per index suffice because future contributions depend only on the band index, not past path state.
Exhaustive enumeration matched the top two scores on 1,350 seeded sequences up to nine bands and small actual replay sequences. **No verified DP search defect.** Scoring and invalid geometric constraints can still produce the wrong highest-scoring path; that is a separate issue.

| Candidate | Diagnostic signature (not automatically a root cause) | Images | Parents | Example |
|---|---|---|---|---|
| C4 | wrong proposal despite sole legal path (margin sentinel 99) | 39 | 13 | all_006186__aug_03_rotation_jpeg |
| C5 | wrong proposal despite sole legal path (margin sentinel 99) | 39 | 13 | all_006186__aug_03_rotation_jpeg |
| C4 | scale smaller than 65% of tallest band | 355 | 117 | all_000158__aug_01_rotation_jpeg |
| C5 | scale smaller than 65% of tallest band | 357 | 117 | all_000158__aug_01_rotation_jpeg |
| C4 | band taller than 1.6 component scales | 331 | 109 | all_000158__aug_01_rotation_jpeg |
| C5 | band taller than 1.6 component scales | 330 | 109 | all_000158__aug_01_rotation_jpeg |
| C4 | wrong row count | 243 | 68 | all_001016__aug_02_perspective |
| C5 | wrong row count | 43 | 16 | all_001016__aug_02_perspective |
| C4 | accepted wrong structure | 291 | 99 | all_000158__aug_01_rotation_jpeg |
| C5 | accepted wrong structure | 90 | 47 | all_000158__aug_01_rotation_jpeg |

These signatures overlap and are not causal labels. failure_categories.csv retains actual first failure reasons. Geometry/background changes cannot be blamed merely because a transformed image failed. Stage 2C's registered component audit found no component loss among its aligned cases, but it did not cover wrong/rejected structures. D1 will audit component retention separately; no universal preservation claim follows.

## Long-page audit and limits
Exact truth/prediction strings, raw leading offsets, dimensions, Unicode values, target lengths, required repeats, network output lengths, and clean argmax/decoded space counts are in long_sequence_audit.json. Row/tier order comes from the frozen annotation reference. Meaningful internal spaces remain in all headline CER calculations. The diagnostic strips spaces only for equality attribution. Offline ideal-ink masks approximate blur/interpolation and are not a human verification of every pixel.

## Targeted experiment decision
Build on C5's normalization/geometry/denoising, retain C4/C5 unchanged. Change only the grouping/role engine: robust band-aware scale, extent-limited fragment merging, bounded whole-band morphology, explicit confidence diagnostics and structured abstention. Compare a small predeclared grid on original tuning parents only; freeze the selected D1 before replaying exposed validation. No untouched positive holdout is used or claimed. Long OCR behavior is documented for subsequent training design; no recognition formatting candidate is justified.

Gallery: [index](../galleries/index.html). Source data, labels, model, original decoder and historical files are read-only.
