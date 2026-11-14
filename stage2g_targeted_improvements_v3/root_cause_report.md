# Stage 2G V3 root-cause analysis — PROVISIONAL

No training, synthetic generation, source-label change, dataset restructuring or protected-test/quarantine inference. The analysis reuses V2 predictions and metrics; new inference fills missing posteriors/features and tests two narrowly scoped inference-only interventions. W05 was already exposed in V2 and motivates these hypotheses, so this is exploratory development, not a fresh independent validation. Writer provenance and Talim copying/transcription remain incompletely verified.

## What the observed errors show

| Approach | Errors | Symbol-only errors | Whitespace-involving edit events | Out-of-role predicted edit events |
|---|---:|---:|---:|---:|
| A | 65 | 54 | 10 | 10 |
| B | 62 | 52 | 10 | 13 |
| C | 62 | 51 | 10 | 11 |
| D | 57 | 48 | 9 | 10 |

The symbol-only and whitespace-involving counts are different edit alignments and cannot be added as an error decomposition. per_tier_comparison.csv shows all 14 exact truth/A/B/C/D strings. character_error_analysis.csv records reference/hypothesis indices for each deterministic substitution, deletion and insertion; character_confusions.csv aggregates by role. Whitespace positions are left intact; no target relabeling or trimming beyond the established leading-space rule occurs.

### Roles and concentration

| Approach | Role | S / D / I | CER | No-whitespace CER |
|---|---|---:|---:|---:|
| A | count | 27 / 4 / 1 | 80.0000% | 80.0000% |
| A | lower | 11 / 4 / 0 | 53.5714% | 73.3333% |
| A | upper | 10 / 8 / 0 | 54.5455% | 91.6667% |
| B | count | 30 / 1 / 0 | 77.5000% | 77.5000% |
| B | lower | 9 / 4 / 0 | 46.4286% | 60.0000% |
| B | upper | 12 / 6 / 0 | 54.5455% | 100.0000% |
| C | count | 27 / 3 / 0 | 75.0000% | 75.0000% |
| C | lower | 10 / 4 / 0 | 50.0000% | 66.6667% |
| C | upper | 9 / 9 / 0 | 54.5455% | 91.6667% |
| D | count | 25 / 3 / 0 | 70.0000% | 70.0000% |
| D | lower | 8 / 4 / 0 | 42.8571% | 53.3333% |
| D | upper | 12 / 5 / 0 | 51.5152% | 100.0000% |

All primary errors belong to W05, two sources. Single-writer data cannot identify a writer-population effect. Source errors and macro/source metrics are retained from V2. The count tier has substantial within-alphabet errors, including over-emission of the legitimate encoded count class `i`; a role prior cannot eliminate a legal class. Upper/lower errors include wrong-role outputs (e.g. upper E/C predicted Z/Y, lower [ predicted G) and wrong same-role identities. File-consistent labels do not prove the writer copied every symbol correctly.

## Verified cross-role shape ambiguity

The inspected source font SHA-256 matches the existing frozen rendering profile. Its upper encoded C and lower encoded Y have exactly the same ordered contour points/on-curve flags after translation; only their font position differs (450 font units vertically). No new glyph was rendered. This verifies a real role-dependent encoding ambiguity, not merely a visual resemblance. CRNN.forward receives only the image, with no role embedding or classifier conditioning. Whole-page baseline position can distinguish the roles, but tier cropping/height normalization and irregular handwriting do not provide a reliable global baseline. Inferring that this is a cause of some cross-role confusions is supported by the direct outline equivalence and observed C→Y error; it does not prove identical raster inputs or explain all errors.

| Upper / lower pair | Exact ordered outline equivalence after translation |
|---|---|
| E / Z | False |
| C / Y | True |
| K / ^ | False |
| G / [ | False |
| I / \ | False |

The other tested pairs are visually similar but are not asserted to have identical outlines. The fixed role mask corrects some E/Z and K/^ confusions, but in the inspected E-space-C tier it changes Z-space-Y to E followed by spaces, still deleting C. Thus a hard role prior helps but cannot recover all likelihood already assigned to the wrong class. No additional alias substitution/probability-folding fix was fitted after observing this result. Reliable role-conditioned recognition is an evidence-supported direction for later controlled work; missing count-glyph correspondence and target semantics remain unresolved. See font_outline_check.py, cross_role_font_outline_evidence.csv and source_font_evidence.json.

## Posterior and sequence evidence

- A count: 9 decoded i versus 1 target i; 2 out-of-role emitted characters; mean blank argmax fraction 0.918.
- B count: 5 decoded i versus 1 target i; 4 out-of-role emitted characters; mean blank argmax fraction 0.918.
- C count: 9 decoded i versus 1 target i; 4 out-of-role emitted characters; mean blank argmax fraction 0.928.
- D count: 4 decoded i versus 1 target i; 3 out-of-role emitted characters; mean blank argmax fraction 0.922.

ctc_error_posteriors.csv links substitution/insertion edits to actual greedy emission runs, with reference/predicted posterior and truth rank at the observed peak. Deletion spans are best paths constrained to the reference, explicitly hypothetical rather than verified glyph boxes. Low truth posterior/rank is evidence of poor model likelihood, not proof of why a particular mark was missed. The earlier diagnosis already established adequate CTC sequence lengths and correct repeat/blank collapse; those analyses are not rerun generically. New writer-split traces reproduce all saved V2 outputs. Only three compatible old frozen-A W05 row0 traces are reused; no W07/quarantine traces are loaded.

Role leakage is directly observable at the output distribution. The CNN-versus-RNN attribution is not uniquely identifiable from a wrong character alone: frozen E1 can misrepresent ink, recurrent context can alter scores, and labels may be semantically unverified. A greedy-search-only explanation is inconsistent with the previous limited beam improvement.

## Crop extraction, small marks, and appearance

Measured 620 selected small native components across 108 training originals and 14 W05 crops; 0 projected model-scale ROIs fall below 1/255 peak darkness and 36 fall below original midcontrast. These are smallest-component bbox proxies, not verified Talim dots. Neighbouring ink can share an ROI; threshold changes can alter detectability. Therefore these counts cannot be interpreted as semantic mark-deletion rates or assigned to OCR errors without reader/photo-mask correspondence. Existing crop hashes and whole-tier mappings remain consistent. Original segmentation masks/photos are not fully packaged, so extraction-stage missing strokes cannot be independently reconstructed or ruled out. No segmentation-failure CER is invented.

The retained original crops and their actual height64 inputs are shown in visual_failure_examples/. Prior V2 contrast/thickness/geometry measurements and the earlier domain-gap report are reused: thin, light, irregular appearance remains a supported mismatch; generic height normalization did not close it. New spacing checks and single-class feature probes address the writer-split-specific questions instead of repeating those generic ablations.

## Intermediate CNN feature probe

{
  "training_tiers": 16,
  "synthetic_tiers": 31,
  "classes": [
    "C",
    "D",
    "K",
    "P",
    "R",
    "S",
    "V",
    "W",
    "X",
    "Z",
    "[",
    "_"
  ],
  "scope": "Only original training-writer isolated-single-character tiers and existing synthetic development tiers. H lacks a matched single-class control and is excluded from this descriptive probe, not from OCR evaluation. No W05 class centroid fitting. No count-tier isolated glyph correspondence is available; count localization uses posterior/whole-tier evidence instead."
}

| Condition | Last-pool mean matched cosine | Last-pool mean matched-minus-best-other margin | Matched nearest prototype |
|---|---:|---:|---:|
| A | 0.7953 | -0.0174 | 7/16 |
| B | 0.8328 | -0.0004 | 9/16 |
| C | 0.8004 | -0.0178 | 7/16 |
| D | 0.8361 | +0.0002 | 9/16 |

CNN activations are pooled within projected visible-ink ROI at each existing pooling block; no new glyph crops, learned probes or training are produced. Prototype comparisons use existing synthetic single-symbol development tiers and isolated training-writer one-character tiers. These small-class-set similarity/retrieval values are descriptive, not OCR accuracies or evidence that the CNN alone causes every error. Receptive fields extend beyond the approximate ROI. Class labels are putative on handwriting until reader verification. There are no reliable isolated count-glyph targets here; no forced segmentation is used to fabricate count correspondences.

## Whitespace and geometry

- handwritten_train: 30/71 upper/lower tiers admit a count-matched horizontal-group/target-position fit; median local cell-pitch CV 0.000 (descriptive).
- handwritten_validation: 4/9 upper/lower tiers admit a count-matched horizontal-group/target-position fit; median local cell-pitch CV 0.075 (descriptive).
- synthetic_development: 18/80 upper/lower tiers admit a count-matched horizontal-group/target-position fit; median local cell-pitch CV 0.008 (descriptive).

The fit uses target character positions offline to inspect consistency, never to repair inference or fit W05 parameters. Multi-component marks can break grouping; unmatched group counts are left unfit, not silently aligned. Spatial gaps are not sufficient evidence to insert a particular number of label spaces on irregular handwriting. Geometry-aware spacing recovery is therefore not implemented. Blank/space emissions and internal label-grid spacing can disagree even when symbol identities are preserved.

## Exact synthetic regression attribution

| Cohort | A errors | B errors | D errors | A→B preprocessing delta | B→D fixed-preprocessing weight delta |
|---|---:|---:|---:|---:|---:|
| clean_historical_regression | 6 | 6 | 7 | +0 | +1 |
| clean_validation | 2 | 2 | 2 | +0 | +0 |
| positive_development | 327 | 331 | 348 | +4 | +17 |
| long_positive_development | 317 | 319 | 329 | +2 | +10 |

Long-positive is a subset of positive development, never extra independent samples. regression_failure_analysis.csv identifies the exact examples and source parents. Three variants of positive_edge_00323 lower retain identical nonspace text while D loses 12 more spaces in total than A: one additional space deletion appears after contrast and eleven after the CNN updates at fixed contrast. Across the entire long cohort, the total edit-count decomposition is +2 preprocessing and +10 weight updates; this also includes a B-only upper I deletion that D corrects. These are correlated variants of one source, not three independent mechanisms. Two short A→U upper errors are already present in B, so they are preprocessing-associated rather than newly caused by D. At fixed normalization B→D, v→u count substitutions increase by four and m→u by two, alongside punctuation/count-token substitutions. The CNN is the only weight block that changed in V2; BiLSTM/classifier weights did not learn new parameters, although their outputs respond to changed CNN features.

changed_regression_counterfactuals.csv adds D with raw preprocessing only on changed development examples, completing a diagnostic 2×2 comparison for those cases. It is a selected-case counterfactual, not a full-cohort D/raw score. Correlation between improved handwriting and worsened synthetic classes does not prove a shared per-class cause; compare regression_character_changes.csv with handwritten confusions, noting unverified handwriting targets.

## Confirmed versus uncertain mechanisms

| Evidence level | Finding | Limit |
|---|---|---|
| Direct | Many wrong identities remain after whitespace removal; count i over-emission includes legal-class errors | Does not distinguish visual shape from copying error for every glyph |
| Direct | Wrong-role emissions contradict authentic role alphabets on these intended-valid tiers | Role restriction can hide invalid notation; preserve raw evidence |
| Direct | D adds space deletions on unchanged synthetic symbol strings and adds other count confusions | One source supplies the long-space variants |
| Direct intervention | A/B isolates the existing normalization; B/D isolates changed CNN weights | No unique neural-stage diagnosis for every handwritten error |
| Descriptive | Single-symbol CNN similarities and small-component/pitch measures show representation/layout differences | Proxy measurements; no independent count-glyph correspondence |
| Uncertain | Missing semantic dots during original extraction, writer copying mistakes, exact per-error CNN/RNN causation | Need original masks/photos and independent Talim-reader localized annotations |

## Targeted decisions

Role-aware contrast calibration is not a third fix: reused training-only results show contrast benefits every role, so it collapses to the already evaluated B. No new thickness or glyph-height setting is tuned. F1 tests the authentic role alphabet while leaving blank available and preserving raw output; it does not suppress the legitimate count class i. F2 shrinks only D's existing CNN delta toward E1, testing one predetermined three-point scalar grid on training writers plus synthetic development. It performs no gradient updates and never tunes alpha on W05. These are separate interventions, not an uncontrolled stack.
