# Stage 2G V3 controlled targeted-fix comparison — PROVISIONAL

No new training or synthetic data generation. Original E1 and all V2 files remain unchanged. One W05 writer/two sources/14 tiers cannot establish population generalization; no IID-tier confidence interval is reported. Sourcewise results provide sensitivity only.

| Approach | Errors / chars | CER | No-whitespace CER | Exact tiers | Gates |
|---|---:|---:|---:|---:|---|
| V2 A | 65/101 | 64.3564% | 80.5970% | 0/14 | reference |
| V2 B | 62/101 | 61.3861% | 77.6119% | 1/14 | True |
| V2 C | 62/101 | 61.3861% | 76.1194% | 0/14 | True |
| V2 D | 57/101 | 56.4356% | 71.6418% | 1/14 | False |
| F1_authentic_role_mask | 59/101 | 58.4158% | 73.1343% | 1/14 | True |
| F2_shrunk_CNN | 61/101 | 60.3960% | 76.1194% | 1/14 | False |

## Targeted changes and calibration

F1 applies the existing fixed contrast normalization to frozen E1 and masks frame-level nonblank classes outside the authentic known tier alphabet; blank remains legal and internal spaces remain legal only for upper/lower. It uses the original greedy collapse. Its raw unmasked counterpart is retained, not overwritten. This is conditional valid-tier recognition, not a safe replacement for Stage2F invalid-notation rejection.

F2 uses (1−alpha)×E1 CNN + alpha×V2-D CNN, with all unmodified/equal tensors and recurrent/classifier parameters preserved exactly. The existing contrast recipe remains fixed. Only .25/.5/.75 are calibrated, solely on training writers and full existing synthetic development. selected alpha is fixed before its single W05 evaluation. It is inference-only shrinkage, not fine-tuning. The separate targets are motivated by observed illegal-role predictions and D's fixed-preprocessing regression respectively; neither is guaranteed to solve legal-class count confusion.

### F1_authentic_role_mask

| Cohort | Errors / chars | CER | No-whitespace CER | Exact tiers |
|---|---:|---:|---:|---:|
| clean_historical_regression | 2/4055 | 0.0493% | 0.0000% | 643/645 |
| clean_validation | 0/380 | 0.0000% | 0.0000% | 63/63 |
| positive_development | 331/2950 | 11.2203% | 0.5882% | 413/431 |
| long_positive_development | 319/2274 | 14.0281% | 0.1287% | 3/9 |

| Role | S / D / I | CER | No-whitespace CER |
|---|---:|---:|---:|
| count | 30 / 1 / 0 | 77.5000% | 77.5000% |
| lower | 8 / 4 / 0 | 42.8571% | 53.3333% |
| upper | 9 / 7 / 0 | 48.4848% | 83.3333% |

Sourcewise outcomes: {"all_003191": {"tiers": 3, "characters": 19, "nonspace_characters": 12, "errors": 9, "cer": 0.47368421052631576, "cer_no_whitespace": 0.6666666666666666, "exact_tiers": 0, "exact_accuracy": 0.0, "substitutions": 8, "deletions": 1, "insertions": 0, "no_space_substitutions": 8, "no_space_deletions": 0, "no_space_insertions": 0, "mean_ctc_loss": 4.485371510187785}, "all_003801": {"tiers": 11, "characters": 82, "nonspace_characters": 55, "errors": 50, "cer": 0.6097560975609756, "cer_no_whitespace": 0.7454545454545455, "exact_tiers": 1, "exact_accuracy": 0.09090909090909091, "substitutions": 39, "deletions": 11, "insertions": 0, "no_space_substitutions": 39, "no_space_deletions": 2, "no_space_insertions": 0, "mean_ctc_loss": 5.592062117193233}}

Failed fixed gates: none. Model: C:\NAQSHKASH\stage2\stage2e_finetuning\checkpoints\selected_model\best.pt; SHA-256: 27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1.

### F2_shrunk_CNN

| Cohort | Errors / chars | CER | No-whitespace CER | Exact tiers |
|---|---:|---:|---:|---:|
| clean_historical_regression | 7/4055 | 0.1726% | 0.1695% | 639/645 |
| clean_validation | 2/380 | 0.5263% | 0.7326% | 61/63 |
| positive_development | 339/2950 | 11.4915% | 0.6787% | 412/431 |
| long_positive_development | 325/2274 | 14.2920% | 0.1287% | 3/9 |

| Role | S / D / I | CER | No-whitespace CER |
|---|---:|---:|---:|
| count | 28 / 2 / 0 | 75.0000% | 75.0000% |
| lower | 9 / 4 / 0 | 46.4286% | 60.0000% |
| upper | 12 / 6 / 0 | 54.5455% | 100.0000% |

Sourcewise outcomes: {"all_003191": {"tiers": 3, "characters": 19, "nonspace_characters": 12, "errors": 10, "cer": 0.5263157894736842, "cer_no_whitespace": 0.75, "exact_tiers": 0, "exact_accuracy": 0.0, "substitutions": 9, "deletions": 1, "insertions": 0, "no_space_substitutions": 9, "no_space_deletions": 0, "no_space_insertions": 0, "mean_ctc_loss": 3.974900960922241}, "all_003801": {"tiers": 11, "characters": 82, "nonspace_characters": 55, "errors": 51, "cer": 0.6219512195121951, "cer_no_whitespace": 0.7636363636363637, "exact_tiers": 1, "exact_accuracy": 0.09090909090909091, "substitutions": 40, "deletions": 11, "insertions": 0, "no_space_substitutions": 40, "no_space_deletions": 2, "no_space_insertions": 0, "mean_ctc_loss": 5.41436780244112}}

Failed fixed gates: minimum_three_fewer_handwritten_edits. Model: C:\NAQSHKASH\stage2\stage2g_targeted_improvements_v3\candidate_models\inference_only_shrunk_CNN.pt; SHA-256: 4b85b6dff9975051b49b96c165721bec79172b9cd97485f30e7bbec2025aeca7.

## Retained model and further evidence

Best gate-passing targeted development configuration: F1_authentic_role_mask. The operational frozen E1 file remains selected and unchanged; no production policy or entry point is modified. Keep raw model text, role-violation signals and provenance available. A role-masked result must not be passed to Stage2F as if it were the untouched OCR evidence. Stage2F rejection regression is not established here.

Before more fine-tuning: independent Talim-reader checks and localized mark/count annotations; original crop masks/photos to verify missing dots; more real writers for independent evaluation; valid/invalid role reliability and rejection-policy tests; and train-only calibration of any spacing rule on verified glyph correspondences. These are evidence needs, not additional experiments performed now. The demonstrated improvement/regression mechanisms are observations; claims that any one feature block or every faint dot explains all errors remain hypotheses.

## Reproduction

Use experiment.py with --stage analysis, then --stage fixes in a new isolated O directory (change only O for a new run). Existing V2 hashes and paths must match. No package installation or network is required. The script reuses saved predictions and compatible posterior traces, fills unavailable traces, and records all calibration choices before W05 evaluation. Earlier 74.23% source-split results are not pooled with this writer-based set. All outputs are in this directory.

A further read-only font check confirms an exact upper-C/lower-Y outline alias under translation in the existing source font. This supports explicit tier-role conditioning rather than expecting generic handwriting fine-tuning to resolve every role-dependent identity from stroke appearance alone. The check does not change either tested fix or its calibration. Run font_outline_check.py before --stage reports to reproduce this additional evidence; no synthetic glyph images are generated.

## Final answers and practical interpretation

1. The largest unresolved handwritten problem is incorrect symbol likelihood, especially count-class confusions: the winning targeted configuration still has 31 count errors over 40 count characters (77.5% CER). A separate structural problem is missing explicit tier-role conditioning, supported by wrong-role emissions and a verified translated-outline C/Y alias. Internal whitespace/grid timing is a third mechanism and dominates the added long-sequence regression in V2 D.
2. Wrong predictions, their emission posteriors, font equivalence, fixed-preprocessing weight effects, and exact space-count changes are direct evidence. Original-extraction missing dots, semantic copying mistakes, and a unique CNN-versus-RNN explanation for each handwritten deletion remain uncertain. No complete semantic mark-loss rate is claimed from component proxies.
3. Implemented only F1 authentic role-constrained decoding and F2 scalar CNN-delta shrinkage. No new training, glyph images, generic augmentations or additional model-adaptation techniques were run.
4. F1 improves A from 65 to 59 errors (64.36% to 58.42% CER) and B from 62 to 59 errors. No-whitespace CER is 73.13%, exact 1/14. Its primary gain is concentrated on all_003801; all_003191 remains at nine errors, so two-source sensitivity is important. F2 gives 61 errors and fails the minimum improvement relative to B.
5. F1 clean historical/clean validation CER is 0.0493%/0%; positive-edge is 11.2203% and long-positive 14.0281%, within unchanged gates. These are exposed synthetic development/regression results, not independent tests or Stage2F rejection-policy results.
6. Retain the frozen E1 weights. Frozen E1 + fixed contrast + authentic role-constrained greedy CTC is the best gate-passing V3 configuration for trusted intended-valid tier crops. Keep original raw text and violation flags; do not replace the operational rejection pipeline or silently sanitize invalid notes. The original selected-model file is unchanged.
7. Before additional fine-tuning, obtain reader-verified localized count/dot examples, original photo/mask correspondence, and independent new-writer evaluation. Explicit role-conditioned recognition deserves investigation because some source glyph shapes are role-dependent aliases; this is a next-step hypothesis, not a further training run performed here.

All final metric, provenance, coefficient-timing, font and checkpoint checks passed; see verification_results.json. The exact candidate role alphabets, original E1 SHA, preprocessing and raw-evidence contract are in recommended_valid_tier_config.json. The 50% inference-only blend is preserved for reproducibility but not promoted.
