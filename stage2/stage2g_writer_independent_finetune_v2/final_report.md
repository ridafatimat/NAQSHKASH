# Stage 2G V2 writer-independent appearance pilot — PROVISIONAL

Writer assignments and Talim copying/transcription verification remain incomplete. Primary validation is 14 original tiers from W05, two sources. These results are not comparable to the previous 74.23%/73.20% source-split pilot.

| Approach | Errors / characters | CER | CER excluding whitespace | Exact tiers |
|---|---:|---:|---:|---:|
| A: Frozen E1 + original | 65 / 101 | 64.3564% | 80.5970% | 0/14 |
| B: Frozen E1 + selected normalization | 62 / 101 | 61.3861% | 77.6119% | 1/14 |
| C: Raw-handwriting fine-tune + original | 62 / 101 | 61.3861% | 76.1194% | 0/14 |
| D: Normalized-handwriting fine-tune + normalization | 57 / 101 | 56.4356% | 71.6418% | 1/14 |

Best observed approach: **D**. Selected under predetermined gates: **C**. Numerical experiment ranking retains frozen E1: False. The operational selected-model file remains frozen E1; see the decision interpretation below.

## Rolewise primary results

| Approach | Role | S / D / I | CER | No-whitespace CER | Exact tiers |
|---|---|---:|---:|---:|---:|
| A | count | 27 / 4 / 1 | 80.0000% | 80.0000% | 0/5 |
| A | lower | 11 / 4 / 0 | 53.5714% | 73.3333% | 0/5 |
| A | upper | 10 / 8 / 0 | 54.5455% | 91.6667% | 0/4 |
| B | count | 30 / 1 / 0 | 77.5000% | 77.5000% | 0/5 |
| B | lower | 9 / 4 / 0 | 46.4286% | 60.0000% | 1/5 |
| B | upper | 12 / 6 / 0 | 54.5455% | 100.0000% | 0/4 |
| C | count | 27 / 3 / 0 | 75.0000% | 75.0000% | 0/5 |
| C | lower | 10 / 4 / 0 | 50.0000% | 66.6667% | 0/5 |
| C | upper | 9 / 9 / 0 | 54.5455% | 91.6667% | 0/4 |
| D | count | 25 / 3 / 0 | 70.0000% | 70.0000% | 0/5 |
| D | lower | 8 / 4 / 0 | 42.8571% | 53.3333% | 1/5 |
| D | upper | 12 / 5 / 0 | 51.5152% | 100.0000% | 0/4 |

## Synthetic regression with each actual inference preprocessing

| Approach | Cohort | Errors / chars | CER | No-whitespace CER | Exact tiers |
|---|---|---:|---:|---:|---:|
| A | clean_historical_regression | 6 / 4055 | 0.1480% | 0.1356% | 640/645 |
| A | clean_validation | 2 / 380 | 0.5263% | 0.7326% | 61/63 |
| A | positive_development | 327 / 2950 | 11.0847% | 0.4525% | 415/431 |
| A | long_positive_development | 317 / 2274 | 13.9402% | 0.0644% | 3/9 |
| B | clean_historical_regression | 6 / 4055 | 0.1480% | 0.1356% | 640/645 |
| B | clean_validation | 2 / 380 | 0.5263% | 0.7326% | 61/63 |
| B | positive_development | 331 / 2950 | 11.2203% | 0.5882% | 413/431 |
| B | long_positive_development | 319 / 2274 | 14.0281% | 0.1287% | 3/9 |
| C | clean_historical_regression | 7 / 4055 | 0.1726% | 0.1695% | 639/645 |
| C | clean_validation | 2 / 380 | 0.5263% | 0.7326% | 61/63 |
| C | positive_development | 335 / 2950 | 11.3559% | 0.5430% | 414/431 |
| C | long_positive_development | 324 / 2274 | 14.2480% | 0.1287% | 3/9 |
| D | clean_historical_regression | 7 / 4055 | 0.1726% | 0.1695% | 639/645 |
| D | clean_validation | 2 / 380 | 0.5263% | 0.7326% | 61/63 |
| D | positive_development | 348 / 2950 | 11.7966% | 0.8597% | 407/431 |
| D | long_positive_development | 329 / 2274 | 14.4679% | 0.0644% | 3/9 |

Long positive is the existing Stage2E subset with target length >=65, nine tiers; it is a subset of positive development, never pooled as extra independent samples. Clean cohorts contain 645 historical and 63 validation tiers; positive development contains 431. These are exposed development/regression cohorts, not independent tests.

## Predetermined gates and candidates

{
  "minimum_error_reduction": 3,
  "clean_cer_delta": 0.001,
  "clean_exact_drop": 0.01,
  "positive_cer_delta": 0.005,
  "long_cer_delta": 0.005
}

Require at least three fewer primary edits, no worse no-whitespace CER or exact-tier accuracy, and no sourcewise CER worsening. Candidate C/D must satisfy gates against its matching frozen pipeline AND absolute gates against A. Exact/nonspace/source gates are included in all_candidates.json. Validation comparisons and candidate selection use only the 14 W05 originals. The 312 augmented validation crops are audited but are not used for training or selection.

- C: checkpoint C:\NAQSHKASH\stage2\stage2g_writer_independent_finetune_v2\checkpoints\C_raw_epoch_1.pt; SHA-256 f393c32074bab8e28b3e95500eec6be5791539e366ee240646088628b7840d92; eligible=True; epoch=1.
- D: checkpoint C:\NAQSHKASH\stage2\stage2g_writer_independent_finetune_v2\checkpoints\D_normalized_epoch_2.pt; SHA-256 45fdcffbaa8dc417678fa062229090a2f322664d33ea84c6c0fb917f602ea565; eligible=False; epoch=2.

## Uncertainty and source sensitivity

Only one validation writer is observed. A meaningful writer-population confidence interval cannot be estimated. Fourteen correlated tiers are not 14 independent writers. The following sourcewise changes also give leave-one-source-out sensitivity (retaining just that source), not confidence limits.

| Approach | Source | CER | Difference from A (percentage points) |
|---|---|---:|---:|
| A | all_003191 | 57.8947% | +0.000 |
| A | all_003801 | 65.8537% | +0.000 |
| B | all_003191 | 52.6316% | -5.263 |
| B | all_003801 | 63.4146% | -2.439 |
| C | all_003191 | 47.3684% | -10.526 |
| C | all_003801 | 64.6341% | -1.220 |
| D | all_003191 | 42.1053% | -15.789 |
| D | all_003801 | 59.7561% | -6.098 |

## Normalization and safety

Selected configuration (fixed before W05 evaluation):

{
  "name": "contrast",
  "contrast": true
}

All parameters were calibrated solely on 108 training-writer originals and 120 existing synthetic development controls. Exact recipes, stroke targets and train-only scores are saved in selected_normalization.json and normalization_calibration_results.csv. No geometry warp is included; earlier domain-gap evidence already found no helpful height/crop normalization. New training/control appearance and vertical-extent proxies are recorded without repeating established CTC/glyph-height/whitespace diagnosis.

Dilation operates on height64 float inputs, uses at most half of a one-pixel cross erosion, and falls back to a smaller strength or no thickness change if original components merge/disappear or hole topology changes at three contrast-relative thresholds. Contrast may darken, never whiten, retained ink. These guards and representative train-only inspection are not proof that every Talim mark is semantically unchanged. No independent Talim reader has verified the dataset. Exact guard parameters, fallbacks and training inputs are recorded.

## Controlled training

Two distinct branches start anew from the SHA-verified frozen E1. Only CNN parameters update, low LR 1e-5, AdamW, weight decay 1e-4, gradient clip1, seed42, maximum two epochs, patience2; BiLSTM/classifier and BN running statistics remain frozen. D sees the same fixed normalized representation during training, validation, replay and regression inference. C sees raw representation. Both use identical original-parent/augmentation/replay schedules and optimizer budgets. Each original is visited once per epoch; at most one safe-list, non-near-duplicate same-parent augmentation is sampled, with inverse-writer-frequency handwritten loss weights and a 50% replay-loss allocation. Replacement tiers without eligible augmentation are not replaced with another parent. Frozen-E1 raw replay posteriors anchor the student with 0.1 mean-time-step KL. No new augmentations or labels are generated; AdaBN and handwriting-style synthetic generation are optional future experiments, not performed.

Training logs include actual exposures, per-domain CTC and distillation losses, and per-epoch validation loss. Checkpoints retain separate raw/normalized lineage, vocabulary, optimizer, RNG and schedule identity. Failed gates and early stops are reported rather than silently replaced by more tuning.

## Stage 2F policy regression

NOT EVALUATED: existing pixel/evaluator APIs hard-lock frozen E1 and original preprocessing; no supported candidate/preprocessor injection. Reuse would require an adapter/lineage change, outside this conditional check. No rejection policy retrained or changed. OCR gates do not certify rejection behavior. See stage2f_interface_assessment.md for the inspected interface evidence.

## Observations versus hypotheses

Observed outcomes are the measured comparison tables, per-source results and gate decisions. The CNN-only adaptation/guarded normalization/distillation combination is an experimental hypothesis; success here would not establish unseen-writer generalization. Remaining confounders include provisional writer identity, unverified copying/targets, one validation writer and development-control exposure. No protected-test or quarantined payload is extracted, inferred or used for selection. Frozen E1 and all prior experiment files are hash-checked after execution. No Git operations or dataset restructuring.

## Reproduction

The single reproducible script is experiment.py. On this machine use the recorded Python/PyTorch/OpenCV environment. For a fresh isolated reproduction, change only O to a new directory, then run `python -B experiment.py --stage audit`, `--stage calibrate`, inspect normalization_train_review.png and record a truthful normalization_visual_review.json, then `--stage run`. A completed phase refuses overwrite. Packages are selectively extracted in their existing internal paths; only allowlisted train/validation members are loaded. Download archives and original Stage2E replay/regression paths must remain present. Input hashes, authoritative metadata, exact normalizer, sampling_schedule.json and training_config.json reproduce the protocol.

## Execution recovery note

{
  "scope_correction": "Two unnecessary frozen-E1 secondary augmented-validation baseline passes from the handoff workflow completed. Their ancillary files are preserved, but excluded from calibration, all model selection and final original-tier comparisons. No candidate augmented-validation evaluation is performed.",
  "recovery": "Stopped a brief preliminary C run before any candidate checkpoint. Four completed updates were durably logged; the exact final in-memory interruption step is not recoverable. Preserved partial history/exposures under protocol_history. Both final branches restart from frozen E1 under identical fixed schedules and original hyperparameters. The final branch exposure totals exclude this discarded transient run.",
  "completed_work_reused": "A/B primary original-tier predictions, A/B full synthetic regression and all calibration/audit outputs reused without rerunning.",
  "parameters_and_gates_changed": false,
  "normalization_changed": false,
  "preliminary_logged_steps": 4
}

## Operational decision: preserve frozen E1 as the default

The numerical gate ranking selects C epoch 1. This is an eligible experimental checkpoint, not an automatic deployment decision. D has the best observed handwriting recognition, but is rejected: positive-edge CER is 11.7966% versus A 11.0847% (+0.7119 percentage points; allowed +0.5), and long-positive CER is 14.4679% versus 13.9402% (+0.5277 points; allowed +0.5).

C passes the predeclared numerical gates, but offers the same 62/101 primary errors as frozen E1 + contrast B. C has 0/14 exact tiers versus B 1/14, one fewer nonspace error (51 versus 52), and more synthetic errors. Thus C does not provide convincing additional model-weight benefit over the simpler normalization-only alternative on this small single-writer evaluation. Retain the existing frozen E1 checkpoint as the operational default. Keep C epoch 1 as the best gate-passing fine-tuned research candidate; no gate values or pass/fail statuses have been changed retrospectively. The original selected_model/best.pt remains byte-identical.

This operational interpretation is separate from the fixed numerical checkpoint ranking. It does not invalidate C's recorded gate pass or conceal D's stronger handwriting result. B can be considered an exploratory normalization-only pipeline; its small gain and Stage 2F limitation do not establish production readiness.

## Complete checkpoint outcome table

| Branch / epoch | Handwritten errors | CER | Fixed gates failed |
|---|---:|---:|---|
| C_raw / 1 | 62 | 61.3861% | none |
| C_raw / 2 | 62 | 61.3861% | positive_development_cer |
| D_normalized / 1 | 61 | 60.3960% | long_positive_development_cer, minimum_three_fewer_handwritten_edits, positive_development_cer |
| D_normalized / 2 | 57 | 56.4356% | long_positive_development_cer, positive_development_cer |

## Training-only normalization comparison

| Configuration | Training CER | Macro-writer training CER | Synthetic calibration CER | Guard fallbacks |
|---|---:|---:|---:|---:|
| identity | 67.5159% | 69.8847% | 0.0000% | 0 |
| contrast | 64.1720% | 66.7388% | 0.0000% | 0 |
| guarded_fixed_thickness | 66.8790% | 69.2505% | 0.0000% | 77 |
| guarded_adaptive_stroke | 66.8790% | 69.2505% | 0.0000% | 76 |
| contrast_guarded_adaptive | 64.3312% | 66.8439% | 0.0000% | 76 |

Contrast alone wins the predetermined training-only selection rule. Guarded fixed/adaptive thickness changes give much smaller gains, and adding guarded thickness to contrast slightly worsens its training score. No W05 result was used to change this choice. The complete fixed formula is in normalization_recipe.json. Blackpoint estimation on each inference image is a deterministic application of the fixed fifth-percentile rule, not label-based retuning.

## Executed exposure totals and verification

Each final branch completed 216 updates: 216 original handwritten exposures (all 108 originals twice), 170 unique sampled augmentations, and 432 replay exposures. Each augmented parent receives at most one variant per epoch; missing eligible augmentation for replacement-capture tiers is not borrowed from another parent. Across a full schedule, effective handwritten loss is equal across the four assigned training writers, and total handwritten/replay loss allocation is 50/50. Raw and normalized branches use the same saved input schedule. The discarded preliminary transient run is recorded separately in the execution recovery note and is not included in these final-branch totals.

Independent verification recomputed edit distances, CER with/without whitespace, sample/target equality, schedule weights, and extracted-file allowlists. It confirmed nonzero CNN updates; unchanged BiLSTM/classifier and BatchNorm running states; exact checkpoint hashes; parameters/gates frozen before first W05 evaluation; and preserved previous input files. See verification_results.json and final_integrity.json. Sourcewise uncertainty remains descriptive: C's primary gain is confined to all_003801 and disappears if that source is left out; D improves both sources but fails regression. Neither fact supports an independent-writer population confidence interval.
