# NAQSHKASH V5b — selected update 750

Training was stopped at the user’s request. Update 750 is the selected generated-only checkpoint; later checkpoints are retained but are not selected. No full overnight fold/final training has been run.

Selected checkpoint: `C:\NAQSHKASH\stage2\stage2g_handwriting_adaptation_v5b\generated_only\training\checkpoints\attempt_001_step_00750.pt`
SHA256: `f394064f18b8360154e8b30b4d9777923e955a226d77ef9125e987dea7ca1b17`

| Model/update | Raw overall CER | Raw macro-writer CER | Raw no-space CER | Raw exact /105 | Mask macro-writer CER | Mask exact /105 |
|---|---:|---:|---:|---:|---:|---:|
| frozen_E1 | 66.6110% | 68.6056% | 83.1461% | 8 | 64.5170% | 10 |
| step_00250 | 59.4324% | 59.4142% | 73.7079% | 10 | 56.0458% | 17 |
| step_00500 | 60.2671% | 60.4259% | 75.0562% | 10 | 56.3467% | 16 |
| step_00750 | 56.5943% | 56.9194% | 70.5618% | 12 | 52.0034% | 17 |
| step_01000 | 57.5960% | 57.5028% | 71.2360% | 10 | 52.7711% | 16 |

| Selected decoder | Writer | CER | No-space CER | Exact tiers |
|---|---|---:|---:|---:|
| raw | W01 | 57.3134% | 73.7052% | 8/55 |
| raw | W03 | 48.2353% | 59.4203% | 2/17 |
| raw | W04 | 66.6667% | 80.9524% | 0/10 |
| raw | W06 | 55.4622% | 65.0602% | 2/23 |
| role_mask | W01 | 52.5373% | 67.3307% | 12/55 |
| role_mask | W03 | 45.8824% | 56.5217% | 2/17 |
| role_mask | W04 | 58.3333% | 69.0476% | 1/10 |
| role_mask | W06 | 51.2605% | 59.0361% | 2/23 |

Per-role, per-writer, count-error and raw/masked prediction files for every evaluated checkpoint remain in `generated_only/training/attempt_001/evaluations/`. All 105 originals were used for development evaluation, never for generated-only gradient updates. W05 is not used for selection.

## Synthetic gates and decision

**FAIL**. The starting checkpoint is frozen E1.

The 105-tier raw macro-writer CER improvement is 11.69 percentage points. The original >=3-point handwriting condition is True. All synthetic gate views passed: False.

PASS requires >=3 percentage points lower raw macro-writer CER than frozen E1, plus every synthetic CER/exact gate across both decoders and both whitespace views. Initialization follows the synthetic-gate result, as requested, independently of informational W05 results. Existing numerical gate tolerances are unchanged.

| Decoder | Cohort | E1 CER | Selected CER | E1 no-space CER | Selected no-space CER |
|---|---|---:|---:|---:|---:|
| raw | clean_historical_regression | 0.1480% | 2.8360% | 0.1356% | 2.6102% |
| raw | clean_validation | 0.5263% | 2.1053% | 0.7326% | 2.1978% |
| raw | positive_development | 11.0847% | 18.2373% | 0.4525% | 8.1448% |
| raw | long_positive_development | 13.9402% | 17.5022% | 0.0644% | 2.7027% |
| role_mask | clean_historical_regression | 0.0493% | 2.5894% | 0.0000% | 2.2712% |
| role_mask | clean_validation | 0.0000% | 1.3158% | 0.0000% | 1.0989% |
| role_mask | positive_development | 11.0847% | 17.2542% | 0.4525% | 6.8326% |
| role_mask | long_positive_development | 13.9402% | 17.5022% | 0.0644% | 2.7027% |

Individual original/no-whitespace checks are saved in `local_run/gate_outputs/update_00750/synthetic_gates.json`.

## Informational W05

| Tiers | Decoder | CER | No-space CER | Exact |
|---|---|---:|---:|---:|
| 14_tiers | raw | 52.4752% | 64.1791% | 1/14 |
| 14_tiers | role_mask | 49.5050% | 59.7015% | 2/14 |
| 13_tiers | raw | 51.5464% | 63.4921% | 1/13 |
| 13_tiers | role_mask | 48.4536% | 58.7302% | 2/13 |

W05 was evaluated only after the gate decision and initialization were fixed. Both decoders use the same posterior for each image. The 13-tier view excludes only `all_003801_row_00_count`. Cached posteriors prevent repeated forwards on resume.

## QA and package validation

Safe4 selected QA round 1: core darkness median 42.5 versus real 39.72; detached-mark diameter median 5.0 px, with accepted equivalent diameters guarded to 4–7 px. Visibility rejections were 4.42% of all proposals and 13.89% of rendered proposals. The <5% acceptance used all proposals; the stricter rendered-only screen failed. All three rounds and both denominators are retained in `quality_safe4/`.

Earlier local fold/final scripts passed their bounded 20-update smoke tests, with no W05 access; those temporary outputs were deleted. The new gates script is inference-only and has its own tiny synthetic-only dry-run verification. No model was deployed; frozen E1 remains the operational default.

Global safe4 style calibration used all four training writers. A Step 2 initialization was also selected using all 105 development tiers; subsequent folds exclude held-out real gradient updates but are not fully nested validation. Count-feature error categories and compact-mark measurements are heuristic rather than independent reader verification.
