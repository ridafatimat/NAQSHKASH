# Stage 2G final handwritten pilot â€” PROVISIONAL

Independent Talim-reader semantic verification remains incomplete. Recognition of manually corrected crops only; 14 validation tiers from three sources cannot establish generalization. Historical 74.23% CER is retained separately and is not the comparable baseline.

| Metric | Frozen E1, final ink-preserving release | Selected fine-tune |
|---|---:|---:|
| CER including spaces | 74.2268% | 73.1959% |
| CER excluding whitespace | 88.7324% | 84.5070% |
| Exact-tier accuracy | 0.0000% | 0.0000% |

Regression CER:

| Cohort | Before | After |
|---|---:|---:|
| clean_historical_regression | 0.1480% | 0.1726% |
| clean_validation | 0.5263% | 0.5263% |
| positive_development | 11.0847% | 12.0339% |

Selected checkpoint: candidate_epoch_2.pt (checkpoint stored privately)

SHA-256: c601b44adcddbe0a41a5e72ae960b296a94c82d6fbef60e2f6db8fdf4294f604

Ready for further development testing under prespecified regression gates: False. No deployment endorsement. Original E1 and prior results preserved. No Git operations.

## Decision and experiment coverage

Retain frozen E1 as the selected production/research reference. The Stage 2G epoch-2 checkpoint is the best candidate within this pilot, but is NOT recommended for promotion or further protected testing: positive-edge CER increased by 0.9492 percentage points, exceeding the prespecified 0.5-point regression allowance. It is retained for diagnosis and reproducibility. No protected test evaluation was performed.

Two epochs completed: 166 optimizer updates, 330 handwritten sample exposures covering every one of the 165 approved-by-crop-QA training tiers twice, and 332 synthetic replay exposures. No validation tier was used for training. Checkpoint epoch 1 had 74.2268% handwritten CER; epoch 2 had 73.1959%, selected by the recorded validation rule. Effective final updates contain three rather than four samples because 165 is odd.

Space-sensitive edit counts: frozen E1 45 substitutions / 24 deletions / 3 insertions (72 edits); candidate 48 / 20 / 3 (71 edits). Thus the primary improvement is only one edit over 97 reference characters. Both models have 0/14 exact-tier matches. Validation loss fell from 8.1060 to 6.9890. Train loss and sample exposure logs are saved separately; training loss mixes handwritten and replay examples and is not directly comparable to handwritten validation loss.

## Per-source handwritten validation

| Source | Before CER | After CER | Before exact | After exact |
|---|---:|---:|---:|---:|
| all_003191 | 73.2143% | 71.4286% | 0.00% | 0.00% |
| all_003556 | 76.4706% | 76.4706% | 0.00% | 0.00% |
| all_009197 | 75.0000% | 75.0000% | 0.00% | 0.00% |

## Audit and limitations

The exact latest Downloads ZIP was imported and all 377 listed checksums matched. All 165 training and 14 validation image/TXT paths, hashes, raw labels, ocr_text and original source annotation mappings passed. Source assignments match existing splits (the old name test maps to protected_test); four excluded training tiers are absent. Height-64 aspect-preserving normalization and true CTC length feasibility were checked for all handwritten samples. Supplementary checks found 179 grayscale PNGs, white outer borders and no duplicate image hash across splits.

All validation tiers and representative crops from every training source were visually inspected: genuine handwritten appearance and retained faint grey strokes/dots were visible. This does not independently establish the correctness of every Talim target or exact source-mask rendering fidelity. The release records keep_faint_v1 consistently; original source photos/masks are not fully packaged for independent pixel reconstruction. Independent Talim-reader semantic verification remains incomplete. Results are PROVISIONAL and limited to the same tiny manually prepared validation cohort. The historical baseline is preserved separately; its numerical equality with the freshly measured final-release baseline does not imply identical inputs.

Final verification confirms checkpoint hashes, unchanged frozen E1 and reused software, all import checks, 14-tier scoring coverage, metric arithmetic, 166 completed steps, all training tier exposures and no validation training. See final_verification.json. Stage 2E/F and previous Stage 2G artifacts were not intentionally written or replaced. Reproducibility instructions and runtime/source hashes are included.
