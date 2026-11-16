# Stage 2D fine-tuning readiness

| Bucket | Images | Unique parents |
|---|---:|---:|
| training_ready | 714 | 179 |
| manual_review | 14 | 5 |
| excluded | 184 | 46 |

1794 verified training-eligible tier crops (1655 unique pixel hashes; 145 distinct target strings) were exported. Counts include clean and existing scanned variants, not independent pages from distinct sources. Parent counts overlap between ready and review buckets; variants of one training parent can have different processing outcomes.

Eligibility requires original training assignment AND Stage 2C tuning partition, independently reverified TXT/JSON/canonical labels, complete offline segmentation alignment, >=98% tier-ink retention and <=2% contamination, >=95% retention/content support for each ideal component >=3px, and repeat-aware CTC feasibility. Algorithm acceptance alone never establishes ground truth. Content support uses Otsu ink within a 1px neighbourhood and is an approximation on degraded pixels. Human-reviewed count is zero.

All 46 previously exposed internal-validation positive parents remain excluded from training; original positive validation/test parents are absent entirely. All variants retain parent ownership. Labels, internal spaces and offsets are unchanged. Image manifests include archive/member paths, exact source hashes, full transcriptions, split, verification status and review/exclusion reason. Tier manifests include local crop paths, PNG/pixel hashes, row/role, raw/normalized label offsets and CTC lengths.

The eligible subset is suitable for preparing a controlled fine-tuning loader/pilot, subject to reviewing representative difficult cases and freezing the training, validation, resource and stopping protocol. Its size does not establish that fine-tuning will improve recognition. Long spacing failures are a recognition target, while rejected/wrong segmentation pages require preprocessing/manual review. No end-to-end deployment approval and no independent generalization claim follows. Do not automatically start training; the checkpoint remains frozen.