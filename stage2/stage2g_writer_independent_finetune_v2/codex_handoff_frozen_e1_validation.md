# Codex handoff: evaluate frozen E1 on the writer-independent validation split (v2)

## Goal

Run one inference-only evaluation of **frozen E1** on the v2 validation writer, W05 Easha.

- No training, fine-tuning or tuning on these results.
- No Git operations.

## Load exactly these

| Item | Source | Identity |
|---|---|---|
| Split package | `naqshkash_stage2g_writer_split_v2_final.zip` | See `checksums.csv` inside |
| **Primary input** | `validation_primary_originals_v2.csv` | 14 tiers |
| Secondary input | `validation_augmented_secondary_v2.csv` | 312 crops, reported separately |
| Crop package A | `naqshkash_stage2g_augmented_training_v1_1.zip` | sha256 `b0550bdc3247f054637565edfbd11135d87572ad85ec03f6122682042407e420` (11 primary + 312 secondary) |
| Crop package B | `naqshkash_stage2g_handwritten_release_v1.zip` | sha256 `7c3c901c4963f9aa4906d14f0868b1fadcdf131f6c49ec4803bfe3e3035f075b` (3 primary: all_003191 row 0) |
| **Checkpoint** | `C:\NAQSHKASH\stage2\stage2e_finetuning\checkpoints\selected_model\best.pt` | sha256 `27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1` |
| Vocabulary and decoder | The original E1 vocabulary and `data/ctc_decode.py` (greedy CTC) | Unchanged |

The checkpoint is the frozen E1 used in Stage 2F.

## Do not load

- `train_manifest_v2.csv`, `test_manifest_v2.csv`, `test_pending_annotation_manifest_v2.csv`, `quarantine_manifest_v2.csv`
- Any protected-test image
- `candidate_epoch_2.pt`, or any Stage 2G fine-tuned checkpoint: they trained on 11 of these validation tiers

## Steps

1. Verify the sha256 of both crop packages and of the checkpoint. Stop on any mismatch.
2. Extract each package into its own new folder.
   - Each ZIP contains a top folder named after its `package_root` column.
   - Resolve `resolved_crop_path` and `resolved_label_path` against the extraction directory.
3. Check every crop and label before inference.
   - For every row, check the crop file's sha256 against `crop_sha256`.
   - Check that the label file content equals `raw_label`.
   - Stop on any mismatch. Do not edit any image or label.
4. Set the target to `ocr_text`, which is `raw_label` with leading spaces stripped only (Stage 2E convention).
   - Keep internal spaces.
   - If any target character is outside the E1 vocabulary, report it. Do not drop the row or remap the character.
5. Preprocess exactly as the Stage 2E evaluator does (`stage2/stage2e_finetuning/src/datasets.py` and `evaluation.py`).
   - Read as 8-bit grayscale.
   - Resize to height 64, keeping the aspect ratio: width = `max(1, round(w*64/h))`.
   - Use the same normalisation.
   - Use no extra enhancement and no per-sample tuning.
6. Decode with greedy CTC (blank = 0), using `model.eval()` and no gradients.

## Report

Write to a new versioned output folder. Do not overwrite existing results.

**Per-tier predictions CSV** with these columns:
- `sample_id`, `source_id`, `tier`, `variant_id`
- `truth`, `prediction`
- edit distance and its substitution / deletion / insertion breakdown
- CER with spaces and CER without spaces
- `exact`

**Metrics JSON:**
- Primary (14 originals): micro CER with spaces, micro CER without spaces, exact-tier accuracy, per-tier-role CER, per-source CER and macro-over-source CER.
- Secondary (312 augmented): the same metrics, reported separately and never pooled with the primary metrics.
- Provenance: the checkpoint sha256, the manifest sha256 and the counts loaded (expected 14 and 312).

## Interpretation notes for the report

- This is a single-writer validation set with 14 tiers, so the metrics are noisy. Give raw error counts next to the percentages.
- The result is not comparable with the earlier Stage 2G pilot figures (CER 74.23% / 73.20%), which used the old source split.
- The quarantined samples are excluded by design: all_006767 and all_003191 row 1, whose writers are unidentified.
