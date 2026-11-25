# NAQSHKASH Stage 2G: Writer-Independent Split v2 (FINAL)

Date: 2026-11-09. Status: **final for v2**. This package contains manifests and documentation only.

**What did not change:**
- No image, crop or label was changed, copied or re-rendered.
- Historical datasets and the 24/3/3 source split are untouched.
- The frozen E1 checkpoint and all Stage 2F/2G outputs are untouched.
- No training, inference or evaluation was run.
- Protected-test images were not opened. Their rows come from `manifest_protected_test.csv` metadata only.

## Writers and split

| Writer | Name | Split | Identity evidence |
|---|---|---|---|
| W01 | user (Me) | train | Confirmed by elimination |
| W03 | Bissam | train | Assignment evidence |
| W04 | Dania | train | Assignment evidence (folder path) |
| W06 | Faiqa | train | Assignment evidence |
| W05 | Easha | **validation** | Assignment evidence (folder path) |
| W02 | Ayesha | **test** | Photo match for 4 sources; content match for 2 protected-test sources |
| W07 | Hareem | **test** | Exact file hash plus user confirmation (2026-11-09) |

Each writer's original tiers and augmented crops are kept together. No writer appears in more than one of train, validation or test, and this is checked in `scripts/build_split.py`.

## Exact sample counts (confirmed and usable sources only)

| Split | Sources | Original tiers | Augmented crops | Total |
|---|---|---:|---:|---:|
| **train** | 15 | 108 | 2,469 | 2,577 |
| **validation** | 2 (all_003801; all_003191 row 0) | 14 | 312 | 326 |
| **test** | 11 | 54 | 1,110 | 1,164 |
| test, pending annotation | 1 (all_009894) | 12 | 0 | 12 |
| quarantine | all_006767; all_003191 row 1 | 9 | 173 | 182 |
| excluded (as in release v1) | all_001623 | 4 | 0 | 4 |
| **all listed samples** | | 201 | 4,064 | 4,265 |

Notes on the table:
- The 4,265 total includes the 4 excluded tiers.
- The manifests list 4,261 rows: 4,229 training-package rows, 14 validation tiers and 18 protected-test tiers.

Per-writer counts:

| Writer | Split | Original | Augmented |
|---|---|---:|---:|
| W01 | train | 55 | 1,590 |
| W03 | train | 17 | 174 |
| W04 | train | 12 (all_004383) | 353 |
| W06 | train | 24 | 352 |
| W05 | validation | 14 | 312 |
| W02 | test | 29 (23 former train + 6 protected holdout) | 667 |
| W07 | test | 25 (17 former train + 8 former validation) | 443 |
| W07 | test, pending | 12 | 0 |

Validation originals by tier: 4 upper, 5 count, 5 lower.

## Recorded limitations

1. **all_006767 is quarantined (final for v2).** Its writer cannot be reliably identified: its reference appears under both DANIA and AYESHA. It is in no split: 6 tiers and 173 augmented crops. Its identity was not guessed.
2. **all_003191 row 1 is quarantined (final for v2).** The rewritten row's writer cannot be reliably identified, so its 3 tiers are in no split. Row 0 (3 tiers) remains in validation.
3. **Replacement captures stay in train with their assigned writers.** These are all_004254 (W06, 12 tiers) and all_007011 (W03, 11 tiers). Their rewriting writer is unconfirmed, but no leak is possible unless W02, W05 or W07 wrote them.
4. **The W02 protected-test sources are matched by content only.** These are all_002266 and all_009421. They are in test either way.
5. **W03, W04, W05 and W06 rest on assignment evidence**, not direct handwriting matches.
6. **Only frozen E1 is clean for the new validation and test splits.** The Stage 2G pilot fine-tune (`candidate_epoch_2.pt`) trained on 11 of the validation originals and 40 of the test originals.
7. **The new test split has been seen during preparation.** Its 40 former-train and 8 former-validation tiers were viewed during crop correction and QA. Only the 6 protected holdout crops are unseen.
8. **Use original tiers for the primary metrics.** Augmented crops are photo variants of the same handwriting, not new writers, so report them separately.
9. **Renderer mismatch in test.** The 6 protected holdout crops use the older renderer from `handwritten_tier_crops_5`. All other crops use `keep_faint_v1`.
10. **Small evaluation sets.** Validation is 14 originals from 1 writer and 2 sources. Test is 54 usable originals from 2 writers. Treat the metrics as noisy.
11. **No Talim reader has verified the labels or the copying** (`talim_reader_verified = False`).

## Files

| File | Use |
|---|---|
| `validation_primary_originals_v2.csv` | 14 validation originals with resolved paths. Primary input for frozen-E1 validation |
| `validation_augmented_secondary_v2.csv` | 312 validation augmented crops. Secondary, reported separately |
| `train_manifest_v2.csv`, `validation_manifest_v2.csv`, `test_manifest_v2.csv` | Full split manifests |
| `test_pending_annotation_manifest_v2.csv`, `quarantine_manifest_v2.csv`, `excluded_tiers_v2.csv` | Samples outside the usable splits |
| `all_samples_v2.csv` | Every listed sample |
| `split_counts_v2.csv` | Per split and per writer counts |
| `writer_split_assignment_v2.csv` | Writer-level split and evidence |
| `source_to_writer_mapping_final_v2.csv` | Mapping with W07 = Hareem, `writer_name`, `proposed_writer_split_v2` and `v2_status` |
| `unconfirmed_items_v2.csv` | Open items and how each is handled |
| `codex_handoff_frozen_e1_validation.md` | Instructions for the frozen-E1 validation run |
| `scripts/`, `checksums.csv` | Build code and file hashes |
