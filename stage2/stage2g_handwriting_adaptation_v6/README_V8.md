# NAQSHKASH V8: recovered tiers

## Data
- Codex recovered 44 tiers. Claude visual QA accepted 39 and rejected 5 (see recovery_qa.csv for each reason):
  - 1 crop narrower than its row
  - 2 count crops that lost oval dots
  - 2 writer/label mismatches (fewer dots drawn than the label says)
- The 39 accepted tiers split as:
  - 36 train tiers: W22 11, W16 10, W18 5, W26 4, W11 2, W19 2, W32 2
  - 3 W09 tiers, with split "validation_extra". These are reported separately and never used for selection.
- The selection validation set is unchanged: the same 65 tiers of W05, W24, W30, W31. This keeps the old numbers comparable.
- The classifier data adds 103 real count glyphs and 66 real marks from the recovered train tiers.

## Steps (in the v6 folder, after extracting this ZIP there)

| # | Command | What it does | Time |
|---|---|---|---|
| 1 | `python -B run_v8.py --check` | Checks inputs and the start model | ~1 min |
| 2 | `python -B run_v8.py` | V8 training | overnight |
| 3 | `python -B train_oval_classifier.py final v2` | writes oval_classifier_v2.pt | ~25 min |
| 4 | `python -B train_shape_classifier.py final v2` | writes shape_classifier_v2.pt | ~20 min |
| 5 | `python -B train_mark_classifier.py final v2` | writes mark_classifier_v2.pt | ~15 min |
| 6 | `python -B run_v8_final.py` | compares A/B/C/D and prints the decision and the freeze command | ~5 min |

Steps 3-5 can run before or after V8, but not at the same time as V8, because they share the CPU.

The old classifier files and the old `collection_v3_inputs.csv` are not modified.
