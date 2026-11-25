# Remaining recognition and semantic errors

Selected E1. Exact per-tier long predictions/spaces are in metrics/long_sequence_predictions.csv. All substitutions, deletions and insertions, parent aggregates, length buckets and actual distortion groups are preserved in metrics/.

The source canonical_row implementation in talim/talim_rules.py uses horizontal character positions and ljust/rjust padding to align symbols with encoded repetition counts; width=max(len(symbol),len(encoded_count)). Missing internal spaces can therefore shift symbol/count association. They are not assumed harmless. No production whitespace normalization or carpet-generation change was made, and no unverified rendered-carpet equivalence is claimed. Any downstream alignment decoder must be separately validated.

The long perspective edge_00323 variant remains rejected by frozen D1 and is not credited with OCR improvement. Other correctly segmented long variants remain in every recognition comparison. Segmentation coverage is identical for E0/E1/E2; fine-tuning cannot repair D1 row/tier mistakes.

See galleries/error_comparison.html and reports/training_curves.png for paired predictions and measured curves.

## Measured character/space attribution

[
  {
    "candidate": "baseline",
    "positive_deleted_spaces": 337,
    "positive_inserted_spaces": 0,
    "positive_space_substitutions": 0,
    "positive_nonspace_deletions": 9,
    "positive_symbol_count_substitutions": 17
  },
  {
    "candidate": "e1",
    "positive_deleted_spaces": 317,
    "positive_inserted_spaces": 0,
    "positive_space_substitutions": 0,
    "positive_nonspace_deletions": 1,
    "positive_symbol_count_substitutions": 9
  },
  {
    "candidate": "e2",
    "positive_deleted_spaces": 329,
    "positive_inserted_spaces": 0,
    "positive_space_substitutions": 0,
    "positive_nonspace_deletions": 3,
    "positive_symbol_count_substitutions": 9
  }
]

Edit traces are deterministic minimum-distance alignments (substitution, deletion, insertion tie preference). Character attribution in ambiguous alignments is diagnostic; official CER/SDI remains unchanged. Count-tier errors are repetition-code recognition errors; upper/lower-tier errors are symbol recognition errors. Exact confusion pairs and distortion/length/role breakdowns are saved, not inferred from image appearance alone.

Top remaining positive substitutions for selected checkpoint:

- tier count: expected 'w', predicted 'z', count 3
- tier lower: expected 'R', predicted 'U', count 2
- tier count: expected ')', predicted '(', count 1
- tier count: expected ')', predicted 't', count 1
- tier count: expected ',', predicted 'z', count 1
- tier count: expected 'u', predicted ')', count 1
