# Dataset quality report — preflight only

{
  "mode": "preflight",
  "source_parents": 30,
  "images": 120,
  "clean_invalid": 30,
  "clean_valid_controls": 30,
  "scanned_style_images": 60,
  "independently_verified_invalid_images": 60,
  "domain_review_images": 0,
  "validity_counts": {
    "VALID": 60,
    "INVALID": 60
  },
  "clean_invalid_by_category": {
    "wrong_tier": 10,
    "invalid_count_token": 10,
    "missing_count": 10
  },
  "split_source_parents": {
    "development_design": 30
  },
  "human_reviewed_images": 0,
  "excluded_parent_candidates": 0,
  "duplicate_images": [],
  "parent_disjoint": true,
  "mutation_family_disjoint": true,
  "byte_hash_disjoint": true
}

{
  "images": 120,
  "semantic_images_checked": 120,
  "protected_hash_only_images": 0,
  "annotations_consistent": true,
  "duplicates": 0,
  "source_parents": 30,
  "parent_disjoint": true,
  "mutation_family_disjoint": true,
  "byte_hash_disjoint": true
}

Generated input pixels and raw structured display agree. Original valid parent
metadata remains valid and separate from mutated raw glyph layouts. No invalid
image has a canonical-valid CTC target. Scope is source48 grayscale canonical
rendering; pointwise illumination PNGs are verified, not a broad scan benchmark.
Readability and semantic invalidity are separate. Unsupported/ambiguous inputs
and quality failures are not silently assigned INVALID.

New source parents cover1/2/3 logical rows, varied pools/directions/counts/widths,
and dense long sequences. All derivatives are one family, not new independent
parents. Reproducible copies with the same seed are not extra independent data.
No human or Talim expert has reviewed these images yet.
