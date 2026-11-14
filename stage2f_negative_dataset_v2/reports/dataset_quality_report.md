# Dataset quality report — FINAL preflight only

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

Only dataset/preflight is accepted. The initial120-image provisional collection
is quarantined and preserved, not counted in these final counts. Its source
families are reserved from future protected generation. No prior project
artifacts were modified.

All original valid parents pass the original validator. The invalid raw display,
raster and intended violation agree, and no fake canonical-valid CTC label is
assigned to an invalid input. Core verification never trusts a mutation label.
The source48 font/grid scope and conservative visual-alias exclusions are frozen.

Variations cover1/2/3 rows, symbol pools/counts/widths and dense long sequences.
Illumination-only lossless PNG is the supported scanned-style profile. This is
not a broad scan/photo robustness benchmark. Additional lossy/spatial transforms
require verification before inclusion. All samples remain NOT_REVIEWED by humans.

Provenance accounting:240 model PNGs were created across prototype iterations;120 are quarantined,120 are eligible final preflight. There are 80 shared raster hashes and 40 unique source families across the two collections; these are not240 independent examples. The accepted final collection still has30 families.
