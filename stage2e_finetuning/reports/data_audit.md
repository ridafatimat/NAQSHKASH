# Stage2E data audit

{
  "original_positive_images": 714,
  "original_positive_parents": 179,
  "original_positive_tiers": 1794,
  "manual_review_images": 14,
  "excluded_images": 184,
  "positive_train_after_duplicate_audit": 1624,
  "positive_training_parents": 179,
  "replay_tiers": 1222,
  "replay_parents": 1220,
  "replay_vocabulary_coverage": 65,
  "evaluation_tiers": 2493,
  "evaluation_by_cohort": {
    "clean_historical_regression": 645,
    "scanned_exploratory": 1354,
    "positive_development": 431,
    "clean_validation": 63
  },
  "selection_tiers": 180,
  "human_verified": 0,
  "independent_test": false,
  "duplicate_policy": "Training exact normalized pixel duplicates against any evaluation cohort and within training removed with explicit provenance; all evaluable hard evaluation tiers retained. Shared glyph shapes across different parents are expected, not unseen-symbol generalization."
}

Positive development is the existing frozen 46-parent exposed internal-validation partition, never reassigned or called unseen. All original positive validation/test parents and exposed Stage2B test parents remain absent. General sources remain exploratory. Added clean validation uses 60 original Stage1 validation parents; replay comes only from its original training manifest. Source labels, crop/input hashes, normalization, vocabulary and repeated-character CTC feasibility were checked. Exact normalized input leakage is removed from training, never by pruning evaluation. Near-identical isolated glyphs remain by design; this is not unseen-font or unseen-character evaluation.

Actual retained training coverage: 713 source images, 179 source parents, 1,624 tier inputs. One of the original714 algorithmically approved image cases has all its tier inputs removed as exact normalized duplicates; no hard evaluation example was removed.
