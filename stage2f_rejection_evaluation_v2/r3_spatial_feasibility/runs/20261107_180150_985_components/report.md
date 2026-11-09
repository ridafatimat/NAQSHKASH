# Cached-crop component feasibility

No OCR, coordinate correction or rejection policy. Runtime-style groups/corridors use only crop pixels and D1 boxes. References score afterward. Group fragments/merges are unscorable, not silently fused using annotations.

~~~json
{
  "reference_glyphs": 906,
  "matched_glyphs": 882,
  "fragmented_reference_glyphs": 24,
  "merged_or_unassigned_groups": 64,
  "association_correct": 502,
  "association_denominator": 502,
  "visible_count_denominator": 534,
  "expected_empty_tokens": 8,
  "detected_empty_tokens": 8,
  "false_empty_tokens": 0,
  "empty_candidates_unscorable": 32,
  "images": 48,
  "parents": 12,
  "position_mean": -0.0782312925170068,
  "position_median_absolute": 0.5,
  "position_p95_absolute": 0.5,
  "normalized_position_median": 0.01020408163265306,
  "normalized_position_p95": 0.023809523809523808,
  "valid_controls_with_false_empty": 0,
  "valid_controls_with_wrong_association": 0,
  "controls_with_incomplete_glyph_grouping": 8,
  "wrong_tier_identification_accuracy": null,
  "alignment_coverage": 0.9735099337748344,
  "association_accuracy_conditional": 1.0,
  "association_scoring_coverage": 0.9400749063670412,
  "empty_token_detection_recall": 1.0
}
~~~

Compare association accuracy and full-denominator coverage with historical_CTC_metrics. Inspect valid-control false-empty and incomplete-grouping counts before any safety claim. Glyph identity is not inferred, so wrong-tier identification accuracy is null. This 12-parent cache has no crops for the 70 valid contradictions; it cannot establish a reduction in their REVIEW burden.
