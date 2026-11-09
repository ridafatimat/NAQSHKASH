# CTC alignment pilot — measured results

This is a development feasibility pilot, not a validated R3 rejection policy. Mismatched reproductions receive zero coverage and are excluded from conditional position/association accuracy.

~~~json
{
  "reference_glyphs": 906,
  "matched_glyphs": 898,
  "ambiguous_glyphs": 0,
  "wrong_or_extra_glyphs": 8,
  "unmapped_glyphs": 0,
  "token_order_correct": 238,
  "token_order_denominator": 364,
  "multi_glyph_token_correct": 72,
  "multi_glyph_token_denominator": 170,
  "association_correct": 404,
  "association_denominator": 520,
  "unique_count_assignments": 534,
  "visible_count_denominator": 534,
  "reference_empty_tokens": 8,
  "focused_visible_token_denominator": 40,
  "focused_token_correct": 14,
  "focused_unscorable": 8,
  "images": 48,
  "parents": 12,
  "reproduction_mismatches": 0,
  "alignment_coverage": 0.9911699779249448,
  "reference_glyphs_uncovered": 8,
  "position_axis": "source_x; vertical support is the full tier, not a localized glyph center",
  "token_order_agreement": 0.6538461538461539,
  "multi_glyph_token_agreement": 0.4235294117647059,
  "association_accuracy_conditional": 0.7769230769230769,
  "focused_token_agreement": 0.35,
  "association_scoring_coverage": 0.9737827715355806,
  "unique_assignment_coverage": 1.0,
  "position_error_pixels": {
    "signed_mean": -32.95013622809379,
    "median_absolute": 25.104790419161645,
    "p95_absolute": 120.93636363636364
  },
  "normalized_position_error": {
    "median": 0.6391951161688011,
    "p95": 3.556042780748663
  },
  "valid_control_association_risk_images": 24,
  "valid_control_order_risk_images": 24,
  "valid_control_incomplete_evidence_images": 0,
  "association_unscorable_images": 0
}
~~~

See metrics.json for category, validity, variant, parent and paired illumination results, per_image.json for ambiguous/unscorable correspondence and gallery/index.html for crop overlays.

Provisional feasibility verdict: NOT_READY_VALID_CONTROL_ERRORS_OR_INCOMPLETE_EVIDENCE. This classifies the experiment evidence, not Talim images. No numerical localization cutoff was preregistered. Assess source-x error tails against glyph widths and association boundaries, coverage, focused mutation-token agreement and valid-control risk together. A low conditional error with poor coverage is insufficient. Zero risk flags with unscorable controls is not safety evidence. Missing-count absence is not measured by CTC blanks. Any reliable-CTC conclusion applies only to this small selected source-renderer pilot; independent image support remains necessary before definitive rejection.
