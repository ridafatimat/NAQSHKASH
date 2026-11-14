# R3.1 development results

REVIEW is not definitive rejection. ACCEPT alone does not authorize carpet generation: inspect carpet_generation_ready and the explicit selected reading.

Reference claims (16,434/16,442 exact offsets, 138 resolved valid pages, approximately 32 invalid accepts) were not used to choose parameters. Scope, coverage, unavailable evidence and all changed cases are reported below and in the CSV/JSON artifacts. No held-out evaluation or freeze was performed.

~~~json
{
  "R3_to_R3_1_transitions": [
    {
      "before": "ACCEPT",
      "after": "ACCEPT",
      "images": 1220
    },
    {
      "before": "ACCEPT",
      "after": "REVIEW_UNCERTAIN",
      "images": 146
    },
    {
      "before": "REVIEW_UNCERTAIN",
      "after": "REVIEW_UNCERTAIN",
      "images": 1434
    }
  ],
  "policy_invariants_pass": true,
  "new_definitive_rejections": 0,
  "R3_reviews_preserved": 1434,
  "measurement_attempted": 2800,
  "measurement_complete": 0,
  "D1_box_equality": {
    "True": 2800
  },
  "measured_tiers": 16444,
  "saved_D1_tier_denominator": 16444,
  "distinct_reading_counts_before": {
    "2": 134,
    "1": 1220,
    "3": 2,
    "4": 10
  },
  "distinct_reading_counts_after": {
    "None": 1366
  },
  "ambiguous_cases_resolved": 0,
  "ambiguous_valid_cases_resolved": 0,
  "ambiguous_remaining": 146,
  "positioning_unavailable_on_prior_ACCEPT": 1366,
  "carpet_generation_ready": 0,
  "carpet_generation_ready_rate": 0.0,
  "carpet_generation_ready_among_ACCEPT": 0.0,
  "selected_reading_status": {
    "not_selected_prior_review_uncertain": 1434,
    "unresolved_R3_1_POSITION_UNAVAILABLE": 146,
    "unique_text_only_position_unavailable": 1220
  },
  "unavailable_reasons": {
    "REPEATED_OCR_CHARACTER_HAS_INCONSISTENT_INK_GROUPS": 2730,
    "INCONSISTENT_GLYPH_START_SPACING": 34,
    "D1_REJECTED:grammar:no_legal_layout": 6,
    "D1_REJECTED:components:unassigned_foreground": 18,
    "D1_REJECTED:confidence:ambiguous_row_or_role": 4,
    "COUNT_GROUPING_UNSUPPORTED_IN_ONE_OR_MORE_ROWS;REPEATED_OCR_CHARACTER_HAS_INCONSISTENT_INK_GROUPS": 6,
    "D1_REJECTED:bands:unresolved_vertical_merge": 2
  },
  "reason_counts": {
    "R3_1_POSITION_UNAVAILABLE": 146
  },
  "scoring": {
    "measured_tiers": 16444,
    "first_ink_measured": 16444,
    "offset_eligible": 0,
    "exact_offsets": 0,
    "within_half_column": 0,
    "conditional_exact_offset_rate": null,
    "offset_errors_columns": [],
    "valid_ACCEPT_denominator": 1180,
    "selected_reading_eligible": 1180,
    "selected_reading_correct": 0,
    "selected_reading_false": 1180,
    "correct_selected_reading_rate": 0.0,
    "false_selected_reading_rate": 1.0,
    "selected_reading_coverage": 1.0,
    "ready_valid_selected_reading_false": 0,
    "invalid_ACCEPT_selected_readings_unscorable": 40,
    "unscorable_offset_reasons": {
      "No verified complete pitch/row/tier correspondence to MUTATED notation": 16444
    },
    "saved_D1_tier_denominator": 16444,
    "attempted_saved_D1_tiers": 16444,
    "first_ink_coverage_all_saved_tiers": 1.0,
    "first_ink_coverage_attempted_saved_tiers": 1.0,
    "offset_scoring_coverage_all_saved_tiers": 0.0,
    "exact_offset_full_saved_tier_rate": 0.0
  },
  "outcomes": {
    "R3": {
      "images": 2800,
      "valid_denominator": 1400,
      "invalid_denominator": 1400,
      "three_way_confusion": {
        "VALID": {
          "ACCEPT": 1318,
          "REJECT_INVALID": 0,
          "REVIEW_UNCERTAIN": 82
        },
        "INVALID": {
          "ACCEPT": 48,
          "REJECT_INVALID": 0,
          "REVIEW_UNCERTAIN": 1352
        }
      },
      "true_invalid_rejection_rate": 0.0,
      "invalid_accepted_rate": 0.03428571428571429,
      "invalid_review_rate": 0.9657142857142857,
      "invalid_images_incorrectly_accepted_as_valid": 48,
      "invalid_interpreted_syntax_passes": 896,
      "invalid_interpreted_syntax_pass_rate": 0.64,
      "valid_acceptance_rate": 0.9414285714285714,
      "false_definitive_rejection_rate": 0.0,
      "valid_review_rate": 0.05857142857142857,
      "review_rate": 0.5121428571428571,
      "exact_transcription_accuracy": 0.9371428571428572,
      "raw_exact_transcription_accuracy": 0.9285714285714286,
      "valid_raw_CER": 0.02392733564013841,
      "valid_interpreted_CER": 0.023823529411764705,
      "valid_transcription_denominator": 1400,
      "valid_character_denominator": 173400,
      "accepted_output_transcription_correctness": 0.9954476479514416,
      "accepted_transcription_denominator": 1318,
      "valid_end_to_end_page_correctness": 0.9371428571428572,
      "semantic_expected_outcome_accuracy": 0.4707142857142857,
      "preprocessing_failures": 0,
      "D1_segmentation_failures": 30,
      "OCR_decoding_failures": 0,
      "pipeline_exceptions": 0,
      "raw_wrong_tier_images": 522,
      "wrong_tier_reviews": 498,
      "corrected_images": 24,
      "contextual_edits": 36,
      "correct_T1_edits": 18,
      "incorrect_T1_edits": 0,
      "unknown_or_unscorable_T1_edits": 18,
      "valid_images_with_worsened_edits": 0,
      "rule_trigger_image_counts": {
        "R3_COMPLETE_TEXT_CONTRADICTION": 782,
        "OBSERVABLE_CHECKS_PASSED": 2272,
        "DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH": 498,
        "R2_INTERNAL_DECODED_COUNT_GAP": 114,
        "D1_NO_SAFE_PROPOSAL": 30,
        "R3_ROW_TOTAL_CONFLICT": 16
      },
      "source_parents": 700,
      "macro_parent": {
        "true_invalid_rejection_rate": 0.0,
        "invalid_accepted_rate": 0.03428571428571429,
        "valid_acceptance_rate": 0.9414285714285714,
        "false_definitive_rejection_rate": 0.0,
        "invalid_review_rate": 0.9657142857142857,
        "valid_review_rate": 0.05857142857142857,
        "exact_transcription_accuracy": 0.9371428571428572,
        "accepted_output_transcription_correctness": 0.9954545454545455,
        "valid_interpreted_CER": 0.010570941668971144
      },
      "parent_worst_case_counts": {
        "unresolved_review": 673,
        "critical_failure": 27
      },
      "outcome_counts_and_rates": {
        "VALID": {
          "ACCEPT": {
            "count": 1318,
            "rate": 0.9414285714285714
          },
          "REJECT_INVALID": {
            "count": 0,
            "rate": 0.0
          },
          "REVIEW_UNCERTAIN": {
            "count": 82,
            "rate": 0.05857142857142857
          }
        },
        "INVALID": {
          "ACCEPT": {
            "count": 48,
            "rate": 0.03428571428571429
          },
          "REJECT_INVALID": {
            "count": 0,
            "rate": 0.0
          },
          "REVIEW_UNCERTAIN": {
            "count": 1352,
            "rate": 0.9657142857142857
          }
        }
      }
    },
    "R3_1": {
      "images": 2800,
      "valid_denominator": 1400,
      "invalid_denominator": 1400,
      "three_way_confusion": {
        "VALID": {
          "ACCEPT": 1180,
          "REJECT_INVALID": 0,
          "REVIEW_UNCERTAIN": 220
        },
        "INVALID": {
          "ACCEPT": 40,
          "REJECT_INVALID": 0,
          "REVIEW_UNCERTAIN": 1360
        }
      },
      "true_invalid_rejection_rate": 0.0,
      "invalid_accepted_rate": 0.02857142857142857,
      "invalid_review_rate": 0.9714285714285714,
      "invalid_images_incorrectly_accepted_as_valid": 40,
      "invalid_interpreted_syntax_passes": 896,
      "invalid_interpreted_syntax_pass_rate": 0.64,
      "valid_acceptance_rate": 0.8428571428571429,
      "false_definitive_rejection_rate": 0.0,
      "valid_review_rate": 0.15714285714285714,
      "review_rate": 0.5642857142857143,
      "exact_transcription_accuracy": 0.9371428571428572,
      "raw_exact_transcription_accuracy": 0.9285714285714286,
      "valid_raw_CER": 0.02392733564013841,
      "valid_interpreted_CER": 0.023823529411764705,
      "valid_transcription_denominator": 1400,
      "valid_character_denominator": 173400,
      "accepted_output_transcription_correctness": 0.9949152542372881,
      "accepted_transcription_denominator": 1180,
      "valid_end_to_end_page_correctness": 0.8385714285714285,
      "semantic_expected_outcome_accuracy": 0.42142857142857143,
      "preprocessing_failures": 0,
      "D1_segmentation_failures": 30,
      "OCR_decoding_failures": 0,
      "pipeline_exceptions": 0,
      "raw_wrong_tier_images": 522,
      "wrong_tier_reviews": 498,
      "corrected_images": 24,
      "contextual_edits": 36,
      "correct_T1_edits": 18,
      "incorrect_T1_edits": 0,
      "unknown_or_unscorable_T1_edits": 18,
      "valid_images_with_worsened_edits": 0,
      "rule_trigger_image_counts": {
        "R3_COMPLETE_TEXT_CONTRADICTION": 782,
        "OBSERVABLE_CHECKS_PASSED": 2272,
        "R3_1_POSITION_UNAVAILABLE": 146,
        "DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH": 498,
        "R2_INTERNAL_DECODED_COUNT_GAP": 114,
        "D1_NO_SAFE_PROPOSAL": 30,
        "R3_ROW_TOTAL_CONFLICT": 16
      },
      "source_parents": 700,
      "macro_parent": {
        "true_invalid_rejection_rate": 0.0,
        "invalid_accepted_rate": 0.02857142857142857,
        "valid_acceptance_rate": 0.8428571428571429,
        "false_definitive_rejection_rate": 0.0,
        "invalid_review_rate": 0.9714285714285714,
        "valid_review_rate": 0.15714285714285714,
        "exact_transcription_accuracy": 0.9371428571428572,
        "accepted_output_transcription_correctness": 0.9949238578680203,
        "valid_interpreted_CER": 0.010570941668971144
      },
      "parent_worst_case_counts": {
        "unresolved_review": 677,
        "critical_failure": 23
      },
      "outcome_counts_and_rates": {
        "VALID": {
          "ACCEPT": {
            "count": 1180,
            "rate": 0.8428571428571429
          },
          "REJECT_INVALID": {
            "count": 0,
            "rate": 0.0
          },
          "REVIEW_UNCERTAIN": {
            "count": 220,
            "rate": 0.15714285714285714
          }
        },
        "INVALID": {
          "ACCEPT": {
            "count": 40,
            "rate": 0.02857142857142857
          },
          "REJECT_INVALID": {
            "count": 0,
            "rate": 0.0
          },
          "REVIEW_UNCERTAIN": {
            "count": 1360,
            "rate": 0.9714285714285714
          }
        }
      }
    }
  }
}
~~~
