# Frozen R3 protected_test results

REVIEW is not definitive REJECT. R3 ACCEPT is not carpet-generation authorization. No tuning or R3.1 is used.

~~~json
{
  "images": 600,
  "valid_denominator": 300,
  "invalid_denominator": 300,
  "three_way_confusion": {
    "VALID": {
      "ACCEPT": 274,
      "REJECT_INVALID": 0,
      "REVIEW_UNCERTAIN": 26
    },
    "INVALID": {
      "ACCEPT": 6,
      "REJECT_INVALID": 0,
      "REVIEW_UNCERTAIN": 294
    }
  },
  "true_invalid_rejection_rate": 0.0,
  "invalid_accepted_rate": 0.02,
  "invalid_review_rate": 0.98,
  "invalid_images_incorrectly_accepted_as_valid": 6,
  "invalid_interpreted_syntax_passes": 204,
  "invalid_interpreted_syntax_pass_rate": 0.68,
  "valid_acceptance_rate": 0.9133333333333333,
  "false_definitive_rejection_rate": 0.0,
  "valid_review_rate": 0.08666666666666667,
  "review_rate": 0.5333333333333333,
  "exact_transcription_accuracy": 0.9,
  "raw_exact_transcription_accuracy": 0.8866666666666667,
  "valid_raw_CER": 0.12256630755108942,
  "valid_interpreted_CER": 0.1224696845258225,
  "valid_transcription_denominator": 300,
  "valid_character_denominator": 41398,
  "accepted_output_transcription_correctness": 0.9854014598540146,
  "accepted_transcription_denominator": 274,
  "valid_end_to_end_page_correctness": 0.9,
  "semantic_expected_outcome_accuracy": 0.45666666666666667,
  "preprocessing_failures": 0,
  "D1_segmentation_failures": 24,
  "OCR_decoding_failures": 0,
  "pipeline_exceptions": 0,
  "raw_wrong_tier_images": 98,
  "wrong_tier_reviews": 90,
  "corrected_images": 8,
  "contextual_edits": 8,
  "correct_T1_edits": 4,
  "incorrect_T1_edits": 0,
  "unknown_or_unscorable_T1_edits": 4,
  "valid_images_with_worsened_edits": 0,
  "rule_trigger_image_counts": {
    "OBSERVABLE_CHECKS_PASSED": 486,
    "DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH": 90,
    "R3_COMPLETE_TEXT_CONTRADICTION": 181,
    "D1_NO_SAFE_PROPOSAL": 24,
    "R2_INTERNAL_DECODED_COUNT_GAP": 25
  },
  "source_parents": 150,
  "macro_parent": {
    "true_invalid_rejection_rate": 0.0,
    "invalid_accepted_rate": 0.02,
    "valid_acceptance_rate": 0.9133333333333333,
    "false_definitive_rejection_rate": 0.0,
    "invalid_review_rate": 0.98,
    "valid_review_rate": 0.08666666666666667,
    "exact_transcription_accuracy": 0.9,
    "accepted_output_transcription_correctness": 0.9854014598540146,
    "valid_interpreted_CER": 0.046761863803081144
  },
  "parent_worst_case_counts": {
    "unresolved_review": 145,
    "critical_failure": 5
  },
  "outcome_counts_and_rates": {
    "VALID": {
      "ACCEPT": {
        "count": 274,
        "rate": 0.9133333333333333
      },
      "REJECT_INVALID": {
        "count": 0,
        "rate": 0.0
      },
      "REVIEW_UNCERTAIN": {
        "count": 26,
        "rate": 0.08666666666666667
      }
    },
    "INVALID": {
      "ACCEPT": {
        "count": 6,
        "rate": 0.02
      },
      "REJECT_INVALID": {
        "count": 0,
        "rate": 0.0
      },
      "REVIEW_UNCERTAIN": {
        "count": 294,
        "rate": 0.98
      }
    }
  }
}
~~~

See metrics.json for parent/category/variant results, transitions, reasons and localization coverage. Broad row highlights do not imply precise glyph localization.
