# Frozen R3 validation_calibration results

REVIEW is not definitive REJECT. R3 ACCEPT is not carpet-generation authorization. No tuning or R3.1 is used.

~~~json
{
  "images": 600,
  "valid_denominator": 300,
  "invalid_denominator": 300,
  "three_way_confusion": {
    "VALID": {
      "ACCEPT": 270,
      "REJECT_INVALID": 0,
      "REVIEW_UNCERTAIN": 30
    },
    "INVALID": {
      "ACCEPT": 12,
      "REJECT_INVALID": 0,
      "REVIEW_UNCERTAIN": 288
    }
  },
  "true_invalid_rejection_rate": 0.0,
  "invalid_accepted_rate": 0.04,
  "invalid_review_rate": 0.96,
  "invalid_images_incorrectly_accepted_as_valid": 12,
  "invalid_interpreted_syntax_passes": 207,
  "invalid_interpreted_syntax_pass_rate": 0.69,
  "valid_acceptance_rate": 0.9,
  "false_definitive_rejection_rate": 0.0,
  "valid_review_rate": 0.1,
  "review_rate": 0.53,
  "exact_transcription_accuracy": 0.8866666666666667,
  "raw_exact_transcription_accuracy": 0.8666666666666667,
  "valid_raw_CER": 0.11572142496905231,
  "valid_interpreted_CER": 0.11553803126862593,
  "valid_transcription_denominator": 300,
  "valid_character_denominator": 43622,
  "accepted_output_transcription_correctness": 0.9851851851851852,
  "accepted_transcription_denominator": 270,
  "valid_end_to_end_page_correctness": 0.8866666666666667,
  "semantic_expected_outcome_accuracy": 0.45,
  "preprocessing_failures": 0,
  "D1_segmentation_failures": 28,
  "OCR_decoding_failures": 0,
  "pipeline_exceptions": 0,
  "raw_wrong_tier_images": 89,
  "wrong_tier_reviews": 77,
  "corrected_images": 12,
  "contextual_edits": 16,
  "correct_T1_edits": 8,
  "incorrect_T1_edits": 0,
  "unknown_or_unscorable_T1_edits": 8,
  "valid_images_with_worsened_edits": 0,
  "rule_trigger_image_counts": {
    "DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH": 77,
    "OBSERVABLE_CHECKS_PASSED": 495,
    "R3_COMPLETE_TEXT_CONTRADICTION": 191,
    "D1_NO_SAFE_PROPOSAL": 28,
    "R2_INTERNAL_DECODED_COUNT_GAP": 16,
    "R3_ROW_TOTAL_CONFLICT": 6
  },
  "source_parents": 150,
  "macro_parent": {
    "true_invalid_rejection_rate": 0.0,
    "invalid_accepted_rate": 0.04,
    "valid_acceptance_rate": 0.9,
    "false_definitive_rejection_rate": 0.0,
    "invalid_review_rate": 0.96,
    "valid_review_rate": 0.1,
    "exact_transcription_accuracy": 0.8866666666666667,
    "accepted_output_transcription_correctness": 0.9851851851851852,
    "valid_interpreted_CER": 0.04197817384524879
  },
  "parent_worst_case_counts": {
    "unresolved_review": 142,
    "critical_failure": 8
  },
  "outcome_counts_and_rates": {
    "VALID": {
      "ACCEPT": {
        "count": 270,
        "rate": 0.9
      },
      "REJECT_INVALID": {
        "count": 0,
        "rate": 0.0
      },
      "REVIEW_UNCERTAIN": {
        "count": 30,
        "rate": 0.1
      }
    },
    "INVALID": {
      "ACCEPT": {
        "count": 12,
        "rate": 0.04
      },
      "REJECT_INVALID": {
        "count": 0,
        "rate": 0.0
      },
      "REVIEW_UNCERTAIN": {
        "count": 288,
        "rate": 0.96
      }
    }
  }
}
~~~

See metrics.json for parent/category/variant results, transitions, reasons and localization coverage. Broad row highlights do not imply precise glyph localization.
