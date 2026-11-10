# R3_HITL_V3 development results

REVIEW is not definitive rejection. This policy permits any adequately supported complete canonical page interpretation. Uniqueness is not required; incomplete or unsupported evidence remains REVIEW. Differences from Stage 1 are listed without forcing target counts. Image validity is not proved by text consistency.

~~~json
{
  "R2": {
    "images": 2800,
    "valid_denominator": 1400,
    "invalid_denominator": 1400,
    "three_way_confusion": {
      "VALID": {
        "ACCEPT": 1376,
        "REJECT_INVALID": 0,
        "REVIEW_UNCERTAIN": 24
      },
      "INVALID": {
        "ACCEPT": 788,
        "REJECT_INVALID": 0,
        "REVIEW_UNCERTAIN": 612
      }
    },
    "true_invalid_rejection_rate": 0.0,
    "invalid_accepted_rate": 0.5628571428571428,
    "invalid_review_rate": 0.43714285714285717,
    "invalid_images_incorrectly_accepted_as_valid": 788,
    "invalid_interpreted_syntax_passes": 896,
    "invalid_interpreted_syntax_pass_rate": 0.64,
    "valid_acceptance_rate": 0.9828571428571429,
    "false_definitive_rejection_rate": 0.0,
    "valid_review_rate": 0.017142857142857144,
    "review_rate": 0.22714285714285715,
    "exact_transcription_accuracy": 0.9371428571428572,
    "raw_exact_transcription_accuracy": 0.9285714285714286,
    "valid_raw_CER": 0.02392733564013841,
    "valid_interpreted_CER": 0.023823529411764705,
    "valid_transcription_denominator": 1400,
    "valid_character_denominator": 173400,
    "accepted_output_transcription_correctness": 0.9534883720930233,
    "accepted_transcription_denominator": 1376,
    "valid_end_to_end_page_correctness": 0.9371428571428572,
    "semantic_expected_outcome_accuracy": 0.49142857142857144,
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
      "OBSERVABLE_CHECKS_PASSED": 2272,
      "DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH": 498,
      "R2_INTERNAL_DECODED_COUNT_GAP": 114,
      "D1_NO_SAFE_PROPOSAL": 30
    },
    "source_parents": 700,
    "macro_parent": {
      "true_invalid_rejection_rate": 0.0,
      "invalid_accepted_rate": 0.5628571428571428,
      "valid_acceptance_rate": 0.9828571428571429,
      "false_definitive_rejection_rate": 0.0,
      "invalid_review_rate": 0.43714285714285717,
      "valid_review_rate": 0.017142857142857144,
      "exact_transcription_accuracy": 0.9371428571428572,
      "accepted_output_transcription_correctness": 0.9528301886792453,
      "valid_interpreted_CER": 0.010570941668971144
    },
    "parent_worst_case_counts": {
      "critical_failure": 403,
      "unresolved_review": 297
    },
    "outcome_counts_and_rates": {
      "VALID": {
        "ACCEPT": {
          "count": 1376,
          "rate": 0.9828571428571429
        },
        "REVIEW_UNCERTAIN": {
          "count": 24,
          "rate": 0.017142857142857144
        },
        "REJECT_INVALID": {
          "count": 0,
          "rate": 0.0
        }
      },
      "INVALID": {
        "ACCEPT": {
          "count": 788,
          "rate": 0.5628571428571428
        },
        "REVIEW_UNCERTAIN": {
          "count": 612,
          "rate": 0.43714285714285717
        },
        "REJECT_INVALID": {
          "count": 0,
          "rate": 0.0
        }
      }
    }
  },
  "Stage1": {
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
      "OBSERVABLE_CHECKS_PASSED": 2272,
      "DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH": 498,
      "R2_INTERNAL_DECODED_COUNT_GAP": 114,
      "D1_NO_SAFE_PROPOSAL": 30
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
        "REVIEW_UNCERTAIN": {
          "count": 82,
          "rate": 0.05857142857142857
        },
        "REJECT_INVALID": {
          "count": 0,
          "rate": 0.0
        }
      },
      "INVALID": {
        "ACCEPT": {
          "count": 48,
          "rate": 0.03428571428571429
        },
        "REVIEW_UNCERTAIN": {
          "count": 1352,
          "rate": 0.9657142857142857
        },
        "REJECT_INVALID": {
          "count": 0,
          "rate": 0.0
        }
      }
    }
  },
  "R3_HITL_V3": {
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
        "REVIEW_UNCERTAIN": {
          "count": 82,
          "rate": 0.05857142857142857
        },
        "REJECT_INVALID": {
          "count": 0,
          "rate": 0.0
        }
      },
      "INVALID": {
        "ACCEPT": {
          "count": 48,
          "rate": 0.03428571428571429
        },
        "REVIEW_UNCERTAIN": {
          "count": 1352,
          "rate": 0.9657142857142857
        },
        "REJECT_INVALID": {
          "count": 0,
          "rate": 0.0
        }
      }
    }
  }
}
~~~

See metrics.json for transitions, category/parent/variant/transcription/coverage/localization metrics; Stage1_differences.csv identifies every changed image. Precise pixel localization has no scoring coverage; broad row/tier overlap is a separate metric. Optional overlays require verified source transforms and may be unavailable. No validation/protected run or freeze was performed.
