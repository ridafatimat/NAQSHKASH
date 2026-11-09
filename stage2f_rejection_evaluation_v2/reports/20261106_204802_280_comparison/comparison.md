# R0 versus R1 — actual saved development results

No automatic winner. REVIEW is not definitive rejection. Inspect invalid false acceptance, valid false rejection, exact/CER, accepted correctness, parent worst cases and intervention harm together.

```json
{
  "R0": {
    "images": 2800,
    "valid_denominator": 1400,
    "invalid_denominator": 1400,
    "three_way_confusion": {
      "VALID": {
        "ACCEPT": 1364,
        "REJECT_INVALID": 0,
        "REVIEW_UNCERTAIN": 36
      },
      "INVALID": {
        "ACCEPT": 884,
        "REJECT_INVALID": 0,
        "REVIEW_UNCERTAIN": 516
      }
    },
    "true_invalid_rejection_rate": 0.0,
    "invalid_accepted_rate": 0.6314285714285715,
    "invalid_review_rate": 0.36857142857142855,
    "invalid_images_incorrectly_accepted_as_valid": 884,
    "invalid_interpreted_syntax_passes": 884,
    "invalid_interpreted_syntax_pass_rate": 0.6314285714285715,
    "valid_acceptance_rate": 0.9742857142857143,
    "false_definitive_rejection_rate": 0.0,
    "valid_review_rate": 0.025714285714285714,
    "review_rate": 0.19714285714285715,
    "exact_transcription_accuracy": 0.9285714285714286,
    "raw_exact_transcription_accuracy": 0.9285714285714286,
    "valid_raw_CER": 0.02392733564013841,
    "valid_interpreted_CER": 0.02392733564013841,
    "valid_transcription_denominator": 1400,
    "valid_character_denominator": 173400,
    "accepted_output_transcription_correctness": 0.9530791788856305,
    "accepted_transcription_denominator": 1364,
    "valid_end_to_end_page_correctness": 0.9285714285714286,
    "semantic_expected_outcome_accuracy": 0.48714285714285716,
    "preprocessing_failures": 0,
    "D1_segmentation_failures": 30,
    "OCR_decoding_failures": 0,
    "pipeline_exceptions": 0,
    "raw_wrong_tier_images": 522,
    "wrong_tier_reviews": 522,
    "corrected_images": 0,
    "contextual_edits": 0,
    "correct_T1_edits": 0,
    "incorrect_T1_edits": 0,
    "unknown_or_unscorable_T1_edits": 0,
    "valid_images_with_worsened_edits": 0,
    "rule_trigger_image_counts": {
      "OBSERVABLE_CHECKS_PASSED": 2248,
      "DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH": 522,
      "D1_NO_SAFE_PROPOSAL": 30
    },
    "source_parents": 700,
    "macro_parent": {
      "true_invalid_rejection_rate": 0.0,
      "invalid_accepted_rate": 0.6314285714285715,
      "valid_acceptance_rate": 0.9742857142857143,
      "false_definitive_rejection_rate": 0.0,
      "invalid_review_rate": 0.36857142857142855,
      "valid_review_rate": 0.025714285714285714,
      "exact_transcription_accuracy": 0.9285714285714286,
      "accepted_output_transcription_correctness": 0.9524158125915081,
      "valid_interpreted_CER": 0.010951507651927395
    },
    "parent_worst_case_counts": {
      "critical_failure": 451,
      "unresolved_review": 249
    }
  },
  "R1": {
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
        "ACCEPT": 896,
        "REJECT_INVALID": 0,
        "REVIEW_UNCERTAIN": 504
      }
    },
    "true_invalid_rejection_rate": 0.0,
    "invalid_accepted_rate": 0.64,
    "invalid_review_rate": 0.36,
    "invalid_images_incorrectly_accepted_as_valid": 896,
    "invalid_interpreted_syntax_passes": 896,
    "invalid_interpreted_syntax_pass_rate": 0.64,
    "valid_acceptance_rate": 0.9828571428571429,
    "false_definitive_rejection_rate": 0.0,
    "valid_review_rate": 0.017142857142857144,
    "review_rate": 0.18857142857142858,
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
      "D1_NO_SAFE_PROPOSAL": 30
    },
    "source_parents": 700,
    "macro_parent": {
      "true_invalid_rejection_rate": 0.0,
      "invalid_accepted_rate": 0.64,
      "valid_acceptance_rate": 0.9828571428571429,
      "false_definitive_rejection_rate": 0.0,
      "invalid_review_rate": 0.36,
      "valid_review_rate": 0.017142857142857144,
      "exact_transcription_accuracy": 0.9371428571428572,
      "accepted_output_transcription_correctness": 0.9528301886792453,
      "valid_interpreted_CER": 0.010570941668971144
    },
    "parent_worst_case_counts": {
      "critical_failure": 457,
      "unresolved_review": 243
    }
  },
  "deltas_R1_minus_R0": {
    "true_invalid_rejection_rate": 0.0,
    "invalid_accepted_rate": 0.008571428571428563,
    "valid_acceptance_rate": 0.008571428571428563,
    "false_definitive_rejection_rate": 0.0,
    "invalid_review_rate": -0.008571428571428563,
    "valid_review_rate": -0.00857142857142857,
    "review_rate": -0.008571428571428563,
    "exact_transcription_accuracy": 0.008571428571428563,
    "valid_interpreted_CER": -0.00010380622837370332,
    "accepted_output_transcription_correctness": 0.000409193207392744
  },
  "critical_regressions": {
    "invalid_false_acceptance_increased": true,
    "valid_false_rejection_increased": false
  },
  "outcome_transitions": [
    {
      "validity": "INVALID",
      "R0": "ACCEPT",
      "R1": "ACCEPT",
      "images": 884
    },
    {
      "validity": "INVALID",
      "R0": "REVIEW_UNCERTAIN",
      "R1": "ACCEPT",
      "images": 12
    },
    {
      "validity": "INVALID",
      "R0": "REVIEW_UNCERTAIN",
      "R1": "REVIEW_UNCERTAIN",
      "images": 504
    },
    {
      "validity": "VALID",
      "R0": "ACCEPT",
      "R1": "ACCEPT",
      "images": 1364
    },
    {
      "validity": "VALID",
      "R0": "REVIEW_UNCERTAIN",
      "R1": "ACCEPT",
      "images": 12
    },
    {
      "validity": "VALID",
      "R0": "REVIEW_UNCERTAIN",
      "R1": "REVIEW_UNCERTAIN",
      "images": 24
    }
  ],
  "selection_decision": "NOT_AUTOMATIC: review critical errors, real transcription correctness and unscorable interventions together",
  "evidence_scope": "development synthetic canonical labels; no Talim expert review; no protected-test inference"
}
```
