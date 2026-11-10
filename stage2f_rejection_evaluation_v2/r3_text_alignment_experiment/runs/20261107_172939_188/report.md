# R3 Stage 1 — measured development text-alignment results

Conditional literal-text contradictions are not image invalidity. The counterfactual changes ACCEPT to REVIEW only, preserves existing REVIEW/REJECT and never modifies R2. Incomplete searches cannot establish contradictions.

~~~json
{
  "images": 2800,
  "parents": 700,
  "plausible_alignment": {
    "images": 1366,
    "parents": 660
  },
  "complete_model_contradiction": {
    "images": 1404,
    "parents": 668
  },
  "all_candidates_fail_in_tested_search": {
    "images": 1404,
    "parents": 668
  },
  "incomplete_or_unscorable": {
    "images": 30,
    "parents": 11
  },
  "multiple_candidates": {
    "images": 248,
    "parents": 75
  },
  "one_page_candidate_in_search": {
    "images": 1220,
    "parents": 592
  },
  "multiple_page_candidates_in_search": {
    "images": 146,
    "parents": 70
  },
  "unique_page_reconstruction_proven_in_model": {
    "images": 1220,
    "parents": 592
  },
  "multiple_common_totals": {
    "images": 0,
    "parents": 0
  },
  "ACCEPT_to_REVIEW": {
    "images": 798,
    "parents": 379
  },
  "invalid_ACCEPT_to_REVIEW": {
    "images": 740,
    "parents": 370
  },
  "valid_ACCEPT_to_REVIEW": {
    "images": 58,
    "parents": 30
  },
  "R2_confusion": {
    "VALID": {
      "ACCEPT": 1376,
      "REVIEW_UNCERTAIN": 24
    },
    "INVALID": {
      "ACCEPT": 788,
      "REVIEW_UNCERTAIN": 612
    }
  },
  "review_only_confusion": {
    "VALID": {
      "ACCEPT": 1318,
      "REVIEW_UNCERTAIN": 82
    },
    "INVALID": {
      "REVIEW_UNCERTAIN": 1352,
      "ACCEPT": 48
    }
  },
  "passing_row_multiplicity": {
    "one_in_search": 3857,
    "multiple_in_search": 264,
    "none_in_search": 1421
  },
  "no_new_definitive_rejection": true
}
~~~

Targeted cohorts (measured, not forced to a hypothesis):

~~~json
{
  "invalid_R2_accept": {
    "images": 788,
    "parents": 394,
    "plausible_alignment": {
      "images": 48,
      "parents": 24
    },
    "complete_model_contradiction": {
      "images": 740,
      "parents": 370
    },
    "all_candidates_fail_in_tested_search": {
      "images": 740,
      "parents": 370
    },
    "incomplete_or_unscorable": {
      "images": 0,
      "parents": 0
    },
    "multiple_candidates": {
      "images": 58,
      "parents": 29
    },
    "one_page_candidate_in_search": {
      "images": 40,
      "parents": 20
    },
    "multiple_page_candidates_in_search": {
      "images": 8,
      "parents": 4
    },
    "unique_page_reconstruction_proven_in_model": {
      "images": 40,
      "parents": 20
    },
    "multiple_common_totals": {
      "images": 0,
      "parents": 0
    },
    "ACCEPT_to_REVIEW": {
      "images": 740,
      "parents": 370
    },
    "invalid_ACCEPT_to_REVIEW": {
      "images": 740,
      "parents": 370
    },
    "valid_ACCEPT_to_REVIEW": {
      "images": 0,
      "parents": 0
    },
    "R2_confusion": {
      "VALID": {},
      "INVALID": {
        "ACCEPT": 788
      }
    },
    "review_only_confusion": {
      "VALID": {},
      "INVALID": {
        "REVIEW_UNCERTAIN": 740,
        "ACCEPT": 48
      }
    },
    "passing_row_multiplicity": {
      "one_in_search": 779,
      "multiple_in_search": 60,
      "none_in_search": 743
    },
    "no_new_definitive_rejection": true
  },
  "pattern_faithful_invalid_R2_accept": {
    "images": 507,
    "parents": 254,
    "plausible_alignment": {
      "images": 14,
      "parents": 7
    },
    "complete_model_contradiction": {
      "images": 493,
      "parents": 247
    },
    "all_candidates_fail_in_tested_search": {
      "images": 493,
      "parents": 247
    },
    "incomplete_or_unscorable": {
      "images": 0,
      "parents": 0
    },
    "multiple_candidates": {
      "images": 38,
      "parents": 19
    },
    "one_page_candidate_in_search": {
      "images": 10,
      "parents": 5
    },
    "multiple_page_candidates_in_search": {
      "images": 4,
      "parents": 2
    },
    "unique_page_reconstruction_proven_in_model": {
      "images": 10,
      "parents": 5
    },
    "multiple_common_totals": {
      "images": 0,
      "parents": 0
    },
    "ACCEPT_to_REVIEW": {
      "images": 493,
      "parents": 247
    },
    "invalid_ACCEPT_to_REVIEW": {
      "images": 493,
      "parents": 247
    },
    "valid_ACCEPT_to_REVIEW": {
      "images": 0,
      "parents": 0
    },
    "R2_confusion": {
      "VALID": {},
      "INVALID": {
        "ACCEPT": 507
      }
    },
    "review_only_confusion": {
      "VALID": {},
      "INVALID": {
        "REVIEW_UNCERTAIN": 493,
        "ACCEPT": 14
      }
    },
    "passing_row_multiplicity": {
      "one_in_search": 473,
      "multiple_in_search": 40,
      "none_in_search": 485
    },
    "no_new_definitive_rejection": true
  },
  "preserved_reversal_invalid_R2_accept": {
    "images": 429,
    "parents": 215,
    "plausible_alignment": {
      "images": 8,
      "parents": 4
    },
    "complete_model_contradiction": {
      "images": 421,
      "parents": 211
    },
    "all_candidates_fail_in_tested_search": {
      "images": 421,
      "parents": 211
    },
    "incomplete_or_unscorable": {
      "images": 0,
      "parents": 0
    },
    "multiple_candidates": {
      "images": 26,
      "parents": 13
    },
    "one_page_candidate_in_search": {
      "images": 6,
      "parents": 3
    },
    "multiple_page_candidates_in_search": {
      "images": 2,
      "parents": 1
    },
    "unique_page_reconstruction_proven_in_model": {
      "images": 6,
      "parents": 3
    },
    "multiple_common_totals": {
      "images": 0,
      "parents": 0
    },
    "ACCEPT_to_REVIEW": {
      "images": 421,
      "parents": 211
    },
    "invalid_ACCEPT_to_REVIEW": {
      "images": 421,
      "parents": 211
    },
    "valid_ACCEPT_to_REVIEW": {
      "images": 0,
      "parents": 0
    },
    "R2_confusion": {
      "VALID": {},
      "INVALID": {
        "ACCEPT": 429
      }
    },
    "review_only_confusion": {
      "VALID": {},
      "INVALID": {
        "REVIEW_UNCERTAIN": 421,
        "ACCEPT": 8
      }
    },
    "passing_row_multiplicity": {
      "one_in_search": 395,
      "multiple_in_search": 28,
      "none_in_search": 433
    },
    "no_new_definitive_rejection": true
  },
  "lost_internal_count_gap": {
    "images": 216,
    "parents": 108,
    "plausible_alignment": {
      "images": 2,
      "parents": 1
    },
    "complete_model_contradiction": {
      "images": 214,
      "parents": 107
    },
    "all_candidates_fail_in_tested_search": {
      "images": 214,
      "parents": 107
    },
    "incomplete_or_unscorable": {
      "images": 0,
      "parents": 0
    },
    "multiple_candidates": {
      "images": 20,
      "parents": 10
    },
    "one_page_candidate_in_search": {
      "images": 0,
      "parents": 0
    },
    "multiple_page_candidates_in_search": {
      "images": 2,
      "parents": 1
    },
    "unique_page_reconstruction_proven_in_model": {
      "images": 0,
      "parents": 0
    },
    "multiple_common_totals": {
      "images": 0,
      "parents": 0
    },
    "ACCEPT_to_REVIEW": {
      "images": 212,
      "parents": 106
    },
    "invalid_ACCEPT_to_REVIEW": {
      "images": 212,
      "parents": 106
    },
    "valid_ACCEPT_to_REVIEW": {
      "images": 0,
      "parents": 0
    },
    "R2_confusion": {
      "VALID": {},
      "INVALID": {
        "ACCEPT": 214,
        "REVIEW_UNCERTAIN": 2
      }
    },
    "review_only_confusion": {
      "VALID": {},
      "INVALID": {
        "REVIEW_UNCERTAIN": 214,
        "ACCEPT": 2
      }
    },
    "passing_row_multiplicity": {
      "none_in_search": 213,
      "one_in_search": 225,
      "multiple_in_search": 20
    },
    "no_new_definitive_rejection": true
  },
  "wrong_tier_substituted_invalid_R2_accept": {
    "images": 38,
    "parents": 19,
    "plausible_alignment": {
      "images": 32,
      "parents": 16
    },
    "complete_model_contradiction": {
      "images": 6,
      "parents": 3
    },
    "all_candidates_fail_in_tested_search": {
      "images": 6,
      "parents": 3
    },
    "incomplete_or_unscorable": {
      "images": 0,
      "parents": 0
    },
    "multiple_candidates": {
      "images": 2,
      "parents": 1
    },
    "one_page_candidate_in_search": {
      "images": 30,
      "parents": 15
    },
    "multiple_page_candidates_in_search": {
      "images": 2,
      "parents": 1
    },
    "unique_page_reconstruction_proven_in_model": {
      "images": 30,
      "parents": 15
    },
    "multiple_common_totals": {
      "images": 0,
      "parents": 0
    },
    "ACCEPT_to_REVIEW": {
      "images": 6,
      "parents": 3
    },
    "invalid_ACCEPT_to_REVIEW": {
      "images": 6,
      "parents": 3
    },
    "valid_ACCEPT_to_REVIEW": {
      "images": 0,
      "parents": 0
    },
    "R2_confusion": {
      "VALID": {},
      "INVALID": {
        "ACCEPT": 38
      }
    },
    "review_only_confusion": {
      "VALID": {},
      "INVALID": {
        "ACCEPT": 32,
        "REVIEW_UNCERTAIN": 6
      }
    },
    "passing_row_multiplicity": {
      "one_in_search": 68,
      "none_in_search": 6,
      "multiple_in_search": 2
    },
    "no_new_definitive_rejection": true
  }
}
~~~

See metrics.json for validity/category/variant/parent/OCR/fidelity breakdowns and per_image.jsonl for row candidates, inferred totals, coverage and failure reasons. Parent counts within conditions overlap and illumination variants are not independent.

Text-space alignment can find a strict canonical reconstruction or disprove all reconstructions under its literal prefix-only model. It cannot recover source geometry, repaired spaces, lost glyphs, true count absence or wrong glyph identities. A passing candidate is a plausible alternative text interpretation, not source validity. Review how many valid controls acquire contradictions and how much ambiguity/coverage remains before designing Stage 2. Use CTC for observed character/run positions and image components for independent glyph/absence support; the measured report must inform that choice. No production rejection policy is selected.
