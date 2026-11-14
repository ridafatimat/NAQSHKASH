# T0 versus T1 — saved development outputs

Review-to-accept transitions alone are not improvements. Inspect exact accuracy, CER, accepted correctness, harmful edits and parent/cohort results.

## T0

```json
{
  "images": 776,
  "scorable_images": 776,
  "characters": 13489,
  "outcome_counts": {
    "ACCEPT": 757,
    "REVIEW_UNCERTAIN": 19
  },
  "exact_transcription_accuracy": 0.9458762886597938,
  "raw_CER": 0.38846467492030545,
  "interpreted_CER": 0.38846467492030545,
  "review_rate": 0.024484536082474227,
  "wrong_tier_reviews": 4,
  "accepted_output_correctness": 0.9696169088507266,
  "accepted_output_denominator": 757,
  "end_to_end_page_correctness": 0.9458762886597938,
  "corrected_images": 0,
  "contextual_corrections": 0,
  "correct_corrections": 0,
  "incorrect_corrections": 0,
  "unscorable_corrections": 0,
  "syntax_failures_resolved": 0,
  "worsened_CER_images": 0,
  "independent_parents": 230,
  "macro_parent": {
    "exact_transcription_accuracy": 0.9478260869565217,
    "interpreted_CER": 0.0312908615651285,
    "review_rate": 0.02717391304347826,
    "end_to_end_page_correctness": 0.9478260869565217
  }
}
```

## T1

```json
{
  "images": 776,
  "scorable_images": 776,
  "characters": 13489,
  "outcome_counts": {
    "ACCEPT": 758,
    "REVIEW_UNCERTAIN": 18
  },
  "exact_transcription_accuracy": 0.9471649484536082,
  "raw_CER": 0.38846467492030545,
  "interpreted_CER": 0.3883905404403588,
  "review_rate": 0.023195876288659795,
  "wrong_tier_reviews": 3,
  "accepted_output_correctness": 0.9696569920844327,
  "accepted_output_denominator": 758,
  "end_to_end_page_correctness": 0.9471649484536082,
  "corrected_images": 5,
  "contextual_corrections": 5,
  "correct_corrections": 1,
  "incorrect_corrections": 0,
  "unscorable_corrections": 4,
  "syntax_failures_resolved": 1,
  "worsened_CER_images": 0,
  "independent_parents": 230,
  "macro_parent": {
    "exact_transcription_accuracy": 0.9521739130434783,
    "interpreted_CER": 0.031073470260780674,
    "review_rate": 0.02282608695652174,
    "end_to_end_page_correctness": 0.9521739130434783
  }
}
```
