# Implementation and lightweight verification

Implemented Stage 2F at C:\NAQSHKASH\stage2\stage2f_negative_rejection.
No model training/fine-tuning, backward pass, optimizer update, full inference,
full evaluation, threshold calibration, background task or Git operation ran.

21 unittest tests passed. See lightweight_tests.txt. Tests include dataset/schema,
frozen partition/hash checks, category/annotation distinction, supported canonical
count encoding, OCR uncertainty, three-way metrics, policy boundaries, positive-only
quality selection on unit-test records, paired candidate comparison and strict
E1 state-dict compatibility with all parameters frozen. Synthetic test records
are temporary unit-test fixtures, not experiment results or trained models.

One smoke run processed exactly these three selected fixtures:

| Image ID | F0 action | Offline transcription agreement |
|---|---|---|
| negative_count_zero_2_clean | ACCEPT | Incorrect: second upper tier decoded X rather than B |
| negative_blank_png_2_clean | REJECT_INVALID | Not applicable |
| negative_corrupt_png_2_clean | REJECT_INVALID | Not applicable |

This verifies integration, not performance. The accepted example has valid image
content despite its mutated JSON count. The OCR error is retained as an error.
Smoke output is in reports/implementation_smoke and excluded from experiment
comparisons/calibration. Its completion file records the precise software used
for that smoke run. Subsequently the adapter gained explicit structural-result
and expected-action fields, and asset ZIP reads were cached; no model or D1
algorithm changed. Final lightweight tests and CLI checks verify these additions.

Preparation: 776 development pages from 230 parents; 434 evaluation records from
99 parents, including one negative family and one missing-image record. No
cross-partition parent or byte-identical image overlap. Frozen test assignments
are excluded. These are exposed development/regression controls, not unseen tests.

The scoped lock verifies 282 historical input files, including original Stage1
and selected E1 checkpoints, integration code/config/manifests and source negative
artifacts. Selected E1 SHA-256 remains:
27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1.
This does not claim an exhaustive rehash of every historical large dataset.

Awaiting user execution: full F0 development/evaluation inference, F0/F1/F2 saved
output metrics, optional positive-only quality calibration and final comparative
report. Content-rejection calibration, independent negative efficacy evidence and
a learned classifier require new, independently labeled invalid-image parents.
