# Tier-context audit

The original Stage1 training path uses `data/manifest_dataset.py` and
`training/manifest_training.py`. The approved `stratified_train_150k.csv` has
image_path, training_label, tier_name, parent_image_id, row_index and tier_index.
Its first row is a count-tier crop, not a page. The dataset returns tier_name;
`data/collate.py` retains tier_names while resizing grayscale pixels/255 to
height64 with aspect ratio preserved and white right padding. Internal spaces
remain target characters; leading source offsets are carried separately.
Training calls `logits = model(images)` (manifest_training.py:370), not a
role-conditioned model. The metadata has not disappeared from the batch: it
simply is not part of the CRNN input.

`models/crnn.py` uses a CNN, two-layer bidirectional LSTM (hidden256) and a
linear classifier for66 outputs:65 fixed vocabulary characters plus blank0.
`data/ctc_decode.py` collapses repeats/blanks and maps IDs to characters without
a tier-role argument. Original Stage1 training was tier-level; this experiment
does not assume an earlier historical page-level model was used for E1.

Stage2E `src/datasets.py` returns FrozenTiers with tier_name=r['role']; its
`src/evaluation.py` calls model(batch['images']) and the original greedy decoder.
E1's architecture and vocabulary are unchanged. E1 positive-only fine-tuning
uses D1 crops, not a role input. The selected checkpoint remains frozen.

D1 `engine.py` produces rows of {row,role,box}; diagnostics retain path,
legal_alternatives, role_margin, rejection and morphology features. The accepted
path uses the existing frozen D1 margin0.2 when alternatives exist; a sole legal
path has no margin. The new adapter reuses that criterion, not a new threshold.
Acceptance is still a heuristic, not independent proof of correct segmentation.

Stage2F `src/evaluate_baseline.py` loops through D1 rows/roles and saves each
crop's row, role, box, prediction and signals. It calls model(tensor) and greedy
CTC without role information. `decoded_rows` retain role keys, and `tiers` and
`segmentation_rows` retain identifiers/geometry. Role context is therefore
available immediately after decoding, even though the pixel model did not use it.

Saved development completion metadata, four curated development observations
and source code were inspected; no full candidate evaluation ran. A single
Stage1 parent JSON and a positive-source parent JSON both specify Talamat.ttf,
IMTCRPET, source48 and grayscale L. This is source rendering size, distinct
from CRNN's resized height64. For manual T1 evaluation every candidate source's
rendering block is checked against the profile before using its ambiguity map;
missing/unknown source metadata blocks correction. Only rendering fields are
extracted; annotations and ground-truth symbols do not enter runtime decisions.

The existing validator still reports unsupported/wrong-tier glyphs; F1 v2 maps
OCR-only violations to review with provenance. T1 wraps that API without changing
its source/config. Raw decisions/evidence are stored separately even when an
interpreted sequence passes syntax. A remaining unresolved context can add a
separately identified T1 review guard, never override independent artifact rejection.

This is an isolated post-CTC interpretation prototype, not retraining, logit
masking, beam decoding, confidence calibration or a vocabulary modification.
