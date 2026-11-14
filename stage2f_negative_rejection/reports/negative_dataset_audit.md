# Negative dataset audit

{
  "categories": 49,
  "source_records": 51,
  "clean_image_files": 50,
  "scanned_image_files": 147,
  "total_image_files": 197,
  "independent_source_parents": 1,
  "missing_image_records": 1,
  "clean_unique_file_hashes": 6,
  "clean_byte_identical_to_base": 45,
  "truth_counts": {
    "valid": 184,
    "empty_input": 1,
    "unadjudicated": 3,
    "corrupted_unreadable": 1,
    "missing_artifact": 1,
    "ambiguous_potentially_valid": 4,
    "unsupported_representation": 4
  },
  "clean_annotation_only_records": 46,
  "existing_frozen_test_pages_excluded": 0,
  "classifier_justified": false
}

Canonical sources: C:\FYP Dataset Generation\datasets\talim_edge_cases_negative_v1
and the corresponding clean/scanned ZIP files. Clean ZIP is an archival copy,
not extra independent samples. Scanned manifest source_id values identify 51
records, NOT independent parents. negative_expectations.json explicitly traces
all 49 mutations to embedded base A11/R19, S20/B10. The second duplicate-ID
fixture changes A11/R19 to A12/R18 and remains in this same derivation family.

50 clean files include one unreadable corrupt PNG; one image is absent.
147 scanned variants derive from 49 readable source records; artifact manifest
retains the missing/corrupt cases. 45 clean files have exactly the base hash.
All clean files were inspected for readability/mode/ink and hash equality;
this is an asset audit, not an OCR evaluation. Scanned hashes and manifest
source hashes were checked. No model inference or classifier training ran.

The inventory's image_ground_truth governs image metrics. The original negative
fixture label is retained verbatim in existing_annotation. JSON count/type/range,
directions, symbols, schema, row sums and canonical line errors are detectable
with the original check_sample validator. TXT mismatches/separators, duplicate
IDs/semantics, missing artifacts and rendering metadata require artifact-level
validation; they cannot be inferred from unchanged image pixels.

Corrupt bytes and blank clean PNG are artifact-level invalid inputs suitable for
input rejection (not learned Talim-content rejection). Wrong dimensions can be
resized valid content and is marked ambiguous. RGB mode is an unsupported source
representation, not invalid Talim; the inference adapter converts to grayscale.
No demonstrably invalid Talim-content images were established. No notation is
declared invalid merely because its JSON mutation mentions an unsupported symbol.
Some unsupported/ambiguous notation exists in annotations, not rendered pixels.
Scanned variants of blank inputs are unadjudicated, because noise is not Talim.

Classifier: NOT JUSTIFIED. One derivation family cannot supply disjoint train,
validation and evaluation negative parents. Metadata mutations on identical
positive pixels introduce contradictory image labels. No classifier modules,
training commands, fake checkpoints, or negative blank-CTC targets are created.

Full recognizer evaluation, confidence calibration and comparative results remain
unexecuted. New independently authored, image-adjudicated invalid parents are
required for content-rejection calibration and independent efficacy evidence.
