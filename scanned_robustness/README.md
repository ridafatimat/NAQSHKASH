# NAQSHKASH Stage 2B — scanned-style robustness

All code and outputs in this module are local to `C:\NAQSHKASH\stage2\scanned_robustness`. Stage 1, Stage 2A, checkpoint, vocabulary, original datasets and Git state were not modified. No training, fine-tuning, dataset regeneration, cloud compute or network download was performed.

## Measured outcome and scope

Read the [final report](reports/stage2b_final_report.md) before using the headline OCR metrics. **Recommendation B: improve preprocessing first**, especially blind row/tier grouping, geometric distortion handling and background/illumination normalization. The current model/pipeline is not ready to claim scanned-page robustness. Conditional recognition errors also remain, so a later fine-tuning experiment may be considered only after crop quality and development/test provenance are controlled.

| Cohort | Mode | Pages processed | Tier CER | Exact tier accuracy |
|---|---|---:|---:|---:|
| Frozen positive test | Blind | 41/147 | 0.926% | 98.780% |
| Frozen positive test | Assisted | 142/147 | 3.828% | 96.108% |
| General exploratory, 100 parents | Blind | 224/300 | 3.345% | 91.771% |
| General exploratory, 100 parents | Assisted | 244/300 | 3.342% | 91.497% |

The blind mode accepts a much smaller/different subset; its lower conditional CER is not evidence it is better overall. Only 14/49 clean positive parents pass this blind extractor even before corruption. Most blind positive failures are a pre-existing segmentation limitation, not a new scan-induced model failure. Assisted matched clean CER is 0% for positive tiers and 0.109% for the general exploratory subset; scanned assisted CER rises by 3.828 and 3.233 percentage points respectively.

Full audit: **30,000 general variants / 10,000 parents**, **978 positive variants / 326 parents**, three variants each. Every image hash, decoded dimensions and external source-label link was verified. Actual recipes, metadata and full provenance are in `reports/dataset_audit.json`.

**Inference scope:** 450 scanned images in two modes: all 147 frozen-positive-test variants, a deterministic 300-variant/100-parent general exploratory sample, and three additional variants of one parent whose logical structure matches a Stage 1 test page but whose image identity is unconfirmed. This last one-parent case is reported separately (two of three pages processed in each mode, six exact tier predictions). The 30,978 images were all audited; they were not all inferred.

No general source parent matched a Stage 1 source image byte-for-byte after scanning all 100,000 Stage 1 image hashes and complete logical structures. The one content-only match does not establish parent identity. **None of the general result cohorts is called an independent Stage 1 test.** Positive variants inherit the frozen Stage 2A assignments; 228 training and 49 validation source parents are excluded. No image-level random split was created. All 49 positive test parents stay in test. This robustness follow-up has now exposed those test variants.

## Exact input paths

Already available and actually used:

```text
C:\FYP Dataset Generation\datasets\talim_10000_scanned.zip
C:\FYP Dataset Generation\datasets\talim_edge_cases_positive_scanned_v1.zip
C:\FYP Dataset Generation\datasets\talim_10000.zip
C:\FYP Dataset Generation\datasets\talim_edge_cases_positive_v1.zip
C:\FYP Dataset Generation\Final dataset\talim_100000.zip
```

The scanned ZIPs reference labels/JSON in the clean source ZIPs; they do not embed the labels. Each also includes a QA montage outside the images folder, excluded from sample counts. A source inventory and official clean-archive hash establish identity with the local clean source files. The original requested general filename `talim_scanned_augmented_v1.zip` was corrected to `talim_10000_scanned.zip`.

Newly downloaded copies were not yet present under `C:\FYP Dataset Generation\Final dataset` at the last check. This did not block evaluation because the exact named archives were already present and fully verified in the datasets directory. `check_downloads.py` compares any later copies against the audited hashes without changing either location. The report records which actual files were used; it never claims a missing download was verified.

Stage 1 checkpoint is read in place from `C:\NAQSHKASH\stage2\positive_edge_cases\checkpoints\stage1\best.pt`, hash `22450cbddd42968149739ea0293a6f4ad105f593e82387d707ecb6418a6bac0f`. Vocabulary is read from `C:\NAQSHKASH\config\vocabulary.json`. No extra checkpoint copy was made.

## Modes and interpretation

Both modes reuse the root Stage 1 deskew/clean/crop/row pipeline, height-64 aspect-preserving resize, grayscale /255 normalization, padding, vocabulary and greedy CTC decoder. Deskew is enabled. No model or shared preprocessing module was edited.

- **Blind:** row detection and tier proposals use pixels only. One/two/three observed bands are interpreted by fixed Talim layout and height/ink-mass heuristics; ambiguous or overlapping bands are rejected. Labels, expected rows/tiers, source IDs, augmentation recipes and predictions never choose or retry segmentation. Ground truth checks structural correspondence only after proposals are final. Mismatches are rejected, not repaired.
- **Annotation-assisted:** known row count may use the existing Stage 2A whitespace fallback, and annotated nonempty tier presence controls original tier-band extraction. Raw labels and offsets are preserved.

A whole page is withheld from OCR metrics if any segmentation, ordered structural alignment or CTC check fails. Failures have their own page coverage records; they are not counted as OCR deletion errors. Structurally matching crops are not a guarantee of perfect pixel-level geometric correspondence because these datasets do not supply transformed tier polygons. Representative images were visually reviewed, and residual crop/warp effects may contribute to conditional recognition differences.

## Reproduce locally in PowerShell

The existing Python 3.12 environment was used; plain python on this machine previously selected Python 3.8. Recorded versions: Python 3.12.10, torch 2.2.2+cpu, NumPy 1.26.4, SciPy 1.17.1, OpenCV 4.13.0. See `reports/environment.json`. These are the measured environment, not an automatically installed dependency set. Tests use standard-library unittest; no pytest installation is needed.

```powershell
Set-Location 'C:\NAQSHKASH'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
```

Check optional new download copies (reports absent/different/identical explicitly):

```powershell
& $Python312 -B .\stage2\scanned_robustness\check_downloads.py
```

Run tests against the saved complete benchmark:

```powershell
& $Python312 -B -m unittest discover -s .\stage2\scanned_robustness\tests -p test_stage2b.py -v
```

Repeat inference against the already audited, frozen manifest, into a NEW output directory:

```powershell
& $Python312 -B .\stage2\scanned_robustness\run_evaluation.py --device cpu --output .\stage2\scanned_robustness\reports\evaluation_reproduction
```

This evaluates the same 450 images and modes; measured initial runtime was approximately 6.3 minutes on the local CPU. Runtime varies with hardware and workload. The command refuses an existing output directory. Model state is checked unchanged, and successful examples are re-inferred without the exact-pixel cache to check deterministic predictions.

`audit.py` is the full read-only-source audit used before inference. It intentionally refuses to overwrite `reports/dataset_audit.json`; do not delete a frozen audit or regenerate its manifest to optimise results. `freeze_protocol.py` records the pre-inference conservative source-identity classification used for this run; it is a protocol setup utility, not an inference or optimisation step. The saved manifests and source hashes are sufficient for the reproduction command above.

Postprocessing existing predictions does not run inference:

```powershell
& $Python312 -B .\stage2\scanned_robustness\analyse_errors.py
& $Python312 -B .\stage2\scanned_robustness\review_examples.py
& $Python312 -B .\stage2\scanned_robustness\make_report.py --recommendation B
```

## Files and evidence

- `common.py`: actual project paths, metrics and hashing helpers; reuses existing Stage 1/2A modules read-only.
- `audit.py`, `freeze_protocol.py`: full integrity/provenance audit and conservative pre-inference cohort protocol.
- `modes.py`: separate image-only blind proposals, annotation-assisted extraction and post-proposal structural/CTC checks.
- `run_evaluation.py`: frozen-model inference, matched clean parents, per-tier predictions, full failure accounting, examples and deterministic replays.
- `analyse_errors.py`, `review_examples.py`: saved-error edit attribution and a reviewed matched clean/degraded crop.
- `make_report.py`, `check_downloads.py`, `tests/test_stage2b.py`: reporting, optional copy verification and 12 passing automated tests.
- `manifests/parents.csv`, `all_variants.csv`, `evaluation_pages.csv`: full parent/variant audit, exclusions and frozen inference selection.
- `reports/evaluation_v1/metrics.json`, `metrics.csv`: all cohort/mode metrics and per-tier/category breakdowns. `category_metrics.csv` and `tier_metrics.csv` provide flattened tables.
- `predictions.csv`, `recognition_errors.csv`, `page_processing.csv`, `segmentation_failures.csv`: every evaluated tier/page and each failure, including source parent, mode and recipe.
- `clean_predictions.csv`, `clean_page_processing.csv`, `clean_scan_coverage.csv`: matched clean controls and pre-existing versus new failure transitions.
- `failure_examples.csv`, `examples/index.md`, image files, `edit_attribution.json`, `qualitative_review.json`: complete error listings and representative visual evidence.
- `reports/tests.json`, `tests.txt`, `protected_before.json`, `final_verification.json`: 12 passing tests, source protection hashes and final integrity assertions. All 6,451 protected project files were unchanged.

The last two groups of prediction/example filenames are inside `reports/evaluation_v1/`. No dataset regeneration or fine-tuning command exists in this module. Synthetic scanned-style results do not establish real handwriting or arbitrary real-camera performance.
