# Stage 2C — preprocessing and image-only segmentation experiments

Active module: `C:\NAQSHKASH\stage2\preprocessing_improvements`.

This module is isolated research code. It does not replace Stage 1 preprocessing or change the original model. All Stage 1/2A/2B files, source images/labels, checkpoint, vocabulary and decoder remain protected. No model training, dataset regeneration, Git operations or cloud execution occurred.

## Final measured status

All six candidates were evaluated on identical frozen cohorts. **69 tests passed, zero failures/errors/skips.** C4 was selected from tuning before validation, but is **not recommended for deployment** because wrong acceptances remain high.

| Candidate | Correct positive validation pages /138 | Correct general validation pages /150 | Wrong acceptances across 288 scanned validation pages |
|---|---:|---:|---:|
| C0 | 42 | 111 | 13 |
| C1 | 42 | 111 | 13 |
| C2 | 94 | 133 | 58 |
| C3 | 42 | 112 | 13 |
| C4 | 94 | 135 | 56 |
| C5 | 94 | 134 | 26 |

General common-subset CER falls from 6.411% (C0) to 1.835% (C4), or 1.812% with normalization-only C1. Positive common-subset CER is unchanged at 3.125%. C4's expanded positive conditional CER is **27.532%**, dominated by one newly covered long parent (710/723 errors); those errors are retained, not excluded. Its clean counterpart already loses 112 internal spaces. See the report and `long_sequence_diagnostic.json` for the distinction between new coverage and OCR regression.

Normal clean regression improves correct segmentation from 91/100 (C0) to 96/100 (C4), but wrong acceptances increase from zero to four. Common-subset clean CER is unchanged at 0.191%. In the separate 90-page diagnostic, no component losses were found among 2,562 evaluated C0 components and 6,254 C4 components; this is conditional on matching structure and is not a guarantee for rejected/misgrouped pages.

All **6,565 protected historical files** retained their hashes. No pipeline was promoted, no original file was edited, and no model was trained. The next step is stronger ambiguity/role rejection and focused long-sequence analysis on development data, not automatic CRNN fine-tuning.

## Read these results first

The measured comparison and deployment recommendation are in [reports/stage2c_final_report.md](reports/stage2c_final_report.md). Candidate selection is recorded in `reports/experiments_v1/selection_before_validation.json`; it is fixed before validation and is not a claim that the selected candidate is safe for deployment.

Always read correct segmentation coverage together with wrong acceptances. High proposal coverage or low OCR CER on an easier accepted subset is insufficient. All candidates remain experimental until their safety and clean-regression results justify promotion.

## Frozen development cohorts

| Source pool | Tuning parents | Internal-validation parents | Clean-regression parents |
|---|---:|---:|---:|
| Frozen positive training pool | 182 | 46 | 0 |
| Unexposed general supplementary development | 100 | 50 | 0 |
| Confirmed original Stage 1 training | 0 | 0 | 100 |

478 parents, 1,612 existing pages, six candidates, 9,672 page-candidate outcomes. Each positive/general source contributes its clean original and all three existing augmentations. All variants stay with their source parent. Exposed Stage 2B parents and all original positive validation/test parents are excluded. The general cohort is development/exploratory only, never an independent test. The clean regression uses original Stage 1 training parents; it is not a new held-out clean test.

`protocol/design.json`, `protocol/lock.json`, `manifests/parents.csv` and `manifests/pages.csv` define the frozen inputs. Archives remain in their original local directories and are read in place. SHA256 locks cover the archives, model, vocabulary, original decoder, old split manifests and development protocol. `protocol/software_lock.json` preserves the ten files used by the inference implementation. Historical project files are fingerprinted in `protocol/protected_before.json`.

## Candidate definitions

- **C0:** exact Stage 2B blind proposal baseline with original Stage 1 preprocessing.
- **C1:** background/illumination normalization, original grouping.
- **C2:** connected-component tier grouping, joint image-only row/role grammar and adaptive whitespace-limited padding.
- **C3:** existing Stage 1 rotational deskew plus guarded image-only baseline-convergence perspective correction, original grouping.
- **C4:** combined normalization, grouping and geometry.
- **C5:** C4 plus gated edge-preserving noise handling and stricter ambiguity rejection.

Parameters are fixed in the frozen implementation. The first pilot touched only tuning parents. The full run selects from scanned tuning coverage, then wrong acceptances, then common-subset CER. Validation and normal clean regression are evaluated afterward without changing the saved selection.

## Deployment/evaluation separation

The deployable API is:

```python
process_image(image_pixels, frozen_configuration)
```

It accepts no row counts, tier annotations, labels, augmentation recipes, IDs or OCR predictions. It returns proposed rows/tier roles, boxes, transformations, image-derived diagnostics, and acceptance/rejection. No expected text or CRNN hypothesis chooses a segmentation. Existing source-level functions are reused read-only.

`evaluation/reference.py` is offline-only. It derives ideal tier-ink annotations from the documented source renderer slots and transfers only those masks using recorded augmentation geometry. Source slots were validated against every selected clean source ink pixel and nonempty tier label. Original TXT/JSON/canonical Talim correspondence was independently checked for all 478 selected parents. No source images were rerendered or regenerated.

Reference-correct segmentation requires the right row/role structure, >=98% intended tier-ink retention and <=2% neighbouring-tier contamination for every crop. Forward-mapped reference points prevent clipping from disappearing from the denominator. These are ideal-ink geometric checks, not manually verified polygons for every blurred/JPEG image. A separate pre-registered 90-page diagnostic tests individual components >=3 pixels for >=95% retention. It is diagnostic only and does not reselect candidates.

A "safe rejection" means the algorithm abstains instead of supplying an unverified crop; it does not mean the valid page is inherently unreadable. Conditional OCR uses only completely reference-correct, CTC-feasible pages. Wrong acceptances remain visible separately. Common-subset OCR uses the same pages/tier labels across all six candidates. Clean-image regression and parent-bootstrap coverage changes are also reported.

## Local commands

Use the existing measured Python 3.12 installation; plain python previously selected Python 3.8. Dependencies already present were sufficient. No package installation was performed. Environment versions are recorded in `reports/environment.json`.

```powershell
Set-Location 'C:\NAQSHKASH'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
```

Read-only protocol verification:

```powershell
& $Python312 -B .\stage2\preprocessing_improvements\protocol\verify.py
```

Reproduce the full evaluation into a NEW directory (can be CPU-intensive on the widest pages):

```powershell
& $Python312 -B .\stage2\preprocessing_improvements\evaluation\launch.py --output .\stage2\preprocessing_improvements\reports\experiments_reproduction
```

The launcher checks locked inputs, inference code and historical files before inference. The evaluator refuses an existing output directory. It uses the same frozen cohorts and writes selection before validation. No optimizer exists in this workflow. `freeze.py` intentionally refuses to overwrite the existing protocol; do not delete or regenerate a frozen split to improve scores.

Run all Stage 2C unittest tests against the completed v1 artifacts:

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
& $Python312 -B -m unittest discover -s .\stage2\preprocessing_improvements\tests -p 'test_*.py' -v
```

The saved comprehensive suite also includes relevant unchanged Stage 1 preprocessing, model-forward and CTC tests. Its exact command and evidence are in `reports/test_command.ps1`, `tests.txt`, `tests.xml` and `tests.json`. Test temporary files stay under Stage 2C; no training test was run.

Rebuild summaries and visual/report artifacts from the saved completed v1 experiment (no new model inference):

```powershell
& $Python312 -B .\stage2\preprocessing_improvements\evaluation\summarize.py
& $Python312 -B .\stage2\preprocessing_improvements\evaluation\visualize_selected.py
& $Python312 -B .\stage2\preprocessing_improvements\evaluation\report.py
```

`component_audit.py` replays preprocessing, but not OCR, for the pre-registered 90-page preservation diagnostic. `audit_annotations.py` checks the selected original TXT/JSON/rules correspondence. Neither modifies original data or candidate parameters.

## Source files

- `core.py`: local paths, hashing, read-only baseline integration and source/software lock checks.
- `pipeline.py`: image+configuration API and within-page processing cache.
- `preprocessing/photometric.py`: illumination normalization and conservative denoising.
- `preprocessing/geometry.py`: existing deskew plus guarded projective baseline correction.
- `segmentation/engine.py`: component-aware bands, joint row/role grammar, confidence margin and safe crop padding.
- `protocol/freeze.py`, `verify.py`, `audit_annotations.py`: cohort definition, fail-closed history verification and source annotation correspondence.
- `evaluation/reference.py`: offline reference geometry and crop correctness.
- `evaluation/pilot.py`, `run.py`, `launch.py`: tuning-only pilot and frozen full experiments.
- `evaluation/summarize.py`, `component_audit.py`, `visualize_selected.py`, `report.py`: comparisons, component audit, visual evidence and final reporting.
- `tests/test_contracts.py`, `tests/test_safety_and_experiments.py`: contracts, safety, provenance, deterministic inference and outcome accounting.

## Main artifacts

`reports/experiments_v1/pages.csv` records every page-candidate outcome, including proposals, correct segmentation, wrong structure/geometry, CTC feasibility, OCR totals, timing and evaluation order. `predictions.csv` retains every scored tier's expected/predicted text, offsets, geometric checks and edit counts. Source labels and internal spaces are never rewritten to improve scores.

`summary.csv`/JSON contain per-candidate cohort metrics and common-subset OCR; `augmentation_metrics.csv` retains actual recipe breakdowns; `parent_bootstrap.json` reports source-parent uncertainty. `selection_before_validation.json` and `completion.json` document selection order, model immutability and uncached deterministic replays.

`examples/selected_candidate/index.md` contains selected-candidate before/after panels and full-resolution/left-right details. Additional C0/C2/C5 panels are in the experiment examples directory. Diagnostic component losses and per-page counts are preserved separately.

`reports/final_verification.json` and `reports/stage2c_final_report.md` state the measured outcome and remaining limits. No independent-test, handwriting or arbitrary real-camera generalisation is claimed. The observed validation results must not become a new tuning set while still being called untouched validation.
