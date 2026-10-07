# NAQSHKASH

NAQSHKASH is an OCR-based Final Year Project for recognizing Kashmiri Carpet Talim notation.

## Preprocessing Package

The `preprocessing` package provides an end-to-end preprocessing pipeline for Talim documents:

```text
preprocessing/
├── __init__.py        # Unified package exports
├── deskew.py          # Rotational deskewing and skew angle estimation
├── crop.py            # Margin detection and document content bounding-box cropping
├── rows.py            # 3-tier logical row detection and separation
└── pipeline.py        # Integrated end-to-end preprocessing pipeline
```

### Preprocessing Capabilities:
1. **Rotational Deskewing (`preprocessing.deskew`)**: Corrects small global in-plane rotation using projection-profile analysis with confidence gating.
2. **Margin Cropping (`preprocessing.crop`)**: Removes outer blank borders and dark scanner margins while preserving text strokes.
3. **Logical Row Detection & Separation (`preprocessing.rows`)**: Extracts complete 3-tier Talim logical rows (Upper symbols, Center stitch counts, Lower symbols) as single units.
4. **End-to-End Pipeline (`preprocessing.pipeline`)**: Seamlessly coordinates deskewing, margin cropping, row detection, row extraction, and local-to-global coordinate mapping into structured `PreprocessingResult` dataclasses.

For full details on Ayesha's cropping & row detection design, see [README_AYESHA.md](file:///c:/Users/hp/Desktop/NAQSHKASH/README_AYESHA.md).

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running Tests and Validation

### 1. Pytest Test Suite
Run all unit and integration tests across cropping, row segmentation, and end-to-end pipeline:

```bash
pytest tests/ -v
```

### 2. Deskew Validation
Run the deskew evaluation script:

```bash
python tests/test_deskew.py
```

### 3. Preprocessing Benchmark & Visual Artifacts
Generate visual debug plots and run dataset benchmarking:

```bash
python run_ayesha_preprocessing.py
```

## Repository Policy

Large datasets, generated outputs, reports, checkpoints and ZIP files are excluded from Git and should be stored separately.
