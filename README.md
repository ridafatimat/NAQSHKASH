# NAQSHKASH

NAQSHKASH is an OCR-based Final Year Project for recognizing Talim notation.

## Current preprocessing module

### Rotational Deskewing

The current preprocessing module corrects small global in-plane rotation in Talim page images.

Main file:

```text
preprocessing/deskew.py
```

The deskewing pipeline:
- works on general user-supplied Talim images
- is not dependent on synthetic metadata
- supports clean, scanned, photo-style and handwritten inputs
- uses projection-profile based skew estimation
- uses confidence-based correction
- leaves uncertain images unchanged rather than applying an unsafe rotation

Perspective / keystone correction is intentionally handled separately and is not part of the deskew module.

## Validation

Deskew validation is available in:

```text
tests/test_deskew.py
```

The validation script supports:
- controlled ground-truth rotation tests
- degradation tests
- category-based testing
- confidence reporting
- output review overlays
- CSV evaluation reports

## Installation

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Run validation

From the project root:

```bash
python tests/test_deskew.py
```

## Repository policy

Large datasets, generated outputs, reports, checkpoints and ZIP files are excluded from Git and should be stored separately.
