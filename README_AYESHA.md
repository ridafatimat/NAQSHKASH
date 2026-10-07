# NAQSHKASH Preprocessing — Ayesha Amer

**Author:** Ayesha Amer  
**Team:** NAQSHKASH FYP  
**Phase:** Preprocessing (Days 1–3 of 14-Day Plan)

---

## 1. Overview & Scope

This package implements the core preprocessing pipeline for Kashmiri Carpet Talim document recognition:
- **Margin Cropping:** Content margin detection and bounding-box cropping (`preprocessing.crop`).
- **Row Detection & Separation:** 3-tier logical row detection and row separation (`preprocessing.rows`).
- **Row Normalization:** Grayscale conversion, 64px aspect-preserving resizing (`cv2.INTER_AREA`), float32 normalization in `[0.0, 1.0]` (`preprocessing.normalize`).
- **Model Preparation Pipeline:** End-to-end chained execution returning structured model-ready row dataclasses (`preprocessing.model_prep`).

### Strict Scope Boundaries
- **In Scope:** Document cropping, logical row grouping, row slicing, aspect-preserving 64px resizing, `[0.0, 1.0]` normalization, test suite, and verification tools.
- **Teammate Scopes:** 
  - **Rida (Days 1–3, 4–5):** Rotational deskewing (`preprocessing.deskew`), DataLoader, batching, variable-width padding (Day 5), and CTC preparation.
  - **Hareem (Days 1–3, 4–6):** Noise reduction, ruling-line removal hook (`clean_fn`), and CRNN model architecture.

---

## 2. Kashmiri Carpet Talim Row Definition

In Talim notation, **one logical row is defined as a 3-tier block**:
1. **Upper Tier (optional):** UP-direction color symbols (e.g. `CN`, `A`, `Q`).
2. **Center Tier (mandatory):** Count glyphs indicating stitch quantities (e.g. `jjj`, `eijf`, `l`).
3. **Lower Tier (optional):** DOWN-direction color symbols (e.g. `R`, `_`, `]`, `T`).

```text
+-------------------------------------------------------------+
| Upper Tier:  [ C N ]              [ Q E ]       (Symbols)   |
| Center Tier: [ j j j ]            [ j e i f ]   (Counts)    | ===> 1 Logical Row
| Lower Tier:  [       _ ]          [     R _ ]   (Symbols)   |
+-------------------------------------------------------------+
  <---------------- Inter-row Whitespace Gap ---------------->
+-------------------------------------------------------------+
| Upper Tier:  [   N ]                                        |
| Center Tier: [ j j j ]                                      | ===> Next Logical Row
| Lower Tier:  [ ]   U ]                                      |
+-------------------------------------------------------------+
```

**Rule Conformance:**  
- `detect_rows` and `separate_rows` output the **entire 3-tier block as ONE row image**.
- Sub-lines are never split into separate outputs, and adjacent logical rows are never merged.

---

## 3. Coordinate Convention & Polarity Standards

### Coordinate Convention
All bounding boxes throughout this codebase strictly follow **`(ymin, xmin, ymax, xmax)`**:
- `ymin`: Top row pixel index (inclusive)
- `xmin`: Left column pixel index (inclusive)
- `ymax`: Bottom row pixel index (exclusive)
- `xmax`: Right column pixel index (exclusive)

```python
row_crop = image[ymin:ymax, xmin:xmax]
```

### Polarity & Normalization Convention
- **`0.0`** = Dark text ink (Black)
- **`1.0`** = Light paper background (White)
- **Data Type:** `float32` in `[0.0, 1.0]`

> [!IMPORTANT]
> **Contract for Rida's DataLoader (Day 5):** When padding variable-width row tensors to batch maximum width, **pad with `1.0`** (white paper background), never `0.0` (which would represent black ink).  
> **Contract for Hareem's Cleaning Module:** Any row-level image cleaning / ruling-line removal stage must preserve this polarity convention (`0.0` = ink, `1.0` = background).

---

## 4. Package Architecture & Public APIs

```text
preprocessing/
├── __init__.py        # Public API exports (additive and backward-compatible)
├── deskew.py          # Rida: Rotational deskewing and skew estimation
├── crop.py            # Margin removal and content bounding-box extraction
├── rows.py            # 3-tier logical row detection and extraction
├── pipeline.py        # Shared: Base crop-and-separate pipeline (deskew integrated)
├── normalize.py       # Grayscale conversion, 64px resizing, and [0, 1] normalization
└── model_prep.py      # End-to-end model-ready row preprocessing pipeline
```

### Public API Functions

#### `to_grayscale(image)`
- **Input:** 2D grayscale, 3D BGR/RGB, or float array.
- **Output:** 2D `uint8` array of shape `(H, W)` in `[0, 255]`. Never mutates input.

#### `resize_to_height(image, height=64, min_width=16)`
- **Input:** 2D grayscale array.
- **Output:** Resized 2D array of shape `(64, W)` where $W \ge \text{min\_width}$.
- **Interpolation:** Uses `cv2.INTER_AREA` for downscaling (ensuring thin strokes, dots, and diacritics survive) and `cv2.INTER_LINEAR` for upscaling.
- **No Padding:** Strictly preserves variable width without horizontal padding.

#### `normalize(image)`
- **Input:** Image array.
- **Output:** 2D `float32` array in `[0.0, 1.0]` with zero NaNs/Infs.

#### `preprocess_row(row_image, target_height=64, min_width=16)`
- **Chained Row Transform:** `to_grayscale` $\rightarrow$ `resize_to_height(64)` $\rightarrow$ `normalize`.
- **Output:** `(64, W)` float32 array in `[0.0, 1.0]`.

#### `preprocess_image(image, clean_fn=None, target_height=64, min_width=16, apply_deskew=True, ...)`
- **Full End-to-End Pipeline:** Calls `crop_and_separate_rows` $\rightarrow$ optional `clean_fn` per row $\rightarrow$ `preprocess_row`.
- **Output:** `ModelPrepResult` holding `List[ModelReadyRow]`.

---

## 5. Usage Example

```python
import numpy as np
from PIL import Image
from preprocessing import preprocess_image

# Load raw Talim page image
raw_image = np.array(Image.open("datasets/baseline_normal/images/talim_000001.png"))

# Run End-to-End Model Preparation Pipeline
result = preprocess_image(raw_image, target_height=64, apply_deskew=True)

print(f"Original Shape:   {result.original_shape}")
print(f"Cropped Shape:    {result.cropped_image.shape}")
print(f"Deskew Angle:     {result.deskew_angle:.2f}° (Corrected: {result.deskew_corrected})")
print(f"Total Rows:       {result.num_rows}")

for row in result.rows:
    print(f"  Row {row.row_index}: tensor_shape={row.tensor_image.shape}, dtype={row.tensor_image.dtype}, range=[{row.tensor_image.min():.2f}, {row.tensor_image.max():.2f}]")
```

---

## 6. Output Contract for Rida's DataLoader (Days 4–5)

Each `ModelReadyRow` object in `result.rows` provides:
1. `row.tensor_image`: `np.ndarray` of shape `(64, W)`, dtype `float32`, values $\in [0.0, 1.0]$.
2. `row.height`: Fixed at `64`.
3. `row.width`: Variable width $W \ge 16$.
4. `row.bbox`: Local bounding box in cropped space `(ymin, xmin, ymax, xmax)`.
5. `row.global_bbox`: Global bounding box in deskewed/input space `(ymin, xmin, ymax, xmax)`.

```python
# Rida's Day 4-5 DataLoader Consumption:
# Convert row.tensor_image (shape (64, W)) directly to torch.FloatTensor (1, 64, W)
import torch

tensor_input = torch.from_numpy(row.tensor_image).unsqueeze(0)  # Shape: (1, 64, W)
# Pad width with 1.0 (white background) during batch collation
```

---

## 7. Verification & Tests

### Running the Pytest Suite
```bash
pytest tests/ -v
```
All 36 unit and integration tests pass across cropping, row segmentation, deskew integration, resizing, grayscale conversion, and normalization.

### Running the Model Preparation Verification Script
```bash
python verify_model_prep.py
```
*Verifies:*
1. Dot and thin-stroke retention under 64px area downscaling (100% dot retention).
2. Dtype (`float32`), shape (`(64, W)`), value range (`[0.0, 1.0]`), zero NaNs/Infs across datasets.
3. Row width distribution and CTC receptive field requirements.
4. Saves visual inspection artifacts to `debug_outputs/model_prep/`.

---

## 8. Dependencies
The implementation uses existing environment dependencies (from `requirements.txt`):
- `numpy`
- `opencv-python` (`cv2`)
- `scipy` (`scipy.ndimage`)
- `torch`
- `pytest`
