# NAQSHKASH Preprocessing — Ayesha Amer (Day 1 & Day 2)

**Author:** Ayesha Amer  
**Team:** NAQSHKASH FYP  
**Phase:** Preprocessing (Days 1 & 2 of 14-Day Plan)

---

## 1. Overview & Scope

This package implements the foundational preprocessing stages for Kashmiri Carpet Talim document recognition:
- **Day 1 Scope:** Content margin detection and bounding-box cropping (`preprocessing.crop`).
- **Day 2 Scope:** 3-tier logical row detection and row separation (`preprocessing.rows`).
- **Pipeline Integration:** Chained end-to-end wrapper returning structured dataclasses (`preprocessing.pipeline`).

### Strict Scope Boundaries
- **In Scope (Days 1–2):** Standalone margin cropping, 3-tier logical row detection, unresized/un-normalized row slicing, bounding-box metadata mapping, test suite, and benchmark runner.
- **Teammate Scopes (Untouched):** Deskewing / Rotation correction (Rida), Noise reduction & Ruling-line removal (Hareem), DataLoader & Model training (Rida / Hareem).
- **Day 3 Hand-Off Scope (Next Step):** Resizing row crops to fixed 64px height, grayscale conversion, tensor normalization, and dataset integration.

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

## 3. Coordinate Convention

All bounding boxes throughout this codebase strictly follow the **`(ymin, xmin, ymax, xmax)`** coordinate convention:
- `ymin`: Top row pixel index (inclusive)
- `xmin`: Left column pixel index (inclusive)
- `ymax`: Bottom row pixel index (exclusive)
- `xmax`: Right column pixel index (exclusive)

**Python Slice Equivalent:**  
```python
row_crop = image[ymin:ymax, xmin:xmax]
```

---

## 4. Package Architecture & Public APIs

```text
preprocessing/
├── __init__.py        # Exports public API functions and dataclasses
├── crop.py            # Day 1: Margin removal and content bounding-box extraction
├── rows.py            # Day 2: 3-tier logical row detection and extraction
└── pipeline.py        # Chained pipeline execution and coordinate mapping
```

### Public API Functions

#### `crop_margins(image, padding=10, return_bbox=False, noise_filter=True)`
- **Input:** 2D Grayscale or 3D BGR/RGB numpy array (uint8).
- **Immutability Guarantee:** The input array is never mutated in-place.
- **Robustness:** Handles dark scan borders, uneven illumination (Otsu thresholding), and filters isolated 1–2px speckle noise.
- **Fallback:** Returns a clean full-image copy with `(0, 0, H, W)` on blank/uniform images.

#### `detect_rows(image, padding=5, min_line_gap_merge=3, subline_height_ratio=0.8)`
- **Input:** Cropped or full document image (2D/3D uint8).
- **Algorithm:** Combines horizontal projection profiling, subline fragment merging, and anchor count tier clustering.
- **Output:** List of row bounding boxes `[(ymin, xmin, ymax, xmax), ...]` ordered strictly from top to bottom.

#### `separate_rows(image, row_bboxes=None, padding=5)`
- **Input:** Image array and optional pre-computed bounding boxes.
- **Output:** `List[RowData]` containing:
  - `row_index`: 0-indexed position from top.
  - `image`: Cropped row array (**unresized, un-normalized, original dtype & channels preserved**).
  - `bbox`: `(ymin, xmin, ymax, xmax)` local coordinates.
  - `height`, `width`: Dimensions of the crop.

#### `crop_and_separate_rows(image, crop_padding=10, row_padding=5, noise_filter=True)`
- **Chained Pipeline:** Executes `deskew` (optional/integrated) $\rightarrow$ `crop_margins` $\rightarrow$ `detect_rows` $\rightarrow$ `separate_rows`.
- **Output:** `PreprocessingResult` dataclass holding:
  - `deskewed_image`: Image array after rotational deskewing.
  - `deskew_angle`: Detected skew angle in degrees.
  - `cropped_image`: Image with outer margins removed.
  - `crop_bbox`: Crop bounds in original/deskewed image coordinates.
  - `rows`: List of `RowData` items.
  - `num_rows`: Count of detected rows.
  - `global_row_bboxes`: Bounding boxes mapped back to original uncropped image space.

---

## 5. Usage Example

```python
import numpy as np
from PIL import Image
from preprocessing import crop_and_separate_rows

# Load input document (Grayscale or Color)
raw_image = np.array(Image.open("datasets/baseline_normal/images/talim_000001.png"))

# Execute Preprocessing Pipeline
result = crop_and_separate_rows(raw_image, crop_padding=10, row_padding=5)

print(f"Original Shape: {result.original_shape}")
print(f"Cropped Shape:  {result.cropped_image.shape}")
print(f"Detected Rows:  {result.num_rows}")

for row in result.rows:
    print(f"  Row {row.row_index}: bbox={row.bbox}, shape={row.image.shape}")
```

---

## 6. Integration Contract for Day 3 (Ayesha) & Teammates (Rida, Hareem)

### Incoming Hand-off from Rida & Hareem (Day 3):
- Rida's deskewed images and Hareem's noise-cleaned / ruling-line-removed images can be passed directly into `crop_and_separate_rows(cleaned_image)` without any modification.

### Outgoing Hand-off to Day 3 (Ayesha):
```python
# Day 3 transformation loop:
for row in result.rows:
    # 1. Convert row.image to Grayscale (if 3D)
    # 2. Resize row.image to fixed height 64px while maintaining aspect ratio
    # 3. Normalize pixel values to [0, 1] or [-1, 1] for CRNN feature extractor
    pass
```

---

## 7. Verification & Benchmarking

### Running the Pytest Suite
```bash
pytest tests -v
```
*Tests verify:* Blank images, edge-touching text, dark borders, speckle noise, ruling lines, tight row spacing, single-symbol rows, tilt, gray vs BGR, input immutability, and top-to-bottom ordering.

### Running the Dataset Benchmark & Visualization Tool
```bash
python run_ayesha_preprocessing.py
```
*Generates debug visualizations in `debug_outputs/`:*
1. `*_crop_overlay.png`: Document crop boundary.
2. `*_row_overlay.png`: Color-coded row bounding boxes.
3. `*_projection_plot.png`: Horizontal projection profile with row division cut lines.

> **Note on `debug_outputs/`:** The `debug_outputs/` directory is intended for visual inspection and should be left unstaged in git.

---

## 8. Dependencies
The implementation uses existing environment dependencies:
- `numpy >= 1.24`
- `scipy >= 1.10` (specifically `scipy.ndimage`)
- `pillow >= 10.0`
- `matplotlib >= 3.7` (for visualization generation)
- `pytest >= 7.0` (for test suite)
