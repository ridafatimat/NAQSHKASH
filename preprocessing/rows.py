"""
preprocessing.rows
==================
Module for logical row detection and row separation in Kashmiri Carpet Talim OCR.

Author: Ayesha Amer (Day 2 Scope)
Project: NAQSHKASH FYP

Logical Row Definition in Talim:
---------------------------------
In Kashmiri Carpet Talim notation, one logical row represents a 3-tier block:
  1. Upper tier (optional): UP-direction symbols (e.g. 'CN', 'A')
  2. Center tier (mandatory): Count glyphs (e.g. 'jjj', 'eijf')
  3. Lower tier (optional): DOWN-direction symbols (e.g. 'R', '_', ']')
Followed by inter-row whitespace separation before the next logical row.

This module detects and separates each logical row as ONE complete unit (preserving
the upper symbol, count glyphs, and lower symbol together), without splitting sub-lines
apart or merging distinct logical rows.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy import ndimage

from .crop import _binarize_ink


@dataclass
class RowData:
    """
    Data structure representing an extracted logical row.
    
    Attributes
    ----------
    row_index : int
        0-indexed position from top to bottom (0 is the topmost row).
    image : np.ndarray
        Cropped row image array (original resolution, channels, and dtype preserved;
        no resizing or normalization is applied in Day 2 scope).
    bbox : Tuple[int, int, int, int]
        Bounding box of this row in `(ymin, xmin, ymax, xmax)` format relative to the input image.
    height : int
        Height of the row crop in pixels (`ymax - ymin`).
    width : int
        Width of the row crop in pixels (`xmax - xmin`).
    metadata : Dict[str, Any]
        Additional extraction metadata (e.g. sub-line count, anchor position).
    """
    row_index: int
    image: np.ndarray
    bbox: Tuple[int, int, int, int]
    height: int
    width: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def detect_rows(
    image: np.ndarray,
    padding: int = 5,
    min_line_gap_merge: int = 3,
    subline_height_ratio: float = 0.8,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect bounding boxes of logical Talim rows using horizontal projection profiling,
    structural sub-tier clustering, and gap valley analysis.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array (2D grayscale or 3D BGR/RGB). The input is never modified.
    padding : int, default=5
        Safety padding (in pixels) added around each detected row bounding box.
    min_line_gap_merge : int, default=3
        Maximum gap in pixels between sub-line fragments (e.g. stroke discontinuities)
        to be merged into a single text line.
    subline_height_ratio : float, default=0.8
        Relative height ratio threshold used to identify anchor count tiers.
        
    Returns
    -------
    List[Tuple[int, int, int, int]]
        List of bounding boxes in `(ymin, xmin, ymax, xmax)` format, ordered from
        top to bottom (Row 0, Row 1, ...).
        
    Notes
    -----
    Coordinate Convention:
        `ymin` : Top row index (inclusive)
        `xmin` : Left column index (inclusive)
        `ymax` : Bottom row index (exclusive)
        `xmax` : Right column index (exclusive)
        Slicing: `image[ymin:ymax, xmin:xmax]`
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")
    if image.size == 0:
        return []

    H, W = image.shape[:2]
    
    # 1. Binarize to obtain clean ink mask
    ink_mask = _binarize_ink(image)
    if not np.any(ink_mask):
        return [(0, 0, H, W)]

    clean_mask = ndimage.binary_opening(ink_mask, structure=np.ones((2, 2), dtype=bool))
    if not np.any(clean_mask):
        clean_mask = ink_mask

    y_indices, x_indices = np.where(clean_mask)
    if len(y_indices) == 0:
        return [(0, 0, H, W)]

    content_ymin = int(y_indices.min())
    content_ymax = int(y_indices.max() + 1)
    content_xmin = int(x_indices.min())
    content_xmax = int(x_indices.max() + 1)

    # 2. Horizontal projection profile across the text content area
    cropped_mask = clean_mask[content_ymin:content_ymax, content_xmin:content_xmax]
    proj = np.sum(cropped_mask, axis=1)

    # 3. Detect contiguous text scanlines
    labeled, num_features = ndimage.label(proj > 0)
    slices = ndimage.find_objects(labeled)

    if not slices:
        return [(0, 0, H, W)]

    # 4. Merge sub-line fragments separated by <= min_line_gap_merge
    merged_lines: List[List[int]] = []
    for s in slices:
        s_ymin, s_ymax = s[0].start, s[0].stop
        if not merged_lines:
            merged_lines.append([s_ymin, s_ymax])
        else:
            prev_ymin, prev_ymax = merged_lines[-1]
            if s_ymin - prev_ymax <= min_line_gap_merge:
                merged_lines[-1][1] = s_ymax
            else:
                merged_lines.append([s_ymin, s_ymax])

    if len(merged_lines) == 1:
        r_ymin = max(0, content_ymin - padding)
        r_ymax = min(H, content_ymax + padding)
        r_xmin = max(0, content_xmin - padding)
        r_xmax = min(W, content_xmax + padding)
        return [(r_ymin, r_xmin, r_ymax, r_xmax)]

    # 5. Analyze subline properties: heights, masses, centers, and inter-line gaps
    line_heights = [m[1] - m[0] for m in merged_lines]
    line_masses = [float(np.sum(proj[m[0]:m[1]])) for m in merged_lines]
    line_centers = [(m[0] + m[1]) / 2.0 for m in merged_lines]
    gaps = [merged_lines[i + 1][0] - merged_lines[i][1] for i in range(len(merged_lines) - 1)]

    max_h = max(line_heights)
    min_h = min(line_heights)
    has_height_disparity = (max_h >= 1.2 * min_h)

    row_line_groups: List[Tuple[int, int]] = []

    if has_height_disparity:
        # Strategy A: Anchor count tier detection (for Talim documents with tall count glyphs)
        count_candidates = [
            i for i, h in enumerate(line_heights)
            if h >= max_h * subline_height_ratio
        ]
        if not count_candidates:
            count_candidates = [int(np.argmax(line_heights))]

        # Cluster count candidates that are within 2.0 * max_h into a single anchor
        clustered_counts: List[int] = []
        min_dist = max_h * 2.0
        for idx in count_candidates:
            if not clustered_counts:
                clustered_counts.append(idx)
            else:
                prev_idx = clustered_counts[-1]
                if line_centers[idx] - line_centers[prev_idx] < min_dist:
                    if (line_heights[idx] > line_heights[prev_idx] or
                        (line_heights[idx] == line_heights[prev_idx] and
                         line_masses[idx] > line_masses[prev_idx])):
                        clustered_counts[-1] = idx
                else:
                    clustered_counts.append(idx)

        num_anchors = len(clustered_counts)
        if num_anchors <= 1:
            row_line_groups = [(0, len(merged_lines) - 1)]
        else:
            for k, c_idx in enumerate(clustered_counts):
                if k == 0:
                    start_line = 0
                else:
                    prev_c = clustered_counts[k - 1]
                    mid_lines = list(range(prev_c + 1, c_idx))
                    mid_y = (line_centers[prev_c] + line_centers[c_idx]) / 2.0
                    start_line = c_idx
                    for m in mid_lines:
                        if line_centers[m] >= mid_y:
                            start_line = m
                            break

                if k == num_anchors - 1:
                    end_line = len(merged_lines) - 1
                else:
                    next_c = clustered_counts[k + 1]
                    mid_lines = list(range(c_idx + 1, next_c))
                    mid_y = (line_centers[c_idx] + line_centers[next_c]) / 2.0
                    end_line = c_idx
                    for m in mid_lines:
                        if line_centers[m] < mid_y:
                            end_line = m

                row_line_groups.append((start_line, end_line))

    else:
        # Strategy B: Gap valley clustering (for uniform line heights or <= 3 lines)
        unique_gaps = sorted(list(set(gaps)), reverse=True)
        best_cuts: List[int] = []
        best_score = -1e9

        for t in unique_gaps:
            cuts = [i for i, g in enumerate(gaps) if g >= t]
            group_sizes = []
            last_cut = -1
            for c in cuts:
                group_sizes.append(c - last_cut)
                last_cut = c
            group_sizes.append(len(merged_lines) - 1 - last_cut)

            if max(group_sizes) > 3 or min(group_sizes) <= 0:
                continue

            avg_group_size = np.mean(group_sizes)
            score = avg_group_size * 10.0 + t
            if score > best_score:
                best_score = score
                best_cuts = cuts

        last_cut = -1
        for c in best_cuts:
            row_line_groups.append((last_cut + 1, c))
            last_cut = c
        row_line_groups.append((last_cut + 1, len(merged_lines) - 1))

    # 6. Construct final bounding boxes with horizontal extent & padding
    row_bboxes: List[Tuple[int, int, int, int]] = []
    for start_l, end_l in row_line_groups:
        r_ymin = content_ymin + merged_lines[start_l][0]
        r_ymax = content_ymin + merged_lines[end_l][1]

        # Calculate horizontal ink bounds specifically within this row's vertical slice
        row_ink = clean_mask[r_ymin:r_ymax, :]
        r_yidx, r_xidx = np.where(row_ink)
        if len(r_xidx) > 0:
            r_xmin = max(0, int(r_xidx.min()) - padding)
            r_xmax = min(W, int(r_xidx.max()) + 1 + padding)
        else:
            r_xmin = max(0, content_xmin - padding)
            r_xmax = min(W, content_xmax + padding)

        r_ymin = max(0, r_ymin - padding)
        r_ymax = min(H, r_ymax + padding)

        row_bboxes.append((r_ymin, r_xmin, r_ymax, r_xmax))

    return row_bboxes


def separate_rows(
    image: np.ndarray,
    row_bboxes: Optional[List[Tuple[int, int, int, int]]] = None,
    padding: int = 5,
) -> List[RowData]:
    """
    Extract logical row crops from an image based on detected row bounding boxes.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array (2D or 3D). The input is never modified.
    row_bboxes : Optional[List[Tuple[int, int, int, int]]], default=None
        Pre-computed row bounding boxes `(ymin, xmin, ymax, xmax)`. If None,
        `detect_rows(image, padding=padding)` is called automatically.
    padding : int, default=5
        Padding passed to `detect_rows` if `row_bboxes` is not supplied.
        
    Returns
    -------
    List[RowData]
        Ordered list of `RowData` objects containing individual row crops, coordinates,
        and dimensions. Original resolution, channels, and dtypes are strictly preserved.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(image)}")

    if row_bboxes is None:
        row_bboxes = detect_rows(image, padding=padding)

    results: List[RowData] = []
    for idx, (ymin, xmin, ymax, xmax) in enumerate(row_bboxes):
        row_img = image[ymin:ymax, xmin:xmax].copy()
        rh, rw = row_img.shape[:2]
        results.append(
            RowData(
                row_index=idx,
                image=row_img,
                bbox=(ymin, xmin, ymax, xmax),
                height=rh,
                width=rw,
                metadata={"num_channels": 1 if row_img.ndim == 2 else row_img.shape[2]},
            )
        )

    return results
