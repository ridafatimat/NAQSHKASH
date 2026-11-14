"""Stage 2-only structural fallback; shared Stage 1 remains untouched.

Uses GT row count already available in this offline tier-preparation task.
It never reads character values, model predictions or sample IDs. This is
annotation-assisted OCR preparation, not a deployable blind page segmenter.
"""
import numpy as np
from scipy import ndimage
from inspect_local import STAGE1
from preprocessing.pipeline import crop_and_separate_rows
from preprocessing.crop import _binarize_ink
from preprocessing.rows import separate_rows


def whitespace_row_boxes(image, expected_rows, padding=5):
    if expected_rows < 1:
        raise ValueError('Expected positive row count')
    mask = _binarize_ink(image)
    clean = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool))
    if not clean.any():
        clean = mask
    if not clean.any():
        raise ValueError('No foreground ink')
    active = clean.any(axis=1)
    transitions = np.diff(np.pad(active.astype(np.int8), (1, 1)))
    bands = list(zip(np.where(transitions == 1)[0].tolist(), np.where(transitions == -1)[0].tolist()))
    merged = []
    for start, end in bands:
        if merged and start - merged[-1][1] <= 3:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    if len(merged) < expected_rows:
        raise ValueError('Fewer ink bands than expected logical rows')
    gaps = [(merged[i+1][0] - merged[i][1], i) for i in range(len(merged)-1)]
    if expected_rows == 1:
        cuts = []
    else:
        ranked = sorted(gaps, reverse=True)
        chosen, others = ranked[:expected_rows-1], ranked[expected_rows-1:]
        if others and min(g for g, i in chosen) <= max(g for g, i in others):
            raise ValueError('Ambiguous equal whitespace at logical-row boundary')
        cuts = sorted(i for gap, i in chosen)
    boxes = []
    begin = 0
    height, width = image.shape[:2]
    for end in cuts + [len(merged)-1]:
        y0, y1 = merged[begin][0], merged[end][1]
        ys, xs = np.where(clean[y0:y1])
        boxes.append((max(0,y0-padding),max(0,int(xs.min())-padding),
                      min(height,y1+padding),min(width,int(xs.max())+1+padding)))
        begin = end+1
    return boxes, dict(bands=merged, gaps=[g for g,i in gaps], boundary_after_bands=cuts)


def segment_page(image, expected_rows):
    result = crop_and_separate_rows(image, apply_deskew=False)
    old_count = len(result.rows)
    if old_count == expected_rows:
        return result, dict(method='stage1_unchanged', original_rows=old_count, recovered=False)
    boxes, geometry = whitespace_row_boxes(result.cropped_image, expected_rows)
    rows = separate_rows(result.cropped_image, row_bboxes=boxes)
    if len(rows) != expected_rows:
        raise ValueError('Fallback row count mismatch')
    result.rows = rows
    result.num_rows = len(rows)
    cy, cx, _, _ = result.crop_bbox
    result.global_row_bboxes = [(cy+y0,cx+x0,cy+y1,cx+x1) for y0,x0,y1,x1 in boxes]
    return result, dict(method='stage2_whitespace_fallback',original_rows=old_count,recovered=True,**geometry)
