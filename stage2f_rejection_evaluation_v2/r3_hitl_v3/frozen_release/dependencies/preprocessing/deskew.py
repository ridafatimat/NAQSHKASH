"""
NAQSHKASH - Rotational deskewing (preprocessing stage)
=======================================================

Scope
-----
Corrects small, global, IN-PLANE ROTATION of Talim page images.

Out of scope (handled by later stages, never "fixed" here):
  * perspective / keystone distortion   -> perspective-correction stage
  * page curl, per-line drift           -> line-level dewarping / deskew
  * 90 / 180 degree orientation errors  -> orientation-detection stage

Method: projection-profile skew estimation (Postl 1986; Baird 1987)
-------------------------------------------------------------------
1. Build a CONTENT MASK used only for estimation. The returned image is
   never cleaned, so no Talim symbol, dot or stroke is ever removed.
     a. grayscale, optional downscale (speed only),
     b. background flattening: divide by a morphologically closed copy of
        the page so shadows and uneven lighting stop looking like ink,
     c. Otsu threshold on the flattened image,
     d. ignore a thin margin at the image border (scanner/camera edges),
     e. keep only symbol-sized connected components; drop page edges,
        borders, long ruling lines, shadow blobs and specks.
2. For each candidate angle in [-max_angle, +max_angle] rotate the
   foreground POINT COORDINATES (not the image; no clipping, no
   interpolation) and build the horizontal projection profile.
   Score = profile energy (sum of squared row counts). Ink packed into
   horizontal rows gives high energy; tilted rows smear it out.
3. Coarse search (0.5 deg) then fine search (0.05 deg).
4. Confidence = (best score - median score) / best score. If confidence
   is low, the best angle sits on the edge of the search range, there is
   no content, or the angle is negligible, the image is returned
   UNCHANGED. Not rotating is always safer than rotating wrongly.
5. Rotate the ORIGINAL image once, expanding the canvas (no clipping) and
   filling new area with the estimated paper colour (no white wedges).

Angle convention
----------------
skew_angle > 0 : content is rotated counter-clockwise as displayed
                 (text rows rise to the right).
Correction     : rotate_image(image, -skew_angle).
The search and rotate_image share this convention, so the correction
direction is right by construction; no "try both directions" step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ============================================================
# CONFIGURATION AND RESULT TYPES
# ============================================================

@dataclass
class DeskewConfig:
    max_angle: float = 15.0            # search range, +/- degrees
    min_angle: float = 0.2             # below this, do not resample the image
    coarse_step: float = 0.5           # degrees
    fine_step: float = 0.05            # degrees
    min_confidence: float = 0.40       # calibrated; re-check on real images
    analysis_max_side: int = 1600      # estimation-only downscale target
    max_points: int = 150_000          # subsample foreground points above this
    border_margin_frac: float = 0.015  # ignored outer margin (fraction of short side)
    max_component_span_frac: float = 0.40   # wider/taller -> edge, border, ruling line
    max_component_area_frac: float = 0.02   # larger area  -> shadow / blob
    min_component_area: int = 3             # smaller      -> speck / noise
    min_foreground_points: int = 200
    expand_canvas: bool = True


@dataclass
class DeskewResult:
    image: np.ndarray            # corrected image, or an untouched copy
    skew_angle: float            # estimated skew of the INPUT (deg, CCW +)
    applied_rotation: float      # rotation actually applied (0.0 if none)
    confidence: float            # 0..1 prominence of the best angle
    corrected: bool
    reason: str                  # CORRECTED | ALREADY_STRAIGHT | LOW_CONFIDENCE
                                 # | OUT_OF_RANGE | NO_CONTENT
    matrix: np.ndarray | None = field(default=None, repr=False)
    # 2x3 affine used for the correction. Lets later stages (annotations,
    # bounding boxes, CRNN/Transformer line crops) map coordinates between
    # the original and deskewed image with cv2.transform / invertAffineTransform.


# ============================================================
# HELPERS
# ============================================================

def _to_grayscale(image: np.ndarray) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None.")
    if image.ndim == 2:
        return image.copy()
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _paper_color(image: np.ndarray):
    """Median colour; for a document page this is the paper colour."""
    if image.ndim == 2:
        return int(np.median(image))
    return tuple(int(v) for v in np.median(image.reshape(-1, image.shape[2]), axis=0))


# ============================================================
# STEP 1 - CONTENT MASK (estimation only)
# ============================================================

def build_content_mask(image: np.ndarray, cfg: DeskewConfig | None = None) -> np.ndarray:
    """Binary mask (255 = probable Talim ink) at analysis resolution."""
    cfg = cfg or DeskewConfig()
    gray = _to_grayscale(image)

    h, w = gray.shape
    scale = min(1.0, cfg.analysis_max_side / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = gray.shape

    # Background flattening. Closing with a kernel larger than a stroke
    # erases dark ink and leaves the illumination of the paper.
    k = max(15, (min(h, w) // 30) | 1)
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), k / 4)
    flat = cv2.divide(gray, np.maximum(background, 1), scale=255)

    _, mask = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    m = int(round(cfg.border_margin_frac * min(h, w)))
    if m > 0:
        mask[:m, :] = 0
        mask[-m:, :] = 0
        mask[:, :m] = 0
        mask[:, -m:] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    cw = stats[:, cv2.CC_STAT_WIDTH]
    ch = stats[:, cv2.CC_STAT_HEIGHT]
    area = stats[:, cv2.CC_STAT_AREA]
    keep = (
        (area >= cfg.min_component_area)
        & (area <= cfg.max_component_area_frac * h * w)
        & (cw <= cfg.max_component_span_frac * w)
        & (ch <= cfg.max_component_span_frac * h)
    )
    keep[0] = False
    filtered = np.where(keep[labels], 255, 0).astype(np.uint8)

    # If filtering removed almost everything (e.g. every symbol touches a
    # ruling line), fall back to the unfiltered mask. Under pure rotation,
    # ruling lines are parallel to the text, so they still give the right
    # angle; the confidence score reflects the weaker evidence.
    if cv2.countNonZero(filtered) < cfg.min_foreground_points:
        return mask
    return filtered


# ============================================================
# STEPS 2-4 - PROJECTION-PROFILE SEARCH
# ============================================================

def _profile_score(xs: np.ndarray, ys: np.ndarray, angle_deg: float) -> float:
    """Energy of the horizontal projection after undoing a skew of angle_deg."""
    t = np.radians(angle_deg)
    # Image y points down, so a CCW-skewed row has y decreasing with x.
    # yr is each point's row coordinate after rotating the cloud back.
    yr = ys * np.cos(t) + xs * np.sin(t)
    yr = yr - yr.min()
    hist = np.bincount(yr.astype(np.int64)).astype(np.float64)
    return float(np.dot(hist, hist))


def _search(xs, ys, lo, hi, step):
    angles = np.arange(lo, hi + step / 2, step)
    scores = np.array([_profile_score(xs, ys, a) for a in angles])
    return angles, scores


def estimate_skew(image: np.ndarray, cfg: DeskewConfig | None = None) -> dict:
    """
    Estimate skew with diagnostics.

    Returns dict: angle (deg, CCW +), confidence (0..1),
                  at_boundary (bool), n_points (int).
    """
    cfg = cfg or DeskewConfig()
    mask = build_content_mask(image, cfg)
    ys, xs = np.nonzero(mask)
    result = dict(angle=0.0, confidence=0.0, at_boundary=False, n_points=int(len(xs)))
    if len(xs) < cfg.min_foreground_points:
        return result

    if len(xs) > cfg.max_points:
        idx = np.random.default_rng(0).choice(len(xs), cfg.max_points, replace=False)
        xs, ys = xs[idx], ys[idx]
    xs = xs.astype(np.float64) - xs.mean()
    ys = ys.astype(np.float64) - ys.mean()

    angles, scores = _search(xs, ys, -cfg.max_angle, cfg.max_angle, cfg.coarse_step)
    i = int(np.argmax(scores))
    peak, typical = float(scores[i]), float(np.median(scores))
    confidence = 0.0 if peak <= 0 else (peak - typical) / peak

    fine_angles, fine_scores = _search(
        xs, ys, angles[i] - cfg.coarse_step, angles[i] + cfg.coarse_step, cfg.fine_step)
    best = float(fine_angles[int(np.argmax(fine_scores))])

    result.update(
        angle=0.0 if abs(best) < 1e-9 else best,
        confidence=float(confidence),
        at_boundary=bool(i == 0 or i == len(angles) - 1),
    )
    return result


def estimate_skew_angle(image: np.ndarray, cfg: DeskewConfig | None = None) -> float:
    """Backward-compatible: estimated skew in degrees (CCW positive)."""
    return estimate_skew(image, cfg)["angle"]


# ============================================================
# STEP 5 - ROTATION
# ============================================================

def rotation_matrix(shape, angle: float, expand_canvas: bool = True):
    """2x3 matrix rotating CCW by `angle` degrees about the image centre."""
    h, w = shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    if not expand_canvas:
        return M, (w, h)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(np.ceil(h * s + w * c))
    new_h = int(np.ceil(h * c + w * s))
    M[0, 2] += new_w / 2.0 - w / 2.0
    M[1, 2] += new_h / 2.0 - h / 2.0
    return M, (new_w, new_h)


def rotate_image(image: np.ndarray, angle: float, expand_canvas: bool = True,
                 fill=None) -> np.ndarray:
    """Rotate CCW by `angle` degrees; new area is filled with paper colour."""
    if image is None:
        raise ValueError("Input image is None.")
    M, size = rotation_matrix(image.shape, angle, expand_canvas)
    if fill is None:
        fill = _paper_color(image)
    return cv2.warpAffine(image, M, size, flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


# ============================================================
# MAIN ENTRY POINTS
# ============================================================

def deskew(image: np.ndarray, cfg: DeskewConfig | None = None) -> DeskewResult:
    """Estimate skew; correct only when the estimate is trustworthy."""
    cfg = cfg or DeskewConfig()
    est = estimate_skew(image, cfg)
    angle, conf = est["angle"], est["confidence"]

    def unchanged(reason: str) -> DeskewResult:
        return DeskewResult(image=image.copy(), skew_angle=angle, applied_rotation=0.0,
                            confidence=conf, corrected=False, reason=reason)

    if est["n_points"] < cfg.min_foreground_points:
        return unchanged("NO_CONTENT")
    if conf < cfg.min_confidence:
        return unchanged("LOW_CONFIDENCE")
    if est["at_boundary"]:
        return unchanged("OUT_OF_RANGE")
    if abs(angle) < cfg.min_angle:
        return unchanged("ALREADY_STRAIGHT")

    M, size = rotation_matrix(image.shape, -angle, cfg.expand_canvas)
    corrected = cv2.warpAffine(image, M, size, flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=_paper_color(image))
    return DeskewResult(image=corrected, skew_angle=angle, applied_rotation=-angle,
                        confidence=conf, corrected=True, reason="CORRECTED", matrix=M)


def deskew_image(image: np.ndarray, max_angle: float = 15.0, min_angle: float = 0.2,
                 expand_canvas: bool = True):
    """Backward-compatible wrapper: returns (corrected_image, detected_angle)."""
    cfg = DeskewConfig(max_angle=max_angle, min_angle=min_angle, expand_canvas=expand_canvas)
    result = deskew(image, cfg)
    return result.image, result.skew_angle
