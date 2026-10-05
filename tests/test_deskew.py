"""
NAQSHKASH - Deskew validation
==============================

Two kinds of evaluation, kept strictly separate:

PART A - Ground-truth sweep (accuracy)
    Straight source pages from tests/ground_truth_sources/ are rotated by
    KNOWN angles and degraded (blur, noise+JPEG, low-res, shadow). The
    deskew module never sees the true angle; only this script does.
    Metric: |estimated - true| in degrees.

PART B - Real / category folders (behaviour on real inputs)
    Images in tests/input, clean, scanned, phone_style, real.
    If tests/labels.csv has a manually measured angle for a file
    (columns: filename,true_angle ; CCW positive), it is scored like Part A.
    Otherwise the result is UNVERIFIED and a review overlay (horizontal
    guide lines) is saved so a human can check it in seconds.

Why the residual re-check is NOT used as PASS/FAIL:
    Re-estimating skew on the corrected image with the same estimator only
    shows the estimator agrees with itself. It cannot detect the estimator
    being wrong (it reports ~0 after any rotation it chose). It is recorded
    as `self_check` for debugging only.

Status values
    PASS        error <= 0.5 deg
    ACCEPTABLE  0.5 < error <= 1.0 deg
    FAIL        error > 1.0 deg
    HARM        image was rotated and ended up MORE skewed than the input
    MISSED      true skew > 1 deg but the module declined to correct
    UNVERIFIED  no label; see review overlay
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.deskew import deskew, estimate_skew, rotate_image  # noqa: E402


# ============================================================
# PATHS AND SETTINGS
# ============================================================

TEST_ROOT = PROJECT_ROOT / "tests"
GT_SOURCE_DIR = TEST_ROOT / "ground_truth_sources"   # straight pages only
LABELS_PATH = TEST_ROOT / "labels.csv"

CATEGORY_DIRS = {
    "basic": TEST_ROOT / "input",
    "clean": TEST_ROOT / "clean",
    "scanned": TEST_ROOT / "scanned",
    "phone_style": TEST_ROOT / "phone_style",
    "real": TEST_ROOT / "real",
}

OUTPUT_ROOT = TEST_ROOT / "robust_output"
REVIEW_ROOT = TEST_ROOT / "review_overlays"

GT_REPORT_PATH = TEST_ROOT / "deskew_ground_truth.csv"
FULL_REPORT_PATH = TEST_ROOT / "deskew_report.csv"
FAILURE_REPORT_PATH = TEST_ROOT / "deskew_failures.csv"

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

PASS_LIMIT = 0.5          # deg
ACCEPTABLE_LIMIT = 1.0    # deg
MISSED_LIMIT = 1.0        # true skew above this should have been corrected

GT_ANGLES = [-12.0, -8.0, -5.0, -3.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 5.0, 8.0, 12.0]


# ============================================================
# DEGRADATIONS (test-side only; mimic scans and phone photos)
# ============================================================

def _shadow(img, strength=0.45):
    h, w = img.shape[:2]
    g = np.linspace(1 - strength, 1, w)[None, :] * np.linspace(1, 1 - strength * 0.4, h)[:, None]
    if img.ndim == 3:
        g = g[..., None]
    return np.clip(img * g, 0, 255).astype(np.uint8)


def _noise_jpeg(img, sigma=8, quality=25):
    rng = np.random.default_rng(0)
    noisy = np.clip(img + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _lowres(img, factor=0.35):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * factor)), max(1, int(h * factor))),
                      interpolation=cv2.INTER_AREA)


DEGRADATIONS = {
    "none": lambda im: im,
    "blur": lambda im: cv2.GaussianBlur(im, (7, 7), 0),
    "noise_jpeg": _noise_jpeg,
    "low_res": _lowres,
    "shadow_blur": lambda im: cv2.GaussianBlur(_shadow(im), (5, 5), 0),
}


# ============================================================
# HELPERS
# ============================================================

def iter_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)


def load_labels() -> dict[str, float]:
    if not LABELS_PATH.exists():
        return {}
    labels = {}
    with LABELS_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                labels[row["filename"].strip()] = float(row["true_angle"])
            except (KeyError, ValueError):
                continue
    return labels


def score(true_angle: float, result) -> tuple[str, float, float]:
    """Return (status, estimation_error, true_residual)."""
    est_error = abs(result.skew_angle - true_angle)
    # Real skew left in the output: input skew plus the rotation applied.
    true_residual = abs(true_angle + result.applied_rotation)

    if result.corrected and true_residual > abs(true_angle) + 0.1:
        return "HARM", est_error, true_residual
    if not result.corrected and abs(true_angle) > MISSED_LIMIT:
        return "MISSED", est_error, true_residual
    if true_residual <= PASS_LIMIT:
        return "PASS", est_error, true_residual
    if true_residual <= ACCEPTABLE_LIMIT:
        return "ACCEPTABLE", est_error, true_residual
    return "FAIL", est_error, true_residual


def review_overlay(image: np.ndarray, spacing: int = 40) -> np.ndarray:
    """Draw horizontal guide lines; tilted text is obvious against them."""
    vis = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for y in range(spacing, vis.shape[0], spacing):
        cv2.line(vis, (0, y), (vis.shape[1] - 1, y), (0, 0, 255), 1)
    return vis


def summarise(rows, key, title):
    print()
    print(title)
    print("-" * 104)
    print(f"{'group':16s} {'n':>4s} {'MAE':>6s} {'p95':>6s} {'max':>6s} "
          f"{'PASS':>6s} {'<=1deg':>7s} {'HARM':>5s} {'MISSED':>7s} {'FAIL':>5s} {'skipped':>8s}")
    groups = sorted({r[key] for r in rows}) + ["ALL"]
    for g in groups:
        R = [r for r in rows if g == "ALL" or r[key] == g]
        scored = [r for r in R if r["status"] != "UNVERIFIED"]
        if not scored:
            print(f"{g:16s} {len(R):4d}   (no ground truth - see review overlays)")
            continue
        res = np.array([float(r["true_residual"]) for r in scored])
        st = [r["status"] for r in scored]
        skipped = sum(1 for r in scored if r["reason"] != "CORRECTED")
        print(f"{g:16s} {len(scored):4d} {res.mean():6.2f} {np.percentile(res, 95):6.2f} "
              f"{res.max():6.2f} {st.count('PASS') / len(st) * 100:5.0f}% "
              f"{(st.count('PASS') + st.count('ACCEPTABLE')) / len(st) * 100:6.0f}% "
              f"{st.count('HARM'):5d} {st.count('MISSED'):7d} {st.count('FAIL'):5d} {skipped:8d}")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))   # union, ordered
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# PART A - GROUND-TRUTH SWEEP
# ============================================================

def run_ground_truth() -> list[dict]:
    sources = iter_images(GT_SOURCE_DIR)
    print(f"\n[PART A] Ground-truth sweep: {len(sources)} straight source page(s) x "
          f"{len(GT_ANGLES)} angles x {len(DEGRADATIONS)} degradations")
    if not sources:
        print(f"  No sources in {GT_SOURCE_DIR}. Add straight (unrotated) Talim pages.")
        return []

    rows = []
    for src_path in sources:
        src = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if src is None:
            continue
        base = estimate_skew(src)
        if abs(base["angle"]) > 0.3:
            print(f"  WARNING {src_path.name}: source itself measures {base['angle']:.2f} deg; "
                  "ground truth for this page is approximate.")
        for true_angle in GT_ANGLES:
            rotated = rotate_image(src, true_angle)
            for deg_name, degrade in DEGRADATIONS.items():
                result = deskew(degrade(rotated))
                status, est_err, true_res = score(true_angle, result)
                rows.append({
                    "source": src_path.name, "degradation": deg_name,
                    "true_angle": true_angle, "estimated": round(result.skew_angle, 3),
                    "estimation_error": round(est_err, 3), "true_residual": round(true_res, 3),
                    "confidence": round(result.confidence, 3), "reason": result.reason,
                    "status": status,
                })
    summarise(rows, "degradation", "PART A summary by degradation (residual = real skew left in output, deg)")
    return rows


# ============================================================
# PART B - REAL / CATEGORY FOLDERS
# ============================================================

def run_categories() -> list[dict]:
    labels = load_labels()
    print(f"\n[PART B] Category folders ({len(labels)} manual label(s) loaded)")
    rows = []
    for category, folder in CATEGORY_DIRS.items():
        images = iter_images(folder)
        print(f"\n  [{category.upper()}] {len(images)} image(s)")
        for path in images:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                rows.append({"category": category, "filename": path.name, "estimated": "",
                             "confidence": "", "reason": "UNREADABLE", "applied_rotation": "",
                             "true_angle": "", "estimation_error": "", "true_residual": "",
                             "self_check": "", "status": "FAIL", "output_path": ""})
                continue

            t0 = time.perf_counter()
            result = deskew(image)
            ms = (time.perf_counter() - t0) * 1000
            self_check = estimate_skew(result.image)["angle"]   # debugging only

            out_path = OUTPUT_ROOT / category / path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), result.image)
            review_path = REVIEW_ROOT / category / path.name
            review_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(review_path), np.hstack([
                review_overlay(cv2.resize(image, (int(image.shape[1] * result.image.shape[0] / image.shape[0]),
                                                  result.image.shape[0]))),
                review_overlay(result.image)]))

            if path.name in labels:
                true_angle = labels[path.name]
                status, est_err, true_res = score(true_angle, result)
            else:
                true_angle, est_err, true_res, status = "", "", "", "UNVERIFIED"

            row = {
                "category": category, "filename": path.name,
                "estimated": round(result.skew_angle, 3),
                "confidence": round(result.confidence, 3),
                "reason": result.reason,
                "applied_rotation": round(result.applied_rotation, 3),
                "true_angle": true_angle,
                "estimation_error": "" if est_err == "" else round(est_err, 3),
                "true_residual": "" if true_res == "" else round(true_res, 3),
                "self_check": round(self_check, 3),
                "status": status,
                "output_path": str(out_path.relative_to(PROJECT_ROOT)),
            }
            rows.append(row)
            print(f"    {path.name:34s} est={result.skew_angle:7.2f}  conf={result.confidence:4.2f}  "
                  f"{result.reason:16s} [{status}]  {ms:5.0f} ms")

    summarise(rows, "category", "PART B summary (labelled images only; unlabelled -> review overlays)")
    return rows


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    for folder in [GT_SOURCE_DIR, *CATEGORY_DIRS.values(), OUTPUT_ROOT, REVIEW_ROOT]:
        folder.mkdir(parents=True, exist_ok=True)

    print("\nNAQSHKASH - DESKEW VALIDATION")
    print("=" * 104)

    gt_rows = run_ground_truth()
    cat_rows = run_categories()

    write_csv(GT_REPORT_PATH, gt_rows)
    write_csv(FULL_REPORT_PATH, cat_rows)
    failures = [r for r in gt_rows + cat_rows if r["status"] in ("FAIL", "HARM", "MISSED")]
    write_csv(FAILURE_REPORT_PATH, failures)

    print()
    print("=" * 104)
    print(f"Ground-truth report : {GT_REPORT_PATH}")
    print(f"Category report     : {FULL_REPORT_PATH}")
    print(f"Failures            : {FAILURE_REPORT_PATH}  ({len(failures)} row(s))")
    print(f"Review overlays     : {REVIEW_ROOT}   (left = input, right = output)")
    print()


if __name__ == "__main__":
    main()
