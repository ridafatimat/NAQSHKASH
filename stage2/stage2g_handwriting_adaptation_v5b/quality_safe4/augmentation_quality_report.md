# Safe4 augmentation QA

Selected round: **1**, the first round meeting the proposal-level criteria. Exactly three rounds were generated; no additional rounds were run.

Visibility rejection is counted as visibility-rejected augmentation proposals divided by all sampled proposals. The stricter conditional rate among rendered proposals is also shown explicitly; it exceeds 5% in these rounds. Every accepted component individually satisfies the darkest-pixel threshold of 120. The original conditional-screen reports are retained unchanged.

| Round | Core median | Dot median | Visibility / all proposals | Visibility / rendered | Proposal-level acceptance |
|---|---:|---:|---:|---:|---|
| 1 | 42.50 | 5.00 | 4.42% | 13.89% | True |
| 2 | 33.00 | 5.00 | 4.67% | 14.59% | True |
| 3 | 29.00 | 5.00 | 3.25% | 9.31% | True |

| Domain | Measure | Median | p10 | p90 | N |
|---|---|---:|---:|---:|---:|
| augmented | stroke_width | 4.00 | 2.00 | 4.00 | 200 |
| augmented | ink_darkness | 42.50 | 19.00 | 76.00 | 200 |
| augmented | dot_diameter | 5.00 | 4.00 | 6.48 | 765 |
| real_handwriting | stroke_width | 2.80 | 2.00 | 6.00 | 105 |
| real_handwriting | ink_darkness | 39.72 | 17.15 | 65.54 | 105 |
| real_handwriting | dot_diameter | 6.96 | 4.00 | 13.04 | 170 |
| clean_synthetic | stroke_width | 8.00 | 6.00 | 11.63 | 200 |
| clean_synthetic | ink_darkness | 3.00 | 3.00 | 3.00 | 200 |
| clean_synthetic | dot_diameter | 13.49 | 5.00 | 16.49 | 776 |

Core darkness uses grayscale at distance-transform ridges; real reference crops use the existing V3 contrast, matching the supplied real reference (median 39.72, p10–p90 17.15–65.54). Dot diameter is the geometric mean of outer bounding-box dimensions. Real compact-mark detection is heuristic, so its p90 includes larger compact marks and should not be interpreted as independently annotated semantic dots. All accepted augmented detached-mark diameters are individually guarded to 4–7 pixels.

Detached compact marks are redrawn without skeletonization. Filled and hollow marks use sampled stroke thickness for the ring; when no interior remains, the mark is filled. Body width sampling stays at 1.5–4 pixels. Enlarged dots influence the whole-image stroke statistic; it is not a body-only stroke measurement. Geometry, component correspondence, body holes, boundary support, visibility and dot diameter are all guarded at height 64. Unconfirmed variants remain disabled. Whole-tier affine scale/pitch is retained from V5; accepted angle/scale distributions reflect geometric rejection.

![Overlaid distributions](round_01/distributions.png)
![200 paired samples, sheet 1](round_01/contact_sheets/pairs_00.jpg)
![200 paired samples, sheet 2](round_01/contact_sheets/pairs_01.jpg)
![200 paired samples, sheet 3](round_01/contact_sheets/pairs_02.jpg)
![200 paired samples, sheet 4](round_01/contact_sheets/pairs_03.jpg)
![200 paired samples, sheet 5](round_01/contact_sheets/pairs_04.jpg)
![200 paired samples, sheet 6](round_01/contact_sheets/pairs_05.jpg)
![200 paired samples, sheet 7](round_01/contact_sheets/pairs_06.jpg)
![200 paired samples, sheet 8](round_01/contact_sheets/pairs_07.jpg)
![200 paired samples, sheet 9](round_01/contact_sheets/pairs_08.jpg)
![200 paired samples, sheet 10](round_01/contact_sheets/pairs_09.jpg)
