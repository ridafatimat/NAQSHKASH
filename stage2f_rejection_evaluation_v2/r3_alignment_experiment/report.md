# CTC alignment pilot — implementation status

**REAL EXPERIMENT NOT RUN. Alignment accuracy is unknown.**

Prepared 12 independent development parents, four per invalidity category, and all four clean/illumination valid/invalid variants: 48 images. Source widths span 6–23 renderer columns. The isolated CPU extractor will forward only the affected row, at most 144 tiers total for a completed run.

The implementation reuses frozen D1/E1 inference, resize, decoding, deterministic seeds, dataset selection and immutable storage. It captures model tensors and complete coordinate transforms, checks reproduction against saved raw predictions, and scores source-x localization and count-token associations offline against each image's actual visible notation. No rejection rule, new CTC target or model training is implemented.

Lightweight verification: 16 synthetic/mock tests passed. These test CTC blanks versus literal spaces, repeated labels, inverse/rounded/projective geometry, ambiguous edit correspondence, geometry-only association, missing tokens, hook cleanup, reproduction mismatch exclusion, cached asset hashes, gallery generation, actual metadata-only development selection, offline-reference isolation and a tiny mocked cached-evaluation/report round trip. They provide no empirical E1 alignment result.

No real image from the 48-image selection was opened or forwarded during implementation. No validation/protected image bytes were accessed. Existing D1/E1/T1/F1/R2 code, checkpoints, datasets, saved observations and historical locks remain unchanged. No Git operations were performed.

Run the exact commands in README.md. The subsequent runs/<tag>_evaluation/report.md and metrics.json will contain the measured feasibility result, category/parent metrics, focused mutation-token scores, reproducibility mismatches and valid-control risk. The gallery is generated from the captured real crops only after that manual run. Until then, no claim that CTC is accurate enough for R3 validation is justified.
