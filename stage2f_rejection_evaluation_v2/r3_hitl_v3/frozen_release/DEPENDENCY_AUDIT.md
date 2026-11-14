# Dependency audit

The six existing dependency files were insufficient for standalone policy/inference use. Added byte-identical local dependencies include:

- F1 common paths/IO, confidence, rejection pipeline/policy and original FrozenPipeline.observe implementation.
- T1 runtime, contextual interpretation, tier-aware decoder, evaluation measurements, verified ambiguity registry and its audit-summary checksum, rendering profile.
- R2 structural guard, shared observation journal, raw-inference deterministic seeding/environment helper, evaluation-case scorer and unchanged protected DevelopmentDataset.
- D1 pipeline/config/engine, photometric/geometry modules, core imports, preprocessing package, and its transitive data/build_full_tier_dataset, positive-edge segmentation/inspection/evaluation and scanned modes imports. Annotation-assisted helpers are imported by historical D1 dependencies but are NOT called by the blind D1 path.
- E1 selected best.pt (SHA256 27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1), CRNN definition, vocabulary, CTC decoder, resize/collate, training vocabulary-loader definition and its data utility imports. Training functions are NOT executed.
- Canonical talim package and original Talamat config/count-converter references; no copied or invented mapping replaces the canonical rules.

The exact file/source/hash inventory is in release_manifest.json. Additional dependency hashes not present in the approved run's direct dependency map are explicitly recorded at freeze; imported definitions are verified through isolated lazy-import preflight. All declared approved source/config/dependency hashes matched before snapshot additions. Every existing snapshot matched both its CSV hash and original source.

Frozen imports cannot fall back to working project modules. Absolute paths in byte-identical source are relocated by the isolated loader, not by editing snapshots. Installed third-party packages are external runtime dependencies and must match the recorded environment. No model is initialized by import preflight. Original evaluator CLIs and development-only loader protections are preserved unchanged; the authorized held-out adapter is separate.
