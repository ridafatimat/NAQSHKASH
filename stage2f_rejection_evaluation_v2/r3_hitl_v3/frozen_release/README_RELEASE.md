# Final Stage 2F release: R3_HITL_V3

The selected policy is the byte-identical R3_HITL_V3 from approved development run `20261107_192326_500_058300e0`: valid 1318 ACCEPT / 82 REVIEW / 0 REJECT; invalid 48 ACCEPT / 1352 REVIEW / 0 REJECT; 2800 images / 700 parents; zero Stage 1 decision differences. R3.1 and its targeted fix are preserved experiments and are excluded.

## Integrity and provenance

The original eleven snapshots, file_hashes.csv, snapshot_hashes.csv and DEVELOPMENT_CHECKPOINT_NOTES.txt are preserved without editing. The notes describe the historical checkpoint when freeze was pending; release_manifest.json and RELEASE_SHA256.txt now describe the final release. No original source, old result or dataset manifest is modified.

release_manifest.json records every frozen file SHA/size/original source, all held-out adapter hashes, dataset manifest/parent/protected-seal metadata, split counts, checkpoint identity and snapshot, inference software versions, approved development completion/result/raw hashes, and generation restrictions. Extra local transitive dependencies and canonical external reference files are copied byte for byte. No ZIP-dependent lock is added. Site packages remain installed software, not project source; exact approved Python/PyTorch/NumPy/OpenCV/Pillow versions and frozen SciPy version are checked before inference.

Supported entrypoints are verify_freeze.py and the separate heldout_evaluation/src/heldout_cli.py. The original snapshotted experiment CLIs are historical source artifacts, not new held-out entrypoints. FrozenRuntime executes their reusable APIs from copied bytes. It relocates only filesystem roots/import resolution: C:\NAQSHKASH to dependencies/, and C:\FYP Dataset Generation to external/Talamat/. A pathlib proxy is scoped to copied project modules, and explicit importlib loads are redirected too. Working-project imports are rejected. No canonical, preprocessing, recognition, T1/F1/R2/R3 logic is rewritten or monkeypatched. The original DevelopmentDataset remains development-only and is never used to load a held-out split.

`verify_freeze.py --replay-approved-development` checks all sealed hashes and reproduces the 2800 already-approved decisions from derived cached-evidence fixtures, comparing exact decision/finding hashes and outcome counts. A 16-case saved-raw post-OCR smoke check also reproduces R2/R3 through the copied T1/F1/R2/R3 APIs. This is integrity preflight: no image pixels, D1 execution, E1 inference or new development experiment. Fixtures are cryptographically bound to approved source artifacts. Full raw development outputs remain preserved at their original locations and are not runtime dependencies.

## ACCEPT and carpet generation

R3 allows ACCEPT if at least one supported complete canonical interpretation exists. It does not require uniqueness and does not establish physical alignment. Thus an ACCEPT may have multiple distinct carpet readings. **Do not send an ambiguous reading to carpet generation or select the first candidate.** Separate authorized alignment resolution or human confirmation is required.

The held-out reporting wrapper marks `carpet_generation_ready=false` and `selected_carpet_reading=null`; this does not change the frozen R3 outcome. Even a unique text-model reading is not independent physical alignment evidence. REVIEW is not definitive REJECT. No autonomous invalid-image rejection claim is made.

## Held-out procedure

See ../heldout_evaluation/README.md for exact commands. There are no compatible saved v2 validation/protected-test observations at freeze. Manual frozen CPU inference is required, then saved-observation scoring. Validation and protected test each contain 600 images / 150 independent parents. Global sample ID, image hash, parent and semantic-key ownership is disjoint across all three splits.

No held-out inference or evaluation is executed during release preparation. Validation is run once, then protected test once only after explicit separate authorization and validation review. Release/adapter code must not be tuned after exposure. If a genuine implementation defect is discovered, stop, preserve all partial/completed artifacts and report it; do not silently change the frozen release.
