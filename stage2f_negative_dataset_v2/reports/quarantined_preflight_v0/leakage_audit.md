# Leakage and source-ownership audit

Preflight parents are newly created using namespace v2_preflight_design_only and
seed20261106 through the existing canonical generator. They are not mutations of
historical parent images. Their semantic keys were checked against a bounded
index containing 1858 distinct old keys, from1380 Stage1 source
records,150 general source records,326 positive-edge source records and two old
fixture content templates. No exact semantic overlap was found. Known Stage2C/D
parents and Stage2E train/development/replay references are included; Stage2F
shares these sources. Old protected-source metadata is used for exclusion hashes
only, never generation/rule-selection/model-performance evidence.

The complete100,000-parent Stage1 and10,000-parent general archives were NOT swept
by Codex. Therefore global exact nonoverlap is not yet established. Full generation
refuses a bounded index and requires the manually created full history index with
expected100000/10000/326 record coverage and source archive fingerprints. It then
excludes prior semantic keys and preflight source keys. The index covers known
canonical sources, not unknown external data or all perceptual near-duplicates.

Preflight has30 independent source families and120 images, all development_design.
All mutation/control/scan siblings have one owner and split. No byte-identical
images were found. Full mode assigns ownership before mutation/augmentation:
70% development_design,15% validation_calibration,15% protected_test. Construction
QA finishes before sealing; post-seal validation is hash-only for protected
image/annotation/parent files, and galleries exclude protected examples.

This is new synthetic canonical content under an audited source distribution,
not independent real-world test evidence. No current exposed evaluation image
has been relabeled as a fresh test. Historical datasets/splits/results and
E1/D1/T1/F1 remain untouched; the scoped preservation lock covers427 files.
