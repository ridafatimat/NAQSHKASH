# NAQSHKASH root integration audit

Stage-1 was integrated into the Rida branch from the reviewed snapshot at
7fdc1d9441c8e52967c86f3e5e7a5009e31e7d3f using GitHub APIs only. Component commits
record the real reorganization; existing contributor history is preserved.

| Conflicting paths | Resolution |
| --- | --- |
| preprocessing/crop.py | Keep the newer root implementation, including scanner-border removal and polarity fixes. |
| preprocessing/__init__.py | Keep all existing exports, including normalize and model-preparation APIs. |
| Other preprocessing modules | Root and Stage-1 blobs are identical; retain one root copy. |
| data/vocabulary.py | Preserve the row class and canonical fallback; add explicit for_tiers factory and strict tier label builder. |
| data/encoder.py | Keep the richer existing encoder; it already supports the tier loss contract. |
| data/ctc_decode.py | Keep structured/tolerant row decoding; add strict tier decoding and an alias to the shared collapse algorithm. |
| Existing tests | Preserve root tests; port only portable dataset paths and add missing tier-layout checks. |
| README_AYESHA.md | Preserve the newer root documentation and append vocabulary integration guidance. |
| Saved results | Move unchanged Git blobs to results/stage1/; scoped Git attributes preserve bytes. |

Training, evaluation and overfit entry points explicitly construct tier
vocabularies, preserving the saved 66-class checkpoint mapping and blank token.
Ayesha's default 67-class row mapping, caller ordering, serialization, parsing,
normalization and model-ready preprocessing remain available.

No implementation remains nested under stage1/. Talim references, fixed config,
tier datasets, CRNN, training, evaluation, tests and notebooks use root paths.
The cloud workflow runs the complete combined root suite, including original
contributor tests, training/resume and evaluation fixtures, vocabulary
compatibility checks and original result-blob integrity assertions.

The previous nested pipeline passed 77 tests with 28 external-data checks
skipped. Current root verification is recorded in the pull request and GitHub
Actions, rather than presenting those earlier numbers as root test results.
Full-dataset GPU training and final evaluation were not rerun during integration.

The saved full-validation and test metrics describe held-out synthetic tiers,
not real handwriting. Exact training reproduction requires the original
external corpus and approved manifests. Datasets, checkpoints, ZIP backups and
credentials are excluded from Git. The checkpoint release remains unpublished
pending owner approval.
