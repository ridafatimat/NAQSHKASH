# Stage-1 GitHub transfer audit

The Stage-1 pipeline was transferred through GitHub APIs from the previously
reviewed FYP-Dataset_Generation pull request #6 into NAQSHKASH, based on the
latest Rida branch. Each component is introduced in its own current commit.
Existing Rida history and root-level source files are preserved; only the root
README link and artifact ignore rules are extended.

The tested pipeline remains under stage1/ as a reproducible snapshot. The two
original Talamat rule reference files are included under talim/reference/ because
NAQSHKASH does not contain the companion Talamat application. Their source is
recorded in talim/reference/README.md. No domain mappings are re-created.

Original saved evaluation JSON/CSV, training metrics, numerical failure logs and
integrity metadata are transferred with their original Git blob hashes. Source
file hashes are checked after every component tree is built. The README and
rule reference path are adapted for this destination. No datasets, model
checkpoints, ZIP archives, credentials or generated inventories are included.

Before transfer, the exported pipeline passed 77 tests with 28 external-dataset
checks skipped, using Python 3.12, CPU PyTorch 2.5.1 and NumPy 1.26.4. Original
archives passed CRC checks, saved metrics were checked arithmetically, and
source/checkpoint metadata credential-pattern scans had no findings. Those are
the prior export's test results; the remote transfer has its own GitHub Actions
check. Original full-dataset evaluation is not rerun.

The evaluation remains held-out synthetic-tier performance, not real-handwriting
accuracy. Exact training reproduction requires the original external dataset
and approved manifests. The model release remains unpublished pending approval.

This transfer uses GitHub API operations without local checkout or file edits.