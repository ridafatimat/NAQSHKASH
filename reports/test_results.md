# Stage 2D test results

Final comprehensive run: **109 passed, 0 failures, 0 errors, 0 skipped, 1 explicitly deselected**. JUnit elapsed 21.69 seconds.

The 29 Stage 2D tests cover exact role-search alternatives, ambiguity rejection, unusual valid layouts, rows, extent-limited grouping, dots and source-pixel preservation, crop boundaries, tiny components in the wrong intended tier, CTC/spacing diagnostics, deterministic frozen inference, parent isolation, source-label and crop hashes, fine-tuning eligibility, historical preservation and reproduction namespaces. Relevant unchanged Stage 2B/2C and Stage 1 preprocessing, model-forward and CTC numerics tests are included. No training loop test was run.

One old Stage 2C test, test_software_and_protected_project_are_unchanged, assumes equality of the entire project tree and therefore rejects any authorized new stage directory. Its source was not changed; it is explicitly deselected. Stage 2D verifies inclusion and exact hashes of all 7,051 protected historical files, all 486 inventoried Stage 2C artifacts and the frozen Stage 2D runtime files. This replaces the obsolete whole-tree equality assertion without relaxing protection of any old file.

Initial focused run: 11/12 passed; disconnected-dot edge_00001 failed because a 5px gap exceeded a 4px limit. The training-parent gap correction is recorded, with original code/results/configuration and failure evidence preserved in reports/attempt1. The focused rerun passed 12/12 before final selection.

Evidence: test_command.ps1, focused_tests.txt, tests.txt, tests.xml and tests.json. Temporary files stayed under this module, bytecode/cache writes were disabled, and the local Stage 2A pytest dependency directory was read only. No historical test source, model, labels, datasets or Git state was changed.
