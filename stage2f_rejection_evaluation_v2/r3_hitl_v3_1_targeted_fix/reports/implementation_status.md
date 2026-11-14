# Targeted-fix implementation status

Separate revision `targeted_pitch_fix_1`: corrected unsupported pitch assumptions and selected-carpet scoring only. Original R3.1 implementation/run and frozen R3/D1/E1/Stage 1 are preserved. Candidate policy, monotonicity, geometry checks, minimum support, raster allowance and candidate tolerance remain unchanged.

Three cached cases and synthetic regressions support the local correction. Full development metrics remain pending the user's one final manual run. No historical results are overwritten or rescored, no OCR/training/held-out evaluation or Git/GitHub operations occur, and no new architecture or threshold tuning is introduced.

See targeted_fix_diagnosis.md, README.md and the final executed test_results.json. If the final run cannot establish reliable positioning, stop and use the verified R3 fallback.

Final lightweight suite: **71 passed, 0 failed**. No complete development evaluation was executed.
