# Rerun corrected F1 from saved development outputs

The v2 correction changes image decision mapping only. The canonical validator
still flags wrong-tier and unsupported OCR glyphs with
`DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH`. OCR-only violations now mean
`REVIEW_UNCERTAIN`. They do not independently establish source-image invalidity.
The existing pixel adapter's `unreadable` and `no_ink` conditions still mean
`REJECT_INVALID`, scoped as invalid input artifacts rather than Talim-content
proof. D1 uncertainty and missing artifacts keep their existing review handling.
F0 actions/schema are unchanged. F2 uses the same v2 mapping, with its existing
separate confidence policy. No confidence/length thresholds were changed.

Each F1/F2 decision retains original reason codes and validator evidence plus
`reason_details` (source, tier/row, predicted characters/text, evidence scope and
evidence basis), `syntax_validation`, policy version and independent-image flag.
Multiple findings are retained; existing independent invalid artifact evidence
takes precedence over OCR-only uncertainty. Generic caller-supplied invalidity
flags and evaluation ground truth are never used as runtime evidence.

Only the four identified saved development observations were replayed for
regression tests. The full 776-image development set was not evaluated, and no
434-image evaluation outcomes were examined. Historical baseline/F1 result bytes
were hash-protected. The original validator and Talim rules are unchanged.

In VS Code, open C:\NAQSHKASH and use PowerShell. Run this block yourself:

```powershell
Set-Location 'C:\NAQSHKASH'
$F = 'C:\NAQSHKASH\stage2\stage2f_negative_rejection'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH="$F\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$Output = "$F\metrics\20261106_154728_812_dev_F1_policy_v2_$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"
& $Python312 -B "$F\src\evaluate_rejection.py" --baseline-dir "$F\metrics\20261106_154728_812_dev_baseline" --candidate F1 --decision-policy "$F\configs\f1_ocr_uncertainty_policy_v2.yaml" --output-dir $Output
if ($LASTEXITCODE -ne 0) { throw 'Corrected F1 saved-output evaluation failed' }
Write-Output "Corrected F1 outputs: $Output"
```

This reads saved OCR observations; it performs no OCR inference or model loading.
The timestamp gives a new output folder; existing folders are never overwritten.
Output files: `results.json`, `summary.json`, `groups.json`, `per_image.csv`,
`completion.json`. JSON retains nested evidence; CSV uses JSON strings for nested
fields. `results.json` records the v2 configuration/path/hash separately from
the original baseline lineage and optional F2 confidence policy.

Expected regression behavior: each of the four wrong-tier cases moves from
REJECT_INVALID to REVIEW_UNCERTAIN, with its original reason and evidence intact.
No new ACCEPT is intended. Full aggregate counts remain unmeasured until you
run the command; do not report predicted aggregate counts as executed results.

For F2 later, use the same `--decision-policy` argument, `--candidate F2`, and
the existing confidence `--policy` argument. No calibration was run by this
correction. Review the development results and freeze this version before any
application to the evaluation partition.
