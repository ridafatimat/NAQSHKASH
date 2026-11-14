# Manual R2 development evaluation

Run in a VS Code PowerShell terminal. This reuses the completed raw run and R0/R1 results discovered during diagnosis. It does not execute OCR, preprocessing, T1 decoding or an image-processing pass. Only development metadata and saved observations/results are read. No package installation is needed beyond the existing evaluation environment.

```powershell
Set-Location 'C:\NAQSHKASH'
$R = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$R\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$Out = "$R\metrics\$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')_R2"

& $Python312 -B "$R\src\evaluate_r2.py" `
  --split development_design `
  --raw-dir "$R\metrics\20261106_204802_280_raw" `
  --r0-dir "$R\metrics\20261106_204802_280_R0" `
  --r1-dir "$R\metrics\20261106_204802_280_R1" `
  --output-dir $Out
if ($LASTEXITCODE -ne 0) { throw 'R2 failed; stop and inspect the error.' }

Get-Content "$Out\summary.json"
Get-Content "$Out\comparison.csv"
Write-Host "Full comparison: $Out\comparison.json"
Write-Host "Category results: $Out\by_group.json"
Write-Host "Parent results: $Out\by_parent.json"
```

`results.json` / `per_image.csv` retain raw and interpreted OCR, reference metrics only for valid images, previous evidence, R2 findings and decisions. `comparison.json` contains R0/R1/R2 summaries and parent/group metrics, per-category transitions and regressions against both baselines. `completion.json` seals the result hash. REVIEW is not counted as definitive rejection. Existing results are read-only; the output directory must be new. The original R0/R1 loader is intentionally unchanged and will not load R2; use the provided R2 comparison outputs.

Optional focused tests, without inference:

```powershell
& $Python312 -B -m unittest discover -s "$R\tests" -p test_r2_structural.py -v
if ($LASTEXITCODE -ne 0) { throw 'Focused R2 tests failed.' }
```

Limitations: R2 can only downgrade an ACCEPT with an internal decoded count gap to REVIEW. It cannot recover a gap omitted by OCR, infer edge counts, verify token boundaries, prove wrong-tier glyph identity or create independent image-level semantic rejection evidence. Do not present its eventual review increase as successful definitive rejection. No R2 evaluation was executed during implementation.
