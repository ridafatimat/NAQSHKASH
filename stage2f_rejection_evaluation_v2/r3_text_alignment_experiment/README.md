# R3 Stage 1: text-space alignment feasibility

Implementation is ready; **the full 2,800-image saved-development evaluation has NOT been run**. No image bytes, model inference, new rejection rules or new OCR caches are needed. The evaluator discovers one completed R2 run under the existing metrics folder and verifies its shared raw lineage. If multiple R2 runs exist, use the optional --r2-dir argument explicitly.

## Exact conventions and what is unknown

Inspected frozen functions: talim/talim_rules.py::canonical_row, stage2f_negative_rejection/src/talim_validator.py::validate_aligned_row, data/tier_dataset.py, data/collate.py, and the original FrozenPipeline.observe. Existing integration.py, R2 adapter/evaluator, development loader and R3 fidelity projection are reused without modification.

- canonical_row uses one-character trusted symbols. Each run has width max(symbol length, encoded-count length), puts the symbol at the run start, right-justifies its encoded count, and rstrips every completed tier. Current encodings 1..259 are nonempty and blank-free. Therefore run width equals token length; the canonical count line is contiguous, begins at column zero and ends at the row width. The first canonical symbol also starts at zero. Empty optional symbol tiers are empty strings.
- Training removes **only leading ASCII spaces** from tier labels and stores leading_offset in label metadata. Canonical text has already removed trailing blanks. Internal spaces remain meaningful. Runtime resize preserves crop aspect ratio at height 64; the page inference loop decodes CTC strings without explicitly lstripping/rstripping them or recovering the label's offset.
- D1 boxes share a processed-row horizontal extent; strings do not retain per-character pixel positions or reliable original leading cell offsets. OCR can insert/delete characters and spaces. T1/R2 do not reconstruct horizontal alignment. Source geometry is **not** determined by the available strings.
- The canonical source comments still describe the multi-character alignment convention as pending domain confirmation. This experiment tests that existing implementation; it does not assert a new historical Talim convention.

## Candidate contract

The hypothesis is **literal, label-style leading-stripped decoded content plus missing leading offsets**: preserve every supplied character and space, and add only nonnegative ASCII-space prefixes. No characters, missing counts, inner spaces or trailing blanks are repaired. If a nonempty observed string already starts with blanks, it is recorded unchanged but marked unsupported: the cache does not establish whether those blanks encode a relative layout offset or common crop padding, and normalization could remove observed characters. They are never silently lstripped or treated as a known physical offset. Training expects leading-stripped labels, but actual CTC output is not guaranteed to follow that contract. Missing optional upper/lower keys and empty strings mean an empty optional tier for this text hypothesis; whitespace-only optional tiers are unsupported rather than silently made empty. Missing/empty count OCR, control whitespace, malformed layouts and unavailable D1/OCR are unsupported.

For any *passing canonical reconstruction*, common translation can be removed. The count offset must then be zero by canonical grammar, not by an observed crop position. This is necessary-condition pruning of possible valid reconstructions; **it does not assert the printed count starts under the first predicted symbol**. Both symbol tiers' offsets are searched; neither is arbitrarily chosen as the physical anchor. At least one first visible symbol must lie at canonical column zero.

If W is the observed count-string length, a nonempty symbol tier of length L can only fit at offsets 0..W-L. Enumerate both origin branches: upper starts at zero and lower varies; lower starts at zero and upper varies. Deduplicate (0,0). Empty tiers have no distinct offset choices. Observed leading blanks are an unsupported representation, not normalized into this search. These bounds and branches contain **every passing reconstruction under the stated leading-stripped prefix-only model**: no source cell width, renderer layout, label, mutation category or expected row count is used.

Default computational limits are added offsets <=128 and <=258 tested candidates/row. They are resource budgets, not confidence thresholds or learned grammar. A natural domain outside either cap is marked incomplete. Both bounds are recorded and may be overridden manually; there is no automatic sweep. Unknown/ambiguous canonical-validator results also prevent a contradiction claim. All successful offsets, inferred runs/totals, failure-reason counts and up to six illustrative failing candidates are retained. The failing examples are sampled evidence, not a claim that only six candidates were searched.

Every tested candidate calls the unchanged validate_aligned_row(upper,count,lower) with **no expected_cells argument**. For multiple rows, intersect totals inferred solely from successful candidate runs. The output records multiple possible totals, row ambiguity, and the number of page-level candidate combinations sharing a total without enumerating their Cartesian product. A single candidate in a capped search is not proven unique.

Completeness has two separate meanings:

1. **Conditional text-model completeness:** all offsets that could yield a strict canonical reconstruction of these exact strings were covered and the validator was decisive.
2. **Image-geometry completeness:** always false. Offset search cannot certify faithful OCR, distinguish a missing printed count from an OCR omission, or restore actual source geometry.

A passing candidate proves existence of a text interpretation, not source-image validity. No passing candidate with complete conditional coverage proves only a contradiction in this literal representation. Incomplete/unsupported searches cannot prove no valid interpretation. A complete contradiction in any row suffices for a conditional page contradiction; disjoint row totals require exhaustive decisive row sets. No ground-truth total selects a candidate.

## Manual evaluation in VS Code PowerShell

```powershell
Set-Location 'C:\NAQSHKASH'
$T = 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_text_alignment_experiment'
$Python312 = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$env:PYTHONPATH = "$T\src;C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\src;C:\NAQSHKASH\stage2\positive_edge_cases\test_dependencies"
$Out = "$T\runs\$(Get-Date -Format 'yyyyMMdd_HHmmss_fff')"

# Integrity/schema checks only; no candidate enumeration.
& $Python312 -B "$T\src\evaluate_text_alignment.py" --preflight
if ($LASTEXITCODE -ne 0) { throw 'Stage 1 preflight failed.' }

# Analyze saved text for all 2,800 development images; no inference/image reads.
& $Python312 -B "$T\src\evaluate_text_alignment.py" --output-dir $Out
if ($LASTEXITCODE -ne 0) { throw 'Stage 1 evaluation failed.' }

Get-Content "$Out\report.md"
Write-Host "Detailed metrics: $Out\metrics.json"
Write-Host "Per-image evidence: $Out\per_image.jsonl"
Write-Host "Comparison: $Out\comparison.csv"
```

The output directory must be new and below this experiment's runs folder. With multiple completed R2 runs, append --r2-dir 'C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\metrics\ACTUAL_R2_DIRECTORY' to both commands. Do not substitute a validation/protected run. The strict loader enforces the exact frozen development IDs, manifest, 700 parents and 2,800 observations. Candidate-incomplete/unsupported pages are included and reported, not dropped.

Optional resource bounds: --max-offset 128 --max-candidates 258. Any changed bounds are written into output provenance. Inputs stay unchanged. Identical runtime text/availability pages share in-memory computations; no duplicate OCR cache is written.

Lightweight tests:

```powershell
& $Python312 -B -m unittest discover -s "$T\tests" -v
if ($LASTEXITCODE -ne 0) { throw 'Stage 1 tests failed.' }
```

## Measurements, counterfactual and leakage prevention

The default text is R2's saved T1-interpreted tiers, not annotation text. Availability comes from saved D1/E1 observations. Analysis receives only text, availability and computational budgets. Image identifiers, validity, source layout, categories, mutation parameters and source pair identities enter only later, in offline scoring/aggregation.

The evaluator preserves original R2 decisions. It computes a separately named, **diagnostic REVIEW-only counterfactual**: ACCEPT becomes REVIEW for a decisive, conditionally complete literal-text contradiction; prior REVIEW/REJECT remain unchanged. No definitive rejection is added. This experiment does not install a production adapter.

Metrics distinguish witnessed alignments, no witnesses in tested search, complete-model contradictions, resource/validator uncertainty, unique/multiple row and page candidates, and common inferred totals. Breakdowns include validity, invalid category, clean/illumination, every parent, OCR availability and offline pattern fidelity. Parent counts are distinct within a condition and may overlap across conditions; illumination variants are not independent parents.

Targeted offline cohorts independently identify R2 unsafe accepts, faithful-pattern unsafe accepts, preserved reversed-token unsafe accepts, collapsed internal count gaps and substituted wrong-tier cases. The background counts 788, 507 and 429 are reference cohorts, **not desired remaining-ACCEPT results**. No outcome of 48 invalid accepts or any other improvement is imposed. Report valid controls gaining REVIEW alongside reduced unsafe acceptance. Runtime offsets never use these cohort labels or reference counts.

Original source labels and mutation details are not duplicated into evidence files; only offline scoring flags and lineage/category labels accompany text-candidate evidence. Historical invalid-image CER fields remain untouched. Completion hashes, source-run hashes, code hashes and search budgets make the run reproducible. No historical integrity lock is weakened or rewritten.

## Stage 2 dependency

Stage 2 design should follow the **measured Stage 1 report**, not a presumed improvement. Count-order errors may be exposed if the exact recognized glyphs and run spacing rule out every offset; alternative legal tokenizations may instead pass. Retained middle gaps can contradict the literal count representation, but leading/trailing omissions or collapsed gaps may admit valid-looking alternative text. Wrong glyph substitutions and lost characters cannot be repaired by adding prefixes. Synthetic tests demonstrate both false text contradictions on a valid image's space-collapsed OCR and plausible reconstructions of known-invalid source geometry.

CTC extraction can investigate character/run positions; image components provide independent glyph identity and readable-region/absence evidence. The relative need for either or both depends on measured ambiguity, coverage, unresolved cohorts and valid-control REVIEW burden. Neither should be promoted to definitive image rejection from text-only evidence.
