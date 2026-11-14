# Stage 2F — isolated tier-aware interpretation prototype

T0 preserves E1 greedy predictions and F1 v2. T1 optionally interprets verified
font aliases using reliable saved D1 role metadata, then calls the unchanged
canonical validator and F1 v2. E1, D1, vocabulary/blank0, annotations and previous
results are read-only. No training or inference module exists in this workspace.

The new all-vocabulary font audit covers65 characters/2080 pairs at24,32,48,64,96
source pixels. B/X is exact at48 but not32; C/Y is not exact at48. Only B/X is in
the project-scope mapping. T1's default requires confirmed source48, matching
local font hash, grayscale L and integer lt rendering. Unknown fonts/renderings
or missing/conflicting tier context do not get forced replacements. A character
already legal in its tier is never changed. No partial tier repair is performed.

This is label interpretation after CTC, not a new model or logit decoder. All raw
text, F1 reasons/evidence, edits and interpreted decisions remain available.
Passing syntax is not proof of exact transcription or source-image validity.

See reports/tier_context_audit.md and reports/glyph_ambiguity_audit.md for actual
inspection findings; galleries/glyph_comparison.html contains internal rendered
comparisons. No font file is copied or distributed.

RUN_TIER_AWARE_IN_VSCODE.md contains the exact manual commands. The CLI permits
only the saved776-image development baseline. The exposed434 partition cannot
be selected. Neither candidate's full evaluation has been run by Codex; metrics
directories stay empty until manual execution. Tests use synthetic unit fixtures
and four previously identified DEVELOPMENT observations only.

Modules serve these purposes:

- audit_glyphs.py: bounded internal font audit and verified mapping preparation.
- tier_aware_decoder.py: pure, configurable per-tier interpretation API.
- contextual_pipeline.py: role/rendering provenance checks and optional F1 adapter.
- measurements.py: evaluation-only CER/exact/parent metrics and correction assessment.
- evaluate_tier_aware.py: manual T0/T1 saved-development evaluation.
- compare_tier_aware.py: paired metrics and error galleries from actual saved runs.
- verify_artifacts.py and runtime.py: frozen input/experiment hashes and shared I/O.

Audit/preparation scripts refuse to overwrite existing generated artifacts. Do
not rerun audit_glyphs.py into this frozen workspace; create a separately reviewed
version if source font/rules/configuration change. Configuration .yaml files use
the JSON subset of YAML1.2; no PyYAML dependency is introduced.
