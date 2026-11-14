# Canonical Talim rule audit — generation gate

Scope: the project's existing fixed-font, three-tier canonical implementation.
AUTO_VERIFIED means independently established within this explicit code/render
scope, not a claim of expert adjudication of all authentic historical Talim.
No grammar, font, production renderer, D1, E1, T1 or F1 source is modified.

Inspected sources:

- C:\FYP Dataset Generation\dataset_tools\talim_rules.py: source-derived
  symbol_direction/is_valid_symbol_direction, encode_count, canonical_row/text.
- C:\FYP Dataset Generation\config.py and Utils\talamNumConverter.py:
  original symbol pools and authentic count mapping (1..259).
- C:\FYP Dataset Generation\dataset_tools\talim\generator.py:
  generate_talim, controlled/random rows, authentic counts, fixed row totals.
- C:\FYP Dataset Generation\dataset_tools\talim\renderer.py:
  _cell_metrics and render_talim_text, grayscale48, integer lt anchor,
  exactly three tier lines plus separator, cell and row spacing.
- C:\FYP Dataset Generation\dataset_tools\validate_datasets.py:
  check_sample and artifact checks, canonical reconstruction and directions.
- C:\NAQSHKASH\talim\talim_rules.py: linked project rule API; its
  canonical_row explicitly marks multi-character alignment pending domain confirmation.
- Retained original positive/negative supplement generator:
  C:\Users\hp\OneDrive\Documents\ChatGPT\FYP\edge_audit_staging\dataset_tools\edge_case_audit.py,
  make/write_sample/negatives. The deployed dataset_tools folder lacks that script.
- Stage2F reports/post_eval_diagnosis and tier-aware glyph_ambiguity_audit:
  invalid JSON is not image invalidity, B/X visible aliases, one-family problem.
- Existing talamat_scan_scripts_v1_1/talamat_scan_augment.py: apply_recipe.

| Rule ID | Existing rule and source | Valid / invalid demonstration | Visible evidence and independent verification | OCR/segmentation limitations |
|---|---|---|---|---|
| CANON_SYMBOL_DIRECTION | talim_rules.is_valid_symbol_direction and canonical_row put exclusive-UP and exclusive-DOWN symbols in their defined tier | Valid UP glyph above count; same glyph moved below count when it has NO lower-tier raster equivalent | Rerender changes both locations. Inverse raster matcher considers every legal lower glyph and all aliases; no canonical row parse exists | Requires correct page tier context; a tight crop alone loses role. OCR-only wrong-tier output is never ground-truth invalidity |
| CANON_COUNT_ENCODING | encode_count accepts integer1..259 and produces the authentic token set; canonical_row places one token per run | 11 -> ja is valid; aj under that same single symbol is not any authentic token | Reverse a two-glyph token only when absent from all259 encodings. Parser tries both one- and two-glyph counts and every symbol boundary, so it cannot simply assume the generator's run grouping | Must preserve token order and symbol locations. Recognizer normalization can lose grouping; not promised detectable by current OCR |
| CANON_REQUIRED_SYMBOL_COUNT | canonical_row requires a supported positive count for EACH symbol run; renderer requires a nonempty count line | Symbol with ja; remove the entire ja while preserving the symbol and other instructions | Actual center cells become white. Parser tries all supported counts/boundaries and finds no complete interpretation; another valid instruction is not assumed absent merely from the assigned label | Blank count cells may be missed by segmentation; geometry is essential and proof uses fixed rendering scope |
| AUTHENTIC_COUNT_RANGE (guard, not standalone image category) | validate_count/encode_count: strictly typed integer1..259 | 259 valid; 0,260,boolean/string invalid as STRUCTURED values | Those JSON changes alone have no image evidence. No nonexistent 300/fallback glyph is invented | Excluded from image-invalid generation without authentic visible notation |
| CANON_ROW_STRUCTURE (verification, not extra quota category) | canonical_text and renderer use upper/count/lower with row separator; canonical_row defines run columns | Existing canonical row; malformed missing-count case above | Inverse parser covers every active column and intersects possible row totals across logical rows | Arbitrary unusual spacing or parser-unsupported dialect is not automatically invalid |

Independent verifier does not import mutations, does not read mutation category
or validity labels to classify pixels, and receives no original symbol sequence.
It uses the declared rendering GRID (not a guessed OCR layout) to recover each
cell against the complete supported glyph atlas. It enumerates all role-legal
symbol aliases and every authentic1..259 count token, rejecting a sample only
when no complete canonical interpretation exists. Control validity, meaningful
pixel change and exact raw-annotation/raster agreement are additional gates.
This proves canonical-format image invalidity, not a universal impossibility
under undocumented dialects, arbitrary fonts/scales or alternative page layouts.

B/X at48 is an explicit excluded wrong-tier ambiguity: both label choices have
the same visible mask, so moving/relabeling that mark can form valid notation.
The mutator screens aliases and the independent parser independently tries them.
The atlas is built from the local font, which is read in place, never copied.
Visual QA additionally excludes C/Y, E/Z, G/[, I/\\, J/], K/^ and L/_ lookalikes.
The parser considers these as plausible legal alternatives despite shade/hinting
differences. The first provisional set was quarantined when this stronger
observability requirement failed; see glyph_observability_audit.md. Wrong-tier
parents now use only the14 remaining visibly distinct supported shapes. These
are dataset-admission constraints, not changes to the canonical grammar or T1.

Alignment/spacing disputes, adjacent-identical run conventions as an isolated
negative category, numeric row totals known only from metadata, ambiguous O,
foreign/fallback characters and parser-unsupported layouts are NOT certified
semantic-negative categories. Alignment's historical authenticity remains
NEEDS_DOMAIN_REVIEW. Missing marks that yield another valid count/symbol or
leave a valid alternative parse are EXCLUDED. Unknown raster cells are
UNSUPPORTED_OR_AMBIGUOUS / NEEDS_DOMAIN_REVIEW, never automatic INVALID.
