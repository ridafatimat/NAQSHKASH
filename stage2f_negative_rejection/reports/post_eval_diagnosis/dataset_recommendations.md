# Dataset and Stage 2F recommendations

This exposed evaluation is diagnostic evidence only. No dataset generation,
policy/threshold tuning, model update or new experiment was performed.

1. Keep this release as a separate artifact/schema validator regression suite.
   Do not train a valid/invalid image classifier using its negative fixture label.
   184 files have valid content with annotation-only defects;4 more have valid
   content inherited from an RGB-source representation fixture. Invalid JSON/TXT,
   duplicate IDs and missing metadata cannot be learned reliably from unchanged
   pixels. One missing image must remain an artifact record, not an OCR target.
2. Preserve current frozen manifests/results. Review the new diagnostic labels
   before creating a separately versioned correction manifest. Correct the
   wrong_dimensions description (black replacement, not resize), scanned blanks
   (actually white), and inherited representation labels. Keep historical and
   diagnostic classifications in separate columns and document the denominator.
3. Before collecting/training rejection data, define separate tasks: authentic
   invalid notation, unsupported notation/layout, unreadable/blank/corrupt input,
   valid difficult Talim, and annotation/artifact defects. Retain uncertainty
   rather than assigning invalidity to unreadable or disputed handwriting.
4. After authorization, author independent base pages, then apply supported
   image-visible mutations and actually rerender/edit the pixels. Candidate
   cases under existing canonical rules: visible unsupported glyphs; symbol in
   an incompatible tier where the glyph/context is independently distinguishable;
   missing count/symbol relationships or contradictory row structure verified
   by the domain expert. Do not invent grammar restrictions or label an isolated
   B/X dot invalid merely by changing its Unicode ID: their visible masks match.
   Keep separate matched valid controls and retain the exact mutation evidence.
5. Do not treat annotated cells_per_row, duplicate IDs, numeric types or JSON
   field omissions as visible semantic violations unless the deployed task
   independently supplies that information. Encode any required context explicitly
   and annotate which evidence makes a case objectively invalid.
6. Obtain domain-expert image-level adjudication, an authentic intended
   transcription, row/tier/component geometry and an invalidity reason/evidence
   for each new base. Keep clean valid positives, hard short glyph-only tiers,
   long/complex valid rows, unsupported representations and unreadable captures.
   No length-dependent rejection rule is justified by this diagnosis.
7. Split by independent base lineage BEFORE augmenting. Group exact/grayscale
   duplicates, semantic duplicates, all mutation siblings and scan variants.
   The current51 source records are not51 independent parents. Track content
   template IDs separately from independent lineage. Report parent macro and
   image/variant-weighted metrics separately. Collect enough independent valid
   and invalid families in development and held-out evaluation before deciding
   whether a classifier is statistically justified; do not choose an arbitrary
   sample count or train from copies of this single family.
8. Review the B/X role-context issue as an interface/design question before any
   training. Possible future experiments, only after approval and on permitted
   development data: preserve tier role/page context at recognition/interpretation,
   or investigate a provenance-aware alias representation. Do not silently map
   X to B, suppress the validator reason or change F1/F2 based on this exposed
   evaluation. Similar-looking [ / G and _ / L need their own reviewed evidence;
   they are not exact font-mask aliases.

Recommended next step: review this diagnosis and approve an image-level dataset
specification plus a separate role/context investigation plan. Keep existing
canonical rules and F1 v2 uncertainty policy unchanged. Only after that approval
should a new versioned dataset be generated/adjudicated, a parent-disjoint protocol
be frozen, and any model/classifier experiment be considered. None of the current
434 exposed evaluation images can be relabeled as a fresh unseen test.
