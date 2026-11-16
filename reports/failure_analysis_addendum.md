# Post-freeze diagnostic addendum

The pre-implementation failure_analysis.md remains byte-identical to its diagnosis-complete hash.

Residual failures are concentrated in two valid lower-symbol parents: edge_00310 (tuning, four variants) and edge_00307 (exposed-validation, clean only). Full-resolution gallery inspection confirms splitting of disconnected lower glyph parts into extra count rows. This is a component-grouping/row-structure failure; no OCR is scored on those wrong crops. It does not establish a dynamic-programming search defect.

The long perspective rejection edge_00323 leaves unassigned foreground under the proposal projection threshold. It is a valid positive page and remains a coverage failure, not a successful rejection of an invalid document. It previously contributed 479/758 character errors; its exclusion explains much of the conditional CER reduction. No labels/spaces were rewritten, and no candidate was tuned on this observed outcome.

Scale/merge signatures in the original diagnosis overlap; they are evidence, not disjoint causal counts. Geometry/background artifacts are not blamed without an isolating experiment. Remaining long-sequence mistakes on aligned crops are frozen recognizer/spacing/distortion findings; they were not repaired by choosing text-compatible crops.

The 90-page comparison and intended-tier component audit strengthen preservation evidence. Their ideal-ink and >=3px scope does not certify every subpixel/damaged mark or every real-world capture. The five wrong acceptances across all cohorts prevent automatic deployment approval despite zero wrong scanned validation proposals.
