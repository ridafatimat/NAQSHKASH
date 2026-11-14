# Conservative glyph observability gate

Visual QA of the first provisional preflight found that exact grayscale equality
was insufficient: a Y/C-style double-dot mark could be a legal opposite-tier
glyph even though its raster shades differed at48. Those provisional labels are
NOT accepted. The120-image provisional set is preserved under
dataset/quarantined_preflight_v0 with a QUARANTINED marker; the validator refuses
it. It is not part of the final preflight and must not be trained/evaluated as a
verified invalid collection. Its30 source families are also reserved from future
protected generation. No previous project dataset was changed.

The local font contact sheet is reports/symbol_role_reference.png. Conservative
visually equivalent/near-equivalent role pairs are B/X, C/Y, E/Z, G/[, I/\\,
J/], K/^, L/_. These pairings are DATASET ADMISSION EXCLUSIONS, not runtime
character correction rules, and do not change T1 or canonical Talim membership.
They cover dot/bar/hook shapes that should not be called visibly invalid merely
because of font hinting or antialias differences. Treating them as possible legal
alternatives is fail-closed: it can exclude weak negatives rather than invent one.

The wrong-tier mutator excludes these aliases. Independently, the raster parser
also considers their possible legal labels when searching all canonical rows.
Thus a bad mutator cannot pass one merely by assigning INVALID. Wrong-tier
examples can use only the remaining clearly distinct patterns, such as mark/bar
relationships with dots on the opposite side, triangles of dots in different
orientations, distinct circles/angles or dot-plus-stroke arrangements. The
actual glyph and all allowed opposite-tier alternatives are visible in the audit.

The final small preflight is regenerated only after these stricter gates and
focused regressions pass. All provisional images/annotations are preserved for
traceability. No full generation, model inference, training or threshold tuning
is performed. This conservatism belongs to generation QA; F1/T1 remain unchanged.
