# R3 CTC alignment availability and feasibility
Date: 2026-11-07. Static source/checkpoint/schema inspection; no CRNN forward pass, alignment extraction or retraining.

## What exists

Inspected models/crnn.py, data/collate.py, data/ctc_decode.py, the original FrozenPipeline.observe in stage2f_negative_rejection/src/evaluate_baseline.py, confidence.py, D1 pipeline_d1.py and geometry.py, the source renderers, and the frozen selected E1 checkpoint. Checkpoint tensors were read for metadata/shapes only; no CRNN was constructed or executed.

Checkpoint: C:/NAQSHKASH/stage2/stage2e_finetuning/checkpoints/selected_model/best.pt
SHA256: 27f2da50795e2fca8f327afefdce10645f3e60e5320060a8cee93994fabdcdf1

The checkpoint is stage2e_v1, step 406. Six 3x3 CNN convolutions feed a two-layer bidirectional LSTM (256 hidden units per direction), followed by a 66-output classifier. The fixed vocabulary has 65 nonblank symbols, including a literal space at index 1; CTC blank is 0. Checkpoint classifier.weight is [66,512], and both LSTM layers/directions are present.

The 2,800 saved observations contain preprocessing status, accepted tier/row boxes, diagnostics, decoded rows, per-tier strings and aggregate confidence. Each of 16,444 recognized tiers saves row, role, box, prediction, resized_width and scalar signals (frames, decoded_tokens, blank_fraction, CTC sequence score, margins, geomean). **No frame logits, probability vectors, argmax paths, character-frame indices or spans are saved.** Sequence scores are marginalizations over paths; they cannot be inverted into a path. Frame counts and strings are insufficient. The D1 transform matrix and processed image are also discarded by this observation adapter.

## Exact output length and nominal CNN geometry

For an actual resized tier width W >= 4:
- all six convolutions: horizontal kernel 3, stride 1, padding 1, so width is unchanged;
- the first two pools: kernel/stride 2, no padding, floor division;
- the remaining four pools: horizontal kernel/stride 1, so width is unchanged;
- **T = floor(floor(W/2)/2) = floor(W/4)**.

The output_lengths helper clamps to at least one, but a physically too-narrow tensor is not made valid by that clamp. FrozenPipeline rejects widths below four before forwarding. All 16,444 saved tier frame counts match floor(resized_width/4), with zero discrepancies. Observed resized widths range from 129 to 11,260.

With input pixel-center coordinates starting at 0.5, the nominal CNN lattice centers are 2 + 4t, and the horizontal receptive-field width is 42 resized pixels. These are convolutional geometry calculations, **not measured CTC glyph centers**. BiLSTM classification depends on the entire tier, so a prediction spike can shift within/across a glyph's nominal region. Quantization, pooling and interpolation introduce additional uncertainty.

## Resize, crop, padding and source mapping

FrozenPipeline:
1. Decode original bytes as grayscale.
2. D1 applies image-only photometric flattening/optional denoising, rotational deskew and possibly projective correction.
3. D1 returns processed pixels and a source-to-processed 3x3 matrix M. The engine proposes row/tier boxes [x0,y0,x1,y1] in the processed image; boxes use shared row-level horizontal ink extent plus padding, and band-derived vertical limits.
4. Crop those pixels; normalize intensities by 255.
5. Resize height Hc to 64 and width Wc to Wr = max(1, round(Wc * 64/Hc)), with PyTorch bilinear interpolation and align_corners=False.
6. Forward a single tier. **There is no batch padding in this saved inference route.** The training/general collator pads on the right with white; its true widths must be honored if later batching is introduced.

In pixel-index-center coordinates, a resized coordinate u maps to crop coordinate (u+0.5)*Wc/Wr - 0.5. Add crop origin x0 (and the analogous y mapping). Use the actual rounded Wr ratio, not merely 64/Hc. Finally apply inverse M in homogeneous coordinates to map to the source image. A projective map depends on y: a time-frame column maps to a region/polygon, not one universal source x. Use an interval spanning the tier height until glyph-local vertical support is independently established.

D1's internal general preprocessing object has margin-crop metadata, but this D1 route returns its full processed deskewed_image to the band engine; it does not pass cropped_image into OCR. Do not add an unrelated margin-crop offset a second time. Canvas expansion/translation from deskew/perspective belongs in M.

Leading blanks: training labels remove leading ASCII spaces and retain a leading_offset as label metadata. The saved page OCR route does not recover a corresponding character-cell offset; it simply decodes each crop. A crop origin is a pixel offset, not the removed-label offset. Character index times cell width is therefore not a valid reconstruction rule.

D1 row/tier IDs and boxes are available. M, deskew affine/canvas translation, processed dimensions/pixels and character positions are not. Fifty-two saved page observations report perspective_applied, but even perspective_applied=False does not certify that deskew was identity. Mapping general saved frame coordinates to source pixels cannot be made exact by guessing missing matrices.

## Ground-truth coordinates for offline measurement

The original renderer returns canvas width/height, not per-glyph boxes. Dataset v2 saves layout (font hash/size, padding, extra-left, cell width, line height, row gap, rows/columns) and mutated raw_display, not an explicit glyph-box array. For this documented fixed-grid source renderer, source cell anchors can be reconstructed:
- x = padding + extra_left + column * cell_width;
- y = padding + logical_row * (3*line_height + row_gap) + tier_index * line_height.

Actual ink boxes require the font's anchor=lt glyph bbox/mask; cell bounds are not ink bounds. Use the **mutated** display and actual font hash, never the valid parent's glyph placements. Illumination variants preserve source coordinates because their only operation is pointwise brightness. Unknown rendering, blur, nonrigid warps and arbitrary scans are outside that exact reference contract.

Thus development frame-to-glyph accuracy is measurable **after extracting model paths and complete coordinate transforms**. Renderer metadata can be an offline scoring reference, not an inference-time grid oracle for R3. It cannot measure current alignment because current paths are absent.

## Minimum proposed extraction experiment — NOT EXECUTED

A bounded diagnostic could select 12 independent development parents, four per invalidity category, with both valid/invalid and clean/illumination images: 48 images. Select representative short/medium cases first and state that long-page behavior remains unmeasured. Extract only the three tiers of the affected logical row, up to 144 tier forwards, with a declared CPU/time budget. Include wrong-tier misrecognition, preserved reversals, internal-gap loss and edge-count absence; selection uses labels only to construct the offline study.

Use the same frozen CPU model, vocabulary, D1 and resize function. An isolated diagnostic wrapper can observe existing D1 return values and register a model forward hook; no frozen source change is necessary. Capture:
- immutable image/checkpoint/software hashes and row/tier IDs;
- original/processed dimensions, M, crop box and dimensions, actual resized dimensions, true widths and any padding;
- float32 per-frame logits or log-probabilities [T,66], argmax path and blank ID;
- standard CTC collapsed-character frame runs/spans, retaining repeat boundaries and literal-space class indices.

Keep this evidence in a separate diagnostic cache, not the existing raw run. Check that recomputed greedy strings match saved strings before comparing alignments. Disagreement is a reproducibility finding; never overwrite or force a match. This is a targeted second inference pass because the missing tensors cannot be recovered from the current cache.

For an initial alignment, use greedy nonblank spans and optionally CTC alignment/posteriors to the model's **own decoded sequence**. Use source glyph boxes only to score associations, with ambiguous edit correspondences marked unscorable. Do not supply nonexistent missing-count targets or force invalid source strings into fabricated CTC targets. A recognition error does not become a correctly localized glyph just because forced alignment assigns it a time.

Repeated characters need separating CTC blanks to survive collapse. Blank runs can occur inside ink or between repeated labels; **blank is not a visible space**. A literal space is a separate model class. BiLSTM context and discriminative spike training do not guarantee optical centering. Report spans, uncertainty, omissions and ordering failures rather than assuming 4t is a center.

## Metrics and provisional acceptance criteria

None of these alignment accuracies has yet been measured:
- signed/absolute source x error against ink-box centers; median and upper-tail error in pixels and normalized by glyph ink width, stratified by tier/category/length;
- recognized-character coverage against all visible reference glyphs, missing/extra characters, ambiguous correspondence rate and transform availability;
- within-token glyph-order agreement and token-boundary agreement, including reversals;
- symbol/count association accuracy against source positions, with unresolved association separate from error;
- false structural findings on matched valid controls, with false REVIEW and false definitive REJECT separated;
- clean/illumination agreement paired by parent; parent-level summaries/uncertainty, not inflated variant sample sizes.

Provisional engineering gates: exact frame/width bookkeeping; verified reversible coordinate transforms; interval/uncertainty support lying within a unique symbol-count association region; correct ordering on the informative pilot cases; zero false definitive structural violations on pilot valid controls; consistent associations across both illumination variants. Coverage and excluded cases must be reported alongside apparent precision. These are candidate feasibility gates, not tuned thresholds or statistical certification. A numerical pixel-error cutoff needs measured distributions and boundary margins; none is justified now. Passing a 12-parent pilot does not certify a rejection policy.

Count order might become recoverable if character spans reliably separate neighboring symbol/count runs. Missing-count absence still needs independent readable-region/occupancy evidence: no CTC emission cannot prove no printed count. Wrong-tier glyph substitutions may remain wrong with perfectly localized frames. CTC geometry and CTC confidence come from the same model and are not independent confirmations of glyph identity.

