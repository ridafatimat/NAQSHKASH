# Coordinate and association audit
Date: 2026-11-07. Existing cache: C:\NAQSHKASH\stage2\stage2f_rejection_evaluation_v2\r3_alignment_experiment\runs\20261107_021046_514_extraction.

## Existing measured baseline

The saved 48-image / 12-parent pilot reproduces all OCR strings. It reports 898/906 glyphs matched, mean source-x error -32.950 px, median absolute error 25.105 px, p95 absolute error 120.936 px, and conditional association accuracy 404/520 (77.69%). Every one of the 24 valid controls has an association/order-risk flag. Coverage is not localization accuracy.

## Mapping mathematics

Reviewed frozen CRNN horizontal layers, resize_preserve_aspect_ratio, D1 crop return/matrix, map_span and reference_glyphs:
- Two horizontal 2x2 stride-2 pools give T=floor(Wr/4); remaining pools affect height only.
- Starting with pixel-edge center 0.5, symmetric padded 3x3 convolutions preserve the origin and the two pools produce nominal centers 2+4t. Horizontal CNN receptive width is 42 resized pixels.
- In CV2 pixel-center coordinates, resize inversion is x_crop=(u+0.5)*Wc/Wr-0.5. Substituting u=1.5+4t gives x_processed=x0+(2+4t)*Wc/Wr-0.5, exactly the implementation.
- Actual rounded resized width is used, no batch padding is present, and source mapping applies inverse D1 homography. Source ink centers use (bbox_left+bbox_right-1)/2 with the same pixel-center convention.
- Renderer anchors are padding+extra_left+column*cell_width and the documented row/tier vertical offset. PIL masks use anchor=lt both in rendering and reference extraction. Cell centers are not substituted for ink centers.

All **48 matrices are identity**; all cached width, height and frame-count checks agree with the exact resize/floor formula. No missing projective/deskew transform explains this pilot's bias. The synthetic lattice/roundtrip tests pass.

**No mathematically justified map_span correction was found. No coordinate rescore was performed.** A half-pixel convention adjustment would not explain tens or hundreds of pixels of error, and a fitted +33px offset is prohibited and unjustified.

## The bias is not constant

| Matched group | Glyphs | Mean signed error | Median signed error | Range |
|---|---:|---:|---:|---:|
| Upper | 212 | -32.963 | -21.420 | -376.094 .. -8.664 |
| Count | 534 | -26.390 | -26.782 | -120.936 .. -16.077 |
| Lower | 152 | -55.979 | -22.125 | -376.281 .. -10.125 |
| First matched glyph in each tier | 144 | -82.272 | -27.938 | -376.281 .. -19.907 |
| Later matched glyphs | 754 | -23.531 | -23.250 | -31.871 .. -8.664 |

In **all 144 tiers**, the first visible-character span begins at frame zero; its mapped nominal point lies before the corresponding reference ink's left edge. Some first visible glyphs occur far to the right of the shared row crop. A globally contextual BiLSTM can emit their leading-stripped labels at the first time step despite that blank image region. This is an observed prediction-timing/localization failure, not evidence that the source boxes should be shifted. Later emissions also precede ink centers systematically.

The available tensors could support other preregistered alignment analyses, but forcing logits onto ground-truth positions or choosing a constant offset from this cache would not constitute a mapping correction. The present audit does not do either.

## Separate algorithmic limitation: offline oracle control

As an OFFLINE control only, the unchanged monotone nonempty association algorithm was fed true source ink centers and all true visible glyphs, without changing any cached CTC coordinate:
- 526/534 count glyph assignments correct.
- Zero valid controls have association errors; all 274 valid-control count assignments are correct.
- The remaining eight assignment errors are in missing-count cases. The algorithm forces a nonempty count group per observed symbol, so it cannot represent the actual empty token.

This oracle is not a deployable result or an apples-to-apples invalid-image improvement: it uses true positions and restores recognized-glyph coverage. On valid controls, all glyphs were already recognized, so the zero-error oracle shows that the stored CTC positions, rather than an inevitable failure of the association objective on valid geometry, explain their failures. Missing-count handling is a separate algorithmic defect: do not force a nonempty group when the image supports an empty region.

## Consequence

Do not use the current CTC spike centers to override Stage 1 reviews or to reject invalid images. Prioritize independently observed ink positions; retain glyph grouping and absence ambiguity. Fixing a real coordinate bug would permit a separately labeled rescore, but this cache provides no justification for such a correction.
