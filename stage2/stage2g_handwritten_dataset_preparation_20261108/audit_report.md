# Stage 2G handwritten dataset audit

## Result
920 images / 30 source IDs: 900 augmented JPGs and 20 original-photo PNGs. All 30 IDs have image, TXT and JSON references across the three supplied collections. No missing triplets, unreadable/corrupt images, exact-byte duplicates, decoded-pixel duplicates or duplicate ZIP members were found. See inventories and hashes for reproducible evidence. Perceptual duplicate detection was not performed.

All 30 sources pass transcription consistency checks after UTF-8 BOM/CRLF normalization and ignoring terminal newline delimiters on both sides. Internal spaces, blank lines, tier order, symbols, run positions, directions, encoded counts and row cell totals are checked. Two provisional mismatch flags (all_005114 and all_009197) were traced to an empty final lower tier plus terminal newline normalization; these were checker errors, corrected before release. No label was changed.

The selected_sources pack has 20 IDs. remaining.zip has 10 IDs, including six overlaps and four additional IDs. source_ground_truth has 20 IDs, including 14 overlaps and six additional IDs. Matching annotation copies were checked; overlaps are not extra sources. source_image_inventory.json records every source image copy and decode result.

## Provenance and metadata
All 30 brightness representatives and all 20 archive PNGs were visually inspected. They show irregular handwritten marks on lined/plain paper, consistent with photographed handwriting. No archive representative was positively identified as photographed printed Talim. This visual classification does not establish a verified chain of custody or writer identity. All augmentations are derivatives, not additional independent physical captures.

Crucial distinction: the annotation collections' clean PNG references are synthetic. The pack README describes printing generated Talim, pilot README explicitly calls source_ground_truth PNGs synthetic, and JSON names Talamat.ttf/IMTCRPET rendering. The 20 PNGs inside the DATASET archive instead visibly show handwritten photographs. Their file extension does not determine provenance. Synthetic annotation-reference images were not added to the 920-image manifest.

capture_manifest.csv records writer_01 for 10 raw captures. No reliable archive-to-raw-capture parent mapping is supplied. writer_id is therefore blank in the training manifest, while metadata_capture_writer preserves that limited metadata association. original_sheet_id is unknown. source_split.csv preserves print_template_sheet_id where available; it is NOT the identity of the handwritten physical sheet. No identities were invented.

## Completeness and disposition
Accepted: 0. Excluded: 0. Manual review: 920. All images decode, but this does not prove that every symbol, count and tier is complete. The manifest is ready for review and downstream filtering, not an approved training release. accepted_manifest.csv is deliberately header-only. Do not train from the unfiltered manifest.

Every image has a full-resolution extracted copy and appears in a per-source contact sheet. Human semantic comparison of every image to a verified physical original remains outstanding. Some representative captures have marks near or at image edges. Partial crops and strong transformations require particular care. Available phase3 code removes 2–6% of one edge and resizes to the old dimensions; fixed-canvas rotations can clip content. These scripts are contextual evidence only: the archive lacks per-image seed, matrices and authenticated recipe/parent links. A recipe name alone cannot prove preservation. No OCR or recognition model was used to resolve ambiguous marks.

Review review_queue.csv against original captures, TXT/JSON and images. Record reviewer/decision/notes in a copy. Confirm handwriting provenance, parent/source identity, complete tiers, all symbols/counts and legibility. Reject content-losing crops with full labels; any partial transcription needs a separately reviewed annotation in a later stage. This preparation does not automatically ingest review decisions.

## Isolation and validation
All supplied originals were opened read-only. input_hashes.csv records source files, ZIP containers/members, metadata and contextual generator scripts; tests re-hash them after preparation. Writes were confined to this new Stage 2G output folder and audit-script workspace files. E1 and Stage 2F were not modified or run. No Git/GitHub operation, OCR, fine-tuning, training or protected-test evaluation occurred.

See source_split_report.md, summary.json, manifest.csv, source_split.csv, duplicates.json, source_image_inventory.json and test_results.txt. prepare.py reproduces the audit into a NEW output directory (existing output is refused). Tests read local artifacts and inputs only.
