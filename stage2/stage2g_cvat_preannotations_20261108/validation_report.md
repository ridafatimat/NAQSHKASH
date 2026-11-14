# CVAT provisional pre-annotations — task 2748744

Input: C:\NAQSHKASH\naqshkash_cvat_30_originals.zip, unchanged SHA-256 4c47f897710e340004d28d965f96c696d7fed84a607e6f12a1bb8fc56b63197c.
30 readable originals, 30 unique source filenames, all 30 represented in XML; image dimensions read from the exact ZIP bytes. Lexical image IDs 0–29; filename matching is authoritative because the live CVAT frame order was not queried.

197 polygons: {'Upper': 59, 'Count': 84, 'Lower': 54}. Images with no proposed shapes: []. Every shape passes finite/bounded coordinates, nonzero area, convex/simple polygon, supported label and required attribute checks. XML parse and ZIP CRC round-trips pass. The archive contains annotations.xml only, with CVAT image XML version 1.1. Live server import has not been tested.

ALL annotations are PROVISIONAL_NEEDS_REVIEW; source=auto. row_index is zero-based, approximate and based only on top-to-bottom image geometry; group_id groups candidates from the same row. These are editing starting points, not ground truth. Some faint, touching, slanted, cropped or overlapping marks will be missing, merged or assigned the wrong role. Prior baseline diagnostics already demonstrated these risks. Do not train on these shapes before review. No TXT, JSON labels/token counts, OCR, E1 model or synthetic image renders are used. The original shared handwritten_candidates pixel function is extracted without importing its model integration; only faint-blue sensitivity and horizontal extent are adjusted locally. Convex ink-envelope polygons approximate slant and irregular borders.

Only this new directory was written. Original ZIP/photos, E1, Stage 2F and prior Stage 2G outputs are unchanged. No Git operations. The requested scope explicitly covers all 30 annotation images; this is image-geometry annotation preparation, not protected-test model evaluation.

## Import instructions
1. Open task 2748744, NAQSHKASH - 30 Original Handwritten Pages. Confirm the 30 image filenames match image_shape_summary.csv. If annotation work already exists, export a backup before replacing it.
2. In task label settings, for EACH of Upper, Count and Lower, add two text attributes: review_status (default PROVISIONAL_NEEDS_REVIEW) and row_index (default unknown). Preserve the label names/case. Ensure label shape type is Polygon or Any. Save. The XML declares these attributes but an existing task must have matching attribute definitions.
3. On the TASK page choose Actions > Upload annotations. Select CVAT for images 1.1 (or the CVAT for images entry shown by your version), choose Local file, select naqshkash_cvat_task_2748744_provisional.zip, and confirm Upload/Submit. Use task annotation import, not dataset/task creation or a job-only import.
4. Wait for the import request to complete. Open a job; check that polygons overlay the correct filename and that review_status is PROVISIONAL_NEEDS_REVIEW. If CVAT reports filename/attribute mismatches, fix the task schema or file mapping rather than importing against numeric frame IDs blindly.
5. Review EVERY page, correcting/add/delete polygons and row_index from visible handwriting. In particular inspect omitted faint strokes, count descenders and overlapping tiers. row_index must follow actual rows, not OCR text. Mark review_status VERIFIED only after human verification. No shapes are automatically approved.

Format reference: https://docs.cvat.ai/docs/dataset_management/formats/format-cvat/
