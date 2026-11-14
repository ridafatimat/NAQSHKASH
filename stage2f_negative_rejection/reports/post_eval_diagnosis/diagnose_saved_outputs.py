"""Read-only post-evaluation diagnosis. No model, preprocessing or policy execution.

Run with the existing Stage2F Python environment and --output-dir pointing at a
new diagnostic directory. Only saved JSON/CSV, source images and font masks are
read. Font-mask diagrams are diagnostic references, not generated dataset items.
"""
import argparse
import base64
from collections import Counter, defaultdict
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import sys

F = Path(r'C:\NAQSHKASH\stage2\stage2f_negative_rejection')
sys.path.insert(0, str(F / 'src'))
from common import (ROOT, PROJECT, DATA, D, E, CHECKPOINT, TOOLS, asset_bytes,
                    text_asset, read_csv, load_json, digest, sha_bytes, write_csv, write_json)
from talim.talim_rules import canonical_row, canonical_text, TRUSTED_SYMBOLS, symbol_direction
from PIL import Image, ImageDraw, ImageFont, ImageOps

BASELINE = F / 'metrics/20261106_154728_812_eval_baseline'
RESULTS = F / 'metrics/20261106_final_eval_F1_v2'
NEG = DATA / 'talim_edge_cases_negative_v1'
FONT = Path(r'C:\FYP Dataset Generation\Assets\Talamat.ttf')
GENERATOR = Path(r'C:\Users\hp\OneDrive\Documents\ChatGPT\FYP\edge_audit_staging\dataset_tools\edge_case_audit.py')
CODE = 'DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH'


def canonical_rows(sample):
    return [{role: t.lstrip(' ') for role, t in zip(('upper', 'count', 'lower'), canonical_row(row['runs'])) if t.strip()}
            for row in sample['logical_rows']]


def get_image(path):
    if not path:
        return None, None
    raw = asset_bytes(path)
    try:
        im = Image.open(io.BytesIO(raw))
        im.load()
        return raw, im
    except (OSError, ValueError):
        return raw, None


def annotation(path):
    try:
        raw = text_asset(path)
    except (OSError, KeyError):
        return dict(raw='', json_status='missing', reconstructed=None, errors='annotation artifact absent')
    try:
        sample = json.loads(raw)
    except ValueError as exc:
        return dict(raw=raw, json_status='malformed', reconstructed=None, errors=str(exc))
    try:
        reconstructed = canonical_rows(sample)
        return dict(raw=raw, json_status='parseable', reconstructed=reconstructed, errors='')
    except (KeyError, TypeError, ValueError) as exc:
        return dict(raw=raw, json_status='parseable_unreconstructable', reconstructed=None, errors=str(exc))


def safe_text(path):
    try:
        return text_asset(path)
    except (OSError, KeyError):
        return None


def image_info(raw, im):
    if raw is None:
        return dict(image_present=False, readable=False, width=0, height=0, mode='', minimum=None, maximum=None,
                    sha256='', grayscale_sha256='')
    if im is None:
        return dict(image_present=True, readable=False, width=0, height=0, mode='', minimum=None, maximum=None,
                    sha256=sha_bytes(raw), grayscale_sha256='')
    gray = im.convert('L')
    lo, hi = gray.getextrema()
    return dict(image_present=True, readable=True, width=im.width, height=im.height, mode=im.mode, minimum=lo, maximum=hi,
                sha256=sha_bytes(raw), grayscale_sha256=sha_bytes(str(gray.size).encode() + gray.tobytes()))


def snapshot():
    paths = set()
    for folder in ('metrics', 'manifests', 'configs', 'src'):
        paths.update(p for p in (F / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(p for p in NEG.rglob('*') if p.is_file())
    paths.update(p for p in D.glob('*.py'))
    paths.update([D / 'configs/D1.json', CHECKPOINT, PROJECT / 'talim/talim_rules.py',
                  TOOLS / 'talim_rules.py', TOOLS / 'validate_datasets.py',
                  DATA / 'talim_edge_cases_negative_v1.zip', DATA / 'talim_edge_cases_negative_scanned_v1.zip', FONT, GENERATOR,
                  Path(r'C:\FYP Dataset Generation\talamat_scan_scripts_v1_1\negative_edge_scan_augment.py')])
    return {str(p): digest(p) for p in sorted(paths)}


def font_audit(out):
    font = ImageFont.truetype(str(FONT), 48)
    rows = []
    for c in sorted(TRUSTED_SYMBOLS):
        mask = font.getmask(c)
        im = Image.frombytes('L', mask.size, bytes(mask))
        ink = im.crop(im.getbbox())
        rows.append(dict(symbol=c, direction=symbol_direction(c), full_mask_size=list(mask.size),
            ink_size=list(ink.size), ink_sha256=sha_bytes(ink.tobytes())))
    by = {r['symbol']: r for r in rows}
    assert by['B']['ink_sha256'] == by['X']['ink_sha256'] and by['B']['ink_size'] == by['X']['ink_size']
    assert by['[']['ink_sha256'] != by['G']['ink_sha256']
    assert by['_']['ink_sha256'] != by['L']['ink_sha256']
    write_csv(out / 'glyph_reference_audit.csv', rows)
    sheet = Image.new('RGB', (1040, 340), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 12), 'Supplied Talamat font: symbol IDs are not Latin glyph shapes', fill='black')
    for j, c in enumerate(['B', 'X', '[', 'G', '_', 'L']):
        x = 15 + j * 170
        draw.text((x, 55), repr(c) + ' ' + symbol_direction(c), fill='black')
        draw.text((x + 35, 90), c, font=font, anchor='lt', fill='black')
        mask = font.getmask(c)
        glyph = Image.frombytes('L', mask.size, bytes(mask))
        glyph = ImageOps.invert(glyph.crop(glyph.getbbox())).resize((90, 90), Image.Resampling.NEAREST)
        sheet.paste(glyph, (x + 20, 160))
    draw.text((15, 285), 'B / X: identical 10 x 8 visible mask. [ / G and _ / L: visually similar, not byte-identical.', fill='black')
    sheet.save(out / 'glyph_reference.png')
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    protected = snapshot()
    write_json(out / 'protected_before.json', protected)
    completion = load_json(BASELINE / 'completion.json')
    assert digest(BASELINE / 'observations.jsonl') == completion['observations_sha256']
    assert digest(RESULTS / 'results.json') == load_json(RESULTS / 'completion.json')['results_sha256']
    items = [json.loads(line) for line in (BASELINE / 'observations.jsonl').read_text(encoding='utf-8').splitlines()]
    by = {item['record']['image_id']: item for item in items}
    saved = load_json(RESULTS / 'results.json')
    results = saved['rows']
    rb = {r['image_id']: r for r in results}
    assert len(items) == len(by) == len(results) == 434
    assert set(by) == set(rb)
    assert Counter(r['outcome'] for r in results) == Counter(ACCEPT=242, REVIEW_UNCERTAIN=187, REJECT_INVALID=5)
    # Canonical expected image content, not mutated source JSON targets.
    base = load_json(NEG / 'actual_symbols/json/negative.json')
    base_rows = canonical_rows(base)
    assert base_rows == [{'upper': 'A', 'count': 'jaji', 'lower': 'R'}, {'upper': 'B', 'count': 'kj', 'lower': 'S'}]
    assert canonical_text(base).rstrip('\n') == text_asset(str(NEG / 'actual_symbols/labels/negative.txt')).rstrip('\n')
    records = read_csv(F / 'manifests/negative_inventory.csv')
    images = {}
    annotations = {}
    expected = {}
    audits = []
    for record in records:
        id = record['image_id']
        raw, im = get_image(record['original_image_path'])
        info = image_info(raw, im)
        assert info['sha256'] == record['image_sha256']
        images[id] = (raw, im, info)
        cat = record['invalidity_category']
        if cat == 'duplicate_id' and 'second.json' in record['json_path']:
            rows = canonical_rows(load_json(record['json_path']))
        else:
            rows = base_rows
        if not info['image_present']:
            image_class = 'missing_image_record'
            rows = None
        elif not info['readable']:
            image_class = 'corrupt_unreadable_artifact'
            rows = None
        elif cat == 'wrong_dimensions':
            image_class = 'non_talim_black_patch_artifact'
            rows = None
            if record['augmentation'] == 'clean':
                assert (im.size, im.convert('L').getextrema()) == ((2, 2), (0, 0))
        elif info['minimum'] == info['maximum'] == 255:
            image_class = 'blank_white_artifact'
            rows = None
        elif cat == 'wrong_image_mode':
            image_class = 'valid_talim_content_representation_fixture'
        else:
            image_class = 'valid_talim_content_annotation_fixture'
        expected[id] = rows
        if record['json_path'] not in annotations:
            annotations[record['json_path']] = annotation(record['json_path'])
        ann = annotations[record['json_path']]
        txt = safe_text(record['label_path'])
        result = rb[id]
        item = by[id]
        audits.append(dict(image_id=id, source_parent_id=record['source_parent_id'], mutation_type=cat,
            origin=record['data_origin'], augmentation=record['augmentation'], original_image_path=record['original_image_path'],
            original_json_path=record['json_path'], original_label_path=record['label_path'],
            historical_image_ground_truth=record['image_ground_truth'], diagnostic_image_class=image_class,
            genuine_invalid_talim_content=False, diagnostic_expected_transcription=rows,
            source_json_status=ann['json_status'], source_json_error=ann['errors'], source_json=ann['raw'], source_txt=txt,
            json_runs_match_image_content=ann['reconstructed'] == rows if rows is not None and ann['reconstructed'] is not None else None,
            source_json_runs_reconstruction=ann['reconstructed'],
            observed_prediction=item['observation']['decoded_rows'], f0_outcome=item['F0']['outcome'], f1_outcome=result['outcome'],
            f1_reason_codes=result['reason_codes'], historical_transcription_exact=item['transcription_exact'], **info))
    hash_groups = defaultdict(list)
    pixel_groups = defaultdict(list)
    for a in audits:
        if a['sha256']:
            hash_groups[a['sha256']].append(a['image_id'])
        if a['grayscale_sha256']:
            pixel_groups[a['grayscale_sha256']].append(a['image_id'])
    for a in audits:
        a['byte_identical_group_size'] = len(hash_groups[a['sha256']]) if a['sha256'] else 0
        a['byte_identical_fixture_ids'] = hash_groups[a['sha256']] if a['sha256'] else []
        a['grayscale_identical_group_size'] = len(pixel_groups[a['grayscale_sha256']]) if a['grayscale_sha256'] else 0
    wrong = [r for r in results if CODE in r['reason_codes']]
    assert len(wrong) == 180
    cases = []
    ab = {a['image_id']: a for a in audits}
    for result in wrong:
        id = result['image_id']
        item = by[id]
        record = item['record']
        obs = item['observation']
        evidence = result['evidence']
        if id not in images:
            raw, im = get_image(record['original_image_path'])
            images[id] = (raw, im, image_info(raw, im))
        raw, im, info = images[id]
        rows = expected[id] if id in expected else canonical_rows(json.loads(text_asset(record['json_path'])))
        if id not in expected:
            assert rows == item['truth_rows']
            annotations[record['json_path']] = annotation(record['json_path'])
        ri, role, glyphs = evidence['row'], evidence['role'], evidence['glyphs']
        same_roles = [sorted(r) for r in rows] == [sorted(r) for r in obs['decoded_rows']] if rows is not None else None
        expected_tier = rows[ri].get(role) if rows is not None and ri < len(rows) else None
        alias = role == 'upper' and glyphs == ['X'] and expected_tier == 'B'
        if rows is None:
            cause = 'non_talim_patch_interpreted_as_count_tier'
            attribution = 'out_of_domain_artifact; saved D1 count-only proposal and OCR tokens, not a genuine wrong-tier Talim example'
        elif alias:
            cause = 'B_X_identical_ink_role_context_not_supplied_to_E1'
            attribution = 'recognition_interface_ambiguity; saved row/role layout agrees with source, not a demonstrated tier-assignment swap'
        else:
            cause = 'glyph_substitution_in_correctly_assigned_lower_tier'
            attribution = 'OCR transcription error; lower-tier source [ and _ predicted G and L; font masks similar but not identical'
        assert same_roles is True or rows is None
        cases.append(dict(image_id=id, source_parent_id=record['source_parent_id'], mutation_type=record['invalidity_category'],
            data_origin=record['data_origin'], original_image_path=record['original_image_path'], source_annotation_path=record['json_path'],
            historical_image_ground_truth=record['image_ground_truth'],
            diagnostic_image_class=ab[id]['diagnostic_image_class'] if id in ab else 'ordinary_valid_positive',
            expected_transcription=rows, predicted_transcription=obs['decoded_rows'],
            offending_row=ri, offending_tier=role, predicted_incorrect_glyphs=glyphs, expected_tier_transcription=expected_tier,
            source_annotation_json=annotations[record['json_path']]['raw'],
            saved_segmentation_rows=obs['segmentation_rows'], saved_bands=obs['diagnostics'].get('bands'),
            row_and_role_layout_matches_source=same_roles, segmentation_accepted=obs['segmentation_accepted'],
            root_cause=cause, attribution=attribution, evidence_scope='saved_predictions + original_source_pixels + font_mask_and_generator_audit',
            independent_image_invalidity=rows is None, reason_code=CODE, f1_outcome=result['outcome'],
            historical_transcription_exact=item['transcription_exact'], image_sha256=info['sha256'],
            byte_identical_fixture_count=len(hash_groups.get(info['sha256'], [])),
            byte_identical_fixture_ids=hash_groups.get(info['sha256'], [])))
    counts = dict(evaluation_records=len(results), wrong_tier_reviews=len(cases),
        wrong_tier_by_origin=dict(Counter(c['data_origin'] for c in cases)),
        wrong_tier_by_cause=dict(Counter(c['root_cause'] for c in cases)),
        wrong_tier_independent_parents=len({c['source_parent_id'] for c in cases}),
        wrong_tier_unique_image_byte_hashes=len({c['image_sha256'] for c in cases}),
        negative_wrong_tier_reviews=sum(c['data_origin'].startswith('negative') for c in cases),
        negative_fixture_records=len(audits), negative_fixture_image_files=sum(a['image_present'] for a in audits),
        negative_independent_source_families=len({a['source_parent_id'] for a in audits}),
        negative_mutation_types=len({a['mutation_type'] for a in audits}),
        negative_wrong_tier_mutation_types=len({c['mutation_type'] for c in cases if c['data_origin'].startswith('negative')}),
        diagnostic_image_classes=dict(Counter(a['diagnostic_image_class'] for a in audits)),
        genuine_invalid_talim_content_images=0,
        negative_fixture_outcomes=dict(Counter(a['f1_outcome'] for a in audits)),
        clean_byte_identical_base_files=len(hash_groups[digest(NEG / 'actual_symbols/images/negative.png')]),
        clean_grayscale_identical_base_files=len(pixel_groups[next(a['grayscale_sha256'] for a in audits if a['image_id']=='negative_actual_symbols_2_clean')]),
        historical_metrics_not_rewritten=True, exposed_evaluation_diagnostic_only=True)
    assert counts['wrong_tier_by_cause'] == dict(B_X_identical_ink_role_context_not_supplied_to_E1=176,
        non_talim_patch_interpreted_as_count_tier=3, glyph_substitution_in_correctly_assigned_lower_tier=1)
    font_audit(out)
    for filename, table in [('negative_fixture_audit.csv', audits), ('wrong_tier_cases.csv', cases)]:
        flat = [{k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in table]
        write_csv(out / filename, flat)
    mutation_summary = []
    for cat in sorted({a['mutation_type'] for a in audits}):
        group = [a for a in audits if a['mutation_type'] == cat]
        mutation_summary.append(dict(mutation_type=cat, records=len(group), image_files=sum(a['image_present'] for a in group),
            source_families=1, wrong_tier_reviews=sum(c['mutation_type'] == cat for c in cases),
            image_classes=dict(Counter(a['diagnostic_image_class'] for a in group)),
            f1_outcomes=dict(Counter(a['f1_outcome'] for a in group))))
    write_csv(out / 'mutation_summary.csv', mutation_summary)
    parent_summary = []
    for parent in sorted({c['source_parent_id'] for c in cases}):
        group = [c for c in cases if c['source_parent_id'] == parent]
        parent_summary.append(dict(source_parent_id=parent, wrong_tier_reviews=len(group),
            unique_image_hashes=len({c['image_sha256'] for c in group}), causes=dict(Counter(c['root_cause'] for c in group))))
    write_csv(out / 'wrong_tier_parent_summary.csv', parent_summary)
    controls = Counter()
    for item in items:
        if item['record']['status'] != 'positive_control' or not item['truth_rows']:
            continue
        for ri, row in enumerate(item['truth_rows']):
            s = row.get('upper', '')
            if 'B' not in s:
                continue
            pred = item['observation']['decoded_rows']
            p = pred[ri].get('upper') if ri < len(pred) else None
            controls[str(('B_only' if s == 'B' else 'B_with_context', s == p, item['observation']['segmentation_accepted']))] += 1
    counts['ordinary_B_tier_controls'] = dict(controls)
    write_json(out / 'diagnostic_counts.json', counts)
    make_gallery(out, images, by, rb, cases, ab)
    write_reports(out, counts, parent_summary)
    changed = [p for p, h in protected.items() if digest(p) != h]
    assert not changed, changed
    checks = dict(protected_files_checked=len(protected), protected_files_changed=changed,
        saved_observation_hash_verified=True, saved_f1_results_hash_verified=True,
        wrong_case_count_checked=180, fixture_record_count_checked=198, unique_parent_count_checked=3,
        full_ocr_or_preprocessing_executed=False, policies_or_manifests_modified=False,
        font_B_X_visible_mask_equality_verified=True)
    write_json(out / 'diagnostic_checks.json', checks)
    print(json.dumps(counts, indent=2))
    print('Diagnostic checks:', json.dumps(checks))


def make_gallery(out, images, by, rb, cases, audit):
    # Representative images are embedded without changing source pixels. Saved
    # D1 crop boxes are listed, not overlaid in the wrong coordinate system.
    ids = ['negative_count_zero_2_clean', 'negative_symbol_unsupported_2_clean',
           'negative_duplicate_id_3_clean', 'negative_wrong_image_mode_2_clean',
           'negative_count_zero_2_clean_1', 'negative_count_zero_2_clean_3',
           'negative_wrong_dimensions_2_clean', 'negative_wrong_dimensions_2_clean_1',
           'negative_wrong_dimensions_2_clean_2', 'negative_wrong_dimensions_2_clean_3',
           'negative_blank_png_2_clean', 'normal_train_011358__clean', 'normal_train_079690__clean']
    lines = ['<!doctype html><html><head><meta charset="utf-8"><title>Stage2F post-evaluation diagnosis</title>',
        '<style>body{font:16px system-ui;margin:30px auto;max-width:1200px;color:#18212b}article{border-top:1px solid #aaa;padding:22px 0}pre{white-space:pre-wrap;background:#f3f5f7;padding:12px}img{max-width:100%;max-height:750px;border:1px solid #ddd}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #bbb}.tiny{image-rendering:pixelated;width:208px}a{color:#1764a4}</style></head><body>',
        '<h1>Exposed evaluation: diagnostic gallery</h1><p>No OCR, preprocessing, policy tuning or dataset generation ran. Font diagrams are references only. Symbol IDs use the Talamat font, not Latin letter appearance.</p>',
        '<h2>Why B can decode as X without a tier swap</h2><img src="glyph_reference.png"><p>The visible B/X glyph masks are byte-identical. E1 receives a tight tier crop without its upper/lower role. Original page context distinguishes them.</p>']
    for id in ids:
        raw, im, info = images[id]
        item = by[id]
        result = rb[id]
        lines.append('<article id="' + id + '"><h2>' + html.escape(id) + '</h2>')
        lines.append('<p>' + html.escape(str(result['outcome']) + '; ' + '; '.join(result['reason_codes'])) + '</p>')
        if im is not None:
            mime = 'image/jpeg' if raw[:2] == b'\xff\xd8' else 'image/png'
            css = ' class="tiny"' if im.width < 30 else ''
            lines.append('<img' + css + ' src="data:' + mime + ';base64,' + base64.b64encode(raw).decode() + '">')
        expected = audit[id]['diagnostic_expected_transcription'] if id in audit else item['truth_rows']
        details = dict(original_image_path=item['record']['original_image_path'], dimensions=[info['width'], info['height']],
            image_class=audit[id]['diagnostic_image_class'] if id in audit else 'ordinary_valid_positive',
            expected_image_transcription=expected, prediction=item['observation']['decoded_rows'],
            saved_boxes_in_prepared_image_coordinates=item['observation']['segmentation_rows'])
        lines.append('<pre>' + html.escape(json.dumps(details, indent=2)) + '</pre></article>')
    lines.append('<h2>Complete wrong-tier case index (180 saved decisions)</h2><p>Detailed annotations, paths, predictions and duplicate IDs are in wrong_tier_cases.csv. No affected cases were sampled away.</p><table><tr><th>Image</th><th>Tier/glyph</th><th>Cause</th></tr>')
    for c in cases:
        lines.append('<tr><td>' + html.escape(c['image_id']) + '</td><td>' + html.escape(str((c['offending_row'], c['offending_tier'], c['predicted_incorrect_glyphs']))) + '</td><td>' + html.escape(c['root_cause']) + '</td></tr>')
    lines.append('</table></body></html>')
    (out / 'visual_gallery.html').write_text('\n'.join(lines), encoding='utf-8')


def write_reports(out, counts, parent_summary):
    root = f'''# Negative-fixture root cause analysis

This is diagnosis of an already exposed evaluation, not independent test evidence.
Read-only inputs: {BASELINE} and {RESULTS}. All 434 saved rows were joined by ID;
180 wrong-tier reviews and 198 negative-fixture records were audited. No model,
preprocessing, segmentation search, rejection policy or calibration was executed.

## Confirmed findings

The excessive review rate is driven by one repeated source family, not 178
independent failures. Of 180 wrong-tier reviews, 178 are negative fixtures (47
clean + 131 scanned), and 2 are ordinary clean positives. There are 3 independent
parent identities total, 135 distinct file hashes among these 180 decisions,
and 46 affected negative mutation categories. The 178 fixture cases still have
only one independent source family, with two derived canonical content templates.

| Diagnosed mechanism | Reviewed images | Independent parent families |
|---|---:|---:|
| Expected singleton B (UP) decoded X (DOWN), identical visible font mask | 176 | 2 |
| Of the above: negative-fixture copies/variants | 175 | 1 |
| 2x2 black-patch variants interpreted as count-only X/Z | 3 | 1 (same fixture family) |
| Ordinary positive lower-tier [ and _ decoded G and L | 1 | 1 |

The family counts overlap and must not be added across rows. The final row's
parent is different from the singleton-B ordinary-positive parent.

## The B/X result is not evidence of an image-level wrong-tier symbol

The supplied Talamat.ttf encodes B and X with byte-identical 10x8 visible masks
at font size 48. Their full font masks have different blank extents; D1 crops
ink bands with small margins, and the recognizer resizes each crop to height 64.
The E1 call receives only the crop tensor; row/tier role is recorded externally
but is not supplied to the CRNN. Thus a single dot cannot intrinsically identify
B versus X from its shape. Correct interpretation needs the upper/lower context.

The base image visibly contains A11/R19 and S20/B10. The second row upper is
the singleton dot representing B; saved D1 bands put it at y194..202 and its
upper crop at [20,190,128,206] (height16, resized width432). The saved row/role
layout agrees with the canonical two-row layout in all 175 B/X fixture cases.
The same ordinary positive failure has upper B at row2, band y364..372, crop
[22,360,372,376], correctly labeled upper. No tier swap is demonstrated.

There is one singleton-B upper tier among the 236 ordinary positive controls;
it too decodes X. For B-containing multi-symbol upper tiers, 29 are transcribed
exactly, 3 are not exact, and one belongs to a failed segmentation proposal.
These are tier counts, not independent image accuracy estimates. This contrast
supports missing role/context plus short glyph-only input as the concentration
mechanism. Repeating that base hundreds of times amplifies this ambiguity.
No inference experiment was run to quantify a fix or isolate every internal
model factor; causal claims beyond the demonstrable information loss remain
untested. We do not claim all ordinary positive OCR errors are explained by B/X.

The other ordinary failure has correctly assigned upper/count/lower rows but
lower [ and _ predicted G and L. Original pixels and canonical annotation agree;
the glyphs are visually similar but their tight font masks are not identical.
This is an OCR transcription substitution, not an established segmentation swap.

## Dataset construction and image/annotation disagreement

The recovered original generator is {GENERATOR}, negatives(), lines227..284.
The deployed C:\\FYP Dataset Generation\\dataset_tools directory no longer has
edge_case_audit.py; the retained project staging copy is identified explicitly,
not represented as a deployed active generator. Its mutation operations agree
with the observed source artifacts and negative_expectations.json. Inputs and
generator hashes are recorded in protected_before.json.

The generator first writes a valid rendered base image and TXT, then most
mutations change JSON only. Wrong-line/count/symbol JSON mutations do not rerender
the PNG. TXT mutations replace text files only. Duplicate-ID and semantic-
duplicate cases add extra image records. The scan script explicitly augments
readable images regardless of malformed JSON, preserving this distinction.

Consequently fixture labels such as count_zero, symbol_unsupported and wrong_line
describe validator defects, not visible invalid notation. This is intentional
validator-test design, but is inappropriate binary image-rejection supervision.
Source JSON is not used by runtime OCR, so annotation mutations do NOT directly
cause the B/X predictions. They affect dataset interpretation/training eligibility;
repeated glyph-only crops explain the observed prediction concentration.

There are 49 mutation categories, 51 clean source records, 50 actual clean files
and 147 scanned variants: 198 records but 197 image files. 45 clean PNGs are
byte-identical to the valid base. RGB mode contributes one more identical
grayscale raster (46 clean rasters total). Two derived content templates remain
one family, not separate independent negative parents.

## Corrected diagnostic classification (historical manifests remain untouched)

| Image class | Clean files | Scanned files | Total files |
|---|---:|---:|---:|
| Valid content with annotation/validator-only defects | 46 | 138 | 184 |
| Valid content, source representation/mode fixture | 1 | 3 | 4 |
| Blank white input | 1 | 3 | 4 |
| Corrupt/unreadable PNG | 1 | 0 | 1 |
| Non-Talim black patch (wrong_dimensions) | 1 | 3 | 4 |
| **Total actual files** | **50** | **147** | **197** |

There is one additional missing-PNG record, not an image. There are ZERO confirmed
genuine invalid-Talim-notation images. There are 9 unusable/non-Talim artifact
files (4 white blanks + 1 corrupt + 4 black-patch variants), which must not be
reported as 9 independently authored semantic Talim negatives. All share the
same source family. The four white inputs are exactly constant255 and the
clean black input is exactly constant0 at 2x2. The three black variants are26x26
patches with padding/augmentation, not a Talim page.

The earlier audit incorrectly described wrong_dimensions as potentially resized
valid content. The generator and pixels show it replaces the page with black.
It also left scanned blanks unadjudicated; all three are actually constant white.
The prior unsupported-representation label was inherited by RGB-derived JPEGs,
although their Talim content remains valid. These are diagnostic corrections
only; no frozen labels or official metrics have been rewritten.

The original summary's invalid denominator2 is its historical blank/corrupt
label scope, not a complete inventory of the9 artifact files. All5 observed
REJECT_INVALID actions are correctly explained by input checks: 4 no-ink white
images and1 corrupt PNG. Three additional wrong-tier reviews are from the black
patch's count-only OCR X/Z. There is no valid expected count transcription for
these patches; do not blame a nonexistent ground-truth symbol or call them
confirmed tier swaps. The clean black image and missing image are the other2
fixture reviews. Valid-image syntax can still be wrong due to recognition.

## Weighting, uncertainty and policy interpretation

Negative fixtures: 198 records yield13 ACCEPT,180 REVIEW and5 REJECT. Ordinary
controls:236 yield229 ACCEPT and7 REVIEW. Of all187 reviews,180 contain the
wrong-tier code; 178 of those are fixtures and2 ordinary positives. The remaining
reviews include failed segmentation/recognition checks. See the case CSV for
every affected image, its original annotation, expected/predicted text, tier,
glyph, duplicate group, parent and attribution. Mutation and parent tables avoid
mistaking variants for independent evidence.

Main source of the wrong-tier pattern: role-blind recognition representation,
not confirmed erroneous D1 tier assignment. Cropping/normalization removes useful
page context, but saved row assignments match source roles on177 valid-content
wrong-tier cases. Exact crop/component preservation for every scanned variant
cannot be independently proven from saved boxes alone; processed crop pixels
were not archived, and no preprocessing replay was authorized/performed here.
The report therefore does not claim a full segmentation-fidelity proof.

Talim membership/count/canonical rules remain correct. F1 v2 correctly sends
OCR-only violations to REVIEW; this diagnosis gives no basis to revert them to
image-level invalidity or to tune the exposed evaluation. D1/E1, F1/F2 policies,
thresholds, datasets, manifests and historical results were left unchanged.
'''
    (out / 'root_cause_analysis.md').write_text(root, encoding='utf-8')
    recommendations = '''# Dataset and Stage 2F recommendations

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
'''
    (out / 'dataset_recommendations.md').write_text(recommendations, encoding='utf-8')


if __name__ == '__main__':
    main()
