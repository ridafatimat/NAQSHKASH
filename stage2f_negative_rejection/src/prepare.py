"""Bounded negative metadata audit and protocol preparation. No model inference."""
import argparse
import base64
from collections import Counter, defaultdict
import csv
import html
import io
import json
from pathlib import Path
import zipfile
from common import *

NEG = DATA / 'talim_edge_cases_negative_v1'
SCAN = DATA / 'talim_edge_cases_negative_scanned_v1.zip'
PREFIX = 'talim_edge_cases_negative_scanned_v1/'
PARENT = 'negative_embedded_base_A11_R19_S20_B10'
FIELDS = ['image_id', 'source_parent_id', 'data_origin', 'status', 'invalidity_category',
          'expected_system_outcome', 'partition', 'existing_annotation', 'original_image_path',
          'preprocessing_config', 'label_path', 'json_path', 'image_sha256', 'image_ground_truth',
          'annotation_scope', 'existing_partition', 'exposure', 'model_rejection_eligible', 'augmentation']


def zip_csv(z, member):
    return list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig'))))


def image_info(raw):
    from PIL import Image
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            return dict(readable=True, mode=im.mode, width=im.width, height=im.height,
                        has_ink=im.convert('L').getextrema()[0] < 255,
                        pixel_sha256=sha_bytes(im.convert('L').tobytes()))
    except (OSError, ValueError):
        return dict(readable=False, mode='', width='', height='', has_ink=False, pixel_sha256='')


def inventory():
    expectations = load_json(NEG / 'negative_expectations.json')
    if {e['source_valid_sample'] for e in expectations} != {'embedded base: A11/R19, S20/B10'}:
        raise ValueError('Negative provenance changed; manually review family grouping')
    reasons = {r['sample_id']: r for r in expectations}
    base_hash = digest(NEG / 'actual_symbols/images/negative.png')
    with zipfile.ZipFile(SCAN) as z:
        source_manifest = zip_csv(z, PREFIX + 'negative_source_manifest.csv')
        aug = zip_csv(z, PREFIX + 'augmentation_manifest.csv')
        indexed = {r['source_filename']: r for r in source_manifest}
        records = []
        for category in sorted(reasons):
            folder = NEG / category
            for ordinal, meta in enumerate(read_csv(folder / 'metadata.csv'), 2):
                image = folder / meta['png_path']
                rel = image.relative_to(NEG).as_posix()
                source = indexed.get(rel)
                raw = image.read_bytes() if image.is_file() else None
                info = image_info(raw) if raw is not None else image_info(b'')
                same = raw is not None and sha_bytes(raw) == base_hash
                # Mutations to JSON/TXT are not pixels. Non-identical artifact
                # variants need image adjudication, not an inherited invalid label.
                truth = ('valid' if same else 'corrupted_unreadable' if category == 'corrupt_png'
                         else 'empty_input' if category == 'blank_png' else 'missing_artifact' if raw is None
                         else 'unsupported_representation' if category == 'wrong_image_mode'
                         else 'ambiguous_potentially_valid' if category == 'wrong_dimensions'
                         else 'valid' if category == 'duplicate_id' else 'unadjudicated')
                scope = 'metadata_or_annotation_only' if truth == 'valid' else truth
                # Count artifact failure as rejection, unsupported/ambiguous as review.
                expected = ('ACCEPT' if truth == 'valid' else 'REJECT_INVALID' if truth in
                            ('corrupted_unreadable', 'empty_input') else 'REVIEW_UNCERTAIN')
                record = dict(image_id=f'negative_{category}_{ordinal}_clean', source_parent_id=PARENT,
                    data_origin='negative_clean', status='negative_fixture', invalidity_category=category,
                    expected_system_outcome=expected, partition='evaluation',
                    existing_annotation=json.dumps(dict(metadata=meta, expectation=reasons[category]), ensure_ascii=False),
                    original_image_path=str(image) if raw is not None else '',
                    preprocessing_config=str(D / 'configs/D1.json'), label_path=str(folder / meta['txt_path']),
                    json_path=str(folder / meta['json_path']), image_sha256=sha_bytes(raw) if raw is not None else '',
                    image_ground_truth=truth, annotation_scope=scope, existing_partition='unsplit_validator_fixture',
                    exposure='historical adversarial fixture; not independent unseen evidence',
                    model_rejection_eligible=str(truth in ('corrupted_unreadable', 'empty_input')), augmentation='clean')
                records.append(dict(record, source_record_id=source['source_id'] if source else '',
                                    image_present=raw is not None, byte_identical_to_base=same, **info))
                if source:
                    variants = [a for a in aug if a['source_id'] == source['source_id']]
                    for a in variants:
                        member = PREFIX + a['augmented_filename']
                        variant = z.read(member)
                        if sha_bytes(variant) != a['augmented_sha256'] or sha_bytes(raw) != a['source_sha256']:
                            raise ValueError('Negative scanned source/hash mismatch: ' + member)
                        vtruth = truth
                        # Noise on a blank source is not demonstrably invalid content.
                        if truth in ('empty_input', 'corrupted_unreadable'):
                            vtruth = 'unadjudicated'
                        records.append(dict(record, image_id=f"{record['image_id']}_{a['variant_index']}",
                            data_origin='negative_scanned', original_image_path=str(SCAN) + '::' + member,
                            image_sha256=sha_bytes(variant), augmentation=a['augmentation_type'],
                            image_ground_truth=vtruth, annotation_scope=scope if vtruth == truth else 'unadjudicated',
                            expected_system_outcome=expected if vtruth == truth else 'REVIEW_UNCERTAIN',
                            model_rejection_eligible='False', source_record_id=source['source_id'], image_present=True,
                            byte_identical_to_base=False, **image_info(variant)))
        return records, source_manifest, aug


def positive_controls():
    pages = read_csv(PROJECT / 'stage2/preprocessing_improvements/manifests/pages.csv')
    assignments = load_json(PROJECT / 'stage2/positive_edge_cases/manifests/parent_assignments_complete_v1.json')['assignments']
    sources = {
        'positive': (DATA / 'talim_edge_cases_positive_v1.zip', DATA / 'talim_edge_cases_positive_scanned_v1.zip', 'talim_edge_cases_positive_v1/'),
        'normal': (Path(r'C:\FYP Dataset Generation\Final dataset\talim_100000.zip'), None, 'datasets/talim_100000/')}
    controls = []
    excluded = []
    for p in pages:
        if p['dataset'] not in sources:
            continue
        original = assignments.get(p['parent_id'], 'train') if p['dataset'] == 'positive' else p['parent_id'].split('_')[0]
        if original == 'test':
            excluded.append(p)
            continue
        clean, scan, prefix = sources[p['dataset']]
        partition = ('development' if p['partition'] == 'tuning' else 'evaluation') if p['dataset'] == 'positive' else (
            'development' if int(sha_bytes(p['parent_id'].encode())[:8], 16) % 2 == 0 else 'evaluation')
        controls.append(dict(image_id=p['dataset'] + '_' + p['case_id'], source_parent_id=p['dataset'] + ':' + p['parent_id'],
            data_origin=p['dataset'] + '_' + p['kind'], status='positive_control', invalidity_category='none',
            expected_system_outcome='ACCEPT', partition=partition,
            existing_annotation='canonical source JSON/TXT; offline scoring only',
            original_image_path=str(clean if p['kind'] == 'clean' else scan) + '::' + p['image_member'],
            preprocessing_config=str(D / 'configs/D1.json'), label_path=str(clean) + '::' + prefix + p['source_json'].replace('json/', 'labels/').replace('.json', '.txt'),
            json_path=str(clean) + '::' + prefix + p['source_json'], image_sha256=p['image_sha256'],
            image_ground_truth='valid', annotation_scope='valid_positive', existing_partition=p['partition'],
            exposure='Stage2C/D/E exposed development or training regression; NOT unseen',
            model_rejection_eligible='False', augmentation=p['augmentation']))
    return controls, excluded


def historical_lock():
    # Bounded contract snapshot: code/config/manifests used by this stage,
    # original Stage1 and selected E1. No historical dataset sweep.
    paths = {CHECKPOINT, PROJECT / 'stage2/positive_edge_cases/checkpoints/stage1/best.pt',
             NEG / 'negative_expectations.json', DATA / 'talim_edge_cases_negative_v1.zip', SCAN,
             TOOLS / 'validate_datasets.py', TOOLS / 'talim_rules.py',
             Path(r'C:\FYP Dataset Generation\config.py'), Path(r'C:\FYP Dataset Generation\Utils\talamNumConverter.py')}
    for directory in [PROJECT / 'models', PROJECT / 'data', PROJECT / 'talim',
                      PROJECT / 'preprocessing', D, PROJECT / 'config', E / 'configs', E / 'manifests',
                      PROJECT / 'stage2/preprocessing_improvements/manifests',
                      PROJECT / 'stage2/positive_edge_cases/manifests']:
        for p in directory.glob('*'):
            if p.is_file() and p.suffix in ('.py', '.json', '.csv', '.yaml'):
                paths.add(p)
    for p in NEG.rglob('*'):
        if p.is_file():
            paths.add(p)
    return {str(p): digest(p) for p in sorted(paths)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--validate-only', action='store_true')
    args = ap.parse_args()
    if args.validate_only:
        print(json.dumps(validate_prepared(), indent=2))
        return
    if (ROOT / 'manifests/negative_inventory.csv').exists():
        raise FileExistsError('Preparation already exists; preserve the frozen protocol')
    for directory in ('configs', 'protocol', 'manifests', 'reports', 'galleries', 'checkpoints', 'metrics', 'tests'):
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    records, sources, aug = inventory()
    positives, excluded = positive_controls()
    write_csv(ROOT / 'manifests/negative_inventory.csv', records)
    manifests = [{k: r[k] for k in FIELDS} for r in records] + positives
    for part in ('development', 'evaluation'):
        write_csv(ROOT / f'manifests/{part}_manifest.csv', [r for r in manifests if r['partition'] == part], FIELDS)
    summary = []
    for cat in sorted({r['invalidity_category'] for r in records}):
        group = [r for r in records if r['invalidity_category'] == cat]
        summary.append(dict(category=cat, records=len(group), image_files=sum(bool(r['original_image_path']) for r in group),
                            independent_parent_families=1, clean=sum(r['augmentation'] == 'clean' for r in group),
                            scanned=sum(r['augmentation'] != 'clean' for r in group),
                            image_truths=';'.join(sorted({r['image_ground_truth'] for r in group}))))
    write_csv(ROOT / 'reports/negative_category_summary.csv', summary)
    stats = dict(categories=len(summary), source_records=len(sources), clean_image_files=sum(r['data_origin'] == 'negative_clean' and r['image_present'] for r in records),
                 scanned_image_files=len(aug), total_image_files=sum(r['image_present'] for r in records),
                 independent_source_parents=1, missing_image_records=sum(not r['image_present'] for r in records),
                 clean_unique_file_hashes=len({r['image_sha256'] for r in records if r['data_origin'] == 'negative_clean' and r['image_present']}),
                 clean_byte_identical_to_base=sum(r['byte_identical_to_base'] for r in records),
                 truth_counts=dict(Counter(r['image_ground_truth'] for r in records)),
                 clean_annotation_only_records=sum(r['data_origin'] == 'negative_clean' and r['annotation_scope'] == 'metadata_or_annotation_only' for r in records),
                 existing_frozen_test_pages_excluded=len(excluded), classifier_justified=False)
    write_json(ROOT / 'reports/audit_counts.json', stats)
    write_json(ROOT / 'configs/historical_lock.json', historical_lock())
    manifest_lock = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'manifests').glob('*.csv')}
    write_json(ROOT / 'configs/manifest_lock.json', manifest_lock)
    # Self-contained clean fixture gallery; scanned entries link to archive members.
    gallery = ['<!doctype html><meta charset="utf-8"><title>Negative fixture audit</title>',
               '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}article{border-top:1px solid #aaa;padding:20px}img{max-width:400px;max-height:350px}pre{white-space:pre-wrap}</style>',
               '<h1>Negative fixtures: annotation errors are not image invalidity</h1>']
    shown_scanned = set()
    for r in records:
        if r['data_origin'] == 'negative_scanned':
            if r['invalidity_category'] in shown_scanned:
                continue
            shown_scanned.add(r['invalidity_category'])
        gallery.append('<article><h2>' + html.escape(r['image_id']) + '</h2><p>' + html.escape(r['image_ground_truth']) + '</p>')
        if r['readable']:
            mime = 'image/jpeg' if r['data_origin'] == 'negative_scanned' else 'image/png'
            gallery.append('<img src="data:' + mime + ';base64,' + base64.b64encode(asset_bytes(r['original_image_path'])).decode() + '">')
        gallery.append('<pre>' + html.escape(r['existing_annotation']) + '</pre></article>')
    (ROOT / 'galleries/negative_examples.html').write_text('\n'.join(gallery), encoding='utf-8')
    audit_text = '# Negative dataset audit\n\n' + json.dumps(stats, indent=2) + '''

Canonical sources: C:\\FYP Dataset Generation\\datasets\\talim_edge_cases_negative_v1
and the corresponding clean/scanned ZIP files. Clean ZIP is an archival copy,
not extra independent samples. Scanned manifest source_id values identify 51
records, NOT independent parents. negative_expectations.json explicitly traces
all 49 mutations to embedded base A11/R19, S20/B10. The second duplicate-ID
fixture changes A11/R19 to A12/R18 and remains in this same derivation family.

50 clean files include one unreadable corrupt PNG; one image is absent.
147 scanned variants derive from 49 readable source records; artifact manifest
retains the missing/corrupt cases. 45 clean files have exactly the base hash.
All clean files were inspected for readability/mode/ink and hash equality;
this is an asset audit, not an OCR evaluation. Scanned hashes and manifest
source hashes were checked. No model inference or classifier training ran.

The inventory's image_ground_truth governs image metrics. The original negative
fixture label is retained verbatim in existing_annotation. JSON count/type/range,
directions, symbols, schema, row sums and canonical line errors are detectable
with the original check_sample validator. TXT mismatches/separators, duplicate
IDs/semantics, missing artifacts and rendering metadata require artifact-level
validation; they cannot be inferred from unchanged image pixels.

Corrupt bytes and blank clean PNG are artifact-level invalid inputs suitable for
input rejection (not learned Talim-content rejection). Wrong dimensions can be
resized valid content and is marked ambiguous. RGB mode is an unsupported source
representation, not invalid Talim; the inference adapter converts to grayscale.
No demonstrably invalid Talim-content images were established. No notation is
declared invalid merely because its JSON mutation mentions an unsupported symbol.
Some unsupported/ambiguous notation exists in annotations, not rendered pixels.
Scanned variants of blank inputs are unadjudicated, because noise is not Talim.

Classifier: NOT JUSTIFIED. One derivation family cannot supply disjoint train,
validation and evaluation negative parents. Metadata mutations on identical
positive pixels introduce contradictory image labels. No classifier modules,
training commands, fake checkpoints, or negative blank-CTC targets are created.

Full recognizer evaluation, confidence calibration and comparative results remain
unexecuted. New independently authored, image-adjudicated invalid parents are
required for content-rejection calibration and independent efficacy evidence.
'''
    (ROOT / 'reports/negative_dataset_audit.md').write_text(audit_text, encoding='utf-8')
    write_protocol(manifests)
    print(json.dumps(stats, indent=2))


def write_protocol(rows):
    text = '''# Stage 2F evaluation protocol (frozen before inference)

The negative source family is assigned entirely to evaluation-only diagnostic
coverage. It has no negative development counterpart: splitting mutations or
augmentations across partitions would leak the source parent. These evaluation
fixtures are historically exposed adversarial data, NOT an independent test.

Positive edge controls use the historical Stage2D tuning vs validation assignment:
tuning goes to development; validation goes to evaluation. E1 already used these
for training/selection, so both are exposed evidence. Clean synthetic historical
training regression parents split by deterministic SHA256 parity of parent ID.
All variants of a parent stay together. Frozen positive test assignments are
excluded entirely; no test images are read or copied. General scanned exploratory
parents are not included in this focused clean/positive control manifest.

Source annotations are OFFLINE scoring data only. They never enter preprocessing,
OCR, structural decisions or confidence. A source's fixture category never drives
runtime decisions. All 198 fixture records (197 files + one missing image) remain
in the inventory, including metadata-only cases. Metadata-only fixtures are scored
as valid image controls where pixels match valid base content. Unsupported/ambiguous
image representations expect REVIEW, and are outside binary content metrics.

F0: D1 proposal + E1 greedy OCR; unreadable/no-ink input rejected, failed D1
proposal sent to REVIEW (D1 itself only exposes proposal accepted/not accepted).
F1: F0 plus observable tier checks; stripped tier offsets prohibit cross-tier
character-position checks. Aligned canonical validation is available only when
true alignment is explicitly supplied. F2 adds configurable quality thresholds;
low confidence causes REVIEW, never a claim of proven invalid content.
F3 absent because no classifier experiment is justified.

Three outcomes are always retained. Rejection recall = REJECT_INVALID / invalid
image inputs; FAR = ACCEPT / invalid inputs; review rate reported separately.
Positive acceptance = ACCEPT / valid images; false rejection = REJECT_INVALID /
valid images; positive review = REVIEW / valid images. Nonacceptance = reject +
review, separately named. Unsupported/ambiguous/unadjudicated/missing entries
are excluded from binary metrics and remain in the three-way/expected-outcome
tables. An absent artifact is not an attempted OCR image. Metrics distinguish
artifact-invalid from Talim-content-invalid; absent content denominators are null.

Accepted-output correctness = exact nonempty tier text + row/role agreement with
source canonical annotation among accepted valid pages with scorable truth.
Leading blank cells are stripped independently exactly as E1 targets; internal
spaces remain significant. This verifies transcription/roles, not pixel alignment.
End-to-end page correctness requires ACCEPT and exact complete row/role/text
agreement for valid pages, or the expected action for invalid/uncertain inputs.
Wrong segmentation cannot be hidden by discarding failed pages. Counts are
page/image weighted plus macro parent rates; variant duplicates are not independent.
Reason accuracy is null unless image-level reason-code labels were independently
provided; source mutation labels are not runtime reason ground truth.

Thresholds may use development outputs only. The calibration tool checks the exact
frozen development manifest, image IDs, input hashes and lineage. Default content
calibration refuses this dataset because it has no independent invalid development
parents. An explicitly requested --allow-positive-only quality gate uses a positive
development quantile and is labeled quality-only, NOT calibrated validity or
negative recall optimization. The evaluation partition is never used to select it.
No classifier or recognizer training is performed by any Stage 2F script.
'''
    (ROOT / 'protocol/evaluation_protocol.md').write_text(text, encoding='utf-8')
    leakage = '''# Leakage audit

All negative records share one original embedded family. All clean/scanned
variants and duplicate IDs/semantics stay in evaluation. Variant IDs are not
independent source parents. Negative split is not a train/test experiment.

Historical positive assignment maps and Stage2C parent/page manifests were read.
Protected test parents are excluded; historical tuning/validation assignments
are preserved. Stage2E full-development model selection is explicitly recorded
as exposed, including clean regression. No new independent test is claimed.

Overlap checks: negative content byte hashes against the historical selected
page source/image hashes; source parent IDs are namespaced by data origin.
E1 training/evaluation tier manifests describe cropped tiers, so raw page hash
nonmatches cannot exclude semantic or near-duplicate overlap with training.
The Stage1 full dataset was NOT swept. Global semantic nonoverlap remains unknown;
use exposed-development interpretation even if selected-page hash overlaps are zero.
Complete historical lock covers integration code/config/manifests, source fixtures,
original Stage1 and E1 checkpoints; it is a scoped contract lock, not a claim that
every historical large ZIP/derived image was rehashed. No historical files are written.
'''
    historical = read_csv(PROJECT / 'stage2/preprocessing_improvements/manifests/pages.csv')
    hashes = {p[k] for p in historical for k in ('source_sha256', 'image_sha256')}
    overlaps = [r['image_id'] for r in rows if r['status'] == 'negative_fixture' and r['image_sha256'] in hashes]
    leakage += '\nSelected historical page byte-hash matches: ' + repr(overlaps) + '\n'
    (ROOT / 'reports/leakage_audit.md').write_text(leakage, encoding='utf-8')


def validate_prepared():
    lock = load_json(ROOT / 'configs/manifest_lock.json')
    for name, h in lock.items():
        if digest(ROOT / name) != h:
            raise ValueError('Frozen manifest drift: ' + name)
    dev = read_csv(ROOT / 'manifests/development_manifest.csv')
    ev = read_csv(ROOT / 'manifests/evaluation_manifest.csv')
    if {r['source_parent_id'] for r in dev} & {r['source_parent_id'] for r in ev}:
        raise ValueError('Parent leakage')
    if ({r['image_sha256'] for r in dev if r['image_sha256']} &
            {r['image_sha256'] for r in ev if r['image_sha256']}):
        raise ValueError('Byte-identical images cross partitions')
    ids = [r['image_id'] for r in dev + ev]
    if len(ids) != len(set(ids)) or any(set(FIELDS) - set(r) for r in dev + ev):
        raise ValueError('Manifest ID/schema error')
    assignments = load_json(PROJECT / 'stage2/positive_edge_cases/manifests/parent_assignments_complete_v1.json')['assignments']
    archive_members = defaultdict(set)
    for r in dev + ev:
        if r['source_parent_id'].startswith('positive:') and assignments[r['source_parent_id'].split(':', 1)[1]] == 'test':
            raise ValueError('Protected test exposure')
        path = r['original_image_path']
        if not path:
            continue
        if '::' in path:
            archive, member = path.split('::', 1)
            archive_members[archive].add(member)
        elif not Path(path).is_file():
            raise FileNotFoundError(path)
    for archive, members in archive_members.items():
        with zipfile.ZipFile(archive) as z:
            for member in members:
                z.getinfo(member)  # one directory lookup per archive, no image/data sweep
    return dict(manifest_rows=len(ids), development=len(dev), evaluation=len(ev),
                parent_disjoint=True, frozen_test_excluded=True, historical_files_verified=verify_integrity())


if __name__ == '__main__':
    main()
