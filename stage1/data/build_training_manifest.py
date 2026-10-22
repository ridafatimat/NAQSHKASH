"""Read-only tier index builder. Only output_dir is written; pixels are never decoded.

Copied scalar semantics (no project imports): data/tier_dataset.py __getitem__
(universal newline read, rstrip CR/LF, lstrip ASCII space); data/collate.py
resize_preserve_aspect_ratio and data/width_sampler.py resized_widths;
models/crnn.py CRNN.output_lengths; data/check_ctc_feasibility.py minimum_ctc_steps.
No transform is configured by training/train_crnn.py or tier_dataloader.py.
Batch padding is white on the right to batch maximum; no width cap/alignment.
Parent IDs are explicit metadata, cross-checked against layout, never inferred.
--limit is a TOTAL scan budget divided across the three splits. Inventories and
build reconciliation always cover all metadata; sample statistics cover scanned rows.
Baseline vocabulary comes from existing validation_report.json TRAIN vocabulary;
the full scan independently verifies it against actual TRAIN targets.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from PIL import Image

SPLITS = ('test', 'train', 'validation')  # required lexical split ordering
FIELDS = ['split', 'tier_id', 'image_path', 'label_path', 'raw_label',
          'training_label', 'tier_name', 'image_width', 'image_height',
          'resized_width', 'label_length', 'parent_image_id', 'row_id',
          'row_index', 'tier_index', 'leading_offset', 'raw_label_length',
          'ctc_required_timesteps', 'crnn_output_length', 'ctc_feasible',
          'unknown_characters', 'image_status', 'label_status']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def labels(path):
    raw = path.read_text(encoding='utf-8').rstrip('\r\n')
    return raw, raw.lstrip(' ')


def resized_width(w, h):
    return max(1, int(round(w * (64 / float(h)))))


def ctc_required(text):
    return len(text) + sum(a == b for a, b in zip(text, text[1:]))


def stats(values):
    values = sorted(values)
    if not values:
        return None
    def quantile(q):
        pos = (len(values) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return dict(count=len(values), min=values[0], max=values[-1],
                quantiles={str(q): quantile(q) for q in (0.01, .05, .25, .5, .75, .9, .95, .99)})


def relative_path(root, split, value):
    # Metadata paths are relative by contract. Lexical validation avoids slow
    # Windows filesystem resolution for every one of 1.3 million references.
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in value:
        raise ValueError('Unsafe metadata path: ' + value)
    return (PurePosixPath(split)/path).as_posix()


def inventory(root, split, folder):
    found = set()
    for directory, _, files in os.walk(root/split/folder):
        rel = Path(directory).relative_to(root).as_posix()
        found.update(rel+'/'+name for name in files)
    return found


def scan_one(args):
    root, split, meta, vocabulary = args
    out = dict.fromkeys(FIELDS, '')
    out.update(split=split, tier_id=meta['tier_id'], tier_name=meta['tier_name'],
               parent_image_id=meta['parent_sample_id'], row_id=meta['row_id'],
               row_index=meta['row_index'], tier_index=meta['tier_index'])
    for key in ('image_path', 'label_path'):
        out[key] = relative_path(root, split, meta[key])
    image = root / out['image_path']
    label = root / out['label_path']
    issues = []
    try:
        # Image.open parses headers only; no load(), verify(), or pixel arrays.
        with Image.open(image) as im:
            w, h = im.size
        if w < 1 or h < 1:
            raise ValueError('Nonpositive dimensions')
        rw = resized_width(w, h)
        out.update(image_width=w, image_height=h, resized_width=rw,
                   crnn_output_length=max(1, rw // 2 // 2), image_status='ok')
        if str(w) != meta['image_width'] or str(h) != meta['image_height']:
            issues.append('metadata_dimension_mismatch')
    except FileNotFoundError:
        out['image_status'] = 'missing'
    except (OSError, ValueError) as e:
        out['image_status'] = 'unreadable_header'
        issues.append('image_error:' + str(e))
    try:
        raw, target = labels(label)
        unknown = ''.join(sorted(set(target) - vocabulary))
        out.update(raw_label=raw, training_label=target, label_length=len(target),
                   raw_label_length=len(raw), leading_offset=len(raw)-len(target),
                   ctc_required_timesteps=ctc_required(target), unknown_characters=unknown,
                   label_status='ok')
        if out['image_status'] == 'ok':
            out['ctc_feasible'] = int(out['crnn_output_length'] >= ctc_required(target))
        if raw != meta['label']:
            issues.append('metadata_label_mismatch')
        if str(len(raw)) != meta['label_length']:
            issues.append('metadata_label_length_mismatch')
    except FileNotFoundError:
        out['label_status'] = 'missing'
    except (OSError, UnicodeError) as e:
        out['label_status'] = 'unreadable'
        issues.append('label_error:' + str(e))
    return out, issues


def build(root, output, limit=None, workers=16):
    root, output = Path(root).resolve(), Path(output).resolve()
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    # Output may be root/manifests or outside dataset, never a source split.
    if output == root or any(output == root/s or root/s in output.parents for s in SPLITS):
        raise ValueError('Output must be separate from original dataset split files')
    report = json.loads((root/'validation_report.json').read_text(encoding='utf-8'))
    vocab = set(report['splits']['train']['vocabulary'])
    if len(vocab) != 65:
        raise ValueError('Expected existing 65 symbols plus CTC blank; inspect vocabulary')
    summary = dict(columns=FIELDS, limit=limit, partial=limit is not None,
                   vocabulary_source='validation_report.json:splits.train.vocabulary',
                   vocabulary=sorted(vocab), vocabulary_classes_including_blank=len(vocab)+1,
                   quantile_method='linear interpolation at (n-1)*q', splits={},
                   semantics=__doc__, source_sha256={})
    selected = []
    parents = {}
    all_ids = Counter()
    for si, split in enumerate(SPLITS):
        base = root/split
        print('Reading metadata/inventory:', split, flush=True)
        for name in ('metadata.csv', 'layout_manifest.csv', 'completed_pages.txt', 'tier_build_failures.csv'):
            summary['source_sha256'][split+'/'+name] = hashlib.sha256((base/name).read_bytes()).hexdigest()
        rows = sorted(read_csv(base/'metadata.csv'), key=lambda r: r['tier_id'])
        layout_rows = read_csv(base/'layout_manifest.csv')
        layout = {r['row_id']: r for r in layout_rows}
        completed_list = (base/'completed_pages.txt').read_text(encoding='utf-8').splitlines()
        completed = set(completed_list)
        failures = read_csv(base/'tier_build_failures.csv')
        failed = {r['sample_id'] for r in failures}
        pc = Counter(r['parent_sample_id'] for r in rows)
        parents[split] = set(pc)
        ids = Counter(r['tier_id'] for r in rows)
        all_ids.update(ids)
        expected_images = {relative_path(root, split, r['image_path']) for r in rows}
        expected_labels = {relative_path(root, split, r['label_path']) for r in rows}
        images = inventory(root, split, 'images')
        label_files = inventory(root, split, 'labels')
        layout_errors = []
        actual_tiers = Counter((r['row_id'], r['tier_name']) for r in rows)
        for r in rows:
            lr = layout.get(r['row_id'])
            if not lr or any(lr[k] != r[k] for k in ('parent_sample_id', 'row_index')):
                layout_errors.append(r['tier_id'])
        expected_absent = []
        unexpected_present = []
        for lr in layout_rows:
            for tier in ('upper', 'count', 'lower'):
                count = actual_tiers[(lr['row_id'], tier)]
                if lr[tier+'_present'] == '1' and count == 0:
                    expected_absent.append(lr['row_id']+'_'+tier)
                if lr[tier+'_present'] == '0' and count:
                    unexpected_present.append(lr['row_id']+'_'+tier)
        info = dict(metadata_rows=len(rows), distinct_parents=len(pc),
                    tiers=dict(sorted(Counter(r['tier_name'] for r in rows).items())),
                    tiers_per_parent=stats(pc.values()), completed_pages=len(completed),
                    duplicate_completed_entries=len(completed_list)-len(completed),
                    completed_without_tiers=sorted(completed-set(pc)),
                    parents_not_completed=sorted(set(pc)-completed),
                    tiers_parent_not_completed=sum(v for p,v in pc.items() if p not in completed),
                    failure_rows=len(failures), failure_reasons=dict(Counter(r['reason'] for r in failures)),
                    failed_pages_with_tiers=sorted(failed & set(pc)),
                    failed_pages_without_tiers=sorted(failed-set(pc)),
                    failed_pages_not_completed=sorted(failed-completed),
                    completed_without_tiers_and_no_failure=sorted(completed-set(pc)-failed),
                    images_without_metadata=sorted(images-expected_images),
                    metadata_missing_images=sorted(expected_images-images),
                    labels_without_metadata=sorted(label_files-expected_labels),
                    metadata_missing_labels=sorted(expected_labels-label_files),
                    duplicate_tier_ids={k:v for k,v in ids.items() if v>1},
                    duplicate_image_paths=len(rows)-len(expected_images),
                    duplicate_label_paths=len(rows)-len(expected_labels),
                    duplicate_layout_rows=len(layout_rows)-len(layout),
                    layout_parent_or_index_mismatches=layout_errors,
                    layout_expected_tiers_absent=expected_absent,
                    layout_unexpected_tiers_present=unexpected_present)
        summary['splits'][split] = info
        n = len(rows) if limit is None else limit//3 + int(si < limit%3)
        selected.extend((root, split, r, vocab) for r in rows[:n])
    summary['duplicate_tier_ids_global'] = {k:v for k,v in all_ids.items() if v>1}
    summary['parent_overlap'] = {a+'/'+b: sorted(parents[a]&parents[b]) for i,a in enumerate(SPLITS) for b in SPLITS[i+1:]}
    output.mkdir(parents=True, exist_ok=True)
    manifest = output/'full_tier_manifest.csv'
    numeric = {k:[] for k in ('image_width', 'resized_width', 'label_length')}
    chars, raw_chars, train_chars = Counter(), Counter(), Counter()
    counts, tier_counts, problems = Counter(), Counter(), {}
    sample_parents = {s:set() for s in SPLITS}
    representatives = set()
    categories = set()
    per_split = {s:Counter() for s in SPLITS}
    with manifest.open('w', encoding='utf-8', newline='') as f, ThreadPoolExecutor(max_workers=workers) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        # Bounded submissions avoid hundreds of thousands of Future objects.
        for start in range(0, len(selected), 2048):
            for row, issues in pool.map(scan_one, selected[start:start+2048]):
                writer.writerow(row)
                split = row['split']
                counts[split] += 1
                tier_counts[row['tier_name']] += 1
                sample_parents[split].add(row['parent_image_id'])
                for k in numeric:
                    if row[k] != '':
                        numeric[k].append(row[k])
                for key in ('image_status', 'label_status'):
                    if row[key] != 'ok':
                        issues.append(key+':'+row[key])
                if row['label_status'] == 'ok':
                    target = row['training_label']
                    chars.update(target)
                    raw_chars.update(row['raw_label'])
                    if split == 'train':
                        train_chars.update(target)
                    if row['unknown_characters']:
                        issues.append('unknown_vocabulary_characters')
                    if not target:
                        issues.append('empty_training_label')
                    if row['ctc_feasible'] == 0:
                        issues.append('ctc_infeasible')
                    cats = [split+'/'+row['tier_name']]
                    if row['leading_offset']: cats.append('leading_spaces')
                    for c, name in [(' ', 'internal_space'), (',','comma'), ('"','quote')]:
                        if c in target: cats.append(name)
                    if len(representatives) < 100 or any(c not in categories for c in cats):
                        representatives.add((split, row['tier_id']))
                        categories.update(cats)
                for issue in issues:
                    problems.setdefault(issue, []).append(split+'/'+row['tier_id'])
                    per_split[split][issue] += 1
            print('Headers/labels: {}/{}'.format(min(start+2048,len(selected)),len(selected)), flush=True)
    # Read the ENTIRE CSV back: check row order, relative paths, count and
    # representative original label files, explicitly including leading spaces.
    previous = None
    roundtrip_count = 0
    back_count = 0
    with manifest.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            identity = row['split'], row['tier_id']
            if previous is not None and identity < previous:
                raise AssertionError('Manifest order is not deterministic')
            previous = identity
            back_count += 1
            if identity in representatives:
                raw, target = labels(root/row['label_path'])
                assert row['raw_label'] == raw and row['training_label'] == target
                roundtrip_count += 1
    assert back_count == len(selected)
    summary.update(total_rows=back_count, rows_per_split=dict(counts), rows_per_tier=dict(tier_counts),
                   scanned_distinct_parents={s:len(p) for s,p in sample_parents.items()},
                   statistics={k:stats(v) for k,v in numeric.items()},
                   character_counts=dict(sorted(chars.items())), raw_character_counts=dict(sorted(raw_chars.items())),
                   train_character_counts=dict(sorted(train_chars.items())),
                   issue_counts={k:len(v) for k,v in problems.items()}, issue_samples=problems,
                   issues_per_split={s:dict(c) for s,c in per_split.items()},
                   roundtrip=dict(rows_read=back_count, original_labels_checked=roundtrip_count,
                                  categories=sorted(categories), passed=True),
                   train_vocabulary_matches_baseline=(set(train_chars)==vocab) if limit is None else None,
                   manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                   header_only_limitation='Header-readable images may still have corrupt pixel payloads; no pixel integrity claim.')
    summary['validation_counts'] = {
        name: len(problems.get(key, [])) for name,key in {
            'missing_images':'image_status:missing',
            'missing_labels':'label_status:missing',
            'unreadable_image_headers':'image_status:unreadable_header',
            'unreadable_labels':'label_status:unreadable',
            'labels_with_unknown_characters':'unknown_vocabulary_characters',
            'empty_training_labels':'empty_training_label',
            'ctc_infeasible':'ctc_infeasible',
            'metadata_label_mismatches':'metadata_label_mismatch',
            'metadata_dimension_mismatches':'metadata_dimension_mismatch',
            'metadata_label_length_mismatches':'metadata_label_length_mismatch',
        }.items()}
    summary['resize'] = dict(target_height=64, formula='max(1, int(round(width * (64 / float(height)))))',
                             maximum_width=None, alignment_multiple=None,
                             padding='white (1.0), right only, to batch maximum; not part of resized_width',
                             module_inconsistencies=[])
    summary['ctc'] = dict(output_length='max(1, resized_width // 2 // 2)',
                          required='len(training_label) + adjacent repeated character pairs')
    summary['assumptions'] = [
        'parent_image_id is metadata.parent_sample_id, confirmed by builder ZIP member stem and layout rows.',
        'raw_label follows TalimTierDataset: universal newline read then rstrip CR/LF; training_label removes only leading ASCII spaces.',
        'No extra source/domain column: none exists in inspected metadata or layout; all data is synthetic Talim.',
        'No external image transform is configured in the current tier training path.',
        'completed_pages means attempted, not necessarily successful (builder behavior).',
        'Partial runs compare against existing validation report vocabulary; full runs independently verify TRAIN character set.',
    ]
    (output/'manifest_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('total_rows','rows_per_split','rows_per_tier','statistics','issue_counts','roundtrip')}, ensure_ascii=False, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets/Final tier dataset'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    build(args.dataset_root, args.output_dir or args.dataset_root/'manifests', args.limit, args.workers)
