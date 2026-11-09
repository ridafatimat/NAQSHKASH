"""Local evaluation of explicitly aligned tier manifests, reusing Stage 1 preprocessing."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

from inspect_local import HERE, STAGE1, check_checkpoint, sha256


def edit_counts(truth, prediction):
    """Minimum Levenshtein alignment; ties prefer substitution, deletion, insertion."""
    table = [[(0, 0, j) for j in range(len(prediction) + 1)]]
    for i, char in enumerate(truth, 1):
        row = [(0, i, 0)]
        for j, pred in enumerate(prediction, 1):
            if char == pred:
                row.append(table[i - 1][j - 1])
            else:
                s, d, ins = table[i - 1][j - 1]
                candidates = [(s + 1, d, ins)]
                s, d, ins = table[i - 1][j]
                candidates.append((s, d + 1, ins))
                s, d, ins = row[j - 1]
                candidates.append((s, d, ins + 1))
                row.append(min(candidates, key=sum))
        table.append(row)
    s, d, ins = table[-1][-1]
    return dict(substitutions=s, deletions=d, insertions=ins)


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot calculate metrics for an empty dataset')
    result = {key: sum(r[key] for r in rows) for key in
              ('substitutions', 'deletions', 'insertions')}
    result.update(samples=len(rows), characters=sum(len(r['truth']) for r in rows),
                  exact_matches=sum(r['truth'] == r['prediction'] for r in rows))
    result['cer'] = sum(result[k] for k in ('substitutions', 'deletions', 'insertions')) / result['characters']
    result['exact_accuracy'] = result['exact_matches'] / result['samples']
    return result


def read_manifest(path, root, vocabulary):
    """Validate every image and its explicit tier label; never infer labels by ordering."""
    import cv2
    root = Path(root).resolve()
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'sample_id', 'parent_id', 'image_path', 'label_path', 'tier_name',
                    'training_label', 'split', 'category', 'width', 'height', 'image_sha256'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'Missing manifest columns: {required - set(reader.fieldnames or [])}')
        rows = list(reader)
    if not rows:
        raise ValueError('Manifest is empty')
    seen, parents, hashes = set(), {}, {}
    for row in rows:
        if not row['sample_id'] or row['sample_id'] in seen or not row['parent_id']:
            raise ValueError('Missing parent/sample identity or duplicate sample')
        seen.add(row['sample_id'])
        if row['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Unknown split')
        if row['tier_name'] not in {'upper', 'count', 'lower'}:
            raise ValueError('Unknown tier')
        if row['parent_id'] in parents and parents[row['parent_id']] != row['split']:
            raise ValueError('Parent leakage across splits')
        parents[row['parent_id']] = row['split']
        for field in ('image_path', 'label_path'):
            candidate = (root / row[field]).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                raise ValueError(f'Invalid or missing {field}: {candidate}')
        image_path = root / row['image_path']
        digest = sha256(image_path)
        if digest != row['image_sha256']:
            raise ValueError('Image hash mismatch')
        if digest in hashes and hashes[digest] != row['split']:
            raise ValueError('Duplicate image leakage across splits')
        hashes[digest] = row['split']
        raw = (root / row['label_path']).read_text(encoding='utf-8').rstrip('\r\n')
        if raw.lstrip(' ') != row['training_label'] or not row['training_label']:
            raise ValueError('Label alignment mismatch or empty target')
        vocabulary.encode(row['training_label'])
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (int(row['height']), int(row['width'])):
            raise ValueError('Invalid image or dimensions')
        width = max(1, round(image.shape[1] * 64 / image.shape[0]))
        text = row['training_label']
        required_steps = len(text) + sum(a == b for a, b in zip(text, text[1:]))
        if width < 4 or required_steps > width // 4:
            raise ValueError(f'CTC infeasible: {row["sample_id"]}')
    return rows


def main():
    import cv2
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from data.collate import talim_collate_fn
    from data.ctc_decode import greedy_ctc_decode
    from training.manifest_training import load_vocabulary
    from models.crnn import CRNN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New directory; existing results are never overwritten')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()
    if a.batch_size < 1:
        p.error('batch-size must be positive')
    if a.output.exists():
        p.error('Output already exists; choose a new directory')
    torch.set_num_threads(4)
    vocabulary, digest = load_vocabulary(STAGE1 / 'config/vocabulary.json')
    identity = check_checkpoint(a.checkpoint, STAGE1 / 'config/vocabulary.json')
    all_rows = read_manifest(a.manifest, a.dataset_root, vocabulary)
    rows = [r for r in all_rows if r['split'] == a.split]
    if not rows:
        raise ValueError('Selected split is empty')
    class Tiers:
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            image = cv2.imread(str(a.dataset_root / row['image_path']), cv2.IMREAD_GRAYSCALE)
            return dict(image=torch.from_numpy(image.astype(np.float32) / 255).unsqueeze(0),
                        label_text=row['training_label'], tier_name=row['tier_name'], sample_id=row['sample_id'])
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device)
    model = CRNN(66).to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True)['model_state_dict'], strict=True)
    model.eval()
    predictions = []
    lookup = {r['sample_id']: r for r in rows}
    with torch.inference_mode():
        for batch in DataLoader(Tiers(), batch_size=a.batch_size, num_workers=0, collate_fn=talim_collate_fn):
            logits = model(batch['images'].to(device))
            ids = logits.argmax(2).cpu()
            lengths = model.output_lengths(batch['image_widths'])
            for index, sample_id in enumerate(batch['sample_ids']):
                row = lookup[sample_id]
                pred = greedy_ctc_decode(ids[:int(lengths[index]), index].tolist(), vocabulary)
                truth = row['training_label']
                predictions.append(dict(sample_id=sample_id, parent_id=row['parent_id'],
                    image_path=row['image_path'], tier_name=row['tier_name'], category=row['category'],
                    truth=truth, prediction=pred, exact=truth == pred, **edit_counts(truth, pred)))
    result = dict(checkpoint=identity, manifest_sha256=sha256(a.manifest), split=a.split,
                  vocabulary_hash=digest, device=str(device), batch_size=a.batch_size,
                  decoding='greedy CTC, true width truncation', overall=aggregate(predictions))
    for field in ('tier_name', 'category'):
        groups = defaultdict(list)
        for row in predictions:
            keys = row[field].split(';') if field == 'category' else [row[field]]
            for key in keys:
                groups[key or 'unknown'].append(row)
        result['per_' + field] = {key: aggregate(group) for key, group in groups.items()}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for filename, selected in [('predictions.csv', predictions), ('errors.csv', [r for r in predictions if not r['exact']])]:
        with (a.output / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(selected)
    with (a.output / 'metrics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['group', 'name'] + list(result['overall']))
        writer.writeheader()
        writer.writerow(dict(group='overall', name='all', **result['overall']))
        for field in ('tier_name', 'category'):
            for key, values in result['per_' + field].items():
                writer.writerow(dict(group=field, name=key, **values))
    print(json.dumps(result['overall'], indent=2))


if __name__ == '__main__':
    main()
