"""Manually executed frozen D1 + E1 page inference. Never uses negative CTC labels."""
import argparse
import json
import time
from common import *
from confidence import from_logits, page_signals
from rejection_pipeline import decide
from talim_validator import validate_tiers


def load_recognizer(device='cpu'):
    import torch
    from models.crnn import CRNN
    from training.manifest_training import load_vocabulary
    if digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError('Selected E1 checkpoint hash mismatch')
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    vocab, vh = load_vocabulary(PROJECT / 'config/vocabulary.json')
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    if state['char_to_idx'] != vocab.char_to_index or state['vocabulary_hash'] != vh or vocab.blank_index != 0 or len(vocab.char_to_index) != 65:
        raise ValueError('E1 vocabulary incompatibility')
    model = CRNN(66)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.requires_grad_(False)
    model.eval().to(device)
    return model, vocab, device


class FrozenPipeline:
    def __init__(self, device='cpu'):
        import torch
        import cv2
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        self.model, self.vocab, self.device = load_recognizer(device)
        sys.path.insert(0, str(D))
        self.d1 = load_module('stage2f_d1_adapter', D / 'pipeline_d1.py')
        self.config = load_json(D / 'configs/D1.json')

    def observe(self, raw):
        import cv2
        import numpy as np
        import torch
        from data.collate import resize_preserve_aspect_ratio
        from data.ctc_decode import greedy_ctc_decode
        observation = dict(preprocessing_status='unreadable', segmentation_accepted=False,
            segmentation_reason='', segmentation_rows=[], diagnostics={}, ocr_status='not_run',
            decoded_rows=[], tiers=[], confidence={})
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE) if raw else None
        if image is None:
            return observation
        if image.min() == 255:
            observation['preprocessing_status'] = 'no_ink'
            return observation
        out = self.d1.process_image(image, self.config)
        observation.update(preprocessing_status='ok', segmentation_accepted=bool(out['accepted']),
                           segmentation_reason=out['reason'], segmentation_rows=out['rows'], diagnostics=out['diagnostics'])
        if not out['accepted']:
            observation['confidence'] = page_signals([], out['diagnostics'])
            return observation
        rows, tiers = [], []
        with torch.inference_mode():
            for ri, row in enumerate(out['rows']):
                decoded = {}
                for tier in row:
                    x0, y0, x1, y1 = tier['box']
                    crop = out['image'][y0:y1, x0:x1]
                    tensor = torch.from_numpy(crop.astype(np.float32) / 255).unsqueeze(0)
                    tensor = resize_preserve_aspect_ratio(tensor).unsqueeze(0)
                    if tensor.shape[-1] < 4:
                        observation['ocr_status'] = 'crop_too_narrow'
                        return observation
                    logits = self.model(tensor.to(self.device))[:, 0]
                    prediction = greedy_ctc_decode(logits.argmax(-1).tolist(), self.vocab)
                    decoded[tier['role']] = prediction
                    tiers.append(dict(row=ri, role=tier['role'], prediction=prediction, box=tier['box'],
                                      signals=from_logits(logits), resized_width=tensor.shape[-1]))
                rows.append(decoded)
        observation.update(ocr_status='ok', decoded_rows=rows, tiers=tiers,
                           confidence=page_signals(tiers, out['diagnostics']))
        return observation


def expected_rows(record):
    """Offline transcription scoring only, after the label-blind observation."""
    if record['image_ground_truth'] != 'valid':
        return None
    source = record['json_path']
    if record['status'] == 'negative_fixture' and not ('duplicate_id' in source and 'second' in source):
        source = str(DATA / 'talim_edge_cases_negative_v1/actual_symbols/json/negative.json')
    try:
        sample = json.loads(text_asset(source))
        from talim.talim_rules import canonical_row
        return [{role: text.lstrip(' ') for role, text in zip(('upper', 'count', 'lower'), canonical_row(r['runs'])) if text.strip()}
                for r in sample['logical_rows']]
    except (OSError, KeyError, TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='cpu')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--image-id', action='append', default=[])
    args = ap.parse_args()
    # Enforce frozen manifests and keep arbitrary external test inputs out.
    allowed = {ROOT / 'manifests/development_manifest.csv', ROOT / 'manifests/evaluation_manifest.csv'}
    if args.manifest.resolve() not in {p.resolve() for p in allowed}:
        raise ValueError('Use a frozen Stage2F manifest')
    from prepare import validate_prepared
    validate_prepared()
    records = read_csv(args.manifest)
    if args.smoke:
        if not 1 <= len(args.image_id) <= 3 or len(set(args.image_id)) != len(args.image_id):
            raise ValueError('Smoke requires one to three distinct explicitly selected --image-id values')
        selected = {r['image_id']: r for r in records}
        if set(args.image_id) - set(selected):
            raise ValueError('Unknown smoke ID')
        records = [selected[i] for i in args.image_id]
    elif args.image_id:
        raise ValueError('--image-id is only supported with --smoke')
    out = new_output(args.output_dir)
    pipeline = FrozenPipeline(args.device)
    start = time.perf_counter()
    with (out / 'observations.jsonl').open('x', encoding='utf-8') as f:
        for i, record in enumerate(records):
            path = record['original_image_path']
            raw = asset_bytes(path) if path else None
            if raw is not None and sha_bytes(raw) != record['image_sha256']:
                raise ValueError('Input hash mismatch: ' + record['image_id'])
            obs = pipeline.observe(raw) if path else dict(preprocessing_status='missing_artifact',
                segmentation_accepted=False, segmentation_reason='missing artifact', ocr_status='not_run',
                decoded_rows=[], tiers=[], confidence={}, segmentation_rows=[], diagnostics={})
            # Labels first appear here; never passed to observe() or decide().
            prediction = decide(obs, 'F0')
            truth = expected_rows(record)
            item = dict(record=record, observation=obs, F0=prediction, truth_rows=truth,
                        structural_validation=validate_tiers(obs['decoded_rows']),
                        expected_action_match=prediction['outcome'] == record['expected_system_outcome'],
                        transcription_exact=(obs['decoded_rows'] == truth) if truth is not None else None)
            f.write(json.dumps(item, allow_nan=False) + '\n')
            f.flush()
            print(f'{i+1}/{len(records)} {record["image_id"]} {prediction["outcome"]}', flush=True)
    verify_integrity()
    write_json(out / 'completion.json', dict(complete=True, rows=len(records), smoke=args.smoke,
        manifest=str(args.manifest.resolve()), manifest_sha256=digest(args.manifest),
        checkpoint_sha256=CHECKPOINT_SHA, d1_config_sha256=digest(D / 'configs/D1.json'),
        seconds=time.perf_counter() - start, device=pipeline.device, recognizer_frozen=True,
        observations_sha256=digest(out / 'observations.jsonl'),
        software={str(p.relative_to(ROOT)): digest(p) for p in (ROOT / 'src').glob('*.py')}))
    print('Completed:', out)


if __name__ == '__main__':
    main()
