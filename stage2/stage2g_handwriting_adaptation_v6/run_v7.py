"""NAQSHKASH V7: fine-tune V6 with contrast-normalised inputs (train = test input).

Pre-declared changes vs V6 (everything else identical to run_v6.py):
  - start from V6 best (step 2000)
  - every image the student model sees is contrast-normalised first
    (black point -> 0), in training, validation and the final comparison;
    the frozen-E1 KL teacher is unchanged
  - learning rates x0.5 (refinement of an already adapted model)
  - 750-2500 updates, evaluation every 250, patience 4
Same data, same writer split, same selection rule as V6.

Usage (v6 folder):
  python -B run_v7.py --check
  python -B run_v7.py           (resume: python -B run_v7.py --resume)
Afterwards: python -B run_v6_geo.py --checkpoint <V7 best>   (printed at the end)
"""
import sys, json, copy, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R  # noqa: E402
e, torch = R.e, R.torch

V6_DONE = json.loads((HERE / 'outputs/v6_run/training/completion.json').read_text())
CFG7 = copy.deepcopy(R.CFG)
CFG7.update(version='V7-contrast', initial_checkpoint=V6_DONE['best_checkpoint'], initial_checkpoint_sha256=V6_DONE['checkpoint_sha256'])
CFG7['training'].update(min_updates=750, max_updates=2500, evaluate_every=250, patience=4)
CFG7['input_normalisation'] = 'contrast: (x - p5_of_ink) / (1 - p5_of_ink), clipped; applied to every student input'
CFG7['learning_rate_scale'] = 0.5
R.CFG = CFG7

_base_config = e.config
def _config():
    c = copy.deepcopy(_base_config()); c['learning_rates'] = {k: v * CFG7['learning_rate_scale'] for k, v in c['learning_rates'].items()}; return c
e.config = _config


def contrast_batch(x):
    a = x.detach().cpu().numpy().astype(np.float32); out = np.empty_like(a)
    for i in range(a.shape[0]):
        s = a[i]; ink = s[s < 0.98]; black = float(np.percentile(ink, 5)) if ink.size else 0.0
        out[i] = np.clip((s - black) / max(1 - black, 0.05), 0, 1)
    return torch.from_numpy(out).to(x.dtype)


_load = e.v2.load_model
def _load_with_contrast(path=None):
    m, v, s = _load(path)
    if path is not None:                       # student / candidate models only; frozen E1 teacher stays raw
        fwd = m.forward
        m.forward = lambda x, *a, **k: fwd(contrast_batch(x), *a, **k)
    return m, v, s
e.v2.load_model = _load_with_contrast


def report(run, comp):
    tr, val, replay, dev = R.load_inputs(); rows = []
    for name, path in [('V6 + contrast', V6_DONE['best_checkpoint']), ('V7 (contrast-trained)', comp['best_checkpoint'])]:
        m, v, _ = e.v2.load_model(path)
        met, _, _ = R.evaluate(m, v, val, run / 'final_eval', name.split()[0])
        _, sr, _ = e.evaluate(m, v, dev, run / 'regression', name.split()[0], True); reg = e.synthetic_metrics(sr, dev)
        rows.append(dict(model=name, macro_role_mask=met['role_mask']['macro_writer']['cer'], overall_role_mask=met['role_mask']['overall']['cer'],
                         exact_role_mask=met['role_mask']['overall']['exact_accuracy'], count=met['role_mask']['per_role']['count']['cer'],
                         per_writer={w: x['cer'] for w, x in met['role_mask']['per_writer'].items()}, clean_synthetic_CER=reg['clean_historical_regression']['cer']))
    e.dump(run / 'v7_summary.json', dict(rows=rows, best_step=comp['best_step'], checkpoint=comp['best_checkpoint'], checkpoint_sha256=comp['checkpoint_sha256']))
    print('| Model | Macro CER (role mask) | Overall CER | Exact | Count CER | Clean synthetic CER |'); print('|---|---:|---:|---:|---:|---:|')
    for r in rows: print(f"| {r['model']} | {r['macro_role_mask']*100:.1f}% | {r['overall_role_mask']*100:.1f}% | {r['exact_role_mask']*100:.1f}% | {r['count']*100:.1f}% | {r['clean_synthetic_CER']*100:.2f}% |")
    print('per writer:', {r['model']: {w: round(c * 100, 1) for w, c in r['per_writer'].items()} for r in rows})
    print('NEXT: python -B run_v6_geo.py --checkpoint "' + comp['best_checkpoint'] + '"', flush=True)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--check', action='store_true'); a.add_argument('--resume', action='store_true'); args = a.parse_args()
    if args.check:
        tr, val, _, _ = R.load_inputs(); m, v, _ = e.v2.load_model(CFG7['initial_checkpoint'])
        assert e.sha(CFG7['initial_checkpoint']) == CFG7['initial_checkpoint_sha256']
        with torch.inference_mode(): m(e.v2.tensors(val[0])[None])
        print('CHECK OK V7 starts from', CFG7['initial_checkpoint'], '| lr', e.config()['learning_rates'], '| train tiers', len(tr), '| validation', len(val)); return
    run = HERE / 'outputs' / 'v7_run'
    comp = R.train(run / 'training', resume=args.resume)
    report(run, comp)


if __name__ == '__main__':
    main()
