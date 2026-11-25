"""NAQSHKASH V8: V7 recipe + 36 recovered training tiers (inputs v2).

Pre-declared, everything else identical to V7 (run_v7.py):
  - start from V7 best (outputs/v7_run/training/completion.json, step 1750)
  - contrast-normalised student inputs, learning rates x0.5 (inherited from run_v7)
  - training data: collection_v3_inputs_v2.csv (old 160 + 36 recovered train tiers)
  - 750-2500 updates, evaluation every 250, patience 4
  - selection: validation macro-writer raw CER on the SAME 65 tiers (W05 W24 W30 W31)
Test writers are never loaded.

Usage (v6 folder):
  python -B run_v8.py --check
  python -B run_v8.py            (interrupted? python -B run_v8.py --resume)
Afterwards:  python -B run_v8_final.py
"""
import sys, json, argparse
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v7 as V7         # noqa: E402  (applies contrast + lr x0.5 patches, sets R.CFG)
import inputs_v2            # noqa: E402
R, e, torch = V7.R, V7.e, V7.torch
inputs_v2.use(R)

V7_DONE = json.loads((HERE / 'outputs/v7_run/training/completion.json').read_text())
CFG8 = V7.CFG7
CFG8.update(version='V8-recovered', initial_checkpoint=V7_DONE['best_checkpoint'], initial_checkpoint_sha256=V7_DONE['checkpoint_sha256'], inputs='collection_v3_inputs_v2.csv')
CFG8['training'].update(min_updates=750, max_updates=2500, evaluate_every=250, patience=4)
R.CFG = CFG8


def report(run, comp):
    tr, val, replay, dev = R.load_inputs(); rows = []
    for name, path in [('V7', V7_DONE['best_checkpoint']), ('V8', comp['best_checkpoint'])]:
        m, v, _ = e.v2.load_model(path)
        met, _, _ = R.evaluate(m, v, val, run / 'final_eval', name)
        _, sr, _ = e.evaluate(m, v, dev, run / 'regression', name, True); reg = e.synthetic_metrics(sr, dev)
        rows.append(dict(model=name, macro=met['role_mask']['macro_writer']['cer'], overall=met['role_mask']['overall']['cer'], exact=met['role_mask']['overall']['exact_accuracy'],
                         count=met['role_mask']['per_role']['count']['cer'], per_writer={w: round(x['cer'] * 100, 1) for w, x in met['role_mask']['per_writer'].items()},
                         clean_synthetic=reg['clean_historical_regression']['cer']))
    e.dump(run / 'v8_summary.json', dict(rows=rows, best_step=comp['best_step'], checkpoint=comp['best_checkpoint'], checkpoint_sha256=comp['checkpoint_sha256']))
    print('| Model (CRNN only, contrast + role mask) | Macro CER | Overall CER | Exact | Count CER | Clean synthetic CER |'); print('|---|---:|---:|---:|---:|---:|')
    for r in rows: print(f"| {r['model']} | {r['macro']*100:.1f}% | {r['overall']*100:.1f}% | {r['exact']*100:.1f}% | {r['count']*100:.1f}% | {r['clean_synthetic']*100:.2f}% |")
    print('per writer:', {r['model']: r['per_writer'] for r in rows})
    print('NEXT: python -B run_v8_final.py', flush=True)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--check', action='store_true'); a.add_argument('--resume', action='store_true'); args = a.parse_args()
    if args.check:
        tr, val, _, _ = R.load_inputs(); extra = R.load_extra()
        assert e.sha(CFG8['initial_checkpoint']) == CFG8['initial_checkpoint_sha256']
        m, v, _ = e.v2.load_model(CFG8['initial_checkpoint'])
        with torch.inference_mode(): m(e.v2.tensors(val[0])[None])
        rec = [r for r in tr if 'tier_crops_recovered' in r.get('rel_path', '')]
        print('CHECK OK V8 starts from', CFG8['initial_checkpoint'], '| train tiers', len(tr), '(recovered', len(rec), ') | validation', len(val), '| W09 extra', len(extra)); return
    run = HERE / 'outputs' / 'v8_run'
    comp = R.train(run / 'training', resume=args.resume)
    report(run, comp)


if __name__ == '__main__':
    main()
