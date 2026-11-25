"""Checkpoint averaging for V6 (no training).

Pre-declared choice: uniform average of every V6 checkpoint in the plateau
region (steps 1750-3000). Evaluated on VALIDATION with contrast input and the
role-mask decoder, next to the single best checkpoint (step 2000).
Usage (v6 folder):  python -B run_v6_avg.py
"""
import sys, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R          # noqa: E402
import run_v6_post as PP    # noqa: E402
e, torch = R.e, R.torch

STEPS = [1750, 2000, 2250, 2500, 2750, 3000]


def evaluate_contrast(m, v, val):
    rows = []
    for r in val:
        prob = PP.probs(m, PP.variants(e.v2.tensors(r))['contrast'])
        row = e.h.result_row(r, e.h.decode(prob, v, r['role'], True), prob, v); row['writer_id'] = r['writer_id']; rows.append(row)
    met = e.all_metrics(rows)
    le1 = sum(int(x['edit_distance']) <= 1 for x in rows) / len(rows)
    return dict(macro_writer_CER=met['macro_writer']['cer'], overall_CER=met['overall']['cer'], exact=met['overall']['exact_accuracy'], tiers_at_most_1_error=le1,
                per_role={k: x['cer'] for k, x in met['per_role'].items()}, per_writer={k: x['cer'] for k, x in met['per_writer'].items()}), rows


def main():
    tr, val, replay, dev = R.load_inputs()
    ck = HERE / 'outputs/v6_run/training/checkpoints'
    paths = [ck / f'step_{s:05}.pt' for s in STEPS]
    for p in paths: assert p.exists(), p
    states = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    avg = {}
    for k, t in states[0]['model_state_dict'].items():
        avg[k] = (sum(s['model_state_dict'][k].float() for s in states) / len(states)).to(t.dtype) if t.is_floating_point() else t
    out = HERE / 'outputs/v6_avg'; out.mkdir(parents=True, exist_ok=True)
    save = {k: v for k, v in states[1].items() if k not in ('optimizer_state_dict',)}
    save['model_state_dict'] = avg; save['averaged_steps'] = STEPS
    avg_path = out / 'v6_avg_1750_3000.pt'; torch.save(save, avg_path)
    res = {}
    for name, path in [('V6 step 2000 + contrast', paths[1]), ('V6 average 1750-3000 + contrast', avg_path)]:
        m, v, _ = e.v2.load_model(path); m.eval()
        res[name], rows = evaluate_contrast(m, v, val); e.write(out / (name.split()[1] + '_rows.csv'), rows)
        _, sr, _ = e.evaluate(m, v, dev, out / 'regression', name.split()[1], True)
        res[name]['clean_synthetic_CER'] = e.synthetic_metrics(sr, dev)['clean_historical_regression']['cer']
    e.dump(out / 'avg_summary.json', dict(results=res, averaged_checkpoint=str(avg_path), sha256=e.sha(avg_path)))
    print('| Model | Macro CER | Overall CER | Exact | <=1 error | Count | Upper | Lower | Clean synth CER |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for n, x in res.items():
        p = x['per_role']
        print(f"| {n} | {x['macro_writer_CER']*100:.1f}% | {x['overall_CER']*100:.1f}% | {x['exact']*100:.1f}% | {x['tiers_at_most_1_error']*100:.1f}% | {p['count']*100:.1f}% | {p['upper']*100:.1f}% | {p['lower']*100:.1f}% | {x['clean_synthetic_CER']*100:.2f}% |")


if __name__ == '__main__':
    main()
