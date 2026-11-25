"""Geometry-based mark placement for V6 (no training), measured on VALIDATION.

"Geometry places, the model names":
  1. Count crop: ink components clustered in x -> glyph x-centres.
     Used only when the number of geometric glyphs equals the length of the
     model's count prediction for that row.
  2. Upper / lower crop: every ink component (specks < 35% of the median
     component area ignored) is assigned to the nearest glyph centre ->
     the set of glyph slots that carry a mark.
  3. The model's marks (its prediction with spaces removed) are written into
     those slots in left-to-right order, only when #slots == #marks.
     Otherwise the model's own string is kept unchanged.
Settings were fixed on TRAIN crops only (slot accuracy 46/46 rows there).
Input: contrast-normalised crops + role-mask decoding (the current best setup).

Usage (v6 folder):  python -B run_v6_geo.py [--checkpoint path.pt]
"""
import sys, json, collections, argparse
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R          # noqa: E402
import run_v6_post as PP    # noqa: E402
e, cv = R.e, R.cv
GAP_FRAC, SPECK_FRAC, WIDTH_TOL = 0.05, 0.35, 0.03


def ink_components(path):
    g = cv.imread(path, 0); H, W = g.shape
    t, _ = cv.threshold(g, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    b = (g < min(t, 210)).astype(np.uint8)
    n, lab, st, _ = cv.connectedComponentsWithStats(b, 8)
    cs = [dict(x0=int(st[k, 0]), x1=int(st[k, 0] + st[k, 2]), y0=int(st[k, 1]), y1=int(st[k, 1] + st[k, 3]), a=int(st[k, 4])) for k in range(1, n) if st[k, 4] >= 6]
    return cs, W


def glyph_centres(path):
    cs, W = ink_components(path)
    if not cs: return [], W
    cs.sort(key=lambda c: c['x0']); tall = np.percentile([c['y1'] - c['y0'] for c in cs], 80); gap = GAP_FRAC * tall; out = []
    for c in cs:
        if out and c['x0'] <= out[-1]['x1'] + gap: out[-1]['x1'] = max(out[-1]['x1'], c['x1']); out[-1]['x0'] = min(out[-1]['x0'], c['x0'])
        else: out.append(dict(c))
    return [(o['x0'] + o['x1']) / 2 / W for o in out], W


def mark_slots(path, centres):
    cs, W = ink_components(path)
    if not cs: return [], W
    med = np.median([c['a'] for c in cs]); cs = [c for c in cs if c['a'] >= SPECK_FRAC * med]
    ctr = np.array(centres)
    return sorted({int(np.argmin(np.abs(ctr - (c['x0'] + c['x1']) / 2 / W))) for c in cs}), W


def main():
    a = argparse.ArgumentParser(); a.add_argument('--checkpoint'); args = a.parse_args()
    tr, val, replay, dev = R.load_inputs()
    ck = args.checkpoint or json.loads((HERE / 'outputs/v6_run/training/completion.json').read_text())['best_checkpoint']
    m, v, _ = e.v2.load_model(ck); m.eval()
    out = HERE / 'outputs' / 'v6_geo'; out.mkdir(parents=True, exist_ok=True)
    pred, prob = {}, {}
    for r in val:
        p = PP.probs(m, PP.variants(e.v2.tensors(r))['contrast']); prob[r['id']] = p
        pred[r['id']] = e.h.decode(p, v, r['role'], True)
    by = collections.defaultdict(dict)
    for r in val: by[(r['source_id'], r['row_index'])][r['role']] = r
    geo = dict(pred); log = collections.Counter()
    for key, g in by.items():
        if 'count' not in g: log['no_count_tier'] += sum(1 for k in g if k != 'count'); continue
        centres, Wc = glyph_centres(g['count']['path'])
        if len(centres) != len(pred[g['count']['id']]): log['count_geometry_mismatch'] += sum(1 for k in g if k != 'count'); continue
        for role in ('upper', 'lower'):
            if role not in g: continue
            r = g[role]; marks = pred[r['id']].replace(' ', '')
            slots, Wm = mark_slots(r['path'], centres)
            if abs(Wm - Wc) > WIDTH_TOL * Wc: log['width_mismatch'] += 1; continue
            if not marks or len(slots) != len(marks): log['slot_mark_mismatch'] += 1; continue
            s = [' '] * (max(slots) + 1)
            for ch, j in zip(marks, slots): s[j] = ch
            geo[r['id']] = ''.join(s).strip(' '); log['placed'] += 1; log['changed'] += geo[r['id']] != pred[r['id']]
    res = {}
    for name, P in [('contrast', pred), ('contrast+geometry', geo)]:
        rows = []
        for r in val:
            row = e.h.result_row(r, P[r['id']], prob[r['id']], v); row['writer_id'] = r['writer_id']; rows.append(row)
        met = e.all_metrics(rows); e.write(out / f'{name}.csv', rows)
        res[name] = dict(macro_writer_CER=met['macro_writer']['cer'], overall_CER=met['overall']['cer'], exact=met['overall']['exact_accuracy'],
                         le1=sum(int(x['edit_distance']) <= 1 for x in rows) / len(rows), per_role={k: x['cer'] for k, x in met['per_role'].items()}, per_writer={k: x['cer'] for k, x in met['per_writer'].items()})
    e.dump(out / 'geo_summary.json', dict(results=res, placement_log=dict(log), checkpoint=str(ck), settings=dict(gap=GAP_FRAC, speck=SPECK_FRAC, width_tol=WIDTH_TOL)))
    print('placement:', dict(log))
    print('| Variant | Macro CER | Overall CER | Exact | <=1 error | Count | Upper | Lower |'); print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for n, x in res.items():
        p = x['per_role']; print(f"| {n} | {x['macro_writer_CER']*100:.1f}% | {x['overall_CER']*100:.1f}% | {x['exact']*100:.1f}% | {x['le1']*100:.1f}% | {p['count']*100:.1f}% | {p['upper']*100:.1f}% | {p['lower']*100:.1f}% |")
    print('per writer:', {n: {w: round(c * 100, 1) for w, c in x['per_writer'].items()} for n, x in res.items()})


if __name__ == '__main__':
    main()
