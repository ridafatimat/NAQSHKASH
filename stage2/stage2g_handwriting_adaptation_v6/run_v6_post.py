"""NAQSHKASH V6 inference-only improvements, measured on VALIDATION only.

Variants (same V6 checkpoint, no training):
  base        identity input, role-mask greedy decoding (should reproduce v6_summary)
  contrast    per-crop contrast stretch (black point -> 0) before the model
  tta         average posteriors over identity, contrast, darker (gamma), thicker ink
  +spacing    marks of the upper/lower tier are re-placed under/over the count glyphs
              by x-position (the three tier crops of a row share the same width)

Usage (in the v6 folder):  python -B run_v6_post.py
Writes outputs/v6_post/post_summary.md and .json. Test writers are never loaded.
"""
import sys, json, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_v6 as R  # noqa: E402
e, torch, cv = R.e, R.torch, R.cv



def variants(t):
    a = t.numpy()[0].astype(np.float32)
    ink = a[a < 0.98]; black = float(np.percentile(ink, 5)) if ink.size else 0.0
    con = np.clip((a - black) / max(1 - black, 0.05), 0, 1)
    dark = np.clip(con, 0, 1) ** 1.5
    thick = cv.erode(con, np.ones((2, 2), np.uint8))          # dark ink on white -> min filter thickens strokes
    f = lambda x: torch.from_numpy(np.float32(x))[None]
    return dict(identity=t, contrast=f(con), dark=f(dark), thick=f(thick))


def probs(m, t):
    with torch.inference_mode():
        return m(t[None])[:, 0].float().softmax(1).numpy()


def greedy(prob, v, role, drop_space=False):
    """Role-masked greedy CTC; returns text and the centre frame of every emitted char."""
    idx = v.char_to_index; blank = v.blank_index
    allowed = set(e.h.allowed(role)) - ({' '} if drop_space else set())
    keep = [blank] + [i for ch, i in idx.items() if ch in allowed]
    p = np.full_like(prob, -1.0); p[:, keep] = prob[:, keep]
    best = p.argmax(1); inv = {i: ch for ch, i in idx.items()}
    out, frames, prev, start = [], [], blank, 0
    for t, k in enumerate(list(best) + [blank]):
        if k != prev:
            if prev != blank: out.append(inv[prev]); frames.append((start + t - 1) / 2)
            start = t
        prev = k
    T = prob.shape[0]
    return ''.join(out), [(f + 0.5) / T for f in frames]


def place_marks(marks, glyph_x):
    """Monotonic assignment of marks to distinct glyph slots minimising |dx| (DP)."""
    M, N = len(marks), len(glyph_x)
    if M == 0 or N == 0 or M > N: return None
    INF = 1e9; D = np.full((M + 1, N + 1), INF); D[0, :] = 0; B = np.zeros((M + 1, N + 1), int)
    for i in range(1, M + 1):
        for j in range(i, N + 1):
            take = D[i - 1, j - 1] + abs(marks[i - 1][1] - glyph_x[j - 1]); skip = D[i, j - 1]
            if take <= skip: D[i, j] = take; B[i, j] = 1
            else: D[i, j] = skip
    slots = [' '] * N; i, j = M, N
    while i > 0:
        if B[i, j] == 1: slots[j - 1] = marks[i - 1][0]; i -= 1; j -= 1
        else: j -= 1
    return ''.join(slots).rstrip().lstrip(' ')


def run():
    tr, val, replay, dev = R.load_inputs()
    best = json.loads((HERE / 'outputs/v6_run/training/completion.json').read_text())['best_checkpoint']
    m, v, _ = e.v2.load_model(best); m.eval()
    out = HERE / 'outputs' / 'v6_post'; out.mkdir(parents=True, exist_ok=True)
    P = {}
    for r in val:
        t = e.v2.tensors(r); vs = variants(t); pr = {k: probs(m, x) for k, x in vs.items()}
        pr['tta'] = np.mean([pr['identity'], pr['contrast'], pr['dark'], pr['thick']], axis=0)
        P[r['id']] = dict(row=r, prob=pr, width=cv.imread(r['path'], 0).shape[1])
    rows_by = collections.defaultdict(dict)
    for k, d in P.items(): rows_by[(d['row']['source_id'], d['row']['row_index'])][d['row']['role']] = k
    agree = sum(greedy(P[k]['prob']['identity'], v, P[k]['row']['role'])[0] == e.h.decode(P[k]['prob']['identity'], v, P[k]['row']['role'], True) for k in P)
    print('masked-decoder agreement with engine:', agree, '/', len(P), flush=True)
    results = {}; spacing_used = collections.Counter()
    for pv in ['identity', 'contrast', 'tta']:
        for spacing in [False, True]:
            name = {'identity': 'base', 'contrast': 'contrast', 'tta': 'tta'}[pv] + ('+spacing' if spacing else '')
            rows = []
            for k, d in P.items():
                r, prob = d['row'], d['prob'][pv]
                pred = e.h.decode(prob, v, r['role'], True)
                if spacing and r['role'] in ('upper', 'lower'):
                    grp = rows_by[(r['source_id'], r['row_index'])]
                    if 'count' in grp and abs(P[grp['count']]['width'] - d['width']) <= 0.03 * d['width']:
                        _, gx = greedy(P[grp['count']]['prob'][pv], v, 'count')
                        txt, mx = greedy(prob, v, r['role'], drop_space=True)
                        placed = place_marks(list(zip(txt, mx)), gx)
                        if placed is not None: pred = placed; spacing_used[name] += 1
                row = e.h.result_row(r, pred, prob, v); row['writer_id'] = r['writer_id']; rows.append(row)
            met = e.all_metrics(rows); e.write(out / f'{name}.csv', rows)
            results[name] = dict(macro_writer_CER=met['macro_writer']['cer'], overall_CER=met['overall']['cer'], exact=met['overall']['exact_accuracy'],
                                 per_role={k: x['cer'] for k, x in met['per_role'].items()}, per_writer={k: x['cer'] for k, x in met['per_writer'].items()})
    e.dump(out / 'post_summary.json', dict(results=results, spacing_applied_tiers=dict(spacing_used), checkpoint=best, validation_tiers=len(val), test_writers_used=False))
    lines = ['| Variant | Macro-writer CER | Overall CER | Exact | Count | Upper | Lower |', '|---|---:|---:|---:|---:|---:|---:|']
    for n, x in results.items():
        pr_ = x['per_role']; lines.append(f"| {n} | {x['macro_writer_CER']*100:.1f}% | {x['overall_CER']*100:.1f}% | {x['exact']*100:.1f}% | {pr_.get('count',0)*100:.1f}% | {pr_.get('upper',0)*100:.1f}% | {pr_.get('lower',0)*100:.1f}% |")
    (out / 'post_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8'); print('\n'.join(lines)); print('spacing applied to', dict(spacing_used))


if __name__ == '__main__':
    run()
