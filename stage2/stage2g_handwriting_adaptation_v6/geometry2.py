"""Geometry placement of upper/lower marks, shared by validation and the protected test.

Default (uniform=False, classread=False) is EXACTLY the validated V7 geometry step:
  count crop -> geometric glyph centres, used only if #centres == len(CRNN count);
  mark crop  -> ink slots, CRNN marks written into slots only if #slots == #marks;
  optional mark classifier re-reads every placed slot.

Two optional extensions for the rows the default step skips:
  uniform   : #geometric centres != len(CRNN count)  ->  spread len(CRNN count) centres
              evenly across the count tier's ink extent and place marks on those.
  classread : #slots != #CRNN marks  ->  the mark classifier reads every ink slot
              itself (role-restricted); used only if EVERY slot has p >= 0.6, the
              last slot lies inside the count string, and (when both tiers of the
              row are placed) upper and lower never share a slot. Label check on
              all 37 training rows: upper and lower marks never share a glyph.
"""
import collections
import numpy as np, cv2 as cv, torch
import run_v6_geo as G
from mark_fusion import slot_crops, glyph_ref_height, prep, ROLE_IDX, MARKS, THRESHOLD


def uniform_centres(count_path, n):
    cs, W = G.ink_components(str(count_path))
    if not cs or n <= 0: return []
    x0 = min(c['x0'] for c in cs); x1 = max(c['x1'] for c in cs)
    return [(x0 + (k + 0.5) * (x1 - x0) / n) / W for k in range(n)]


def classifier_read(mfx, role, mark_path, count_path, centres, n_count):
    mi = cv.imread(str(mark_path), 0); ref = glyph_ref_height(cv.imread(str(count_path), 0))
    crops = slot_crops(mi, centres, G.SPECK_FRAC)
    if not crops: return None
    jobs = sorted(crops)
    if jobs[-1] >= n_count: return None
    X = torch.from_numpy(np.stack([prep(crops[j], ref) for j in jobs]))[:, None]
    with torch.no_grad(): logit = mfx.m(X)
    idx = ROLE_IDX[role]; p = logit[:, idx].softmax(1).numpy()
    if (p.max(1) < THRESHOLD).any(): return None
    s = [' '] * (jobs[-1] + 1)
    for j, pj in zip(jobs, p): s[j] = MARKS[idx[int(pj.argmax())]]
    return ''.join(s)


def place(rows, pred, mfx=None, uniform=False, classread=False):
    """rows: dicts with id, source_id, row_index, role, path. pred: id -> string. Returns (new_pred, log)."""
    pred = dict(pred); log = collections.Counter(); by = collections.defaultdict(dict)
    for r in rows: by[(r['source_id'], r['row_index'])][r['role']] = r
    for g in by.values():
        n_marks = sum(1 for k in ('upper', 'lower') if k in g)
        if 'count' not in g: log['skip_no_count_tier'] += n_marks; continue
        cp = g['count']['path']; n = len(pred[g['count']['id']])
        cen, Wc = G.glyph_centres(cp)
        if len(cen) != n:
            if not uniform or n == 0: log['skip_count_mismatch'] += n_marks; continue
            cen = uniform_centres(cp, n)
            if not cen: log['skip_count_mismatch'] += n_marks; continue
            log['uniform_rows'] += 1
        placed, how = {}, {}
        for role in ('upper', 'lower'):
            if role not in g: continue
            r = g[role]; mk = pred[r['id']].replace(' ', ''); slots, Wm = G.mark_slots(r['path'], cen)
            if abs(Wm - Wc) > G.WIDTH_TOL * Wc: log['skip_width'] += 1; continue
            if mk and len(slots) == len(mk):
                s = [' '] * (max(slots) + 1)
                for ch, j in zip(mk, slots): s[j] = ch
                p = ''.join(s)
                if mfx is not None:
                    p, k = mfx.fix(p, role, r['path'], cp, cen); log['marks_changed'] += k
                placed[role], how[role] = p, 'crnn'
            elif classread and mfx is not None and slots:
                p = classifier_read(mfx, role, r['path'], cp, cen, n)
                if p is None: log['classread_rejected'] += 1; continue
                placed[role], how[role] = p, 'classread'
            else:
                log['skip_slot_mismatch'] += 1
        if len(placed) == 2:
            U = {i for i, c in enumerate(placed['upper']) if c != ' '}; L = {i for i, c in enumerate(placed['lower']) if c != ' '}
            if U & L:
                for role in ('upper', 'lower'):
                    if how[role] == 'classread': del placed[role]; log['classread_overlap_reverted'] += 1
        for role, p in placed.items():
            pred[g[role]['id']] = p.strip(' '); log['placed_' + how[role]] += 1
    return pred, log
