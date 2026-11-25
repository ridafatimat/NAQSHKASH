"""MARK FUSION: combine the CRNN's own probabilities with the mark classifier, instead of
letting the classifier overrule the CRNN only when p >= 0.6.

For every geometry-placed mark k (k-th mark the CRNN emitted):
  p_crnn(c) = CRNN softmax at the frame where that mark is most confident, restricted to
              the tier's marks and renormalised
  p_cls(c)  = mark classifier on the slot's ink (restricted to the tier's marks)
  new mark  = argmax_c [ log p_crnn(c) + log p_cls(c) ]       (no threshold, no tuning)
If the CRNN's emitted marks cannot be matched one-to-one with the placed marks, the plain
mark classifier step (threshold 0.6) is used for that tier, exactly as today.
"""
import numpy as np, cv2 as cv, torch
from mark_fusion import MarkFixer, slot_crops, glyph_ref_height, prep, ROLE_IDX, MARKS


def crnn_mark_dists(prob, v, role, allowed_chars):
    """Greedy role-masked CTC path; for each emitted non-space char, its distribution over role marks."""
    idx = v.char_to_index; blank = v.blank_index
    keep = [blank] + [i for ch, i in idx.items() if ch in allowed_chars]
    p = np.full_like(prob, -1.0); p[:, keep] = prob[:, keep]; best = p.argmax(1); inv = {i: ch for ch, i in idx.items()}
    role_cols = [idx[MARKS[j]] for j in ROLE_IDX[role]]
    chars, dists, prev, run = [], [], blank, []
    for t, k in enumerate(list(best) + [blank]):
        if k != prev:
            if prev != blank and inv[prev] != ' ':
                f = max(run, key=lambda tt: prob[tt, prev]); d = prob[f, role_cols].astype(np.float64) + 1e-6
                chars.append(inv[prev]); dists.append(d / d.sum())
            run = []
        run.append(t); prev = k
    return ''.join(chars), dists


class FusedMarkFixer(MarkFixer):
    def __init__(self, weights):
        super().__init__(weights); self.ctx = {}; self.used = 0; self.fallback = 0

    def set_context(self, mark_path, chars, dists): self.ctx[str(mark_path)] = (chars, dists)

    def fix(self, placed, role, mark_path, count_path, centres):
        chars, dists = self.ctx.get(str(mark_path), ('', []))
        s = list(placed); jobs = [j for j, ch in enumerate(s) if ch != ' ']
        if ''.join(s[j] for j in jobs) != chars or len(dists) != len(jobs):
            self.fallback += 1; return super().fix(placed, role, mark_path, count_path, centres)
        mi = cv.imread(str(mark_path), 0); ref = glyph_ref_height(cv.imread(str(count_path), 0)); crops = slot_crops(mi, centres)
        if not all(j in crops for j in jobs):
            self.fallback += 1; return super().fix(placed, role, mark_path, count_path, centres)
        X = torch.from_numpy(np.stack([prep(crops[j], ref) for j in jobs]))[:, None]
        with torch.no_grad(): logit = self.m(X)
        idx = ROLE_IDX[role]; pc = logit[:, idx].softmax(1).numpy(); n = 0
        for j, a, b in zip(jobs, pc, dists):
            ch = MARKS[idx[int(np.argmax(np.log(a + 1e-6) + np.log(b)))]]
            if ch != s[j]: s[j] = ch; n += 1
        self.used += 1
        return ''.join(s), n
