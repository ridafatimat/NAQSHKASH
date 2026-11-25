"""Oval sub-type classifier fused with the CRNN count prediction.

For a count tier whose geometric glyph boxes match the CRNN's count length, every
position the CRNN reads as an OVAL glyph (plain O, 1-9 dots, + family, xx family)
is re-read by a small glyph classifier at full crop resolution. The classifier's
answer replaces the CRNN character only if it is also an oval glyph and its
probability >= THRESHOLD. Plain shapes (9, m, 7, ...) are never touched.
"""
import numpy as np, cv2 as cv, torch, torch.nn as nn
from pathlib import Path
OVALS = ['a'] + list('jklmnopqr') + list('stuvwxyz{|') + list("'()*+,")
THRESHOLD = 0.6
GAP_FRAC = 0.05


def prep(g, S=64):
    g = g.astype(np.float32) / 255.0; ink = g[g < 0.98]; blk = float(np.percentile(ink, 5)) if ink.size else 0
    g = np.clip((g - blk) / max(1 - blk, 0.05), 0, 1); h, w = g.shape; sc = (S - 4) / max(h, w)
    r = cv.resize(g, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), interpolation=cv.INTER_AREA)
    c = np.ones((S, S), np.float32); y0 = (S - r.shape[0]) // 2; x0 = (S - r.shape[1]) // 2; c[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return c


class Net(nn.Module):
    def __init__(self, n=len(OVALS)):
        super().__init__(); L = []; ci = 1
        for co in (32, 64, 96, 128):
            L += [nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2)]; ci = co
        self.f = nn.Sequential(*L); self.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x): return self.h(self.f(1 - x).mean((2, 3)))


def glyph_boxes(g):
    t, _ = cv.threshold(g, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU); b = (g < min(t, 210)).astype(np.uint8)
    n, lab, st, _ = cv.connectedComponentsWithStats(b, 8)
    cs = [dict(x0=st[k, 0], x1=st[k, 0] + st[k, 2], y0=st[k, 1], y1=st[k, 1] + st[k, 3]) for k in range(1, n) if st[k, 4] >= 6]
    if not cs: return []
    cs.sort(key=lambda c: c['x0']); tall = np.percentile([c['y1'] - c['y0'] for c in cs], 80); gap = GAP_FRAC * tall; out = []
    for c in cs:
        if out and c['x0'] <= out[-1]['x1'] + gap:
            o = out[-1]; o['x1'] = max(o['x1'], c['x1']); o['y0'] = min(o['y0'], c['y0']); o['y1'] = max(o['y1'], c['y1'])
        else: out.append(dict(c))
    return out


class OvalFixer:
    def __init__(self, weights):
        d = torch.load(weights, map_location='cpu', weights_only=True); assert d['classes'] == OVALS
        self.m = Net(); self.m.load_state_dict(d['state_dict']); self.m.eval()

    def fix(self, pred, crop_path):
        """Return (new_prediction, n_changed, applied)."""
        g = cv.imread(str(crop_path), 0); bx = glyph_boxes(g)
        if len(bx) != len(pred) or not any(ch in OVALS for ch in pred): return pred, 0, False
        idx = [j for j, ch in enumerate(pred) if ch in OVALS]
        X = torch.from_numpy(np.stack([prep(g[max(0, bx[j]['y0'] - 2):bx[j]['y1'] + 2, max(0, bx[j]['x0'] - 2):bx[j]['x1'] + 2]) for j in idx]))[:, None]
        with torch.no_grad(): p = self.m(X).softmax(1).numpy()
        out = list(pred); n = 0
        for j, pj in zip(idx, p):
            k = int(pj.argmax())
            if pj[k] >= THRESHOLD and OVALS[k] != out[j]: out[j] = OVALS[k]; n += 1
        return ''.join(out), n, True
