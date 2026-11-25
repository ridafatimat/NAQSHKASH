"""Mark classifier shared code: preprocessing, network, and fusion with geometry slots.

A mark crop is scaled RELATIVE to the row's count-glyph height (so a dot stays
small and a ring stays big), contrast-normalised and centred on a 64x64 canvas.
Fusion: after geometry has placed the CRNN's marks into glyph slots, the ink of
every slot is re-read by this classifier (restricted to the tier's own marks).
The classifier replaces the CRNN mark only if its probability >= THRESHOLD.
"""
import collections
import numpy as np, cv2 as cv, torch, torch.nn as nn

UPPER = list('ABCDEFGHIJKLMNPQ')
LOWER = list('RSTUVWXYZ[\\]^_')
MARKS = UPPER + LOWER
ROLE_IDX = {'upper': [MARKS.index(c) for c in UPPER], 'lower': [MARKS.index(c) for c in LOWER]}
THRESHOLD = 0.6
S = 64; GLYPH_PX = 40


def prep(g, ref):
    g = g.astype(np.float32) / 255.0; ink = g[g < 0.98]; blk = float(np.percentile(ink, 5)) if ink.size else 0
    g = np.clip((g - blk) / max(1 - blk, 0.05), 0, 1)
    sc = GLYPH_PX / max(ref, 1.0); h, w = g.shape; sc = min(sc, (S - 4) / max(h, w))
    r = cv.resize(g, (max(1, int(round(w * sc))), max(1, int(round(h * sc)))), interpolation=cv.INTER_AREA)
    c = np.ones((S, S), np.float32); y0 = (S - r.shape[0]) // 2; x0 = (S - r.shape[1]) // 2; c[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return c


class Net(nn.Module):
    def __init__(self, n=len(MARKS)):
        super().__init__(); L = []; ci = 1
        for co in (32, 64, 96, 128):
            L += [nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2)]; ci = co
        self.f = nn.Sequential(*L); self.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x): return self.h(self.f(1 - x).mean((2, 3)))


def components(g):
    t, _ = cv.threshold(g, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU); b = (g < min(t, 210)).astype(np.uint8)
    n, lab, st, _ = cv.connectedComponentsWithStats(b, 8)
    return [dict(x0=int(st[k, 0]), x1=int(st[k, 0] + st[k, 2]), y0=int(st[k, 1]), y1=int(st[k, 1] + st[k, 3]), a=int(st[k, 4])) for k in range(1, n) if st[k, 4] >= 6], b.shape[1]


def glyph_ref_height(count_img):
    cs, _ = components(count_img)
    return float(np.percentile([c['y1'] - c['y0'] for c in cs], 80)) if cs else 0.0


def slot_crops(mark_img, centres, speck=0.35):
    cs, W = components(mark_img)
    if not cs: return {}
    med = np.median([c['a'] for c in cs]); cs = [c for c in cs if c['a'] >= speck * med]; ctr = np.array(centres); grp = collections.defaultdict(list)
    for c in cs: grp[int(np.argmin(np.abs(ctr - (c['x0'] + c['x1']) / 2 / W)))].append(c)
    out = {}
    for j, g in grp.items():
        x0 = max(0, min(c['x0'] for c in g) - 2); x1 = max(c['x1'] for c in g) + 2; y0 = max(0, min(c['y0'] for c in g) - 2); y1 = max(c['y1'] for c in g) + 2
        out[j] = mark_img[y0:y1, x0:x1]
    return out


class MarkFixer:
    def __init__(self, weights):
        d = torch.load(weights, map_location='cpu', weights_only=True); assert d['classes'] == MARKS
        self.m = Net(); self.m.load_state_dict(d['state_dict']); self.m.eval()

    def fix(self, placed, role, mark_path, count_path, centres):
        """placed: geometry-placed string (slot j = char j). Returns (new_string, n_changed)."""
        mi = cv.imread(str(mark_path), 0); ref = glyph_ref_height(cv.imread(str(count_path), 0)); crops = slot_crops(mi, centres)
        s = list(placed); jobs = [j for j, ch in enumerate(s) if ch != ' ' and j in crops]
        if not jobs: return placed, 0
        X = torch.from_numpy(np.stack([prep(crops[j], ref) for j in jobs]))[:, None]
        with torch.no_grad(): logit = self.m(X)
        idx = ROLE_IDX[role]; p = logit[:, idx].softmax(1).numpy(); n = 0
        for j, pj in zip(jobs, p):
            k = int(pj.argmax()); ch = MARKS[idx[k]]
            if pj[k] >= THRESHOLD and ch != s[j]: s[j] = ch; n += 1
        return ''.join(s), n
