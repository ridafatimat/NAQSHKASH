"""Count-glyph SHAPE classifier fused with the CRNN (+ oval classifier).

Classes: the 9 plain count glyphs a..i (a = empty oval) and 'D' = decorated oval
(any oval with dots and/or crosses: j..r, s..|, ' ( ) * + ,).
Fusion for one count tier (only when geometric glyph boxes == CRNN length):
  - shape says plain glyph X (p >= THRESHOLD) and CRNN char != X  -> X
  - shape says decorated oval (p >= THRESHOLD) and CRNN char is a plain glyph
      -> ask the oval classifier; take its answer if it is confident
  - otherwise keep the CRNN / oval-classifier character.
"""
import numpy as np, cv2 as cv, torch, torch.nn as nn
from oval_fusion import prep, glyph_boxes, OVALS
PLAIN = list('abcdefghi')
SHAPES = PLAIN + ['D']
THRESHOLD = 0.6


class Net(nn.Module):
    def __init__(self, n=len(SHAPES)):
        super().__init__(); L = []; ci = 1
        for co in (32, 64, 96, 128):
            L += [nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(), nn.MaxPool2d(2)]; ci = co
        self.f = nn.Sequential(*L); self.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, n))
    def forward(self, x): return self.h(self.f(1 - x).mean((2, 3)))


def shape_of(ch): return ch if ch in PLAIN else 'D'


class ShapeFixer:
    def __init__(self, weights, oval_fixer=None):
        d = torch.load(weights, map_location='cpu', weights_only=True); assert d['classes'] == SHAPES
        self.m = Net(); self.m.load_state_dict(d['state_dict']); self.m.eval(); self.ov = oval_fixer

    def fix(self, pred, crop_path):
        g = cv.imread(str(crop_path), 0); bx = glyph_boxes(g)
        if len(bx) != len(pred) or not pred: return pred, 0
        crops = [g[max(0, b['y0'] - 2):b['y1'] + 2, max(0, b['x0'] - 2):b['x1'] + 2] for b in bx]
        X = torch.from_numpy(np.stack([prep(c) for c in crops]))[:, None]
        with torch.no_grad(): p = self.m(X).softmax(1).numpy()
        out = list(pred); n = 0
        for j, pj in enumerate(p):
            k = int(pj.argmax())
            if pj[k] < THRESHOLD: continue
            s = SHAPES[k]
            if s in PLAIN and out[j] != s: out[j] = s; n += 1
            elif s == 'D' and out[j] in PLAIN and self.ov is not None:
                with torch.no_grad(): q = self.ov.m(torch.from_numpy(prep(crops[j]))[None, None]).softmax(1).numpy()[0]
                kk = int(q.argmax())
                if q[kk] >= THRESHOLD and OVALS[kk] != 'a': out[j] = OVALS[kk]; n += 1
        return ''.join(out), n
