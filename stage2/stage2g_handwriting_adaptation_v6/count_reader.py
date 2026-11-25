"""Count-tier BOX READER: for count tiers where the number of geometric glyph boxes differs
from the CRNN's length (the oval and shape classifiers are skipped there today).

Evidence for trusting the boxes (TRAIN crops only): box count equals the label length on
61 of 72 train count tiers (85%).
The boxes are read glyph by glyph:
  shape classifier -> plain glyph a..i, or 'D' (decorated oval) -> oval classifier.
mode 'strict' : used only if EVERY glyph is read with p >= 0.6 (else the CRNN string is kept)
mode 'argmax' : always used (argmax for every glyph)
Both modes only act when |#boxes - CRNN length| <= 2 and #boxes >= 1.
"""
import numpy as np, cv2 as cv, torch
from oval_fusion import prep, glyph_boxes, OVALS
from shape_fusion import SHAPES, PLAIN
THRESHOLD = 0.6


class CountReader:
    def __init__(self, shape_fixer, oval_fixer, mode='strict'):
        self.sm, self.om, self.mode = shape_fixer.m, oval_fixer.m, mode

    def fix(self, pred, crop_path):
        g = cv.imread(str(crop_path), 0); bx = glyph_boxes(g)
        if not bx or len(bx) == len(pred) or abs(len(bx) - len(pred)) > 2: return pred, 0
        X = torch.from_numpy(np.stack([prep(g[max(0, b['y0'] - 2):b['y1'] + 2, max(0, b['x0'] - 2):b['x1'] + 2]) for b in bx]))[:, None]
        with torch.no_grad(): ps = self.sm(X).softmax(1).numpy(); po = self.om(X).softmax(1).numpy()
        out = []
        for s, o in zip(ps, po):
            k = int(s.argmax())
            if SHAPES[k] in PLAIN:
                if self.mode == 'strict' and s[k] < THRESHOLD: return pred, 0
                out.append(SHAPES[k])
            else:
                o2 = o.copy(); o2[OVALS.index('a')] = 0; kk = int(o2.argmax())        # decorated -> not the plain oval
                if self.mode == 'strict' and (s[k] < THRESHOLD or o[kk] < THRESHOLD): return pred, 0
                out.append(OVALS[kk])
        return ''.join(out), 1
