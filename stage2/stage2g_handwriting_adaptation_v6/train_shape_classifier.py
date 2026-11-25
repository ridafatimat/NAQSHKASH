"""Train the count-glyph SHAPE classifier on your laptop (CPU, ~15-25 min).

Data: glyph_training_data.npz (TRAIN writers only: originals + their checked
augmentations) + procedurally drawn ovals (ovalgen.py). No validation/test writer.

  python -B train_shape_classifier.py dev     # holds out 5 train writers, prints accuracy
  python -B train_shape_classifier.py final   # trains on everything, writes shape_classifier.pt
  python -B train_shape_classifier.py final v2   # V8 data -> shape_classifier_v2.pt
"""
import sys, os, time
from pathlib import Path
import numpy as np, cv2 as cv, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oval_fusion import prep, OVALS             # noqa: E402
from shape_fusion import SHAPES, Net, shape_of  # noqa: E402
from ovalgen import render                       # noqa: E402

torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0)
mode = sys.argv[1] if len(sys.argv) > 1 else 'final'
DEV = {'W03', 'W10', 'W15', 'W21', 'W29'}
V2 = 'v2' in sys.argv[2:]; DATA = 'glyph_training_data_v2.npz' if V2 else 'glyph_training_data.npz'; OUT = 'shape_classifier_v2.pt' if V2 else 'shape_classifier.pt'; print('data', DATA, '-> out', OUT)
z = np.load(HERE / DATA); pix = z['pixels']; cols = [z[k] for k in ('writer', 'kind', 'char', 'h', 'w')]
real, off = [], 0
for wr, kind, ch, h, w in zip(*cols):
    h, w = int(h), int(w); real.append((str(wr), str(kind), str(ch), pix[off:off + h * w].reshape(h, w).copy())); off += h * w
train = [d for d in real if mode == 'final' or d[0] not in DEV]
dev = [d for d in real if d[0] in DEV and d[1] == 'original'] if mode == 'dev' else []
gen = []
for ch in OVALS:                                  # 'a' = plain oval, others = decorated
    n = 0
    while n < (300 if ch == 'a' else 60):
        im = render(ch, rng)
        if im is not None: gen.append((SHAPES.index(shape_of(ch)), prep(im))); n += 1
print('real train', len(train), 'procedural', len(gen), 'dev', len(dev), flush=True)


def jitter(g):
    h, w = g.shape; M = cv.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.85, 1.15))
    g = cv.warpAffine(g, M, (w, h), borderValue=255)
    if rng.random() < 0.3: g = cv.erode(g, np.ones((2, 2), np.uint8))
    if rng.random() < 0.3: g = cv.GaussianBlur(g, (3, 3), rng.uniform(0.3, 0.9))
    return g


def epoch_set():
    X = [g for _, g in gen]; Y = [y for y, _ in gen]
    for d in train:
        for _ in range(3 if d[1] == 'original' else 1): X.append(prep(jitter(d[3]))); Y.append(SHAPES.index(shape_of(d[2])))
    return np.stack(X), np.array(Y, dtype=np.int64)


def evaluate(m):
    if not dev: return ''
    m.eval(); X = torch.from_numpy(np.stack([prep(d[3]) for d in dev]))[:, None]
    with torch.no_grad(): p = m(X).softmax(1).numpy()
    ok = np.array([SHAPES[i] == shape_of(d[2]) for i, d in zip(p.argmax(1), dev)]); hi = p.max(1) >= 0.6
    return f'dev acc {ok.mean():.3f} | conf>=0.6: coverage {hi.mean():.3f} acc {ok[hi].mean() if hi.any() else 0:.3f} (n={len(dev)})'


m = Net(); E = int(os.environ.get('EPOCHS', 14)); steps = 200
opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * steps)
lossf = nn.CrossEntropyLoss(); t0 = time.time()
for ep in range(E):
    X, Y = epoch_set(); perm = rng.permutation(len(X)); bs = max(16, len(X) // steps); m.train()
    for k in range(steps):
        ix = perm[k * bs:(k + 1) * bs]
        if len(ix) < 2: continue
        loss = lossf(m(torch.from_numpy(X[ix])[:, None]), torch.from_numpy(Y[ix])); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print(f'epoch {ep} loss {loss.item():.3f} {evaluate(m)} {time.time() - t0:.0f}s', flush=True)
if mode == 'final':
    torch.save(dict(state_dict=m.state_dict(), classes=SHAPES), HERE / OUT); print('saved', OUT)
