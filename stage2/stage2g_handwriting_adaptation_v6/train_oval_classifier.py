"""Train the OVAL classifier on your laptop (CPU, ~20-30 min). Same recipe as oval_classifier.pt:
procedural ovals (ovalgen.py, 450 per class) + real oval glyphs of TRAIN writers (originals x4 + augmentations).

  python -B train_oval_classifier.py dev v2     # holds out 5 train writers, prints accuracy
  python -B train_oval_classifier.py final v2   # trains on everything, writes oval_classifier_v2.pt
"""
import sys, time
from pathlib import Path
import numpy as np, cv2 as cv, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from oval_fusion import OVALS, prep, Net   # noqa: E402
from ovalgen import render, spec           # noqa: E402

torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0)
mode = sys.argv[1] if len(sys.argv) > 1 else 'final'
V2 = 'v2' in sys.argv[2:]; DATA = 'glyph_training_data_v2.npz' if V2 else 'glyph_training_data.npz'; OUT = 'oval_classifier_v2.pt' if V2 else 'oval_classifier.pt'
print('data', DATA, '-> out', OUT, flush=True)
IDX = {c: i for i, c in enumerate(OVALS)}
DEV = {'W03', 'W10', 'W15', 'W21', 'W29'}
z = np.load(HERE / DATA); pix = z['pixels']; real, off = [], 0
for wr, kind, ch, h, w in zip(*[z[k] for k in ('writer', 'kind', 'char', 'h', 'w')]):
    h, w = int(h), int(w); g = pix[off:off + h * w].reshape(h, w).copy(); off += h * w
    if str(ch) in IDX: real.append((str(wr), str(kind), str(ch), g))
train = [x for x in real if mode == 'final' or x[0] not in DEV]
dev = [x for x in real if x[0] in DEV and x[1] == 'original'] if mode == 'dev' else []
gen = []
for ch in OVALS:
    n = 0
    while n < 450:
        im = render(ch, rng)
        if im is not None: gen.append((IDX[ch], prep(im))); n += 1
X_gen = np.stack([g for _, g in gen]); Y_gen = np.array([y for y, _ in gen], dtype=np.int64)
print('real oval glyphs', len(train), 'procedural', len(gen), 'dev', len(dev), flush=True)


def jitter(g):
    h, w = g.shape; M = cv.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-10, 10), rng.uniform(0.85, 1.15))
    g = cv.warpAffine(g, M, (w, h), borderValue=255)
    if rng.random() < 0.3: g = cv.erode(g, np.ones((2, 2), np.uint8))
    if rng.random() < 0.3: g = cv.GaussianBlur(g, (3, 3), rng.uniform(0.3, 0.9))
    return g


def evaluate(m):
    if not dev: return ''
    m.eval(); X = torch.from_numpy(np.stack([prep(x[3]) for x in dev]))[:, None]
    with torch.no_grad(): p = m(X).softmax(1).numpy()
    pred = [OVALS[i] for i in p.argmax(1)]; truth = [x[2] for x in dev]; hi = p.max(1) >= 0.6
    ok = np.array([a == b for a, b in zip(pred, truth)]); dots = np.mean([spec(a)[1] == spec(b)[1] for a, b in zip(pred, truth)])
    return f'dev acc {ok.mean():.3f} dots {dots:.3f} | conf>=0.6: coverage {hi.mean():.3f} acc {ok[hi].mean() if hi.any() else 0:.3f} (n={len(dev)})'


m = Net(); E = 14; steps = 200
opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * steps)
lossf = nn.CrossEntropyLoss(label_smoothing=0.05); t0 = time.time()
for ep in range(E):
    Xr = np.stack([prep(jitter(x[3])) for x in train for _ in range(4 if x[1] == 'original' else 1)])
    Yr = np.array([IDX[x[2]] for x in train for _ in range(4 if x[1] == 'original' else 1)], dtype=np.int64)
    X = np.concatenate([X_gen, Xr]); Y = np.concatenate([Y_gen, Yr]); perm = rng.permutation(len(X)); bs = max(16, len(X) // steps); m.train()
    for k in range(steps):
        ix = perm[k * bs:(k + 1) * bs]
        if len(ix) < 2: continue
        loss = lossf(m(torch.from_numpy(X[ix])[:, None]), torch.from_numpy(Y[ix])); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print(f'epoch {ep} loss {loss.item():.3f} {evaluate(m)} {time.time() - t0:.0f}s', flush=True)
if mode == 'final':
    torch.save(dict(state_dict=m.state_dict(), classes=OVALS), HERE / OUT); print('saved', OUT)
