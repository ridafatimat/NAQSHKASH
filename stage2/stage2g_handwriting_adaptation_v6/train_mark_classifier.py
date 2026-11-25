"""Train the mark classifier on your laptop (CPU, ~10-20 min).

Data: mark_training_data.npz (TRAIN writers only + synthetic font marks), made with
the same geometry slots as run_v6_geo.py. Validation/test writers are not in it.

  python -B train_mark_classifier.py dev     # holds out 5 train writers, prints accuracy (sanity check)
  python -B train_mark_classifier.py final   # trains on everything, writes mark_classifier.pt
  python -B train_mark_classifier.py final v2   # V8 data -> mark_classifier_v2.pt
"""
import sys, os, time, collections
from pathlib import Path
import numpy as np, cv2 as cv, torch, torch.nn as nn
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mark_fusion import MARKS, ROLE_IDX, prep, Net  # noqa: E402

torch.set_num_threads(4); torch.manual_seed(0)
mode = sys.argv[1] if len(sys.argv) > 1 else 'final'
DEV = {'W03', 'W10', 'W15', 'W21', 'W29'}
V2 = 'v2' in sys.argv[2:]; DATA = 'mark_training_data_v2.npz' if V2 else 'mark_training_data.npz'; OUT = 'mark_classifier_v2.pt' if V2 else 'mark_classifier.pt'; print('data', DATA, '-> out', OUT)
def load_data():
    z = np.load(HERE / DATA)          # plain arrays, no pickle: works on numpy 1.x and 2.x
    pix = z['pixels']; cols = [z[k] for k in ('writer', 'role', 'char', 'ref', 'h', 'w')]; out, off = [], 0
    for wr, ro, ch, rf, h, w in zip(*cols):
        h, w = int(h), int(w); img = pix[off:off + h * w].reshape(h, w).copy(); off += h * w
        out.append((str(wr), str(ro), str(ch), img, float(rf)))
    return out
data = load_data()                                         # (writer, role, char, image, ref_height)
rng = np.random.default_rng(0)
real = [d for d in data if d[0] != 'syn']; syn = [d for d in data if d[0] == 'syn']
train = syn + [d for d in real if mode == 'final' or d[0] not in DEV]
dev = [d for d in real if d[0] in DEV] if mode == 'dev' else []
print('train samples', len(train), '(real', sum(d[0] != 'syn' for d in train), ') dev', len(dev), flush=True)


def jitter(g, fade):
    h, w = g.shape; M = cv.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.85, 1.15))
    g = cv.warpAffine(g, M, (w, h), borderValue=255)
    if rng.random() < 0.3: g = cv.erode(g, np.ones((2, 2), np.uint8))
    elif rng.random() < 0.3: g = cv.dilate(g, np.ones((2, 2), np.uint8))
    if rng.random() < 0.4: g = cv.GaussianBlur(g, (3, 3), rng.uniform(0.3, 1.0))
    if fade and rng.random() < 0.6: g = np.uint8(255 - (255 - g.astype(np.float32)) * rng.uniform(0.35, 0.9))
    return g


def epoch_set():
    X, Y, R = [], [], []
    for d in train:
        for _ in range(6 if d[0] != 'syn' else 1):
            X.append(prep(jitter(d[3], d[0] == 'syn'), d[4] * rng.uniform(0.9, 1.1))); Y.append(MARKS.index(d[2])); R.append(d[1])
    return np.stack(X), np.array(Y, dtype=np.int64), R


def masked_logits(logit, roles):
    mask = torch.full_like(logit, -1e9)
    for i, r in enumerate(roles): mask[i, ROLE_IDX[r]] = 0
    return logit + mask


def evaluate(m):
    if not dev: return ''
    m.eval(); X = torch.from_numpy(np.stack([prep(d[3], d[4]) for d in dev]))[:, None]
    with torch.no_grad(): p = masked_logits(m(X), [d[1] for d in dev]).softmax(1).numpy()
    pred = [MARKS[i] for i in p.argmax(1)]; ok = np.array([a == d[2] for a, d in zip(pred, dev)]); hi = p.max(1) >= 0.6
    return f'dev acc {ok.mean():.3f} | conf>=0.6: coverage {hi.mean():.3f} acc {ok[hi].mean() if hi.any() else 0:.3f} (n={len(dev)})'


m = Net(); E = int(os.environ.get('EPOCHS', 20)); steps = 120
opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * steps)
lossf = nn.CrossEntropyLoss(); t0 = time.time()
for ep in range(E):
    X, Y, R = epoch_set(); perm = rng.permutation(len(X)); bs = max(16, len(X) // steps); m.train()
    for k in range(steps):
        ix = perm[(k * bs) % len(X):(k * bs) % len(X) + bs]
        if len(ix) < 2: continue
        logit = masked_logits(m(torch.from_numpy(X[ix])[:, None]), [R[i] for i in ix])
        loss = lossf(logit, torch.from_numpy(Y[ix])); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    print(f'epoch {ep} loss {loss.item():.3f} {evaluate(m)} {time.time() - t0:.0f}s', flush=True)
if mode == 'final':
    torch.save(dict(state_dict=m.state_dict(), classes=MARKS), HERE / OUT); print('saved', OUT)
