"""Procedural hand-drawn Talim ovals: plain / 1-9 dots, + family (one cross, 0-9 dots), xx family (two crosses, 0-5 dots)."""
import numpy as np, cv2 as cv
OVALS = ['a'] + list('jklmnopqr') + list('stuvwxyz{|') + list("'()*+,")
def spec(ch):
    if ch == 'a': return 'none', 0
    if ch in 'jklmnopqr': return 'none', 'jklmnopqr'.index(ch) + 1
    if ch in 'stuvwxyz{|': return 'plus', 'stuvwxyz{|'.index(ch)
    return 'xx', "'()*+,".index(ch)
def _stroke(img, pts, th, val):
    pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2); cv.polylines(img, [pts], False, val, th, cv.LINE_AA)
def render(ch, rng, S=96):
    cross, nd = spec(ch); img = np.zeros((S, S), np.float32)
    a = rng.uniform(0.30, 0.44) * S; b = a / rng.uniform(1.0, 1.7); ang = rng.uniform(-25, 25) * np.pi / 180
    cx, cy = S / 2 + rng.normal(0, 2), S / 2 + rng.normal(0, 2); th = int(rng.integers(1, 5))
    t0 = rng.uniform(0, 2 * np.pi); span = 2 * np.pi * rng.uniform(0.9, 1.08); t = np.linspace(t0, t0 + span, 90)
    wob = 1 + 0.04 * np.sin(t * rng.integers(2, 4) + rng.uniform(0, 6)) + rng.normal(0, 0.008, t.size)
    x = a * wob * np.cos(t); y = b * wob * np.sin(t)
    P = np.stack([cx + x * np.cos(ang) - y * np.sin(ang), cy + x * np.sin(ang) + y * np.cos(ang)], 1); _stroke(img, P, th, 1.0)
    def inside(px, py, m=0.72):
        dx, dy = px - cx, py - cy; u = dx * np.cos(-ang) - dy * np.sin(-ang); v = dx * np.sin(-ang) + dy * np.cos(-ang)
        return (u / (a * m)) ** 2 + (v / (b * m)) ** 2 <= 1
    occupied = []
    def cross_at(px, py, r, kind):
        k = rng.uniform(-0.3, 0.3) + (np.pi / 4 if kind == 'x' else 0)
        for d in (k, k + np.pi / 2):
            _stroke(img, np.array([[px - r * np.cos(d), py - r * np.sin(d)], [px + r * np.cos(d), py + r * np.sin(d)]]), max(1, th - rng.integers(0, 2)), 1.0)
        occupied.append((px, py, r * 1.2))
    if cross == 'plus':
        r = b * rng.uniform(0.35, 0.55); px, py = cx + rng.uniform(-0.35, 0.1) * a if nd else cx, cy + rng.normal(0, 1.5)
        cross_at(px, py, r, rng.choice(['+', 'x']))
    elif cross == 'xx':
        r = b * rng.uniform(0.25, 0.38)
        for s in (-1, 1): cross_at(cx + s * a * rng.uniform(0.25, 0.4), cy + rng.normal(0, 2), r, 'x')
    placed = 0; tries = 0; dr = rng.uniform(1.4, 3.6)
    while placed < nd and tries < 600:
        tries += 1; px, py = cx + rng.uniform(-a, a), cy + rng.uniform(-b, b)
        if not inside(px, py): continue
        if any((px - ox) ** 2 + (py - oy) ** 2 < (orr + dr * 1.6) ** 2 for ox, oy, orr in occupied): continue
        el = rng.uniform(1.0, 2.2) if rng.random() < 0.4 else 1.0
        cv.ellipse(img, (int(px), int(py)), (max(1, int(dr * el)), max(1, int(dr))), rng.uniform(0, 180), 0, 360, 1.0, -1, cv.LINE_AA)
        occupied.append((px, py, dr * el)); placed += 1
    if placed < nd: return None
    dark = rng.uniform(0.35, 1.0); img = 1 - np.clip(img, 0, 1) * dark
    if rng.random() < 0.5: img = cv.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.0))
    img = np.clip(img + rng.normal(0, rng.uniform(0, 0.04), img.shape), 0, 1)
    ys, xs = np.where(img < 0.85); img = img[max(0, ys.min() - 3):ys.max() + 4, max(0, xs.min() - 3):xs.max() + 4]
    return np.uint8(img * 255)
