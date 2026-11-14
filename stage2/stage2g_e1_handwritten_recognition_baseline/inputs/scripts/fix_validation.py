"""Targeted pixel-level corrections for unresolved validation tiers.

Uses the ORIGINAL handwritten photographs via the existing segmentation (no
pipeline rebuild). Only existing pen-ink pixels are reassigned between tiers of
the same row; nothing is drawn, thresholds are not changed, labels are not
edited. The crop renderer is re-implemented from pixel masks and must reproduce
the current crops pixel-for-pixel before any correction is applied.
"""
import sys, math, pickle
sys.path.insert(0, '/tmp/claude-0/-home-claude/6dc2ff89-eb76-5c60-88e7-6c47063238b5/scratchpad/tiers/code')
import numpy as np
import cv2
from crop import rotate, bbox, TIERS

SEGS = '/tmp/claude-0/-home-claude/6dc2ff89-eb76-5c60-88e7-6c47063238b5/scratchpad/tiers/val_segs.pkl'

# (source, row, from_tier, to_tier, host member label id, undilated sub-blob bbox or None for whole member, why)
EDITS = [
    ('all_009197', 1, 'count', 'upper', 34, (93, 557, 234, 613),
     'second dash of the first upper mark (≐) was one blob with the count oval after dilation; undilated it is a separate stroke above the oval'),
    ('all_003556', 0, 'count', 'lower', 15, (561, 321, 583, 340),
     'left dot of the lower ∵ mark was fused into the count glyph (⊗) blob'),
    ('all_003556', 0, 'count', 'lower', 23, None,
     'right dot of the lower ∵ mark was merged into the count glyph by containment'),
]


def tier_masks(seg, row):
    lab, mask = seg['lab'], seg['ink']['mask'].astype(bool)
    out = {}
    for tier, _ in TIERS:
        ids = [m for c in seg['assign'].get((row, tier), []) for m in c['members']]
        out[tier] = (np.isin(lab, ids) & mask).astype(np.uint8)
    return out


def sub_blob(seg, member, box):
    sel = ((seg['lab'] == member) & (seg['ink']['mask'] > 0)).astype(np.uint8)
    if box is None:
        return sel
    n, l, st, _ = cv2.connectedComponentsWithStats(sel, 8)
    hits = [i for i in range(1, n) if (st[i][0], st[i][1], st[i][0] + st[i][2], st[i][1] + st[i][3]) == box]
    assert len(hits) == 1, (member, box)
    return (l == hits[0]).astype(np.uint8)


# Ruling residue wrongly assigned to a tier: (source, row, tier, member, sub-blob bbox, why)
DISCARD = [
    ('all_003556', 0, 'lower', 25, None, 'fragment of the page-wide notebook ruled line at y~438 (1 stroke thick, 188 px long), not handwriting'),
    ('all_003556', 0, 'lower', 26, None, 'fragment of the page-wide notebook ruled line at y~438 (1 stroke thick, 146 px long), not handwriting'),
]

# Tiers rendered with the faint-ink-preserving rule: pen-ink pixels keep their
# original photo value even when lighter than 200; only halo pixels OUTSIDE the
# ink mask are whitened. Used only where the standard rule erases the mark.
KEEP_FAINT = {('all_003556', 0, 'lower')}


def render(seg, pix, keep_faint=False):
    """Identical to crop.render_tier but from a pixel mask (keep_faint=False)."""
    stroke = seg['ink']['stroke']
    keep = cv2.dilate(pix, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(stroke)) | 1,) * 2)).astype(bool)
    keep &= ~(seg['ink']['lines'].astype(bool) & ~seg['ink']['mask'].astype(bool))
    norm = seg['ink']['norm']
    out = np.full_like(norm, 255)
    out[keep] = norm[keep]
    paper = (out > 200) & keep
    if keep_faint:
        paper &= ~pix.astype(bool)
    out[paper] = 255
    return out


def crop_row(seg, row, masks, faint_tiers=()):
    """Identical geometry to crop.crop_page for one row."""
    H = seg['H']; h, w = seg['ink']['mask'].shape
    r = seg['rows'][row]
    angle = math.degrees(math.atan(r['slope'])); center = ((r['x0'] + r['x1']) / 2, r['y_mid'])
    rendered = {}
    for tier, _ in TIERS:
        rimg, _ = rotate(render(seg, masks[tier], keep_faint=tier in faint_tiers), angle, center)
        rpix, _ = rotate(masks[tier] * 255, angle, center, fill=0)
        rendered[tier] = (rimg, rpix > 64)
    boxes = [bbox(v[1]) for v in rendered.values() if bbox(v[1])]
    rx0 = min(b[0] for b in boxes); rx1 = max(b[2] for b in boxes)
    padx = int(round(0.25 * H)); pady = int(round(max(6, 0.15 * H)))
    out = {}
    for tier, _ in TIERS:
        rimg, rpix = rendered[tier]; b = bbox(rpix)
        if b is None:
            out[tier] = None; continue
        x0, y0 = max(0, rx0 - padx), max(0, b[1] - pady); x1, y1 = min(w, rx1 + padx), min(h, b[3] + pady)
        out[tier] = dict(crop=rimg[y0:y1, x0:x1], pix=rpix[y0:y1, x0:x1], window=(x0, y0, x1, y1), angle=angle, center=center)
    return out


def corrected(segs):
    """Return {(sid,row): (before_crops, after_crops, moved_masks)}."""
    from crop import crop_page
    result = {}
    for sid, row in sorted({(e[0], e[1]) for e in EDITS}):
        seg = segs[sid]
        masks = tier_masks(seg, row)
        before = crop_row(seg, row, masks)
        # Faithfulness check against the pipeline that produced the package.
        ref = {r['tier']: r['crop'] for r in crop_page(seg) if r['row_index'] == row}
        for tier in ref:
            a, b = ref[tier], (before[tier]['crop'] if before[tier] else None)
            assert (a is None and b is None) or np.array_equal(a, b), (sid, row, tier)
        moved = []
        for s, rw, frm, to, member, box, why in EDITS:
            if (s, rw) != (sid, row):
                continue
            blob = sub_blob(seg, member, box)
            assert (blob & masks[frm]).sum() == blob.sum() > 0, 'blob not in source tier'
            masks[frm] = masks[frm] & ~blob
            masks[to] = masks[to] | blob
            moved.append((frm, to, blob, why))
        for s, rw, tier, member, box, why in DISCARD:
            if (s, rw) == (sid, row):
                blob = sub_blob(seg, member, box)
                assert (blob & masks[tier]).sum() == blob.sum() > 0
                masks[tier] = masks[tier] & ~blob
                moved.append((tier, 'DISCARDED_RULING', blob, why))
        after = crop_row(seg, row, masks, faint_tiers={t for s2, r2, t in KEEP_FAINT if (s2, r2) == (sid, row)})
        result[(sid, row)] = (before, after, moved)
    return result


if __name__ == '__main__':
    segs = pickle.load(open(SEGS, 'rb'))
    res = corrected(segs)
    for (sid, row), (before, after, moved) in res.items():
        print(sid, row, 'faithful reproduction OK; moved', [(m[0], m[1], int(m[2].sum())) for m in moved])
