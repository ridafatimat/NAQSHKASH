"""Cut tier crops from the segmented pages and build preview sheets.

Crop convention (matches the Stage 1 / Stage 2E tier builder):
  * one image per non-empty tier, full ROW width (union of the row's tiers),
    vertical extent = that tier's ink + padding;
  * label file = the raw tier string exactly as in the reference JSON
    (leading/internal spaces kept; the training loader strips only leading
    spaces at load time).
Handwriting-specific additions: other tiers' ink and ruling are whitened
(mask crop), and the row is rotated so its count baseline is horizontal.
"""
import sys, math
sys.path.insert(0, '/tmp/claude-0/-home-claude/6dc2ff89-eb76-5c60-88e7-6c47063238b5/scratchpad/tiers/code')
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from common import *
from segment import cluster_glyphs, glyph_count, mark_groups

TIERS = (('upper', 'upper'), ('count', 'count_line'), ('lower', 'lower'))


def tier_pixels(seg, cs):
    """Binary mask of one tier's pen ink (no ruling)."""
    lab, mask = seg['lab'], seg['ink']['mask']
    ids = np.array(sorted({m for c in cs for m in c['members']}), dtype=np.int32)
    if ids.size == 0:
        return np.zeros_like(mask)
    sel = np.isin(lab, ids) & mask.astype(bool)
    return sel.astype(np.uint8)


def render_tier(seg, cs):
    """White page with only this tier's ink, in flattened grayscale."""
    stroke = seg['ink']['stroke']
    pix = tier_pixels(seg, cs)
    keep = cv2.dilate(pix, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(stroke)) | 1,) * 2)).astype(bool)
    keep &= ~(seg['ink']['lines'].astype(bool) & ~seg['ink']['mask'].astype(bool))
    norm = seg['ink']['norm']
    out = np.full_like(norm, 255)
    out[keep] = norm[keep]
    # Paper texture inside the halo -> white; keep ink darkness gradation.
    out[(out > 200) & keep] = 255
    return out, pix


def rotate(img, angle_deg, center, fill=255):
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=fill), M


def bbox(binary):
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_page(seg):
    """Return tier records for every logical row/tier of the page."""
    H = seg['H']
    h, w = seg['ink']['mask'].shape
    records = []
    for i, lr in enumerate(seg['lrows']):
        row = seg['rows'][i] if seg['ok'] else None
        angle = math.degrees(math.atan(row['slope'])) if row else 0.0
        center = ((row['x0'] + row['x1']) / 2, row['y_mid']) if row else (w / 2, h / 2)
        rendered = {}
        for tier, key in TIERS:
            cs = seg['assign'].get((i, tier), []) if seg['ok'] else []
            img, pix = render_tier(seg, cs)
            rimg, M = rotate(img, angle, center)
            rpix, _ = rotate(pix * 255, angle, center, fill=0)
            rendered[tier] = (rimg, rpix > 64, pix, cs)
        boxes = [bbox(v[1]) for v in rendered.values() if bbox(v[1])]
        row_x0 = min(b[0] for b in boxes) if boxes else 0
        row_x1 = max(b[2] for b in boxes) if boxes else w
        padx = int(round(0.25 * H)); pady = int(round(max(6, 0.15 * H)))
        for tier, key in TIERS:
            label = lr[key]
            rimg, rpix, pix, cs = rendered[tier]
            b = bbox(rpix)
            ob = bbox(pix)
            rec = dict(source_id=seg['sid'], row_index=i, tier=tier, raw_label=label,
                       ocr_text=label.lstrip(' '), leading_offset=len(label) - len(label.lstrip(' ')),
                       label_nonempty=bool(label.strip()), n_components=len(cs), row_angle_deg=round(angle, 2),
                       page_box=list(ob) if ob else None, crop=None, flags=[])
            if not seg['ok']:
                rec['flags'].append('PAGE_ROWS_NOT_FOUND')
            if rec['label_nonempty'] and b is None:
                rec['flags'].append('MISSING_TIER_INK')
            if not rec['label_nonempty'] and b is not None:
                rec['flags'].append('INK_IN_EMPTY_LABEL_TIER')
            if b is not None:
                x0, y0 = max(0, row_x0 - padx), max(0, b[1] - pady)
                x1, y1 = min(w, row_x1 + padx), min(h, b[3] + pady)
                rec['crop'] = rimg[y0:y1, x0:x1]
                if ob and (ob[0] <= 2 or ob[1] <= 2 or ob[2] >= w - 2 or ob[3] >= h - 2):
                    rec['flags'].append('TOUCHES_PHOTO_BORDER')
            if rec['label_nonempty'] and b is not None:
                exp = glyph_count(label)
                if tier == 'count':
                    got = len(cluster_glyphs(cs, H))
                    if abs(got - exp) > max(1, round(0.15 * exp)):
                        rec['flags'].append(f'COUNT_GLYPHS_{got}_VS_LABEL_{exp}')
                else:
                    got = mark_groups(cs, H)
                    if abs(got - exp) > 1:
                        rec['flags'].append(f'MARK_GROUPS_{got}_VS_LABEL_{exp}')
                rec['observed'] = got; rec['expected'] = exp
            records.append(rec)
    return records


def preview(seg, records, tier_view_path, out_path, decisions=None, width=1000):
    tv = Image.open(tier_view_path).convert('RGB')
    tv = tv.resize((width, round(tv.height * width / tv.width)))
    font = ImageFont.load_default(15)
    parts = []
    for r in records:
        if not r['label_nonempty'] and r['crop'] is None:
            continue
        status = (decisions or {}).get((r['row_index'], r['tier']), {}).get('status', '')
        head = f"R{r['row_index']} {r['tier']:<5}  label={r['raw_label']!r}  {status}  {' '.join(r['flags'])}"
        if r['crop'] is not None:
            c = Image.fromarray(r['crop'])
            s = min(1.0, (width - 10) / c.width, 110 / max(1, c.height))
            c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))))
        else:
            c = Image.new('L', (300, 30), 230)
            ImageDraw.Draw(c).text((5, 8), 'NO CROP', fill=0, font=font)
        p = Image.new('RGB', (width, 22 + c.height + 8), 'white')
        d = ImageDraw.Draw(p); d.text((4, 3), head, fill=(150, 0, 0) if (r['flags'] or 'REVIEW' in status) else (0, 0, 0), font=font)
        p.paste(c.convert('RGB'), (5, 22)); d.rectangle([5, 22, 5 + c.width, 22 + c.height], outline=(200, 200, 200))
        parts.append(p)
    H = tv.height + sum(p.height for p in parts) + 10
    sheet = Image.new('RGB', (width, H), 'white'); sheet.paste(tv, (0, 0)); y = tv.height + 10
    for p in parts:
        sheet.paste(p, (0, y)); y += p.height
    sheet.save(out_path, quality=85)
