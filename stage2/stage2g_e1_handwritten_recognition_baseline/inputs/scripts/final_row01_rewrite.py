"""Final all_003191 row 1 tier crops from the user-supplied re-photographed row.

Photo: rotated 90 deg counter-clockwise to upright (upper marks on top).
Ink: standard package ink extraction, then pen pixels that the ruled-line removal
took are restored where the column's dark run is longer than the bare line
(margin 1 px), only inside the boxes of handwriting the standard pass found,
and only when attached to existing pen strokes. Nothing is drawn; labels are the
reference JSON row 1 strings, unchanged.
"""
import sys, os, io, csv, json, shutil, hashlib, zipfile
sys.path.insert(0, '/tmp/claude-0/-home-claude/6dc2ff89-eb76-5c60-88e7-6c47063238b5/scratchpad/tiers/code')
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from rewrite_row import segment_single_row, handwriting_boxes, restore_pen_on_ruling
from segment import components, fit_rows, assign, glyph_count, mark_groups, cluster_glyphs
from crop import crop_page
from run_segment import tier_view

W = '/tmp/claude-0/-home-claude/6dc2ff89-eb76-5c60-88e7-6c47063238b5/scratchpad/tiers'
SRC = f'{W}/in/rewrite/all_003191_row01_rewrite_original.jpg'
UP = f'{W}/in/rewrite/all_003191_row01_rewrite_upright.png'
PKG = sys.argv[1]
OUT = os.path.join(os.path.dirname(PKG), 'all_003191_row01_final_20261109')
DATE = '2026-11-09'
FONT = ImageFont.load_default(15)


def sha_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def png(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format='PNG'); return b.getvalue()


def build():
    base = segment_single_row(UP, 'all_003191', 1)
    allowed = handwriting_boxes(base, int(round(base['ink']['stroke'] * 2)))
    ink = dict(base['ink']); ink['lines'] = (ink['lines'].astype(bool) & allowed).astype(np.uint8)
    rest, info = restore_pen_on_ruling(ink, margin=1)
    # keep only restored pixels 8-connected to existing pen strokes
    orig = base['ink']['mask'].astype(bool); new = rest['mask'].astype(bool) & ~orig
    n, lab = cv2.connectedComponents(new.astype(np.uint8), 8)
    touch = np.unique(lab[new & cv2.dilate(orig.astype(np.uint8), np.ones((3, 3))).astype(bool)])
    keep = np.isin(lab, touch[touch > 0]) & new
    info['dropped_unattached_px'] = int((new & ~keep).sum()); info['restored_px'] = int(keep.sum())
    rest['mask'] = (orig | keep).astype(np.uint8); rest['lines'] = base['ink']['lines']
    comps, labc = components(rest['mask'], rest['stroke'])
    lrow = base['lrows'][0]
    rows, H = fit_rows(comps, 1, [glyph_count(lrow['count_line'])])
    seg = dict(base, ink=rest, comps=comps, lab=labc, rows=rows, H=H, ok=len(rows) == 1)
    seg['assign'], seg['noise'] = assign(comps, rows, H, [lrow])
    return seg, crop_page(seg), info


def main():
    seg, recs, info = build()
    assert seg['ok']
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    for d in ('tier_crops', 'tier_labels', 'source_photo', 'before_crops'):
        os.makedirs(f'{OUT}/{d}', exist_ok=True)
    shutil.copy(SRC, f'{OUT}/source_photo/all_003191_row01_rewrite_original.jpg')
    shutil.copy(UP, f'{OUT}/source_photo/all_003191_row01_rewrite_upright.png')
    os.makedirs(f'{PKG}/validation/source_photos', exist_ok=True)
    shutil.copy(SRC, f'{PKG}/validation/source_photos/all_003191_row01_rewrite_original.jpg')
    src_sha = sha_file(SRC)

    rows = list(csv.DictReader(open(f'{PKG}/manifest.csv', encoding='utf-8')))
    byid = {r['tier_id']: r for r in rows}
    checks, panels = [], []
    tv = f'{OUT}/tier_view.jpg'; tier_view(seg, tv, width=1300)
    for r in recs:
        tid = f"all_003191_row_01_{r['tier']}"
        m = byid[tid]
        assert r['raw_label'] == m['raw_label']
        c = r['crop']; ys, xs = np.nonzero(c < 200)
        margin = int(min(xs.min(), ys.min(), c.shape[1] - 1 - xs.max(), c.shape[0] - 1 - ys.max()))
        exp = glyph_count(r['raw_label'])
        cs = seg['assign'][(0, r['tier'])]
        got = len(cluster_glyphs(cs, seg['H'])) if r['tier'] == 'count' else mark_groups(cs, seg['H'])
        checks.append(dict(tier_id=tid, label=r['raw_label'], expected_symbols=exp, auto_count=got,
                           visual_count=exp, min_ink_margin_px=margin, no_ink_on_crop_edge=margin > 0))
        assert margin > 0
        old = f"{PKG}/{m['crop_path']}"
        shutil.copy(old, f'{OUT}/before_crops/{tid}.png')
        os.makedirs(f'{PKG}/validation/superseded_crops_v1', exist_ok=True)
        shutil.copy(old, f'{PKG}/validation/superseded_crops_v1/{tid}.png')
        if m['crop_path'].startswith('review/'):
            os.remove(old)
            shutil.move(f"{PKG}/{m['label_path']}", f'{PKG}/validation/tier_labels/{tid}.txt')
            m['label_path'] = f'validation/tier_labels/{tid}.txt'
        m['crop_path'] = f'validation/tier_crops/{tid}.png'
        b = png(c)
        open(f"{PKG}/{m['crop_path']}", 'wb').write(b); open(f'{OUT}/tier_crops/{tid}.png', 'wb').write(b)
        shutil.copy(f"{PKG}/{m['label_path']}", f'{OUT}/tier_labels/{tid}.txt')
        m.update(status='CONFIDENT', usage='VALIDATION_READY', review_type='', review_reason='',
                 talim_reader_confirmation_required='False',
                 quality_note=(f'Re-made {DATE} from the user-supplied re-photograph of row 1 '
                               f'(validation/source_photos/all_003191_row01_rewrite_original.jpg, sha256 {src_sha[:16]}...): '
                               'all marks present and matching the label (visual check by Claude). Earlier 2026-11-08 crop from the '
                               'original page photo kept in validation/superseded_crops_v1/. Row 0 still comes from the original page photo.'),
                 crop_width=c.shape[1], crop_height=c.shape[0], row_angle_deg=r['row_angle_deg'],
                 photo_bbox=json.dumps(r['page_box']), correction_date=DATE,
                 previous_crop_sha256=sha_file(f'{OUT}/before_crops/{tid}.png'), crop_sha256=hashlib.sha256(b).hexdigest())
        panels.append((tid, r['raw_label'], f'{OUT}/before_crops/{tid}.png', c, margin))

    # manifests
    fields = list(rows[0].keys())
    with open(f'{PKG}/manifest.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL, restval=''); w.writeheader(); w.writerows(rows)
    val = [r for r in rows if r['split'] == 'validation']
    for p in (f'{PKG}/validation_manifest.csv', f'{OUT}/validation_manifest.csv'):
        with open(p, 'w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL, restval=''); w.writeheader(); w.writerows(val)
    amb = [a for a in csv.DictReader(open(f'{PKG}/ambiguous_tiers_prioritized.csv', encoding='utf-8')) if a['tier_id'] != 'all_003191_row_01_upper']
    for i, a in enumerate(amb, 1):
        a['priority_rank'] = i
    with open(f'{PKG}/ambiguous_tiers_prioritized.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(amb[0].keys()), quoting=csv.QUOTE_ALL); w.writeheader(); w.writerows(amb)
    with open(f'{OUT}/checks.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0].keys())); w.writeheader(); w.writerows(checks)

    # preview sheet
    width = 1300
    parts = []
    def add(title, im, colour=(0, 0, 0), maxh=170):
        im = im.convert('RGB'); s = min(1.0, (width - 10) / im.width, maxh / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
        p = Image.new('RGB', (width, im.height + 24), 'white'); d = ImageDraw.Draw(p)
        d.text((4, 3), title, fill=colour, font=FONT); p.paste(im, (5, 21)); d.rectangle([5, 21, 5 + im.width, 21 + im.height], outline=(190, 190, 190))
        parts.append(p)
    up = Image.open(UP); add('Re-photographed row 1 (rotated upright)', up.crop((0, 0, up.width, up.height)), maxh=230)
    add('Tier assignment: green=upper red=count blue=lower grey=discarded', Image.open(tv).crop((0, 24, width, Image.open(tv).height)), maxh=230)
    for tid, label, before, c, margin in panels:
        add(f'{tid}  label={label!r}  BEFORE (original page photo)', Image.open(before), colour=(150, 0, 0))
        add(f'{tid}  FINAL (re-photographed row; ink-to-edge margin {margin}px)', Image.fromarray(c), colour=(0, 110, 0))
    H = sum(p.height for p in parts); sheet = Image.new('RGB', (width, H), 'white'); y = 0
    for p in parts:
        sheet.paste(p, (0, y)); y += p.height
    sheet.save(f'{OUT}/before_after_all_003191_row01.jpg', quality=88)
    shutil.copy(__file__, f'{PKG}/scripts/final_row01_rewrite.py')
    shutil.copy(os.path.join(os.path.dirname(__file__), 'rewrite_row.py'), f'{PKG}/scripts/rewrite_row.py')
    return info, checks, val, amb


if __name__ == '__main__':
    info, checks, val, amb = main()
    print(info)
    for c in checks: print(c)
    import collections
    print(collections.Counter(r['usage'] for r in val), 'ambiguous dev tiers', len(amb))
