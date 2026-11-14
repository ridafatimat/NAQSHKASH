"""Bounded font comparison of all 65 vocabulary characters; no OCR or model load."""
import argparse
import base64
from collections import defaultdict
import html
import io
import itertools
from runtime import *
from PIL import Image, ImageDraw, ImageFont, ImageOps

FONT = Path(r'C:\FYP Dataset Generation\Assets\Talamat.ttf')
SIZES = (24, 32, 48, 64, 96)


def mask(character, size, mode='L'):
    font = ImageFont.truetype(str(FONT), size)
    m = font.getmask(character, mode=mode)
    im = Image.frombytes('L', m.size, bytes(m))
    bbox = im.getbbox()
    return im.crop(bbox) if bbox else None


def roles(c):
    return [r for r in ('upper', 'count', 'lower') if c in allowed_characters(r)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()
    for d in ('reports', 'manifests', 'galleries', 'configs'):
        (WORKSPACE / d).mkdir(parents=True, exist_ok=True)
    vocab = load_json(PROJECT / 'config/vocabulary.json')['char_to_idx']
    assert len(vocab) == 65
    masks = {(c, size): mask(c, size) for c in vocab for size in SIZES}
    def same(a, b, size):
        x, y = masks[a, size], masks[b, size]
        return x is not None and y is not None and x.size == y.size and x.tobytes() == y.tobytes()
    pairs = []
    for a, b in itertools.combinations(vocab, 2):
        exact = [size for size in SIZES if same(a, b, size)]
        x, y = masks[a, 48], masks[b, 48]
        distance = None
        if x is not None and y is not None:
            xx, yy = x.resize((32, 32)), y.resize((32, 32))
            distance = sum(abs(i - j) for i, j in zip(xx.tobytes(), yy.tobytes())) / (1024 * 255)
        unique = len(roles(a)) == len(roles(b)) == 1 and roles(a) != roles(b)
        pairs.append(dict(character_a=a, character_b=b, roles_a=roles(a), roles_b=roles(b),
            exact_sizes=exact, exact_at_project_size_48=same(a, b, 48), identical_at_all_tested_sizes=len(exact) == len(SIZES),
            role_resolvable_at_48=same(a, b, 48) and unique,
            shape_distance_at_48=distance, similarity_is_not_equivalence=True))
    serialize_csv(WORKSPACE / 'reports/all_vocabulary_pairs.csv', pairs)
    mapping = []
    for p in pairs:
        if p['role_resolvable_at_48']:
            for a, b in ((p['character_a'], p['character_b']), (p['character_b'], p['character_a'])):
                mapping.append(dict(source=a, target=b, target_role=roles(b)[0], status='verified_exact_in_scope',
                    font_sha256=digest(FONT), verified_sizes=p['exact_sizes'], source_mode='L', anchor='lt_integer',
                    evidence='font_mask_audit; crop-to-visible-ink; project renderer integer anchor lt'))
    assert {(r['source'], r['target']) for r in mapping} == {('B', 'X'), ('X', 'B')}
    serialize_csv(WORKSPACE / 'manifests/verified_role_ambiguities.csv', mapping)
    summary = dict(vocabulary_characters=65, pairs_compared=len(pairs), sizes=list(SIZES),
        font_path=str(FONT), font_sha256=digest(FONT), font_copied=False,
        exact_pairs_by_size={str(s): [[p['character_a'], p['character_b']] for p in pairs if s in p['exact_sizes']] for s in SIZES},
        project_render_size=48, mapping_sha256=digest(WORKSPACE / 'manifests/verified_role_ambiguities.csv'),
        correction_scope='audited source font, L rendering, integer lt anchor, original source size48; scanning may transform an identical source mask',
        selection_data='font and canonical role definitions only; no 434-image evaluation outcomes used')
    write_json(WORKSPACE / 'reports/glyph_audit_summary.json', summary)
    profile = dict(font_sha256=summary['font_sha256'], font_size_px=48, source_mode='L', anchor='lt_integer',
        provenance='canonical_source_rendering_metadata', font_family='IMTCRPET')
    write_json(WORKSPACE / 'configs/rendering_profile.json', profile)
    selected = [p for p in pairs if p['exact_sizes']]
    near = sorted((p for p in pairs if p['shape_distance_at_48'] is not None and not p['exact_at_project_size_48']), key=lambda p:p['shape_distance_at_48'])[:12]
    extra = [p for p in pairs if set((p['character_a'], p['character_b'])) in ({'[', 'G'}, {'_', 'L'})]
    selection = list({(p['character_a'], p['character_b']):p for p in selected + near + extra}.values())
    lines = ['<!doctype html><meta charset="utf-8"><title>Glyph equivalence audit</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}section{border-top:1px solid #aaa;padding:18px}img{image-rendering:pixelated}pre{white-space:pre-wrap}</style>',
        '<h1>Glyph comparison: equivalence is scoped, similarity is insufficient</h1><p>Only B/X at the audited 48px source rendering are enabled. 32px fails exact equivalence. C/Y are not equivalent at48. No font file is copied or distributed.</p>']
    for p in selection:
        lines.append('<section><h2>' + html.escape(repr(p['character_a']) + ' / ' + repr(p['character_b'])) + '</h2><pre>' + html.escape(json.dumps(p, indent=2)) + '</pre>')
        for size in SIZES:
            canvas = Image.new('L', (180, 110), 255)
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 5), str(size) + ' px; visible masks', fill=0)
            for j, c in enumerate((p['character_a'], p['character_b'])):
                im = masks[c, size]
                if im is not None:
                    im = ImageOps.invert(im)
                    im.thumbnail((70, 75))
                    canvas.paste(im, (8 + 90*j, 30))
            stream = io.BytesIO(); canvas.save(stream, 'PNG')
            lines.append('<img alt="font mask pair" src="data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode() + '">')
        lines.append('</section>')
    (WORKSPACE / 'galleries/glyph_comparison.html').write_text('\n'.join(lines), encoding='utf-8')
    report = '''# Glyph ambiguity audit

All65 fixed vocabulary characters (space included), 2080 pairs, were compared at
24,32,48,64,96 source font pixels using the local Talamat.ttf, grayscale L masks,
integer positions and the project's lt anchor convention. Visible ink is trimmed
for equivalence because D1 produces ink-band crops; full font whitespace differs.
The renderer's actual source size48 is confirmed in original Stage1 and positive
source JSON rendering metadata. E1 target-height64 is a resized crop, not proof
that source font size was64. No font file was copied.

Exact visible pairs: size24 B/X and C/Y; size32 none; size48 B/X; size64 B/X and
C/Y; size96 B/X. Thus no pair is claimed byte-identical under every rendering.
Only B/X is exact at the audited project source size48. B is upper-only and X
lower-only; a reliable upper/lower role uniquely resolves that label ambiguity.
The verified mapping is two directed entries. Default T1 scope is original
source size48, L mode, this font hash and integer lt rendering. Uploaded images
without trusted rendering provenance remain uncertain; source32 is blocked.
Scanned variants inherit scope only from verified source48 metadata, not from
an assumed observed pixel size. Equality before the same scan transformation
does not establish that segmentation or the whole transcription is correct.

C/Y, [ / G and _ / L are merely similar at48 and are excluded. Normalized shape
distances rank comparison illustrations only; they are neither calibrated scores
nor correction thresholds. Pairs sharing the same role cannot be resolved from
tier identity alone. Count-tier symbols are drawn from the canonical encoded
1..259 values; no unsupported notation or numeric constraints are invented.

Every correction still requires reliable D1 metadata, one legal target, absence
of conflicting evidence and no remaining unsupported character in that tier.
This is contextual interpretation of a discrete label, not a new CTC decoding
algorithm or proof that source image content is valid. No evaluation labels or
exposed434 outcomes were consulted to construct the mapping.
'''
    (WORKSPACE / 'reports/glyph_ambiguity_audit.md').write_text(report, encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
