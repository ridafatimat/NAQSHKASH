"""Independent inverse renderer/parser. Does NOT import mutations or trust labels.

It identifies actual raster cells against every supported glyph, then searches
all canonical 1..259 run encodings with role/direction constraints. Verification
is scoped to the documented fixed source font/grid, not arbitrary real pages.
"""
from functools import lru_cache
from common_v2 import *
from PIL import ImageDraw,ImageOps


@lru_cache(maxsize=24)
def atlas(cell_width,line_height):
    characters=sorted(set(rules.TRUSTED_SYMBOLS)|set(''.join(rules.encode_count(n) for n in range(1,260))))
    font=ImageFont.truetype(str(FONT),48)
    result={' ':Image.new('L',(cell_width,line_height),255)}
    for c in characters:
        im=Image.new('L',(cell_width,line_height),255)
        ImageDraw.Draw(im).text((0,0),c,font=font,fill=0,anchor='lt')
        result[c]=im
    return result


@lru_cache(maxsize=1)
def visual_aliases():
    audit=read_json(ROOT/'configs/visual_alias_exclusions.json')
    if audit['font_sha256']!=digest(FONT):raise ValueError('Visual-alias audit does not apply to this font')
    return {c:set(group) for group in audit['groups'] for c in group}


def role_candidates(character,role,layout):
    """Any role-legal glyph with the same visible raster, including B/X aliases."""
    glyphs=atlas(layout['cell_width'],layout['line_height'])
    allowed=(rules.EXCLUSIVE_UP_SYMBOLS if role=='upper' else rules.EXCLUSIVE_DOWN_SYMBOLS if role=='lower'
             else set(''.join(rules.encode_count(n) for n in range(1,260))))
    return [c for c in allowed if glyphs[c].tobytes()==glyphs[character].tobytes() or c in visual_aliases().get(character,set())]


def recover_cells(image,layout,augmentation=None):
    image=image.convert('L')
    if image.size!=(layout['width'],layout['height']):raise ValueError('Image/layout dimensions disagree')
    if digest(FONT)!=layout['font_sha256']:raise ValueError('Font differs from frozen renderer')
    if layout['font_size_px']!=48:raise ValueError('Rendering outside verified source48 scope')
    templates=atlas(layout['cell_width'],layout['line_height'])
    # Independently reproduce the documented pointwise brightness transfer on
    # the font atlas, not on a generator-labeled target. No mutation code is used.
    if augmentation:
        operations=augmentation.get('operations',[])
        if augmentation.get('padding')!=0 or len(operations)!=1 or operations[0].get('type')!='brightness':
            raise ValueError('Unverified augmentation profile; no automatic semantic label')
        factor=operations[0]['factor']
        if factor not in (0.92,0.96):raise ValueError('Brightness profile outside frozen evidence scope')
        lut=[int(v*factor) for v in range(256)]
        keys={c:im.point(lut).tobytes() for c,im in templates.items()}
    else:keys={c:im.tobytes() for c,im in templates.items()}
    rows=[];unknown=[]
    for ri in range(layout['rows']):
        row={}
        for ti,role in enumerate(ROLES):
            cells=[]
            for col in range(layout['columns']):
                x=layout['padding']+layout['extra_left']+col*layout['cell_width']
                y=layout['padding']+ri*(3*layout['line_height']+layout['row_gap'])+ti*layout['line_height']
                crop=image.crop((x,y,x+layout['cell_width'],y+layout['line_height']))
                value=crop.tobytes()
                matches=[c for c,k in keys.items() if k==value]
                if not matches:unknown.append(dict(row=ri,tier=role,column=col))
                # Raster shades do not prove a visibly wrong-tier glyph when
                # the local font audit identifies the same perceptual mark.
                matches=sorted(set(matches)|{a for c in matches for a in visual_aliases().get(c,set())})
                cells.append(matches)
            row[role]=cells
        rows.append(row)
    return rows,unknown


def parse_row(row):
    """All legal interpretations; never consult original runs or mutation type."""
    active=[i for i in range(len(row['count_line'])) if any(' ' not in row[k][i] for k in ROLES)]
    if not active:return set(),['EMPTY_LOGICAL_ROW']
    end=max(active)+1
    violations=[]
    for i in range(end):
        for role,allowed in [('upper',set(rules.EXCLUSIVE_UP_SYMBOLS)),('lower',set(rules.EXCLUSIVE_DOWN_SYMBOLS))]:
            if ' ' not in row[role][i] and not set(row[role][i])&allowed:
                violations.append('CANON_SYMBOL_DIRECTION')
        if ' ' in row['count_line'][i]:violations.append('CANON_REQUIRED_SYMBOL_COUNT')
    encodings=[(n,rules.encode_count(n)) for n in range(1,260)]
    states={(0,None,0)}
    finished=set()
    for _ in range(end+1):
        following=set()
        for col,previous,total in states:
            if col==end:finished.add(total);continue
            for role,opposite,allowed in [('upper','lower',rules.EXCLUSIVE_UP_SYMBOLS),('lower','upper',rules.EXCLUSIVE_DOWN_SYMBOLS)]:
                if ' ' not in row[opposite][col]:continue
                for symbol in set(row[role][col])&set(allowed):
                    if symbol==previous:continue
                    for n,token in encodings:
                        width=len(token)
                        if col+width>end:continue
                        if not all(token[j] in row['count_line'][col+j] for j in range(width)):continue
                        if not all(' ' in row[k][col+j] for j in range(1,width) for k in ('upper','lower')):continue
                        following.add((col+width,symbol,total+n))
        states=following
        if not states:break
    if not finished and not violations:violations.append('CANON_COUNT_ENCODING')
    return finished,sorted(set(violations))


def verify_image(image,layout,augmentation=None):
    if image.convert('L').getextrema()[0]==image.convert('L').getextrema()[1]:
        return dict(status='AUTO_VERIFIED',classification='IMAGE_QUALITY_FAILURE',reason_codes=['CONSTANT_IMAGE_NO_TALIM_CONTENT'],
                    evidence=dict(scope='input artifact, not semantic Talim invalidity'),human_reviewed=False)
    cells,unknown=recover_cells(image,layout,augmentation)
    if unknown:return dict(status='NEEDS_DOMAIN_REVIEW',classification='UNSUPPORTED_OR_AMBIGUOUS',reason_codes=['UNMATCHED_RASTER_CELLS'],
                           evidence=dict(unknown_cells=unknown),human_reviewed=False)
    parsed=[parse_row(r) for r in cells]
    common_totals=set.intersection(*(p[0] for p in parsed)) if parsed else set()
    codes=sorted({c for totals,cs in parsed for c in cs})
    if common_totals:
        return dict(status='AUTO_VERIFIED',classification='VALID',reason_codes=['CANONICAL_RASTER_PARSE_EXISTS'],
                    evidence=dict(possible_cells_per_row=sorted(common_totals),row_totals=[sorted(t) for t,_ in parsed]),human_reviewed=False)
    if not codes:codes=['CANON_ROW_SUM_CONTRADICTION']
    return dict(status='AUTO_VERIFIED',classification='INVALID',reason_codes=codes,
        evidence=dict(legal_canonical_interpretations=0,row_totals=[sorted(t) for t,_ in parsed],
                      scope='documented fixed-font three-tier canonical renderer; alternatives include all glyph aliases and authentic counts',
                      raster_cells_checked=layout['rows']*3*layout['columns']),human_reviewed=False)


def display_agrees(image,layout,display,augmentation=None):
    """Pixel evidence must agree with declared raw glyphs, without rerendering."""
    cells,unknown=recover_cells(image,layout,augmentation)
    if unknown or len(display)!=len(cells):return False
    return all(c in cells[ri][role][col] for ri,row in enumerate(display) for role in ROLES
               for col,c in enumerate(row[role].ljust(layout['columns'])))


def verify_matched_negative(original,mutated,layout,raw_display,intended_rule=None):
    """Admission gate independent of mutation procedure and assigned label."""
    if original.size!=mutated.size or original.convert('L').tobytes()==mutated.convert('L').tobytes():
        return dict(status='EXCLUDED',classification='UNPROVEN',reason_codes=['VISUALLY_UNCHANGED_OR_GEOMETRY_MISMATCH'],evidence={},human_reviewed=False)
    positive=verify_image(original,layout)
    negative=verify_image(mutated,layout)
    consistent=display_agrees(mutated,layout,raw_display)
    if (positive['classification']!='VALID' or negative['classification']!='INVALID' or not consistent
            or intended_rule is not None and intended_rule not in negative['reason_codes']):
        return dict(status='EXCLUDED',classification='UNPROVEN',reason_codes=['INVALIDITY_OR_ANNOTATION_NOT_INDEPENDENTLY_ESTABLISHED'],
                    evidence=dict(original=positive,mutated=negative,raw_display_consistent=consistent),human_reviewed=False)
    negative['evidence'].update(matched_control_independently_valid=True,image_pixels_changed=True,raw_display_consistent=True)
    return negative
