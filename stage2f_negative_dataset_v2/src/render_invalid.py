"""Faithful display renderer; it never reconstructs or repairs invalid notation."""
from common_v2 import *
from common_v2 import _cell_metrics
from PIL import ImageDraw


def layout_for(sample):
    font=ImageFont.truetype(str(FONT),48)
    text=rules.canonical_text(sample)
    width,height,left=_cell_metrics(font,text)
    columns=max(len(s) for row in displayed_rows(sample) for s in row.values())
    return dict(font_sha256=digest(FONT),font_size_px=48,padding=24,line_gap=8,row_gap=20,
        cell_width=width,line_height=height+8,extra_left=left,columns=columns,rows=sample['rows'],
        width=48+left+columns*width,height=48+sample['rows']*3*(height+8)+(sample['rows']-1)*20,
        convention='existing Talamat grayscale integer lt grid; matched parent layout is frozen')


def render_display(display, layout):
    """Draw the supplied raw tier strings exactly. Invalid middle gaps survive."""
    if len(display)!=layout['rows']:raise ValueError('Display row count/layout mismatch')
    font=ImageFont.truetype(str(FONT),layout['font_size_px'])
    image=Image.new('L',(layout['width'],layout['height']),255)
    draw=ImageDraw.Draw(image)
    for ri,row in enumerate(display):
        if set(row)!=set(ROLES):raise ValueError('Display must explicitly retain all three tiers')
        for ti,role in enumerate(ROLES):
            if len(row[role])>layout['columns']:raise ValueError('Display extends outside frozen grid')
            for col,char in enumerate(row[role]):
                if char==' ':continue
                draw.text((layout['padding']+layout['extra_left']+col*layout['cell_width'],
                           layout['padding']+ri*(3*layout['line_height']+layout['row_gap'])+ti*layout['line_height']),
                          char,font=font,fill=0,anchor='lt')
    return image


def pixel_difference(a,b):
    if a.size!=b.size:raise ValueError('Matched images must use identical canvas geometry')
    from PIL import ImageChops
    diff=ImageChops.difference(a.convert('L'),b.convert('L'))
    return dict(changed_pixels=sum(v!=0 for v in diff.tobytes()),difference_bbox=list(diff.getbbox()) if diff.getbbox() else None,
                mean_absolute_difference=sum(diff.tobytes())/(a.width*a.height),same_dimensions=True)
