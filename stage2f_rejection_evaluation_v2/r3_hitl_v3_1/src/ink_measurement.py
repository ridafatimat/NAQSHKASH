"""D1-only pixels and saved OCR; no renderer, source annotation or validity input."""
import importlib.util,math,sys
from collections import defaultdict
import numpy as np
from paths_v3_1 import D1,PROJECT,content_hash,digest,load_json


def ink_groups(mask):
    """Contiguous occupied x projections group vertically disconnected marks.

    Separated x fragments are not merged. A projection group is not asserted to
    be a glyph until the independent grouping checks in estimate_pitch pass.
    """
    import cv2
    if mask.ndim!=2 or mask.dtype!=np.bool_ or not mask.size:raise ValueError('Expected nonempty boolean mask')
    edges=np.diff(np.pad(mask.any(axis=0).astype(np.int8),(1,1)))
    result=[]
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        ys,xs=np.where(mask[:,a:b]);top,bottom=int(ys.min()),int(ys.max()+1)
        tile=mask[top:bottom,a:b]
        n=cv2.connectedComponents(tile.astype(np.uint8),connectivity=8)[0]-1
        result.append(dict(box=[int(a),top,int(b),bottom],components=int(n),shape_sha256=content_hash(dict(shape=list(tile.shape),bits=np.packbits(tile).tolist()))))
    return result


def first_ink(mask,box):
    h,w=mask.shape
    if len(box)!=4 or any(type(v) is not int for v in box):raise ValueError('Noninteger tier box')
    x0,y0,x1,y1=box
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Tier box outside processed image')
    groups=ink_groups(mask[y0:y1,x0:x1])
    return dict(status='measured' if groups else 'empty_tier',box=box,first_ink_x=x0+groups[0]['box'][0] if groups else None,
                groups=[dict(g,first_ink_x=x0+g['box'][0]) for g in groups])


def estimate_pitch(count_tiers,minimum_observations=3,raster_uncertainty=1):
    """Median adjacent decoded-count-glyph starts, with conservative checks.

    Raw fragment starts alone never establish pitch. Projection-group count
    must equal the OCR's visible count-character count; repeated characters
    must have identical binarized silhouettes. Adjacent character pairs (no
    decoded space between) supply distances. All starts must agree with one
    lattice to within the fixed one-pixel raster allowance. No fallback pitch.
    """
    observations=[];accepted=[];issues=[];shapes=defaultdict(set)
    for t in count_tiers:
        text=t['text'];chars=[(i,ch) for i,ch in enumerate(text) if ch!=' ']
        groups=t['groups']
        if not text or text.startswith(' ') or any(ch.isspace() and ch!=' ' for ch in text) or len(groups)!=len(chars):
            issues.append(dict(row=t['row'],reason='COUNT_GROUPING_CARDINALITY_OR_TEXT_UNSUPPORTED',groups=len(groups),characters=len(chars)));continue
        for (_,ch),g in zip(chars,groups):shapes[ch].add(g['shape_sha256'])
        accepted.append((t,chars))
        for j in range(1,len(chars)):
            if chars[j][0]-chars[j-1][0]==1:
                observations.append(dict(row=t['row'],left_group=j-1,right_group=j,pixels=groups[j]['first_ink_x']-groups[j-1]['first_ink_x']))
    base=dict(status='unavailable',pitch_pixels=None,pitch_interval=None,observations=observations,usable_observations=len(observations),grouping_issues=issues,
              grouping_assumption='projection/cardinality/repeated-silhouette/lattice checks; not glyph identity recognition')
    if any(len(values)>1 for values in shapes.values()):return dict(base,reason='REPEATED_OCR_CHARACTER_HAS_INCONSISTENT_INK_GROUPS')
    if len(observations)<minimum_observations:return dict(base,reason='INSUFFICIENT_ADJACENT_GLYPH_STARTS')
    values=[o['pixels'] for o in observations];pitch=float(np.median(values))
    if not math.isfinite(pitch) or pitch<=2*raster_uncertainty:return dict(base,reason='DEGENERATE_PITCH')
    if any(abs(v-pitch)>raster_uncertainty for v in values):return dict(base,reason='INCONSISTENT_GLYPH_START_SPACING')
    residuals=[]
    for t,chars in accepted:
        x=t['groups'][0]['first_ink_x'];origin=chars[0][0]
        for (i,_),g in zip(chars,t['groups']):
            residual=g['first_ink_x']-x-(i-origin)*pitch;residuals.append(float(residual))
            if abs(residual)>raster_uncertainty:return dict(base,reason='COUNT_GROUPS_DO_NOT_SUPPORT_ONE_CELL_LATTICE',lattice_residuals=residuals)
            if g['box'][2]-g['box'][0]>=pitch:return dict(base,reason='GROUP_SPANS_CELL_PITCH_OR_TOUCHING_GLYPHS')
    return dict(base,status='measured',reason=None,pitch_pixels=pitch,pitch_interval=[min(values)-raster_uncertainty,max(values)+raster_uncertainty],
                lattice_residuals=residuals,median_absolute_deviation=float(np.median(np.abs(np.array(values)-pitch))))


def measure_processed(processed,rows,saved_rows,decoded_rows,provenance,configuration):
    import cv2
    out=dict(status='unavailable',coordinate_system='D1_processed_pixel_centers',geometry_verified=False,rows=[],pitch=None,
             decoded_rows_sha256=content_hash(decoded_rows),provenance=provenance,box_equality=rows==saved_rows,
             regenerated_boxes=rows,saved_boxes=saved_rows)
    if rows!=saved_rows:return dict(out,reason='D1_BOX_MISMATCH')
    if not isinstance(processed,np.ndarray) or processed.dtype!=np.uint8 or processed.ndim!=2 or not processed.size:return dict(out,reason='UNSUPPORTED_PROCESSED_IMAGE')
    if not rows or len(rows)!=len(decoded_rows):return dict(out,reason='D1_OCR_ROW_MISMATCH')
    if int(processed.min())==int(processed.max()):return dict(out,reason='CONSTANT_IMAGE')
    # This exactly matches the frozen D1 engine's page binarization.
    threshold,binary=cv2.threshold(processed,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    mask=binary>0;counts=[];reasons=[]
    out.update(geometry_verified=True,processed_dimensions=list(processed.shape[::-1]),processed_pixels_sha256=content_hash(dict(shape=list(processed.shape),bytes_sha256=__import__('hashlib').sha256(processed.tobytes()).hexdigest())),
               threshold=dict(method='frozen_D1_page_OTSU',value=float(threshold),fallback_not_used=True))
    for ri,tiers in enumerate(rows):
        measured={};roles=[t.get('role') for t in tiers]
        expected={k for k,v in decoded_rows[ri].items() if v}
        if len(set(roles))!=len(roles) or set(roles)!=expected:reasons.append('D1_OCR_TIER_MISMATCH')
        for t in tiers:
            if t.get('row')!=ri or t.get('role') not in ('upper','count','lower'):reasons.append('UNSUPPORTED_TIER_SCHEMA');continue
            try:m=first_ink(mask,t['box'])
            except ValueError as exc:reasons.append(str(exc));continue
            measured[t['role']]=m
            if m['status']!='measured':reasons.append('EMPTY_TIER')
            if t['role']=='count':counts.append(dict(m,row=ri,text=decoded_rows[ri].get('count','')))
        anchors=[m['first_ink_x'] for m in measured.values() if m['first_ink_x'] is not None]
        out['rows'].append(dict(logical_row_index=ri,row_origin=min(anchors) if anchors else None,tiers=measured))
    pitch=estimate_pitch(counts,configuration['minimum_pitch_observations'],configuration['raster_uncertainty_pixels']);out['pitch']=pitch
    if pitch['status']!='measured':reasons.append(pitch['reason'])
    if pitch['grouping_issues']:reasons.append('COUNT_GROUPING_UNSUPPORTED_IN_ONE_OR_MORE_ROWS')
    if reasons:return dict(out,reason=';'.join(sorted(set(reasons))))
    for r in out['rows']:
        for t in r['tiers'].values():
            distance=t['first_ink_x']-r['row_origin'];p=pitch['pitch_pixels'];lo,hi=pitch['pitch_interval']
            t.update(measured_offset=distance/p,offset_interval=[max(0,distance-configuration['raster_uncertainty_pixels'])/hi,(distance+configuration['raster_uncertainty_pixels'])/lo],
                     offset_uncertainty='one-pixel anchor raster allowance plus observed pitch interval; no glyph-bearing correction')
    return dict(out,status='complete',reason=None)


class D1Only:
    """Loads the original pixel API directly; never calls FrozenPipeline/E1."""
    def __init__(self):
        sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(D1))
        spec=importlib.util.spec_from_file_location('r3_1_frozen_d1',D1/'pipeline_d1.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.configuration=load_json(D1/'configs/D1.json')

    def measure(self,raw,saved_rows,decoded_rows,configuration):
        import cv2
        image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_GRAYSCALE) if raw else None
        if image is None:return dict(status='unavailable',reason='UNREADABLE_IMAGE',rows=[],geometry_verified=False,box_equality=None)
        result=self.module.process_image(image,self.configuration)
        matrix=np.asarray(result['matrix'],dtype=float)
        provenance=dict(input_image_sha256=__import__('hashlib').sha256(raw).hexdigest(),original_dimensions=list(image.shape[::-1]),
                        source_to_processed_matrix=matrix.tolist(),D1_source_sha256=digest(D1/'pipeline_d1.py'),D1_config_sha256=digest(D1/'configs/D1.json'))
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
            provenance['source_to_processed_matrix']=None
            return dict(status='unavailable',reason='INVALID_D1_TRANSFORM',rows=[],geometry_verified=False,box_equality=result['rows']==saved_rows,provenance=provenance)
        if not result['accepted']:return dict(status='unavailable',reason='D1_REJECTED:'+result['reason'],rows=[],geometry_verified=False,box_equality=result['rows']==saved_rows,provenance=provenance,regenerated_boxes=result['rows'],saved_boxes=saved_rows)
        return measure_processed(result['image'],result['rows'],saved_rows,decoded_rows,provenance,configuration)
