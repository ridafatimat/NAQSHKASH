"""Pure CTC geometry and OFFLINE scoring. No model, dataset loader or decisions."""
import math
from collections import defaultdict
import numpy as np


def path_runs(path,vocabulary,blank=0):
    runs=[]
    for i,idx in enumerate(map(int,path)):
        if runs and runs[-1]['class_id']==idx:runs[-1]['end_frame']=i+1
        else:runs.append(dict(class_id=idx,start_frame=i,end_frame=i+1,character=None if idx==blank else vocabulary[idx]))
    return runs,[dict(r,character_index=i) for i,r in enumerate(r for r in runs if r['class_id']!=blank)]


def project(points,matrix):
    a=np.c_[np.asarray(points,dtype=float),np.ones(len(points))]@np.asarray(matrix,dtype=float).T
    if not np.isfinite(a).all() or np.any(np.abs(a[:,2])<1e-10):raise ValueError('Nonfinite/projective-horizon mapping')
    return (a[:,:2]/a[:,2:]).tolist()


def map_span(span,box,resized_width,matrix):
    """CV2 pixel-center coordinates (first pixel is 0); intervals use pixel edges.

    Lattice spans and 42px CNN receptive support are NOT confidence intervals.
    BiLSTM classification is globally contextual.
    """
    x0,y0,x1,y1=box;scale=(x1-x0)/resized_width
    start,end=span['start_frame'],span['end_frame'];inverse=np.linalg.inv(matrix)
    def interval(left,right):
        a=x0+left*scale-.5;b=x0+right*scale-.5
        q=project([(a,y0-.5),(b,y0-.5),(b,y1-.5),(a,y1-.5)],inverse)
        return dict(polygon=q,x_interval=[min(p[0] for p in q),max(p[0] for p in q)])
    center_x=x0+(4*((start+end-1)/2)+2)*scale-.5
    center=project([(center_x,(y0+y1-1)/2)],inverse)[0]
    return dict(source_center=center,processed_center=[center_x,(y0+y1-1)/2],
        lattice=interval(4*start,min(4*end,resized_width)),
        nominal_CNN_support=interval(max(0,4*start+2-21),min(resized_width,4*(end-1)+2+21)))


def unique_correspondences(prediction,reference):
    """All minimum edit paths; do not pick a convenient alignment of repeats.

    Returns each prediction's unique reference index, or None; lists every
    possible index/deletion for auditing. Only exact/B-X-equivalent matches score.
    """
    canonical=lambda c:'B' if c=='X' else c
    a=[canonical(c) for c in prediction];b=[canonical(c) for c in reference];n,m=len(a),len(b)
    f=np.zeros((n+1,m+1),dtype=int);g=f.copy();f[:,0]=range(n+1);f[0,:]=range(m+1)
    for i in range(1,n+1):
        for j in range(1,m+1):f[i,j]=min(f[i-1,j]+1,f[i,j-1]+1,f[i-1,j-1]+(a[i-1]!=b[j-1]))
    g[:,m]=range(n,-1,-1);g[n,:]=range(m,-1,-1)
    for i in range(n-1,-1,-1):
        for j in range(m-1,-1,-1):g[i,j]=min(g[i+1,j]+1,g[i,j+1]+1,g[i+1,j+1]+(a[i]!=b[j]))
    result=[]
    for i in range(n):
        options=[]
        if any(f[i,j]+1+g[i+1,j]==f[n,m] for j in range(m+1)):options.append(None)
        for j in range(m):
            if f[i,j]+(a[i]!=b[j])+g[i+1,j+1]==f[n,m]:options.append(j)
        target=options[0] if len(options)==1 and options[0] is not None and a[i]==b[options[0]] else None
        result.append(dict(reference_index=target,possible_reference_indices=options,
            status='matched' if target is not None else 'ambiguous' if len(options)>1 else 'different_or_extra'))
    return result


def associate(symbols,counts):
    """Diagnostic geometry-only partition; NO canonical validation or rejection.

    Partition count glyphs into nonempty contiguous groups in observed symbol
    x-order, minimizing sum |first-count center - symbol center|. No GT inputs.
    Missing/empty tokens and equal-cost partitions are deliberately unscorable.
    """
    m,n=len(symbols),len(counts)
    if not m or n<m:return dict(status='insufficient_visible_glyphs',groups=[])
    dp={(0,0):(0.,1,[])}
    for j in range(1,m+1):
        for end in range(j,n+1):
            candidates=[]
            for start in range(j-1,end):
                if (j-1,start) in dp:
                    cost,ways,path=dp[j-1,start]
                    candidates.append((cost+abs(counts[start]['x']-symbols[j-1]['x']),ways,path+[(start,end)]))
            if candidates:
                best=min(c[0] for c in candidates);tied=[c for c in candidates if math.isclose(c[0],best,rel_tol=0,abs_tol=1e-9)]
                dp[j,end]=(best,min(2,sum(c[1] for c in tied)),tied[0][2])
    if (m,n) not in dp:return dict(status='unscorable',groups=[])
    cost,ways,path=dp[m,n]
    if ways>1:return dict(status='ambiguous_partition',groups=[],cost_pixels=cost)
    return dict(status='unique',cost_pixels=cost,groups=[list(range(a,b)) for a,b in path])


def reference_glyphs(record,row_index,font):
    """OFFLINE ONLY. Use this image's mutated raw_display, never parent notation."""
    from PIL import Image,ImageDraw,ImageChops
    layout=record['layout'];row=record['raw_display'][row_index];result={}
    for ti,role in enumerate(('upper','count_line','lower')):
        glyphs=[]
        for col,char in enumerate(row[role]):
            if char==' ':continue
            margin=font.size;im=Image.new('L',(font.size*4,font.size*4),255)
            ImageDraw.Draw(im).text((margin,margin),char,font=font,fill=0,anchor='lt')
            bb=ImageChops.difference(im,Image.new('L',im.size,255)).getbbox()
            if bb is None:continue
            x=layout['padding']+layout['extra_left']+col*layout['cell_width']
            y=layout['padding']+row_index*(3*layout['line_height']+layout['row_gap'])+ti*layout['line_height']
            box=[x+bb[0]-margin,y+bb[1]-margin,x+bb[2]-margin,y+bb[3]-margin]
            glyphs.append(dict(character=char,column=col,box=box,x=(box[0]+box[2]-1)/2,width=box[2]-box[0]))
        result['count' if role=='count_line' else role]=glyphs
    return result


def score_image(capture,record,row_index,font,focus_column=None):
    """Labels/renderer metadata enter ONLY here, after immutable extraction."""
    refs=reference_glyphs(record,row_index,font);tiers={t['role']:t for t in capture.get('tiers',[])}
    matches={};metrics=dict(reference_glyphs=sum(map(len,refs.values())),matched_glyphs=0,
        ambiguous_glyphs=0,wrong_or_extra_glyphs=0,unmapped_glyphs=0,position_errors=[],normalized_errors=[],
        token_order_correct=0,token_order_denominator=0,multi_glyph_token_correct=0,multi_glyph_token_denominator=0,
        association_correct=0,association_denominator=0,unique_count_assignments=0,visible_count_denominator=len(refs['count']),
        false_association_risk=False,false_order_risk=False,reference_empty_tokens=0)
    reproduction=capture.get('reproduction',{}).get('all_match',False)
    for role,reference in refs.items():
        tier=tiers.get(role,{});pred=[s for s in tier.get('characters',[]) if s['character']!=' ']
        correspondence=unique_correspondences(''.join(s['character'] for s in pred),''.join(g['character'] for g in reference))
        for p,c in zip(pred,correspondence):
            p=dict(p,correspondence=c);p['reference_index']=c['reference_index']
            if not reproduction:p['reference_index']=None;p['unscorable_reason']='reproduction_mismatch'
            elif c['status']=='ambiguous':metrics['ambiguous_glyphs']+=1
            elif c['status']!='matched':metrics['wrong_or_extra_glyphs']+=1
            elif not p.get('mapping'):metrics['unmapped_glyphs']+=1;p['reference_index']=None
            else:
                g=reference[c['reference_index']];error=p['mapping']['source_center'][0]-g['x']
                metrics['position_errors'].append(error);metrics['normalized_errors'].append(abs(error)/g['width']);metrics['matched_glyphs']+=1
            matches.setdefault(role,[]).append(p)
    # Model-only positions drive candidate associations; reference positions only score them.
    symbols=sorted([dict(s,role=role,x=s['mapping']['source_center'][0]) for role in ('upper','lower') for s in matches.get(role,[]) if s.get('mapping')],key=lambda s:s['x'])
    counts=[dict(s,x=s['mapping']['source_center'][0]) for s in matches.get('count',[]) if s.get('mapping')]
    association=associate(symbols,counts) if reproduction else dict(status='reproduction_mismatch',groups=[])
    # True run boundaries are source symbol columns, exclusively OFFLINE.
    true_symbols=sorted([(g['column'],role,i) for role in ('upper','lower') for i,g in enumerate(refs[role])]);count_owner={}
    owners={}
    for j,(start,role,idx) in enumerate(true_symbols):
        end=true_symbols[j+1][0] if j+1<len(true_symbols) else len(record['raw_display'][row_index]['count_line'])
        inds=[i for i,g in enumerate(refs['count']) if start<=g['column']<end]
        owners[role,idx]=inds
        if not inds:metrics['reference_empty_tokens']+=1
        for i in inds:count_owner[i]=(role,idx)
    assigned_by_owner={}
    if association['status']=='unique':
        for symbol,group in zip(symbols,association['groups']):
            key=(symbol['role'],symbol['reference_index']);expected=owners.get(key)
            assigned=[counts[i] for i in group];metrics['unique_count_assignments']+=len(assigned)
            if symbol['reference_index'] is not None:assigned_by_owner[key]=[c['reference_index'] for c in assigned]
            for c in assigned:
                if symbol['reference_index'] is not None and c['reference_index'] is not None:
                    metrics['association_denominator']+=1;ok=count_owner.get(c['reference_index'])==key
                    metrics['association_correct']+=ok;metrics['false_association_risk']|=not ok
            if expected:
                # Token coverage denominator includes every nonempty source token below.
                exact=[c['reference_index'] for c in assigned]==expected
                if exact:metrics['token_order_correct']+=1
                if len(expected)>1:
                    metrics['multi_glyph_token_correct']+=exact
                    metrics['false_order_risk']|=not exact
    metrics['token_order_denominator']=sum(bool(v) for v in owners.values())
    metrics['multi_glyph_token_denominator']=sum(len(v)>1 for v in owners.values())
    metrics.update(focused_visible_token_denominator=0,focused_token_correct=0,focused_unscorable=0)
    focus=dict(status='not_requested')
    if focus_column is not None:
        keys=[(role,idx) for col,role,idx in true_symbols if col==focus_column]
        if len(keys)!=1:focus=dict(status='unscorable_source_symbol')
        else:
            key=keys[0];expected=owners[key]
            if not expected:focus=dict(status='no_visible_count_to_align',column=focus_column)
            else:
                metrics['focused_visible_token_denominator']=1
                observed=assigned_by_owner.get(key)
                metrics['focused_token_correct']=int(observed==expected)
                metrics['focused_unscorable']=int(observed is None)
                focus=dict(status='correct' if observed==expected else 'unscorable_symbol_or_partition' if observed is None else 'incorrect',
                    column=focus_column,expected_reference_indices=expected,assigned_reference_indices=observed)
    return dict(metrics=metrics,association=association,reference=refs,predictions=matches,
        focus=focus,reproduced=reproduction,absence_observability='NOT_MEASURED: blank frames are not image whitespace; empty counts have no glyph to align')


def summarize(cases):
    additive=('reference_glyphs','matched_glyphs','ambiguous_glyphs','wrong_or_extra_glyphs','unmapped_glyphs',
        'token_order_correct','token_order_denominator','multi_glyph_token_correct','multi_glyph_token_denominator',
        'association_correct','association_denominator','unique_count_assignments','visible_count_denominator','reference_empty_tokens',
        'focused_visible_token_denominator','focused_token_correct','focused_unscorable')
    out={k:sum(c['score']['metrics'][k] for c in cases) for k in additive}
    errors=[v for c in cases for v in c['score']['metrics']['position_errors']];norm=[v for c in cases for v in c['score']['metrics']['normalized_errors']]
    def ratio(a,b):return a/b if b else None
    out.update(images=len(cases),parents=len({c['parent_id'] for c in cases}),reproduction_mismatches=sum(not c['score']['reproduced'] for c in cases),
        alignment_coverage=ratio(out['matched_glyphs'],out['reference_glyphs']),
        reference_glyphs_uncovered=out['reference_glyphs']-out['matched_glyphs'],position_axis='source_x; vertical support is the full tier, not a localized glyph center',
        token_order_agreement=ratio(out['token_order_correct'],out['token_order_denominator']),
        multi_glyph_token_agreement=ratio(out['multi_glyph_token_correct'],out['multi_glyph_token_denominator']),
        association_accuracy_conditional=ratio(out['association_correct'],out['association_denominator']),
        focused_token_agreement=ratio(out['focused_token_correct'],out['focused_visible_token_denominator']),
        association_scoring_coverage=ratio(out['association_denominator'],out['visible_count_denominator']),
        unique_assignment_coverage=ratio(out['unique_count_assignments'],out['visible_count_denominator']),
        position_error_pixels=dict(signed_mean=float(np.mean(errors)),median_absolute=float(np.median(np.abs(errors))),p95_absolute=float(np.percentile(np.abs(errors),95))) if errors else None,
        normalized_position_error=dict(median=float(np.median(norm)),p95=float(np.percentile(norm,95))) if norm else None,
        valid_control_association_risk_images=sum(c['validity']=='VALID' and c['score']['metrics']['false_association_risk'] for c in cases),
        valid_control_order_risk_images=sum(c['validity']=='VALID' and c['score']['metrics']['false_order_risk'] for c in cases),
        valid_control_incomplete_evidence_images=sum(c['validity']=='VALID' and (not c['score']['reproduced'] or c['score']['metrics']['matched_glyphs']<c['score']['metrics']['reference_glyphs'] or c['score']['metrics']['association_denominator']<c['score']['metrics']['visible_count_denominator']) for c in cases),
        association_unscorable_images=sum(c['score']['association']['status']!='unique' for c in cases))
    return out
