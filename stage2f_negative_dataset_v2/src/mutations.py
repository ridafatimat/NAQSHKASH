"""Changes raw visible notation. Never edits JSON alone or repairs the display."""
import copy
from common_v2 import *
from independent_verify import role_candidates


def run_locations(sample):
    for ri,row in enumerate(sample['logical_rows']):
        col=0
        for j,run in enumerate(row['runs']):
            token=rules.encode_count(run['count'])
            yield ri,j,col,run,token
            col+=len(token)


def mutate(sample,category,layout,selector=0):
    if category not in RULES:raise ValueError('Category is not verified for image-invalid generation')
    display=displayed_rows(sample)
    candidates=[]
    all_tokens={rules.encode_count(n) for n in range(1,260)}
    for ri,j,col,run,token in run_locations(sample):
        if category=='wrong_tier':
            source='upper' if run['direction']=='UP' else 'lower'
            target='lower' if source=='upper' else 'upper'
            # B/X (and any other real raster alias) cannot enter this category.
            if not role_candidates(run['symbol'],target,layout):
                candidates.append((ri,j,col,run,token,source,target))
        elif category=='invalid_count_token' and len(token)==2 and token[::-1] not in all_tokens:
            candidates.append((ri,j,col,run,token,'count_line','count_line'))
        elif category=='missing_count':
            candidates.append((ri,j,col,run,token,'count_line','count_line'))
    if not candidates:raise ValueError('No observable verified mutation available for this parent')
    ri,j,col,run,token,source,target=candidates[selector%len(candidates)]
    raw=[{k:list(v.ljust(layout['columns'])) for k,v in row.items()} for row in display]
    if category=='wrong_tier':
        raw[ri][source][col]=' ';raw[ri][target][col]=run['symbol']
    elif category=='invalid_count_token':raw[ri]['count_line'][col:col+2]=list(token[::-1])
    else:raw[ri]['count_line'][col:col+len(token)]=[' ']*len(token)
    mutated=[{k:''.join(v) for k,v in row.items()} for row in raw]
    return mutated,dict(category=category,rule_id=RULES[category],row=ri,run=j,column=col,width=len(token),
        original_symbol=run['symbol'],original_count=run['count'],original_token=token,
        displayed_mutated_token=token[::-1] if category=='invalid_count_token' else '' if category=='missing_count' else token,
        source_tier=source,target_tier=target,mutation_changes='raw rendered glyph placement/content, not an annotation-only change')
