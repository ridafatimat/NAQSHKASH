"""Two segmentation modes. Blind functions accept pixels/derived geometry only."""
import numpy as np
from common import PROJECT
from preprocessing.pipeline import crop_and_separate_rows
from preprocessing.rows import separate_rows
from data.build_full_tier_dataset import make_ink_mask,find_active_runs,merge_close_runs,detect_tier_bands
from segmentation import whitespace_row_boxes

class SegmentationFailure(ValueError):pass


def preprocess(image):
    return crop_and_separate_rows(image,apply_deskew=True)


def blind_tiers(image):
    mask=make_ink_mask(image)
    h,w=mask.shape
    active=mask.sum(axis=1)>=max(1,round(w*.002))
    bands=merge_close_runs(find_active_runs(active),max_gap=max(2,round(h*.02)))
    if not 1<=len(bands)<=3:raise SegmentationFailure('blind_ambiguous_band_count:'+str(len(bands)))
    masses=[int(mask[a:b].sum()) for a,b in bands]
    heights=[b-a for a,b in bands]
    if len(bands)==3:names=['upper','count','lower']
    elif len(bands)==1:names=['count']
    else:
        tall=int(np.argmax(heights));heavy=int(np.argmax(masses))
        if max(heights)>=1.2*min(heights):count_index=tall
        elif max(masses)>=1.15*min(masses):count_index=heavy
        else:raise SegmentationFailure('blind_two_tier_role_ambiguous')
        names=['count','lower'] if count_index==0 else ['upper','count']
    padded=[(max(0,a-4),min(h,b+4)) for a,b in bands]
    if any(a[1]>b[0] for a,b in zip(padded,padded[1:])):raise SegmentationFailure('blind_overlapping_padded_bands')
    return [(name,image[a:b,:]) for name,(a,b) in zip(names,padded)]


def blind_segment(processed):
    # No ground truth, expected structural counts, IDs or predictions enter here.
    return [blind_tiers(row.image) for row in processed.rows]


def assisted_segment(processed,logical_rows):
    rows=processed.rows
    if len(rows)!=len(logical_rows):
        boxes,_=whitespace_row_boxes(processed.cropped_image,len(logical_rows))
        rows=separate_rows(processed.cropped_image,row_bboxes=boxes)
    proposed=[]
    for row,truth in zip(rows,logical_rows):
        names=[name for name,key in [('upper','upper'),('count','count_line'),('lower','lower')] if truth[key].strip()]
        bands=detect_tier_bands(row.image,len(names))
        if bands is None or len(bands)!=len(names):raise SegmentationFailure('assisted_tier_band_mismatch')
        if any(a[1]>b[0] for a,b in zip(bands,bands[1:])):raise SegmentationFailure('assisted_overlapping_padded_bands')
        proposed.append([(name,row.image[a:b,:]) for name,(a,b) in zip(names,bands)])
    return proposed


def align_after_segmentation(proposed,logical_rows,vocabulary):
    # Ground truth is consulted ONLY AFTER blind proposals are final. A mismatch
    # rejects the page; it never changes/retries/ranks the blind segmentation.
    if len(proposed)!=len(logical_rows):raise SegmentationFailure('alignment_row_count_mismatch')
    result=[]
    for ri,(tiers,truth) in enumerate(zip(proposed,logical_rows)):
        expected=[(name,truth[key]) for name,key in [('upper','upper'),('count','count_line'),('lower','lower')] if truth[key].strip()]
        if [n for n,_ in tiers]!=[n for n,_ in expected]:raise SegmentationFailure('alignment_tier_presence_or_role_mismatch')
        for (name,image),(_,raw) in zip(tiers,expected):
            text=raw.lstrip(' ');vocabulary.encode(text)
            h,w=image.shape[:2];width=max(1,round(w*64/h))
            required=len(text)+sum(a==b for a,b in zip(text,text[1:]))
            if not text or not image.size:raise SegmentationFailure('alignment_empty_tier')
            if width<4 or required>width//4:raise SegmentationFailure('ctc_infeasible')
            result.append(dict(row_index=ri,tier_name=name,truth=text,raw_label=raw,leading_offset=len(raw)-len(text),image=image,resized_width=width))
    return result
