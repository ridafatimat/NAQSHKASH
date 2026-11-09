"""R2: a conservative count-gap guard, using decoded text only.

No labels, source pixels, font/grid metadata, pairing, or new thresholds.
Canonical count encodings are contiguous; OCR gaps are suspicion, not proof.
"""
from copy import deepcopy


def count_gap_findings(decoded_rows):
    findings=[]
    for i,row in enumerate(decoded_rows):
        text=row.get('count') if isinstance(row,dict) else None
        if isinstance(text,str) and ' ' in text.strip(' '):
            findings.append(dict(reason_code='R2_INTERNAL_DECODED_COUNT_GAP',row=i,tier='count',
                predicted_text=text,based_on='ocr_output',evidence_strength='OCR_only_uncertainty',
                supported_outcome='REVIEW_UNCERTAIN',direct_image_evidence=False,
                reliable_character_alignment=False,high_confidence_OCR_claim=False,
                independent_consistency_checks=False,
                explanation='Canonical encode_count concatenations have no internal blank cells; an OCR gap needs review. No missing count is inferred.'))
    return findings


def structural_decision(base_decision,decoded_rows):
    """Inputs are an existing R1 decision and interpreted OCR rows only."""
    decision=deepcopy(base_decision);findings=count_gap_findings(decoded_rows)
    if findings:
        if decision['outcome']=='ACCEPT':decision['outcome']='REVIEW_UNCERTAIN'
        decision['reason_codes']=list(dict.fromkeys(decision['reason_codes']+['R2_INTERNAL_DECODED_COUNT_GAP']))
        decision['reason_details']=decision.get('reason_details',[])+findings
        if base_decision['outcome']=='ACCEPT':
            decision['evidence_scope']='OCR_only_structural_uncertainty'
            decision['explanation']='Internal gaps in a decoded count line require review; source-image invalidity is unproven.'
    return decision,findings
