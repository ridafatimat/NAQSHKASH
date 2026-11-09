"""Optional T1 adapter; the original validator and F1 v2 are called unchanged."""
import copy
import math
from runtime import *
from tier_aware_decoder import interpret


def role_evidence(observation, row_index, role, prediction):
    reasons = []
    if (observation.get('preprocessing_status') != 'ok' or not observation.get('segmentation_accepted')
            or observation.get('ocr_status') != 'ok'):
        reasons.append('D1_OR_OCR_NOT_READY')
    diag = observation.get('diagnostics', {})
    alternatives = diag.get('legal_alternatives')
    margin = diag.get('role_margin')
    if alternatives == 1:
        pass  # D1 accepts a sole legal path without a measured margin
    elif (type(alternatives) is not int or alternatives < 2 or type(margin) not in (int, float)
          or not math.isfinite(margin) or margin < load_json(D / 'configs/D1.json')['margin']):
        reasons.append('D1_ROLE_CONFIDENCE_NOT_SUPPORTED')
    if diag.get('rejection'):
        reasons.append('D1_REJECTION_PRESENT')
    rows = observation.get('segmentation_rows', [])
    if row_index >= len(rows):
        matches = []
    else:
        matches = [t for t in rows[row_index] if t.get('row') == row_index and t.get('role') == role]
    tiers = [t for t in observation.get('tiers', []) if t.get('row') == row_index and t.get('role') == role]
    if len(matches) != 1 or len(tiers) != 1 or tiers[0].get('prediction') != prediction:
        reasons.append('D1_ROW_TIER_METADATA_CONFLICT')
    elif tiers[0].get('box') != matches[0].get('box'):
        reasons.append('D1_CROP_METADATA_CONFLICT')
    path = diag.get('path', [])
    if row_index >= len(path) or role not in path[row_index][1]:
        reasons.append('D1_PATH_ROLE_CONFLICT')
    return dict(reliable=not reasons, role=role, row=row_index, source='D1_saved_metadata',
                reason_codes=reasons, legal_alternatives=alternatives, role_margin=margin,
                confidence_rule='existing frozen D1 acceptance criterion; no new threshold')


def source_rendering_context(record):
    """Read rendering metadata only; never pass annotations/labels to interpretation.

    Unknown uploads/origins do not inherit the synthetic font profile. For this
    saved-corpus experiment the source JSON rendering block confirms original
    source size/mode. logical_rows and ground-truth symbols are never used here.
    """
    if record.get('data_origin') not in ('positive_clean', 'positive_scanned', 'normal_clean'):
        return {}
    try:
        block = json.loads(text_asset(record['json_path'])).get('rendering', {})
    except (OSError, KeyError, ValueError, TypeError):
        return {}
    profile = load_json(WORKSPACE / 'configs/rendering_profile.json')
    if (block.get('font_size_px') != profile['font_size_px'] or block.get('image_mode') != profile['source_mode']
            or block.get('font_family') != profile['font_family'] or block.get('font_file') != 'Talamat.ttf'):
        return {}
    return dict(profile, source='original_canonical_rendering_block', transformed_variant=record['data_origin'].endswith('_scanned'))


def process(observation, config, mapping, rendering_context):
    raw = copy.deepcopy(observation)
    raw_decision = decide(raw, 'F1')
    if not config['enabled']:
        return dict(raw_observation=raw, interpreted_observation=copy.deepcopy(raw), raw_f1_decision=raw_decision,
            interpreted_f1_decision=copy.deepcopy(raw_decision), final_decision=copy.deepcopy(raw_decision),
            interpretations=[], corrections=[], uncertain=False, syntax_failure_resolved=False,
            original_reason_codes=raw_decision['reason_codes'], original_evidence=raw_decision)
    interpreted = copy.deepcopy(raw)
    interpretations, edits = [], []
    uncertain = False
    for ri, row in enumerate(raw.get('decoded_rows', [])):
        for role, prediction in row.items():
            evidence = role_evidence(raw, ri, role, prediction)
            tier = next((t for t in raw.get('tiers', []) if t.get('row') == ri and t.get('role') == role), {})
            metadata = dict(row=ri, box=tier.get('box'), rendering_context=rendering_context)
            if raw.get('conflicting_evidence'):
                metadata['conflicting_evidence'] = raw['conflicting_evidence']
            interpretation = interpret(prediction, role, metadata, mapping, evidence)
            interpretations.append(interpretation)
            uncertain |= interpretation['uncertain']
            interpreted['decoded_rows'][ri][role] = interpretation['interpreted_prediction']
            for t in interpreted.get('tiers', []):
                if t.get('row') == ri and t.get('role') == role:
                    t['prediction'] = interpretation['interpreted_prediction']
            edits.extend(dict(edit, row=ri) for edit in interpretation['corrections'])
    interpreted_decision = decide(interpreted, 'F1')
    final = copy.deepcopy(interpreted_decision)
    # An experimental context guard wraps the unchanged F1 decision. It never
    # overrides independent invalidity or removes raw F1 reasons/evidence.
    if uncertain and final['outcome'] == 'ACCEPT':
        final['outcome'] = 'REVIEW_UNCERTAIN'
        final['reason_codes'] = final['reason_codes'] + ['T1_CONTEXT_UNCERTAIN']
        final['reason_details'] = final.get('reason_details', []) + [dict(reason_code='T1_CONTEXT_UNCERTAIN',
            source='tier_aware_interpretation_guard', evidence_scope='contextual_interpretation', based_on='unresolved_runtime_context')]
    raw_syntax = validate_tiers(raw.get('decoded_rows', []))
    new_syntax = validate_tiers(interpreted.get('decoded_rows', []))
    return dict(raw_observation=raw, interpreted_observation=interpreted, raw_f1_decision=raw_decision,
        interpreted_f1_decision=interpreted_decision, final_decision=final, interpretations=interpretations,
        corrections=edits, uncertain=uncertain,
        syntax_failure_resolved=raw_syntax['outcome'] != 'ACCEPT' and new_syntax['outcome'] == 'ACCEPT',
        original_reason_codes=raw_decision['reason_codes'], original_evidence=raw_decision)
