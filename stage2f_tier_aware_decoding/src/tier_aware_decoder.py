"""Conservative, role-aware interpretation of saved greedy OCR characters."""
from runtime import allowed_characters


def interpret(raw_prediction, tier_role, ocr_metadata, verified_mapping, role_evidence=None, enabled=True):
    if not isinstance(raw_prediction, str):
        raise TypeError('OCR prediction must be a string')
    role_evidence = role_evidence or {}
    rendering = ocr_metadata.get('rendering_context', {})
    result = dict(raw_prediction=raw_prediction, interpreted_prediction=raw_prediction, corrected=False,
        corrections=[], proposed_corrections=[], tier_role=tier_role, uncertain=False, reason_codes=[],
        evidence=dict(role_evidence=role_evidence, rendering_context=rendering,
                      row=ocr_metadata.get('row'), box=ocr_metadata.get('box'), based_on='saved_OCR + D1_role + verified_font_equivalence'))
    if not enabled:
        result['reason_codes'] = ['T1_DISABLED']
        return result
    def uncertain(code):
        result['uncertain'] = True
        if code not in result['reason_codes']:
            result['reason_codes'].append(code)
    allowed = allowed_characters(tier_role)
    if not allowed:
        uncertain('T1_MISSING_OR_UNKNOWN_ROLE')
        return result
    if ocr_metadata.get('conflicting_evidence') or role_evidence.get('conflicting_evidence'):
        uncertain('T1_CONFLICTING_EVIDENCE')
        return result
    if role_evidence.get('reliable') is not True or role_evidence.get('role') != tier_role:
        uncertain('T1_ROLE_NOT_RELIABLY_ESTABLISHED')
        return result
    for index, character in enumerate(raw_prediction):
        if character in allowed:
            continue  # never change an already role-legal character
        candidates = []
        for row in verified_mapping:
            if (row['source'] == character and row['target_role'] == tier_role
                    and row['target'] in allowed and len(row['target']) == 1
                    and row['status'] == 'verified_exact_in_scope'
                    and row['font_sha256'] == rendering.get('font_sha256')
                    and rendering.get('font_size_px') in row['verified_sizes']
                    and row['source_mode'] == rendering.get('source_mode')
                    and row['anchor'] == rendering.get('anchor')
                    and rendering.get('provenance') == 'canonical_source_rendering_metadata'):
                candidates.append(row)
        targets = sorted({r['target'] for r in candidates})
        if not targets:
            uncertain('T1_NO_VERIFIED_ROLE_EQUIVALENT')
        elif len(targets) > 1:
            uncertain('T1_MULTIPLE_ROLE_EQUIVALENTS')
        else:
            result['proposed_corrections'].append(dict(index=index, original_character=character,
                replacement_character=targets[0], tier_role=tier_role, reason_code='T1_VERIFIED_GLYPH_ROLE_DISAMBIGUATION',
                evidence=candidates))
    # All-or-nothing within a tier: do not conceal an unresolved violation by
    # partially changing the string. Internal spaces/length are never altered.
    if result['uncertain']:
        return result
    chars = list(raw_prediction)
    for edit in result['proposed_corrections']:
        chars[edit['index']] = edit['replacement_character']
    interpreted = ''.join(chars)
    if set(interpreted) - allowed:
        uncertain('T1_RESULT_NOT_ROLE_LEGAL')
        return result
    result.update(interpreted_prediction=interpreted, corrected=interpreted != raw_prediction,
                  corrections=result['proposed_corrections'])
    result['reason_codes'] = ['T1_VERIFIED_GLYPH_ROLE_DISAMBIGUATION' if result['corrected'] else 'T1_NO_CHANGE_NEEDED']
    return result
