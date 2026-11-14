"""Label-blind F0/F1/F2 decisions, reusable with recorded pixel/OCR outputs."""
from confidence import check
from talim_validator import validate_tiers
from common import ROOT, load_json


DEFAULT_DECISION_POLICY = ROOT / 'configs/f1_ocr_uncertainty_policy_v2.yaml'


def load_decision_policy(path=DEFAULT_DECISION_POLICY):
    configuration = load_json(path)
    if (configuration.get('policy_version') != 'f1_ocr_uncertainty_v2'
            or configuration.get('ocr_only_violation_action') != 'REVIEW_UNCERTAIN'
            or configuration.get('independent_invalid_preprocessing_statuses') != ['unreadable', 'no_ink']):
        raise ValueError('Unsupported decision policy; canonical/image rules must not be changed by configuration')
    return configuration


def _f0_decision(observation):
    """Historical F0 actions and output schema, unchanged."""
    state = observation['preprocessing_status']
    if state in ('unreadable', 'no_ink'):
        return dict(outcome='REJECT_INVALID', reason_codes=['INPUT_' + state.upper()], evidence_scope='image_artifact')
    if state != 'ok' or not observation['segmentation_accepted']:
        return dict(outcome='REVIEW_UNCERTAIN', reason_codes=['D1_NO_SAFE_PROPOSAL'], evidence_scope='segmentation')
    if observation.get('ocr_status') != 'ok':
        return dict(outcome='REVIEW_UNCERTAIN', reason_codes=['OCR_UNAVAILABLE'], evidence_scope='recognition')
    return dict(outcome='ACCEPT', reason_codes=['D1_PROPOSAL_E1_OUTPUT'], evidence_scope='baseline')


def decide(observation, candidate='F1', policy=None, decision_policy=None):
    """Use runtime evidence only, never evaluation labels or claimed invalidity.

    Independent invalidity is limited to the existing pixel adapter's confirmed
    unreadable/no-ink statuses. A caller-supplied generic 'image_invalid' flag or
    source annotation is not an independent image check and is never consulted.
    """
    if candidate not in ('F0', 'F1', 'F2'):
        raise ValueError('Implemented candidates: F0, F1, F2; no trained F3 exists')
    if candidate == 'F0':
        return _f0_decision(observation)
    configuration = decision_policy if decision_policy is not None else load_decision_policy()
    # Validate supplied dictionaries using the same fixed v2 contract as files.
    if (configuration.get('policy_version') != 'f1_ocr_uncertainty_v2'
            or configuration.get('ocr_only_violation_action') != 'REVIEW_UNCERTAIN'
            or configuration.get('independent_invalid_preprocessing_statuses') != ['unreadable', 'no_ink']):
        raise ValueError('Unsupported decision policy')
    findings = []

    def finding(code, source, scope, based_on, supported_outcome, evidence=None):
        evidence = evidence or {}
        text = None
        if based_on == 'ocr_output' and type(evidence.get('row')) is int and evidence.get('role'):
            rows = observation.get('decoded_rows', [])
            if 0 <= evidence['row'] < len(rows) and isinstance(rows[evidence['row']], dict):
                text = rows[evidence['row']].get(evidence['role'])
        findings.append(dict(reason_code=code, source=source, evidence_scope=scope,
            based_on=based_on, supported_outcome=supported_outcome, evidence=evidence,
            row=evidence.get('row'), tier=evidence.get('role'), predicted_text=text,
            predicted_characters=evidence.get('glyphs', sorted(set(text) - {' '}) if isinstance(text, str) else [])))

    state = observation['preprocessing_status']
    if state in configuration['independent_invalid_preprocessing_statuses']:
        finding('INPUT_' + state.upper(), 'pixel_input_adapter', 'image_artifact',
                'independent_image_check', 'REJECT_INVALID', dict(preprocessing_status=state))
    elif state != 'ok' or not observation['segmentation_accepted']:
        finding('D1_NO_SAFE_PROPOSAL', 'D1_segmentation', 'segmentation', 'image_processing_uncertainty',
                'REVIEW_UNCERTAIN', dict(preprocessing_status=state, reason=observation.get('segmentation_reason', '')))
    if observation.get('ocr_status') != 'ok' and state == 'ok' and observation['segmentation_accepted']:
        finding('OCR_UNAVAILABLE', 'E1_recognizer', 'recognition', 'recognition_status', 'REVIEW_UNCERTAIN')

    # Retain available OCR violations even when stronger image evidence or D1
    # uncertainty decides the action. No OCR is executed here.
    validation = None
    if observation.get('ocr_status') == 'ok' or observation.get('decoded_rows'):
        validation = validate_tiers(observation.get('decoded_rows', []))
        if validation['outcome'] != 'ACCEPT':
            for code in validation['reason_codes']:
                finding(code, 'E1_decoded_text/talim_validator', validation['scope'], 'ocr_output',
                        configuration['ocr_only_violation_action'], validation['evidence'])

    ready = state == 'ok' and observation['segmentation_accepted'] and observation.get('ocr_status') == 'ok'
    if candidate == 'F2' and ready:
        if policy is None:
            if validation is not None and validation['outcome'] == 'ACCEPT':
                raise ValueError('F2 requires an explicit policy')
        else:
            for code in check(observation.get('confidence', {}), policy):
                finding(code, 'confidence_policy', 'heuristic_quality', 'model_or_segmentation_heuristic', 'REVIEW_UNCERTAIN')

    invalid = [f for f in findings if f['supported_outcome'] == 'REJECT_INVALID']
    uncertain = [f for f in findings if f['supported_outcome'] == 'REVIEW_UNCERTAIN']
    outcome = 'REJECT_INVALID' if invalid else 'REVIEW_UNCERTAIN' if uncertain else 'ACCEPT'
    primary = (invalid or uncertain or [None])[0]
    explanation = ('The decoded transcription violates Talim syntax; this OCR-only evidence does not independently prove the source image invalid.'
                   if validation is not None and validation['outcome'] != 'ACCEPT' else
                   'The decision uses the existing image, segmentation, recognition and optional quality checks.')
    if invalid:
        explanation = 'Existing independent input checks confirmed an invalid image artifact; any OCR-only findings are retained separately.'
    result = dict(outcome=outcome, reason_codes=list(dict.fromkeys(f['reason_code'] for f in findings)) or ['OBSERVABLE_CHECKS_PASSED'],
                  evidence_scope=primary['evidence_scope'] if primary else 'syntax_and_optional_quality; image_validity_not_proven',
                  reason_details=findings, independent_image_invalidity_confirmed=bool(invalid),
                  decision_policy_version=configuration['policy_version'], explanation=explanation)
    if validation is not None:
        result['syntax_validation'] = validation
        if validation['outcome'] != 'ACCEPT':
            # Preserve the historical validator fields as well as the new provenance.
            result['scope'] = validation['scope']
            result['evidence'] = validation['evidence']
    return result
