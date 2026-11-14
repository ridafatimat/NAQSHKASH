"""Optional heuristic quality signals; none are calibrated validity probabilities."""
import math


def from_logits(logits, blank=0):
    """One [T,C] tensor. Sequence score sums ALL CTC paths for its greedy target."""
    import torch
    from data.ctc_decode import collapse_ctc_sequence
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite nonempty [T,C] logits')
    lp = logits.float().log_softmax(-1)
    top = lp.topk(2, dim=-1).values
    ids = lp.argmax(-1)
    target = collapse_ctc_sequence(ids.tolist(), blank)
    selected = ids != blank
    loss = torch.nn.functional.ctc_loss(lp[:, None, :], torch.tensor(target, dtype=torch.long, device=lp.device),
        torch.tensor([len(lp)]), torch.tensor([len(target)]), blank=blank, reduction='sum', zero_infinity=False)
    return dict(ctc_log_sequence_score=-float(loss), ctc_log_score_per_frame=-float(loss) / len(lp),
                greedy_log_score_per_frame=float(top[:, 0].mean()),
                nonblank_geomean=math.exp(float(top[selected, 0].mean())) if selected.any() else None,
                mean_top2_log_margin=float((top[:, 0] - top[:, 1]).mean()),
                blank_fraction=float((~selected).float().mean()), frames=len(lp),
                decoded_tokens=len(target), calibrated_probability=False)


def page_signals(tiers, diagnostics):
    def minimum(key):
        values = [t['signals'][key] for t in tiers if t['signals'].get(key) is not None]
        return min(values) if values else None
    scores = diagnostics.get('heuristic_scores', [])
    return dict(nonblank_geomean=minimum('nonblank_geomean'),
                ctc_log_score_per_frame=minimum('ctc_log_score_per_frame'),
                mean_top2_log_margin=minimum('mean_top2_log_margin'),
                role_margin=diagnostics.get('role_margin'),
                legal_alternatives=diagnostics.get('legal_alternatives'),
                component_preservation='D1 unassigned-foreground check only; no independent component oracle',
                row_consistency='observable tier grammar only', segmentation_score=scores[0] if scores else None)


def validate_policy(policy):
    if policy.get('schema_version') != 1 or policy.get('action') != 'REVIEW_UNCERTAIN':
        raise ValueError('Policy must use schema_version=1 and REVIEW_UNCERTAIN action')
    allowed = {'nonblank_geomean', 'ctc_log_score_per_frame', 'mean_top2_log_margin', 'role_margin'}
    if set(policy.get('minimum', {})) - allowed:
        raise ValueError('Unsupported confidence signal')
    for key, value in policy.get('minimum', {}).items():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('Non-finite or malformed threshold: ' + key)
    if policy.get('enabled') and not any(v is not None for v in policy.get('minimum', {}).values()):
        raise ValueError('Enabled policy needs at least one threshold')
    return policy


def check(signals, policy):
    validate_policy(policy)
    if not policy.get('enabled'):
        return []
    reasons = []
    for key, threshold in policy['minimum'].items():
        if threshold is None:
            continue
        value = signals.get(key)
        if value is None or not math.isfinite(float(value)):
            reasons.append('CONFIDENCE_MISSING_' + key.upper())
        elif float(value) < threshold:
            reasons.append('CONFIDENCE_LOW_' + key.upper())
    return reasons
