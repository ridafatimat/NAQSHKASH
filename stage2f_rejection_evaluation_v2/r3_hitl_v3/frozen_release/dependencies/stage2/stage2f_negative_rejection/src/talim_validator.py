"""Canonical rules, with explicit separation of annotation and OCR evidence."""
from common import TOOLS, load_module
from talim import talim_rules as rules


def result(outcome, code, evidence=None, scope='decoded_syntax'):
    return dict(outcome=outcome, reason_codes=[code], evidence=evidence or {}, scope=scope)


def validate_annotation(sample):
    """Offline annotation audit only. Never feed source annotations to image policy."""
    # The source validator imports talim_rules by its original top-level name.
    import sys
    sys.modules.setdefault('talim_rules', load_module('stage2f_source_rules', TOOLS / 'talim_rules.py'))
    canonical = load_module('stage2f_canonical_validator', TOOLS / 'validate_datasets.py')
    errors = canonical.check_sample(sample)
    return result('REJECT_INVALID' if errors else 'ACCEPT',
                  'ANNOTATION_CANONICAL_ERRORS' if errors else 'ANNOTATION_CANONICAL_OK',
                  dict(errors=errors), scope='annotation_only')


def validate_tiers(rows):
    """Check only observable tier grammar; no guessed horizontal alignment.

    D1 crops and E1 labels strip each tier's leading blank cells. Pixel columns
    cannot therefore be reconstructed from character indices across tiers.
    Count tokens without preserved run boundaries cannot safely be parsed into
    counts: a concatenation may admit multiple 1..259 tokenizations.
    """
    if not rows:
        return result('REVIEW_UNCERTAIN', 'NO_DECODED_ROWS')
    evidence = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or 'count' not in row:
            return result('REVIEW_UNCERTAIN', 'MISSING_COUNT_TIER', dict(row=i))
        if set(row) - {'upper', 'count', 'lower'}:
            return result('REVIEW_UNCERTAIN', 'UNSUPPORTED_TIER_LAYOUT', dict(row=i))
        for role, text in row.items():
            if not isinstance(text, str) or not text.strip():
                return result('REVIEW_UNCERTAIN', 'EMPTY_DECODED_TIER', dict(row=i, role=role))
            symbols = set(text) - {' '}
            if symbols & rules.AMBIGUOUS_SYMBOLS:
                return result('REVIEW_UNCERTAIN', 'AMBIGUOUS_NOTATION', dict(row=i, role=role))
            allowed = (set(''.join(rules.encode_count(n) for n in range(1, 260)))
                       if role == 'count' else set(rules.EXCLUSIVE_UP_SYMBOLS if role == 'upper' else rules.EXCLUSIVE_DOWN_SYMBOLS))
            if symbols - allowed:
                return result('REJECT_INVALID', 'DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH',
                              dict(row=i, role=role, glyphs=sorted(symbols - allowed)))
        if not any(row.get(k, '').strip() for k in ('upper', 'lower')):
            return result('REVIEW_UNCERTAIN', 'COUNT_WITHOUT_SYMBOL_TIER', dict(row=i))
        evidence.append(dict(row=i, roles=list(row), alignment_verified=False))
    return result('ACCEPT', 'OBSERVABLE_SYNTAX_OK', dict(rows=evidence, image_validity_proven=False))


def validate_aligned_row(upper, count_line, lower, expected_cells=None):
    """Strict check for explicitly aligned canonical text, NOT stripped OCR tiers.

    Count boundaries are supplied by symbol positions. Reuses canonical_row
    and encode_count. A failure is a syntax claim, never image ground truth.
    """
    if not all(isinstance(v, str) for v in (upper, count_line, lower)):
        return result('REVIEW_UNCERTAIN', 'MALFORMED_ALIGNED_TEXT')
    positions = sorted({i for t in (upper, lower) for i, c in enumerate(t) if c != ' '})
    if not positions or positions[0] != 0:
        return result('REJECT_INVALID', 'MISSING_ALIGNED_SYMBOL')
    reverse = {rules.encode_count(n): n for n in range(1, 260)}
    runs = []
    for j, start in enumerate(positions):
        end = positions[j + 1] if j + 1 < len(positions) else len(count_line)
        up = upper[start:start + 1].strip()
        down = lower[start:start + 1].strip()
        if bool(up) == bool(down):
            return result('REJECT_INVALID', 'SYMBOL_POSITION_CONTRADICTION', dict(column=start))
        symbol = up or down
        if symbol in rules.AMBIGUOUS_SYMBOLS:
            return result('REVIEW_UNCERTAIN', 'AMBIGUOUS_NOTATION')
        if not rules.is_valid_symbol_direction(symbol, 'UP' if up else 'DOWN'):
            return result('REJECT_INVALID', 'WRONG_SYMBOL_DIRECTION')
        token = count_line[start:end].lstrip(' ')
        if token not in reverse:
            return result('REJECT_INVALID', 'NONCANONICAL_COUNT_TOKEN', dict(token=token))
        if runs and runs[-1]['symbol'] == symbol:
            return result('REJECT_INVALID', 'ADJACENT_IDENTICAL_RUNS')
        runs.append(dict(symbol=symbol, count=reverse[token]))
    if rules.canonical_row(runs) != (upper, count_line, lower):
        return result('REJECT_INVALID', 'NONCANONICAL_ALIGNMENT')
    if expected_cells is not None and sum(r['count'] for r in runs) != expected_cells:
        return result('REJECT_INVALID', 'ROW_SUM_MISMATCH')
    return result('ACCEPT', 'ALIGNED_CANONICAL_OK', dict(runs=runs), scope='aligned_text')
