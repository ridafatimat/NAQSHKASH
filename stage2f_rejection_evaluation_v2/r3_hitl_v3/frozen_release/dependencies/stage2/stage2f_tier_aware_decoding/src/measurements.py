"""Evaluation-only edit accounting; truth never enters contextual_pipeline."""
from collections import Counter


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j-1] + 1, previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def tier_map(rows):
    return {(i, role): text for i, row in enumerate(rows or []) for role, text in row.items()}


def page_errors(truth, prediction):
    if truth is None:
        return None
    gt, pred = tier_map(truth), tier_map(prediction)
    return dict(edits=sum(distance(gt.get(k, ''), pred.get(k, '')) for k in gt.keys() | pred.keys()),
                characters=sum(len(s) for s in gt.values()), exact=truth == prediction)


def alignment_targets(raw, truth, index):
    """All possible truth characters for a raw position over ALL minimum edits.

    None marks a deletion. Ambiguous edit alignments are not scored as proven
    correct/incorrect contextual edits. This avoids choosing a convenient tie.
    """
    n, m = len(raw), len(truth)
    forward = [[0] * (m+1) for _ in range(n+1)]
    backward = [[0] * (m+1) for _ in range(n+1)]
    for i in range(n+1): forward[i][0] = i
    for j in range(m+1): forward[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            forward[i][j] = min(forward[i-1][j]+1, forward[i][j-1]+1, forward[i-1][j-1]+(raw[i-1] != truth[j-1]))
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i == n: backward[i][j] = m-j
            elif j == m: backward[i][j] = n-i
            else: backward[i][j] = min(backward[i+1][j]+1, backward[i][j+1]+1, backward[i+1][j+1]+(raw[i] != truth[j]))
    total = forward[n][m]
    possible = set()
    for j in range(m+1):
        if forward[index][j] + 1 + backward[index+1][j] == total:
            possible.add(None)
        if j < m and forward[index][j] + (raw[index] != truth[j]) + backward[index+1][j+1] == total:
            possible.add(truth[j])
    return possible


def assess_corrections(truth_rows, raw_rows, corrections):
    truth, raw = tier_map(truth_rows), tier_map(raw_rows)
    assessed = []
    for edit in corrections:
        key = (edit['row'], edit['tier_role'])
        targets = alignment_targets(raw[key], truth[key], edit['index']) if key in truth else {None}
        label = 'uncertain_alignment'
        if len(targets) == 1 and None not in targets:
            label = 'correct' if edit['replacement_character'] in targets else 'incorrect'
        assessed.append(dict(edit, assessment=label, possible_truth_characters=sorted(c for c in targets if c is not None),
                             deletion_possible=None in targets))
    return assessed


def ratio(n, d): return n/d if d else None


def summarize(rows):
    scorable = [r for r in rows if r['interpreted_errors'] is not None and r['interpreted_errors']['characters']]
    accepted = [r for r in scorable if r['decision']['outcome'] == 'ACCEPT']
    assessments = [e['assessment'] for r in rows for e in r['correction_assessments']]
    denominator = sum(r['interpreted_errors']['characters'] for r in scorable)
    return dict(images=len(rows), scorable_images=len(scorable), characters=denominator,
        outcome_counts=dict(Counter(r['decision']['outcome'] for r in rows)),
        exact_transcription_accuracy=ratio(sum(r['interpreted_errors']['exact'] for r in scorable), len(scorable)),
        raw_CER=ratio(sum(r['raw_errors']['edits'] for r in scorable), denominator),
        interpreted_CER=ratio(sum(r['interpreted_errors']['edits'] for r in scorable), denominator),
        review_rate=ratio(sum(r['decision']['outcome'] == 'REVIEW_UNCERTAIN' for r in rows), len(rows)),
        wrong_tier_reviews=sum(r['decision']['outcome'] == 'REVIEW_UNCERTAIN' and 'DECODED_UNSUPPORTED_OR_WRONG_TIER_GLYPH' in r['decision']['reason_codes'] for r in rows),
        accepted_output_correctness=ratio(sum(r['interpreted_errors']['exact'] for r in accepted), len(accepted)),
        accepted_output_denominator=len(accepted),
        end_to_end_page_correctness=ratio(sum(r['decision']['outcome'] == 'ACCEPT' and r['interpreted_errors']['exact'] for r in scorable), len(scorable)),
        corrected_images=sum(bool(r['corrections']) for r in rows), contextual_corrections=sum(len(r['corrections']) for r in rows),
        correct_corrections=assessments.count('correct'), incorrect_corrections=assessments.count('incorrect'),
        unscorable_corrections=assessments.count('uncertain_alignment'),
        syntax_failures_resolved=sum(r['syntax_failure_resolved'] for r in rows),
        worsened_CER_images=sum(r['interpreted_errors']['edits'] > r['raw_errors']['edits'] for r in scorable))


def grouped(rows):
    parents = {p: summarize([r for r in rows if r['source_parent_id'] == p]) for p in sorted({r['source_parent_id'] for r in rows})}
    cohorts = {c: summarize([r for r in rows if c in r['cohorts']]) for c in sorted({c for r in rows for c in r['cohorts']})}
    summary = summarize(rows)
    summary['independent_parents'] = len(parents)
    summary['macro_parent'] = {key: ratio(sum(values), len(values)) for key in
        ('exact_transcription_accuracy', 'interpreted_CER', 'review_rate', 'end_to_end_page_correctness')
        for values in [[p[key] for p in parents.values() if p[key] is not None]]}
    return summary, parents, cohorts
