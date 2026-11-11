"""Manual T0/T1 evaluation from frozen DEVELOPMENT predictions. No inference."""
import argparse
from runtime import *
from contextual_pipeline import process, source_rendering_context
from measurements import page_errors, assess_corrections, grouped


def load_saved(directory):
    if Path(directory).resolve() != BASELINE.resolve():
        raise ValueError('This protocol permits only the frozen 776-image DEVELOPMENT baseline; exposed evaluation is not supported')
    from evaluate_rejection import load_baseline
    items, lineage = load_baseline(directory)
    if lineage['manifest_sha256'] != digest(LEGACY / 'manifests/development_manifest.csv'):
        raise ValueError('Wrong development manifest lineage')
    if len(items) != 776 or any(i['record']['partition'] != 'development' for i in items):
        raise ValueError('Incomplete/non-development records')
    return items, lineage


def evaluate_item(item, config, mapping, rendering_context):
    # Runtime interpretation receives NO record annotations or truth.
    output = process(item['observation'], config, mapping, rendering_context)
    record, truth = item['record'], item['truth_rows']
    raw = output['raw_observation']['decoded_rows']
    interpreted = output['interpreted_observation']['decoded_rows']
    cohorts = [record['data_origin']]
    if truth is not None:
        if max((len(s) for row in truth for s in row.values()), default=0) >= 65:
            cohorts.append('long_tier_ge65_descriptive_only')
        if len(truth) > 1:
            cohorts.append('multi_logical_row_descriptive_only')
    return dict(image_id=record['image_id'], source_parent_id=record['source_parent_id'],
        original_image_path=record['original_image_path'], raw_ocr_rows=raw, interpreted_ocr_rows=interpreted,
        evaluation_truth_rows=truth, raw_errors=page_errors(truth, raw), interpreted_errors=page_errors(truth, interpreted),
        corrections=output['corrections'], correction_assessments=assess_corrections(truth, raw, output['corrections']),
        interpretations=output['interpretations'], interpretation_uncertain=output['uncertain'],
        raw_f1_decision=output['raw_f1_decision'], interpreted_f1_decision=output['interpreted_f1_decision'],
        decision=output['final_decision'], original_reason_codes=output['original_reason_codes'],
        original_evidence=output['original_evidence'], syntax_failure_resolved=output['syntax_failure_resolved'], cohorts=cohorts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--baseline-dir', type=Path, default=BASELINE)
    ap.add_argument('--output-dir', type=Path)
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    integrity = verify_integrity()
    config = load_json(args.config)
    if (config.get('schema_version') != 1 or config.get('candidate') not in ('T0', 'T1')
            or config.get('enabled') != (config['candidate'] == 'T1')):
        raise ValueError('Use the frozen T0 or T1 configuration')
    expected_config = WORKSPACE / ('configs/t0_baseline.yaml' if config['candidate']=='T0' else 'configs/t1_tier_aware.yaml')
    if digest(args.config) != digest(expected_config):
        raise ValueError('Configuration differs from the frozen experiment')
    items, lineage = load_saved(args.baseline_dir)
    mapping = load_registry()
    if args.verify_only:
        print(json.dumps(dict(**integrity, baseline_rows=len(items), complete=True, inference=False), indent=2))
        return
    if args.output_dir is None:
        ap.error('--output-dir is required unless --verify-only')
    out = new_output(args.output_dir)
    rows = []
    for item in items:
        rendering = source_rendering_context(item['record']) if config['enabled'] else {}
        rows.append(evaluate_item(item, config, mapping, rendering))
    summary, parents, cohorts = grouped(rows)
    result = dict(candidate=config['candidate'], config=config, config_sha256=digest(args.config),
                  baseline_lineage=lineage, mapping_sha256=digest(WORKSPACE / 'manifests/verified_role_ambiguities.csv'),
                  code_sha256={p.name:digest(p) for p in (WORKSPACE/'src').glob('*.py')}, rows=rows,
                  evidence_claim='exposed development; no independent test; no model inference')
    write_json(out / 'results.json', result)
    write_json(out / 'summary.json', summary)
    write_json(out / 'by_parent.json', parents)
    write_json(out / 'by_cohort.json', cohorts)
    serialize_csv(out / 'per_image.csv', rows)
    verify_integrity()
    write_json(out / 'completion.json', dict(complete=True, candidate=config['candidate'], rows=len(rows),
                                            results_sha256=digest(out / 'results.json')))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
