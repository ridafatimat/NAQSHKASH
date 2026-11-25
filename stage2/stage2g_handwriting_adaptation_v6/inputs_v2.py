"""Inputs v2 = collection_v3_inputs_v2.csv (old 160 CONFIDENT tiers + 39 recovered tiers, Claude visual QA).
Recovered crops live in handwritten_tier_crops/tier_crops_recovered/.
Validation for SELECTION is unchanged: the same 65 tiers of W05, W24, W30, W31.
W09 (3 recovered tiers, split 'validation_extra') is reported separately and never used for selection.
Usage:  import inputs_v2; inputs_v2.use(R)   (R = run_v6 module)
"""
from pathlib import Path
HERE = Path(__file__).resolve().parent
ASSIGNED_TRAIN = {'W01', 'W03', 'W04', 'W06', 'W08', 'W10', 'W11', 'W12', 'W13', 'W15', 'W16', 'W18', 'W19', 'W21', 'W22', 'W26', 'W29', 'W32'}


def use(R):
    e = R.e

    def load_inputs():
        old, replay, dev = e.inputs()
        w05_old = e.read(e.V5 / 'W05_evaluation_inputs.csv')
        assert len(w05_old) == 14 and {r['writer_id'] for r in w05_old} == {'W05'}
        new = e.read(HERE / 'collection_v3_inputs_v2.csv')
        for r in new:
            p = R.DATA / r['rel_path']
            assert p.is_file(), f'Missing crop: {p}  (extract the V8 package into the v6 folder)'
            assert e.sha(p) == r['sha256'], f'Crop changed: {p}'
            r['path'] = str(p)
        train = old + [r for r in new if r['split'] == 'train']
        val = w05_old + [r for r in new if r['split'] == 'validation']
        tw = {r['writer_id'] for r in train}; vw = {r['writer_id'] for r in val}
        assert tw.isdisjoint(vw) and not (tw | vw) & R.TEST_WRITERS
        assert {r['writer_id'] for r in new if r['split'] == 'train'} <= ASSIGNED_TRAIN
        assert vw == set(R.CFG['validation_writers']) and len(val) == 65, (vw, len(val))
        return train, val, replay, dev

    def load_extra():
        rows = [r for r in e.read(HERE / 'collection_v3_inputs_v2.csv') if r['split'] == 'validation_extra']
        for r in rows: r['path'] = str(R.DATA / r['rel_path']); assert r['writer_id'] == 'W09'
        return rows

    R.load_inputs = load_inputs; R.load_extra = load_extra
    return R
