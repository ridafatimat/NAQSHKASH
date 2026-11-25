"""NAQSHKASH V6: retrain the handwriting model with the collection-v3 writers.

Start:      V5b final model (all_layers, 2000 updates; sha256 0dc250f1...).
Train:      old W01 W03 W04 W06 (originals + their checked augmentations)
            + 13 new collection-v3 writers (CONFIDENT tier crops only).
Validation: W05 (old 14 tiers + new 17) + W24 W30 W31 = 4 writers, used for
            checkpoint selection (macro-writer raw CER).
Test:       W02 W07 + 6 new test writers are never loaded here.
Synthetic:  same safe4 replay + clean KL anchor as V5b; clean synthetic CER is
            reported as a sanity check.

Usage (from this folder):
  python -B run_v6.py --check          inputs + model load only, no training
  python -B run_v6.py                  full run
  python -B run_v6.py --resume         continue an interrupted run
"""
import sys, json, csv, time, random, collections, argparse, os
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V5B = HERE.parent / 'stage2g_handwriting_adaptation_v5b'
sys.path.insert(0, str(V5B / 'local_run'))
import v5b_engine as e  # noqa: E402  (read-only reuse of the V5b engine)

np, torch, cv = e.np, e.torch, e.cv
CFG = json.loads((HERE / 'config_v6.json').read_text(encoding='utf-8'))
DATA = Path(CFG['collection_v3_crops_dir'])
TEST_WRITERS = set(CFG['test_writers'])


# ---------------------------------------------------------------- inputs
def load_inputs():
    old, replay, dev = e.inputs()                                  # V5b asserts stay active
    w05_old = e.read(e.V5 / 'W05_evaluation_inputs.csv')
    assert len(w05_old) == 14 and {r['writer_id'] for r in w05_old} == {'W05'}
    new = e.read(HERE / 'collection_v3_inputs.csv')
    for r in new:
        p = DATA / r['rel_path']
        assert p.is_file(), f'Missing crop: {p}  (check collection_v3_crops_dir in config_v6.json)'
        assert e.sha(p) == r['sha256'], f'Crop changed: {p}'
        r['path'] = str(p)
    train = old + [r for r in new if r['split'] == 'train']
    val = w05_old + [r for r in new if r['split'] == 'validation']
    tw = {r['writer_id'] for r in train}; vw = {r['writer_id'] for r in val}
    assert tw.isdisjoint(vw), ('writer in both train and validation', tw & vw)
    assert not (tw | vw) & TEST_WRITERS, 'test writer present'
    assert vw == set(CFG['validation_writers']), vw
    assert {r['tier_id'] for r in train}.isdisjoint({r['tier_id'] for r in val})
    return train, val, replay, dev


# ---------------------------------------------------------------- helpers
class Cache:
    def __init__(self, limit): self.limit = limit; self.d = collections.OrderedDict()
    def get(self, r):
        k = r['id']
        if k in self.d: self.d.move_to_end(k); return self.d[k]
        assert r.get('writer_id') not in TEST_WRITERS
        t = e.v2.tensors(r); self.d[k] = t
        if len(self.d) > self.limit: self.d.popitem(last=False)
        return t


def light_aug(t, rng, required):
    """Geometry/photometric jitter for real crops that have no stored augmentations.
    Moves and blurs existing ink only; never adds or removes strokes."""
    a = t.numpy()[0]; H, W = a.shape
    sx = float(rng.uniform(0.85, 1.15)); newW = max(16, int(round(W * sx)))
    if newW // 4 < required: return t
    img = cv.resize(a, (newW, H), interpolation=cv.INTER_LINEAR)
    ang = float(rng.uniform(-3, 3)); M = cv.getRotationMatrix2D((newW / 2, H / 2), ang, 1.0)
    img = cv.warpAffine(img, M, (newW, H), flags=cv.INTER_LINEAR, borderValue=1.0)
    g = float(rng.uniform(0.7, 1.3)); img = np.clip(img, 0, 1) ** g
    if rng.random() < 0.3: img = cv.GaussianBlur(img, (3, 3), float(rng.uniform(0.1, 0.6)))
    if rng.random() < 0.3:
        u8 = np.uint8(np.clip(img * 255, 0, 255))
        ok, b = cv.imencode('.jpg', u8, [cv.IMWRITE_JPEG_QUALITY, int(rng.integers(65, 96))]); img = cv.imdecode(b, 0) / 255.0
    return torch.from_numpy(np.float32(img))[None]


def evaluate(m, v, rows, outdir, stem):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True); raw = []; masked = []; m.eval()
    for r in rows:
        assert r.get('writer_id') not in TEST_WRITERS
        with torch.inference_mode(): prob = m(e.v2.tensors(r)[None])[:, 0].float().softmax(1).numpy()
        for restrict, dest in [(False, raw), (True, masked)]:
            row = e.h.result_row(r, e.h.decode(prob, v, r['role'], restrict), prob, v); row['writer_id'] = r['writer_id']; dest.append(row)
    e.write(outdir / (stem + '_raw.csv'), raw); e.write(outdir / (stem + '_role_mask.csv'), masked)
    met = {'raw': e.all_metrics(raw), 'role_mask': e.all_metrics(masked)}; e.dump(outdir / (stem + '_metrics.json'), met)
    return met, raw, masked


# ---------------------------------------------------------------- training
def train(folder, resume=False):
    cfg = CFG; base = e.config(); P = cfg['training']
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completion.json').exists(): return json.loads((folder / 'completion.json').read_text())
    tr, val, replay, dev = load_inputs()
    e.c.seed(cfg['seed']); torch.set_num_threads(base['threads']); rng = np.random.default_rng(cfg['seed']); rand = random.Random(cfg['seed'])
    init = Path(cfg['initial_checkpoint']); assert e.sha(init) == cfg['initial_checkpoint_sha256'], 'initial checkpoint hash mismatch'
    m, v, meta = e.v2.load_model(init); teacher, _, _ = e.v2.load_model(); teacher.requires_grad_(False); teacher.eval()
    opt = torch.optim.AdamW([dict(params=list(getattr(m, n).parameters()), lr=base['learning_rates'][n], name=n) for n in ['cnn', 'rnn', 'classifier']], weight_decay=base['weight_decay'])
    darkness = json.loads((V5B / 'quality_safe4/darkness_reference.json').read_text()); sampler = e.Safe4(base['augmentation'], darkness)
    cache = Cache(cfg['image_cache_size']); tcache = collections.OrderedDict(); targets = {}
    writers = sorted({r['writer_id'] for r in tr})
    orig = {w: [r for r in tr if r['writer_id'] == w and r['sample_type'] == 'original'] for w in writers}
    aug = {w: [r for r in tr if r['writer_id'] == w and r['sample_type'] == 'augmented'] for w in writers}
    runcfg = dict(config=cfg, writers=writers, train_originals=sum(len(x) for x in orig.values()), train_augmented=sum(len(x) for x in aug.values()), validation_tiers=len(val), initial=str(init))
    state = dict(step=0, best=None, best_key=None, best_checkpoint=None, best_step=None, stale=0, norms=[], clip=None, clipped=0, elapsed=0., history=[])
    last = folder / 'last_checkpoint.json'
    if last.exists():
        if not resume: raise RuntimeError('Run folder has progress; use --resume')
        L = json.loads(last.read_text()); s = torch.load(L['checkpoint'], map_location='cpu', weights_only=True)
        assert s['run_config'] == runcfg, 'config changed'
        m.load_state_dict(s['model_state_dict']); opt.load_state_dict(s['optimizer_state_dict']); state = s['run_state']
        rng.bit_generator.state = s['numpy_rng']; rand.setstate(s['python_rng']); torch.set_rng_state(s['torch_rng'])
    else:
        e.dump(folder / 'run_config.json', runcfg)
    print('TRAIN WRITERS', len(writers), writers, 'originals', runcfg['train_originals'], 'validation tiers', len(val), flush=True)
    log = (folder / 'training.jsonl').open('a', encoding='utf-8'); started = time.time(); base_el = state['elapsed']

    def req(truth): return len(truth) + sum(a == b for a, b in zip(truth, truth[1:]))
    def ctc(logits, r):
        if r['id'] not in targets: targets[r['id']] = torch.tensor(v.encode(r['truth']), dtype=torch.long)
        tg = targets[r['id']]
        return torch.nn.functional.ctc_loss(logits.float().log_softmax(2), tg, torch.tensor([logits.shape[0]]), torch.tensor([len(tg)]), blank=0, reduction='mean', zero_infinity=False)

    step = state['step']
    for step in range(state['step'] + 1, P['max_updates'] + 1):
        if step - 1 >= P['min_updates'] and state['stale'] >= P['patience']: step -= 1; break
        m.train(); opt.zero_grad(set_to_none=True)
        for g in opt.param_groups: g['lr'] = base['learning_rates'][g['name']] * min(1, step / base['warmup_steps'])
        batch = []
        for dom, n in P['batch'].items():
            for _ in range(n):
                if dom.startswith('synthetic'):
                    for _try in range(200):
                        r = rand.choice(replay); t = cache.get(r)
                        if dom == 'synthetic_augmented':
                            try: t, _d = sampler.sample(t, rng, r['id'])
                            except RuntimeError: continue
                            if t.shape[-1] // 4 < req(r['truth']): continue
                        break
                    else: raise RuntimeError('synthetic sampling failed')
                else:
                    w = rand.choice(writers)
                    if dom == 'handwritten_augmented' and aug[w]: r = rand.choice(aug[w]); t = cache.get(r)
                    else:
                        r = rand.choice(orig[w]); t = cache.get(r)
                        if dom == 'handwritten_augmented': t = light_aug(t, rng, req(r['truth']))
                batch.append((dom, r, t))
        assert len(batch) == base['batch_size']; rand.shuffle(batch); losses = collections.defaultdict(list)
        for dom, r, t in batch:
            logits = m(t[None]); loss = ctc(logits, r); total = loss / base['batch_size']
            if dom == 'synthetic_clean':
                if r['id'] not in tcache:
                    with torch.no_grad(): tcache[r['id']] = teacher(t[None]).float().softmax(2)
                    if len(tcache) > cfg['image_cache_size']: tcache.popitem(last=False)
                kl = torch.nn.functional.kl_div(logits.float().log_softmax(2), tcache[r['id']], reduction='none').sum(2).mean()
                total = total + base['KL_weight'] * kl / P['batch']['synthetic_clean']
            assert torch.isfinite(total), 'Nonfinite training loss'; total.backward(); losses[dom].append(float(loss.detach()))
        norm = sum(float(p.grad.detach().float().pow(2).sum()) for p in m.parameters() if p.grad is not None) ** .5
        state['norms'] = (state['norms'] + [norm])[-base['clip']['window']:]
        if step == base['clip']['calibration_steps'] + 1 or (step > base['clip']['calibration_steps'] and step % base['clip']['refresh_every'] == 1):
            state['clip'] = float(np.quantile(state['norms'], base['clip']['quantile']) * base['clip']['multiplier'])
        if state['clip'] is not None and norm > state['clip']:
            torch.nn.utils.clip_grad_norm_(m.parameters(), state['clip']); state['clipped'] += 1
        opt.step(); state['step'] = step; state['elapsed'] = base_el + time.time() - started
        log.write(json.dumps(dict(step=step, ctc={k: float(np.mean(a)) for k, a in losses.items()}, norm=norm)) + '\n'); log.flush()
        if step % 50 == 0:
            sp = state['elapsed'] / step
            print('TRAIN', step, '/', P['max_updates'], 'seconds/update', round(sp, 2), 'ETA ceiling hours', round(sp * (P['max_updates'] - step) / 3600, 2), 'clip', round(state['clipped'] / step, 4), flush=True)
        if step % P['evaluate_every'] == 0 or step == P['max_updates']:
            met, _, _ = evaluate(m, v, val, folder / 'evaluations', f'step_{step:05}')
            ch = met['raw']; key = [ch['macro_writer']['cer'], ch['overall']['cer_no_whitespace'], -ch['overall']['exact_tiers']]
            cp = folder / 'checkpoints' / f'step_{step:05}.pt'; cp.parent.mkdir(exist_ok=True)
            improved = state['best_key'] is None or tuple(key) < tuple(state['best_key'])
            state['stale'] = 0 if improved else state['stale'] + 1
            if improved: state['best_key'] = key; state['best_checkpoint'] = str(cp); state['best_step'] = step
            state['history'].append(dict(step=step, macro_writer_cer_raw=ch['macro_writer']['cer'], macro_writer_cer_role_mask=met['role_mask']['macro_writer']['cer'], exact=ch['overall']['exact_tiers'], per_writer={w: x['cer'] for w, x in ch['per_writer'].items()}))
            save = dict(format_version='stage2g_v6', model_state_dict=m.state_dict(), optimizer_state_dict=opt.state_dict(), char_to_idx=meta['char_to_idx'], vocabulary_hash=meta['vocabulary_hash'], run_config=runcfg, run_state=state, numpy_rng=rng.bit_generator.state, python_rng=rand.getstate(), torch_rng=torch.get_rng_state())
            tmp = cp.with_suffix('.tmp'); torch.save(save, tmp); os.replace(str(tmp), str(cp))
            e.dump(last, dict(checkpoint=str(cp), sha256=e.sha(cp), step=step)); e.dump(folder / 'progress.json', state)
            print('EVAL', step, 'macro-writer CER raw', round(ch['macro_writer']['cer'], 4), 'role_mask', round(met['role_mask']['macro_writer']['cer'], 4), 'per writer', {w: round(x['cer'], 3) for w, x in ch['per_writer'].items()}, 'stale', state['stale'], flush=True)
    comp = dict(steps=step, best_checkpoint=state['best_checkpoint'], best_step=state['best_step'], checkpoint_sha256=e.sha(state['best_checkpoint']), history=state['history'], seconds=state['elapsed'])
    e.dump(folder / 'completion.json', comp); return comp


# ---------------------------------------------------------------- report
def report(run, comp):
    tr, val, replay, dev = load_inputs()
    rows = []
    for name, path in [('frozen_E1', None), ('V5b_final (start)', CFG['initial_checkpoint']), ('V6 (new)', comp['best_checkpoint'])]:
        m, v, _ = e.v2.load_model(path)
        met, _, _ = evaluate(m, v, val, run / 'final_eval', name.split()[0])
        _, sr, sm = e.evaluate(m, v, dev, run / 'regression', name.split()[0], True); reg = e.synthetic_metrics(sr, dev)
        rows.append(dict(model=name, val_macro_writer_CER_raw=met['raw']['macro_writer']['cer'], val_macro_writer_CER_role_mask=met['role_mask']['macro_writer']['cer'],
                         val_exact_role_mask=met['role_mask']['overall']['exact_accuracy'], val_per_writer_role_mask={w: x['cer'] for w, x in met['role_mask']['per_writer'].items()},
                         clean_synthetic_CER=reg['clean_historical_regression']['cer'], clean_synthetic_exact=reg['clean_historical_regression']['exact_accuracy']))
    e.dump(run / 'v6_summary.json', dict(rows=rows, best_step=comp['best_step'], checkpoint=comp['best_checkpoint'], checkpoint_sha256=comp['checkpoint_sha256'], test_writers_used=False))
    lines = ['| Model | Val macro-writer CER (raw) | Val CER (role mask) | Val exact (role mask) | Clean synthetic CER |', '|---|---:|---:|---:|---:|']
    for r in rows: lines.append(f"| {r['model']} | {r['val_macro_writer_CER_raw']*100:.1f}% | {r['val_macro_writer_CER_role_mask']*100:.1f}% | {r['val_exact_role_mask']*100:.1f}% | {r['clean_synthetic_CER']*100:.2f}% |")
    (run / 'v6_summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8'); print('\n'.join(lines)); print('V6 COMPLETE', comp['best_checkpoint'], flush=True)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--check', action='store_true'); a.add_argument('--resume', action='store_true'); args = a.parse_args()
    if args.check:
        tr, val, replay, dev = load_inputs(); m, v, _ = e.v2.load_model(CFG['initial_checkpoint'])
        assert e.sha(CFG['initial_checkpoint']) == CFG['initial_checkpoint_sha256']
        for r in tr[:3] + val[:3]: e.v2.tensors(r)
        print('CHECK OK train tiers', len(tr), 'writers', len({r['writer_id'] for r in tr}), '| validation tiers', len(val), 'writers', sorted({r['writer_id'] for r in val})); return
    run = HERE / 'outputs' / 'v6_run'
    comp = train(run / 'training', resume=args.resume)
    report(run, comp)


if __name__ == '__main__':
    main()
