"""MANUAL frozen D1/E1 inference. No candidate decisions or ground-truth inputs."""
import argparse
from integration import *
from dataset_loader import DevelopmentDataset,SPLIT,image_bytes
from observation_store import ObservationStore,exclusive_run_lock


def lineage_plan(dataset,device):
    return dict(schema_version=1,split=SPLIT,dataset_dir=str(dataset.directory),dataset_manifest_sha256=dataset.manifest_sha256,
        selection_sha256=dataset.selection_sha256,requested_device=device,seed=42,
        checkpoint=str(CHECKPOINT),checkpoint_sha256=digest(CHECKPOINT),
        d1_config_sha256=digest(PROJECT/'stage2/stage2d_experiments/configs/D1.json'),
        vocabulary_sha256=digest(PROJECT/'config/vocabulary.json'),experiment_lock_sha256=digest(ROOT/'configs/experiment_lock.json'),
        contract='original FrozenPipeline.observe(bytes), one image/tier at a time; no target loss or labels')


def deterministic_seed(id):
    import random,numpy as np,torch,cv2
    seed=int(hashlib.sha256(('rejection_v2:42:'+id).encode()).hexdigest()[:8],16)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);cv2.setRNGSeed(seed%2147483647)
    torch.use_deterministic_algorithms(True)


def execution_environment(requested_device):
    import platform,torch,numpy,cv2,PIL
    actual='cuda' if requested_device=='auto' and torch.cuda.is_available() else 'cpu' if requested_device=='auto' else requested_device
    if actual=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; use --device cpu')
    return dict(actual_device=actual,python=platform.python_version(),torch=torch.__version__,numpy=numpy.__version__,
                opencv=cv2.__version__,Pillow=PIL.__version__,batching='original single-tier forward',seed_policy='per-record deterministic42')


def infer_pending(dataset,store,pipeline,max_new_records=None,progress_every=25,seed_callback=None):
    identities=[dataset.identity(r) for r in dataset.rows]
    completed=0;new=0
    for row,identity in zip(dataset.rows,identities):
        if store.read(identity) is not None:completed+=1;continue
        if max_new_records is not None and new>=max_new_records:break
        raw=image_bytes(dataset.directory,row)  # verified bytes only, no annotation passed to model
        if seed_callback:seed_callback(identity['image_id'])
        try:observation=pipeline.observe(raw)
        except (ValueError,RuntimeError,OSError) as exc:observation=pipeline_exception(exc)
        store.append(identity,observation,runtime_rendering_context(row))
        new+=1;completed+=1
        if progress_every and completed%progress_every==0:print('Saved',completed,'/',len(identities),flush=True)
    # Recount directly from immutable selected IDs, not inference retries.
    total=sum(store.read(i) is not None for i in identities)
    if total==len(identities):return store.finish(identities)
    return dict(complete=False,saved=total,total=len(identities),new_records=new,resume_required=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--split',choices=(SPLIT,),default=SPLIT)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda','auto'),default='cpu')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--max-new-records',type=int)
    p.add_argument('--progress-every',type=int,default=25)
    a=p.parse_args()
    if a.max_new_records is not None and a.max_new_records<1:p.error('max-new-records must be positive')
    if a.progress_every<0:p.error('progress-every must be nonnegative')
    verify_integrity()
    if not a.output_dir.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Inference output must be inside isolated workspace')
    dataset=DevelopmentDataset(a.dataset,a.split,verify_files=True)
    plan=lineage_plan(dataset,a.device)
    environment=execution_environment(a.device);plan['execution_environment']=environment
    if plan['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Selected E1 checkpoint mismatch')
    store=ObservationStore(a.output_dir,plan,a.resume)
    with exclusive_run_lock(store.directory/'run.lock'):
        pending=[r for r in dataset.rows if store.read(dataset.identity(r)) is None]
        # Do not initialize or run the CRNN when resume is already complete.
        pipeline=make_pixel_pipeline(environment['actual_device']) if pending else None
        result=infer_pending(dataset,store,pipeline,a.max_new_records,a.progress_every,deterministic_seed)
        verify_integrity();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
