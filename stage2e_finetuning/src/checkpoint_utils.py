"""Atomic lineage-preserving Stage2E checkpoints."""
from context import *
import os
def save_checkpoint(path,model,optimizer,config,step,vocab,extra=None):
 path=Path(path);assert path.is_relative_to(E/'checkpoints')
 if path.exists():raise FileExistsError('Preserve existing checkpoint: '+str(path))
 _,vh=load_vocabulary(VOCAB)
 state=dict(format_version='stage2e_v1',model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict() if optimizer else {},char_to_idx=vocab.char_to_index,vocabulary_hash=vh,original_checkpoint_sha256=EXPECTED,config=config,step=step,python_rng=random.getstate(),numpy_rng=dict(name=np.random.get_state()[0],keys=np.random.get_state()[1].tolist(),position=int(np.random.get_state()[2]),has_gauss=int(np.random.get_state()[3]),cached_gauss=float(np.random.get_state()[4])),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],extra=extra or {})
 # NumPy RNG complete state is recorded separately for deterministic sampler schedules.
 path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp')
 torch.save(state,temporary);os.replace(temporary,path)
 return digest(path)
def set_finetuning_mode(model):
 model.train()
 # Freeze running statistics on small/micro batches; affine parameters still train.
 for module in model.modules():
  if isinstance(module,torch.nn.BatchNorm2d):module.eval()
def forward_training(model,images,checkpoint_width=2048):
 if images.shape[-1]<=checkpoint_width:return model(images)
 from torch.utils.checkpoint import checkpoint_sequential
 features=checkpoint_sequential(model.cnn,6,images,use_reentrant=False)
 sequence,_=model.rnn(features.squeeze(2).permute(0,2,1))
 return model.classifier(sequence).permute(1,0,2)

