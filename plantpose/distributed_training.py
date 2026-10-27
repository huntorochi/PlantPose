from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .data import collate_graphs
from .distributed import GlobalBatchSampler, RankLocalGraphLoss, rank_rng, restore_rng
from .config import distributed_settings
from .losses import Criterion
from .models import build_relationformer
from .training import settings, initialize_backbone, optimizer_for, device_batch, save_checkpoints, validation_value
from .training_data import TrainingDataset


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def source_sha():
    digest=hashlib.sha256();root=Path(__file__).resolve().parent
    for p in sorted(root.rglob('*.py')):
        digest.update(p.relative_to(root).as_posix().encode());digest.update(p.read_bytes())
    digest.update((root.parent/'train.py').read_bytes())
    return digest.hexdigest()


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)


def finite_all(value,device):
    ok=torch.tensor(int(bool(torch.isfinite(value).all())),device=device)
    dist.all_reduce(ok,op=dist.ReduceOp.MIN)
    if not ok.item():raise FloatingPointError('Nonfinite DDP loss/gradient on at least one rank')


def phase(ddp,optimizer,loader,device,epoch,training,values,output,global_batch,step,config=None):
    ddp.train(training);totals={};seen=[];start=time.monotonic()
    for index,batch in enumerate(loader):
        images,target,ids=device_batch(batch,device)
        with torch.set_grad_enabled(training):
            if training:optimizer.zero_grad(set_to_none=True)
            if not training:
                h,out=ddp.module.model(images)
                value=validation_value(h,out,ddp.module.model,ddp.module.criterion,target,config,values,device)
                losses={'smd':torch.tensor(value,device=device)}
            else:
                losses=ddp(images,target)
            finite_all(torch.stack(list(losses.values())),device)
            if training:
                losses['total'].backward()
                norm=torch.nn.utils.clip_grad_norm_(ddp.module.model.parameters(),values['clip_grad_norm'])
                finite_all(norm,device);optimizer.step();step+=1
        keys=sorted(losses);numbers=torch.stack([losses[k].detach() for k in keys])
        dist.all_reduce(numbers);numbers/=dist.get_world_size()
        for key,value in zip(keys,numbers):totals[key]=totals.get(key,0.)+float(value)
        seen.extend(ids)
        if dist.get_rank()==0 and (index%100==0 or index+1==len(loader)):
            atomic_json(output/'progress.json',dict(epoch=epoch,phase='train' if training else 'validation',
                        batches=index+1,total_batches=len(loader),global_samples=(index+1)*global_batch,
                        phase_seconds=time.monotonic()-start,updated_unix=time.time(),step=step))
    rows=[None]*dist.get_world_size();dist.all_gather_object(rows,seen)
    combined=[rows[r][i] for i in range(len(rows[0])) for r in range(len(rows))]
    indices=loader.sampler.global_indices()
    if len(indices)!=len(set(indices)):raise RuntimeError('Duplicate dataset indices in DDP epoch')
    # Mix intentionally repeats source images. Occurrences have unique dataset
    # indices; repeated source IDs must not be mistaken for sampler padding.
    expected=[loader.dataset.rows[i][2] for i in indices]
    if combined!=expected:raise RuntimeError('Distributed sample order/coverage differs from global sampler')
    result=dict(losses={k:v/len(loader) for k,v in totals.items()},samples=len(combined),
                batches=len(loader),ordered_ids_sha256=hashlib.sha256('\n'.join(combined).encode()).hexdigest(),
                unique_source_samples=len(set(combined)),
                ordered_indices_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
                seconds=time.monotonic()-start)
    if not training:
        result['metric']=values['validation'];result['value']=result['losses']['smd']
    return result,step


def train_distributed(args):
    config,values=settings(args.config,args.dataset,args.training_mst)
    values,global_batch=distributed_settings(values,int(os.environ.get('WORLD_SIZE',0)))
    local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    rank,world=dist.get_rank(),dist.get_world_size();device=torch.device('cuda',local)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(values['seed']+rank);np.random.seed(values['seed']+rank);torch.manual_seed(values['seed'])
    output=Path(args.output)
    if rank==0:output.mkdir(parents=True,exist_ok=False)
    dist.barrier()
    training=TrainingDataset(args.data,args.dataset,True,values.get('train_limit'))
    validation=TrainingDataset(args.data,args.dataset,False,values.get('validation_limit'))
    samplers=[GlobalBatchSampler(len(data),global_batch,rank,world,values.get('sampler_seed',values['seed']),shuffle)
              for data,shuffle in [(training,True),(validation,values.get('validation_shuffle',False))]]
    loaders=[DataLoader(data,batch_size=global_batch//world,sampler=sampler,drop_last=True,
                        collate_fn=collate_graphs,num_workers=args.workers,pin_memory=True)
             for data,sampler in zip([training,validation],samplers)]
    if not all(len(loader) for loader in loaders):raise ValueError('Empty full-batch split')
    identity=dict(settings=values,source_sha256=source_sha(),model_config_sha256=sha_file(args.config),
                  world_size=world,global_batch=global_batch,per_rank_batch=global_batch//world,workers=args.workers,
                  reduction='rank_local_graph_loss_v1',
                  training_mst=args.training_mst,benchmark=False,matmul_tf32=False,cudnn_tf32=False,
                  torch_version=torch.__version__,cuda_version=torch.version.cuda,
                  train_ids=hashlib.sha256('\n'.join(x[2] for x in training.rows).encode()).hexdigest(),
                  validation_ids=hashlib.sha256('\n'.join(x[2] for x in validation.rows).encode()).hexdigest())
    identities=[None]*world;dist.all_gather_object(identities,identity)
    if any(i!=identity for i in identities):raise ValueError('Cross-rank source/config/input identity mismatch')
    model=build_relationformer(config).to(device)
    if not args.resume:initialize_backbone(model,args.backbone_weights)
    criterion=Criterion(model,config,args.dataset,args.training_mst).to(device)
    ddp=DistributedDataParallel(RankLocalGraphLoss(model,criterion),device_ids=[local],broadcast_buffers=False)
    optimizer,scheduler=optimizer_for(model,values);start=step=0;best=float('inf')
    if args.resume:
        previous=torch.load(args.resume,map_location='cpu')
        if previous['identity']!=identity:raise ValueError('Resume identity mismatch')
        model.load_state_dict(previous['net'],strict=True);optimizer.load_state_dict(previous['optimizer']);scheduler.load_state_dict(previous['scheduler'])
        start,step,best=previous['epoch'],previous['step'],previous['best'];restore_rng(previous['rank_rng'][rank])
    else:
        # DDP broadcasts initial parameters; independent dropout streams per rank.
        torch.manual_seed(values['seed']+rank)
    final=args.epochs or values['epochs']
    if final<=start:raise ValueError('Final epoch must exceed restored epoch')
    if rank==0:
        atomic_json(output/'metadata.json',dict(identity=identity,epochs=final,resumed_epoch=start,train_samples=len(training),validation_samples=len(validation),
                    initialization_sha256=sha_file(args.resume or args.backbone_weights),initialization='resume' if args.resume else 'ImageNet backbone only',
                    drop_last=True,train_consumed_per_epoch=len(samplers[0])*world,
                    validation_consumed_per_epoch=len(samplers[1])*world,original_history_equivalent=False))
        atomic_json(output/'selected_ids.json',dict(train=[x[2] for x in training.rows],validation=[x[2] for x in validation.rows]))
    for epoch in range(start,final):
        samplers[0].set_epoch(epoch)
        train,step=phase(ddp,optimizer,loaders[0],device,epoch+1,True,values,output,global_batch,step,config)
        valid,_=phase(ddp,optimizer,loaders[1],device,epoch+1,False,values,output,global_batch,step,config)
        score=valid['value']
        scheduler.step();improved=score<best;best=min(best,score)
        rngs=[None]*world;dist.all_gather_object(rngs,rank_rng())
        row=dict(epoch=epoch+1,train=train,validation=valid,best=best,step=step)
        if rank==0:
            checkpoint=dict(net=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                            epoch=epoch+1,step=step,best=best,identity=identity,rank_rng=rngs)
            latest=save_checkpoints(output,values['dataset'],args.training_mst,checkpoint,improved)
            with (output/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
        dist.barrier()
    if rank==0:atomic_json(output/'complete.json',dict(epoch=final,step=step,checkpoint=latest.name,checkpoint_sha256=sha_file(latest),identity=identity))
    dist.barrier();dist.destroy_process_group()
