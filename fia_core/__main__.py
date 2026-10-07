"""Run python -m fia_core --help from NCFM-Lab (or release repository)."""
import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import torch
from . import __version__, resnet
from .core import Proxy, fit_head, screen, inject


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def write(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def load_data(files):
    shards=[]
    for name in files:
        obj=torch.load(name,map_location='cpu',weights_only=True)
        if not isinstance(obj,(tuple,list)) or len(obj)!=2:
            raise ValueError(f'{name}: require clean [images, labels]; no trigger-file fallback')
        x,y=obj
        if x.ndim!=4 or x.shape[1]!=3 or y.ndim!=1 or len(x)!=len(y) or not len(x):
            raise ValueError(f'{name}: invalid dimensions')
        if not torch.isfinite(x).all() or x.min() < -1e-6 or x.max() > 1+1e-6:
            raise ValueError(f'{name}: require explicit float images in [0,1]')
        if y.min()<0 or not torch.equal(y,y.long()): raise ValueError('Invalid class ids')
        shards.append((x.float().clamp(0,1),y.long()))
    return shards


def model(args,classes):
    # Architecture derives from checkpoint tensor shape, never its filename.
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if 'model' in state: state=state['model']
    if 'state_dict' in state: state=state['state_dict']
    cleaned={}
    for k,v in state.items():
        k=k.removeprefix('module.').removeprefix('backbone.')
        if k.startswith(('projector.','classifier.','head.','fc.')): continue
        cleaned[k]=v
    if 'conv1.weight' not in cleaned: raise ValueError('Checkpoint has no conv1.weight')
    small=cleaned['conv1.weight'].shape[-1]==3
    backbone,dim=resnet.resnet50(small_input=small)
    backbone.load_state_dict(cleaned,strict=True)  # No silent partial/random backbone.
    return Proxy(backbone,dim,classes,args.preprocess).to(args.device)


def metadata(args):
    return {'version':__version__,'argv':sys.argv[1:],'seed':args.seed,
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'checkpoint_sha256':digest(args.checkpoint),
            'input_sha256':{str(p):digest(p) for p in args.shards},
            'code_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
            'preprocess':args.preprocess,'backbone_buffers':'frozen_eval',
            'historical_results_reproduced':False}


def find(args):
    shards=load_data(args.shards)
    x=torch.cat([a for a,b in shards]);y=torch.cat([b for a,b in shards])
    proxy=model(args,int(y.max())+1).eval()
    with torch.no_grad():
        features=torch.cat([proxy.features(b.to(args.device)).cpu() for b in x.split(args.batch_size)])
    result=screen(features,y,args.low,args.high)
    names=json.loads(Path(args.class_names).read_text()) if args.class_names else None
    result['dataset']=args.dataset
    for row in result['pairs']:
        for side in ('source','target'):
            row[side+'_name']=names[row[side]] if names is not None else str(row[side])
    result['provenance']=metadata(args)
    write(args.output/'candidates.json',result)
    with (args.output/'candidates.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result['pairs'][0]));w.writeheader();w.writerows(result['pairs'])
    torch.save({'features':features,'labels':y},args.output/'clean_features.pt')
    # Both directions need empirical scoring; distances do not select the winner.
    with (args.output/'scores_template.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source','target','seed','split','split_sha256','asr','ftr'])
        for r in result['pairs']:
            if r['eligible']:
                for a,b in [(r['source'],r['target']),(r['target'],r['source'])]:
                    w.writerow([a,b,'','selection','','',''])
    print(f"[FIND] {result['undirected_kept']}/{result['undirected_total']} undirected pairs; no empirical winner yet")


def select(args):
    candidates=json.loads(Path(args.candidates).read_text())
    allowed=set()
    for r in candidates['pairs']:
        if r['eligible']:allowed.update([(r['source'],r['target']),(r['target'],r['source'])])
    split_path=Path(args.selection_split)
    split=json.loads(split_path.read_text())
    if split.get('role')!='selection': raise ValueError('Selection split required')
    by_pair={};seen=set()
    with open(args.scores,newline='') as f:
        for row in csv.DictReader(f):
            a,b,seed=int(row['source']),int(row['target']),int(row['seed'])
            if (a,b) not in allowed: raise ValueError('Scored pair outside candidate interval')
            if row['split']!='selection' or row['split_sha256']!=digest(split_path):
                raise ValueError('Score rows must reference the exact selection manifest, not test data')
            if (a,b,seed) in seen: raise ValueError('Duplicate pair/seed')
            seen.add((a,b,seed))
            asr,ftr=float(row['asr']),float(row['ftr'])
            if not(0<=asr<=100 and 0<=ftr<=100):raise ValueError('ASR/FTR must be percent in [0,100]')
            by_pair.setdefault((a,b),{})[seed]=asr-ftr
    if len(by_pair)<2:raise ValueError('At least two evaluated directions required for comparison')
    seeds=next(iter(by_pair.values())).keys()
    if any(v.keys()!=seeds for v in by_pair.values()):raise ValueError('All evaluated pairs need identical seed sets')
    ranked=sorted([{'source':a,'target':b,'mean_margin':sum(v.values())/len(v),'seeds':sorted(v)}
                   for (a,b),v in by_pair.items()],key=lambda r:(-r['mean_margin'],r['source'],r['target']))
    write(args.output/'selected.json',{'selected':ranked[0],'ranking':ranked,
          'candidate_sha256':digest(args.candidates),'scores_sha256':digest(args.scores),
          'selection_split_sha256':digest(split_path),'scope':'best among evaluated directions only'})
    print('[SELECT]',ranked[0])


def split(args):
    with open(args.images,newline='') as stream:
        rows=list(csv.DictReader(stream))
    groups={};seen=set()
    for row in rows:
        # Content hashes prevent identical copies under different paths crossing splits.
        h=digest(row['path'])
        if h in seen:raise ValueError('Duplicate image content: deduplicate before splitting')
        seen.add(h);row['sha256']=h
        groups.setdefault(int(row['label']),[]).append(row)
    parts={'selection':[],'test':[]};rng=random.Random(args.seed)
    for label,items in sorted(groups.items()):
        items.sort(key=lambda r:r['sha256']);rng.shuffle(items)
        n=int(len(items)*args.fraction)
        if not 0<n<len(items):raise ValueError(f'Class {label} cannot populate both splits')
        parts['selection'].extend(items[:n]);parts['test'].extend(items[n:])
    for role,items in parts.items():
        write(args.output/(role+'.json'),{'role':role,'seed':args.seed,'images':items,
              'input_sha256':digest(args.images),'note':'Create before candidate outcome evaluation; not a retrospective holdout claim'})
    print('[SPLIT]',{k:len(v) for k,v in parts.items()})


def injection(args):
    if args.source==args.target:raise ValueError('Source and target must differ')
    shards=load_data(args.shards)
    classes=int(torch.cat([y for x,y in shards]).max())+1
    if not (0<=args.source<classes and 0<=args.target<classes):raise ValueError('Invalid pair')
    modified=set(args.modify_shards)
    if not modified or min(modified)<0 or max(modified)>=len(shards):raise ValueError('Invalid shard positions')
    if len({p.name for p in args.shards})!=len(args.shards):raise ValueError('Duplicate shard filenames')
    for i,(x,y) in enumerate(shards):
        if i not in modified:continue
        if (y==args.source).sum()<args.items or not (y==args.target).any():
            raise ValueError(f'Shard {i}: insufficient source/target samples')
    manifest=metadata(args);manifest.update({'source':args.source,'target':args.target,
        'quality_mode':args.quality_mode,'steps':args.steps,'items_per_shard':args.items,
        'epsilon':args.epsilon,'head_epochs':args.head_epochs,'modify_shards':sorted(modified),'shards':[]})
    banks=[]
    for i,(x,y) in enumerate(shards):
        if i not in modified:
            shutil.copy2(args.shards[i],args.output/args.shards[i].name)
            continue
        # Fresh identical backbone state for each shard; head RNG explicit.
        torch.manual_seed(args.seed+i)
        proxy=model(args,classes)
        x,y=x.to(args.device),y.to(args.device)
        fit_head(proxy,x,y,args.head_epochs,args.batch_size)
        indices=torch.where(y==args.source)[0][:args.items]
        result=inject(proxy,x[indices],x[y==args.target],args.source,args.target,
                      args.steps,args.epsilon,args.quality_mode)
        banks.append(result['patterns'].cpu())
        poisoned=x.clone();labels=y.clone()
        poisoned[indices]=result['images'];labels[indices]=args.target
        name=Path(args.shards[i]).name
        torch.save([poisoned.cpu(),labels.cpu()],args.output/name)
        torch.save({k:v.cpu() if isinstance(v,torch.Tensor) else v for k,v in result.items()},
                   args.output/(Path(name).stem+'_bank.pt'))
        torch.save(proxy.state_dict(),args.output/(Path(name).stem+'_proxy.pt'))
        manifest['shards'].append({'name':name,'selected_indices':indices.cpu().tolist(),
                                    'output_sha256':digest(args.output/name)})
        write(args.output/'manifest.json',manifest)
        print('[INJECT]',name,'completed',flush=True)
    torch.save(torch.cat(banks),args.output/'trigger_bank.pt')
    manifest['output_sha256']={p.name:digest(p) for p in args.output.glob('*.pt')}
    manifest['complete']=True;write(args.output/'manifest.json',manifest)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for command in ['find','inject']:
        q=sub.add_parser(command)
        q.add_argument('--shards',nargs='+',required=True,type=Path)
        q.add_argument('--checkpoint',required=True,type=Path)
        q.add_argument('--preprocess',required=True,choices=['cifar10','imagenet'])
        q.add_argument('--device',default='cpu');q.add_argument('--seed',type=int,default=42)
        q.add_argument('--batch-size',type=int,default=32)
        if command=='find':
            q.add_argument('--dataset',required=True,choices=['cifar10','cifar100','imagenette','coco'])
            q.add_argument('--class-names',type=Path)
            q.add_argument('--low',type=float,default=.30);q.add_argument('--high',type=float,default=.90)
        else:
            q.add_argument('--source',required=True,type=int);q.add_argument('--target',required=True,type=int)
            q.add_argument('--modify-shards',nargs='+',type=int,required=True,help='Zero-based positions in --shards to modify; others copied unchanged')
            q.add_argument('--items',type=int,default=10);q.add_argument('--steps',type=int,default=4500)
            q.add_argument('--head-epochs',type=int,default=150);q.add_argument('--epsilon',type=float,default=.15)
            q.add_argument('--quality-mode',required=True,choices=['stop_gradient','differentiable'])
    q=sub.add_parser('select');q.add_argument('--candidates',required=True,type=Path)
    q.add_argument('--scores',required=True,type=Path);q.add_argument('--selection-split',required=True,type=Path)
    q=sub.add_parser('split');q.add_argument('--images',required=True,type=Path)
    q.add_argument('--seed',type=int,default=42);q.add_argument('--fraction',type=float,default=.5)
    for q in sub.choices.values():q.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    if hasattr(args,'batch_size') and args.batch_size<1:p.error('Positive batch size required')
    if args.command=='inject' and (args.items<1 or args.head_epochs<1):p.error('Positive items/head epochs required')
    if args.command=='split' and not 0<args.fraction<1:p.error('fraction must be between 0 and 1')
    args.output.mkdir(parents=True,exist_ok=False)
    if hasattr(args,'seed'):torch.manual_seed(args.seed)
    {'find':find,'inject':injection,'select':select,'split':split}[args.command](args)

if __name__=='__main__':main()
