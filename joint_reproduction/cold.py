#!/usr/bin/env python3
"""Rebuild the corrected joint input matrices and run the unchanged joint fitter.

No private project tree, Arrow file or internal approval record is required.
Input arrays, canonical item order, original computational modules and the final
matrix identities are bound by the public scientific contract. Internal workflow
gates are replaced explicitly by these scientific identity checks, not by an
invented independent ACCEPT record. A fit must use the complete bound panel.
"""
from __future__ import annotations
import argparse, ast, hashlib, importlib.util, json, os, sys, time, types
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.dont_write_bytecode = True
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'tools'))
sys.path.insert(0,str(ROOT.parent/'core_reproduction'))
from raw_semantic_identity import load as load_semantic, verify as verify_semantic
from array_identity import array_record
CONTRACT_PATH=ROOT/'input_contract/CONTRACT.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract():
    c=json.loads(CONTRACT_PATH.read_text())
    for name,want in c['code_hashes'].items():
        if sha(ROOT/'original_code'/name)!=want:raise ValueError('ORIGINAL_CODE_CHANGED: '+name)
    if sha(ROOT.parent/'upstream_metadata/UPSTREAM_ITEM_IDENTITY_MASK.json')!=c['mask_sha256']:raise ValueError('MASK_CHANGED')
    if sha(ROOT.parent/'upstream_metadata/RAW_SEMANTIC_IDENTITY.json.gz')!=c['raw_semantic_sha256']:raise ValueError('RAW_SEMANTIC_IDENTITY_CHANGED')
    return c


_BUILDER=None
_VALIDITY=None

def builder():
    global _BUILDER,_VALIDITY
    if _BUILDER is None:
        path=ROOT/'original_code/build_matrix.py';text=path.read_text()
        mod=types.ModuleType('portable_original_matrix_builder');mod.__file__=str(path)
        sys.modules[mod.__name__]=mod;exec(compile(text,str(path),'exec'),mod.__dict__);_BUILDER=mod
        vp=ROOT/'original_code/prepare_data.py';tree=ast.parse(vp.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validity')
        ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(vp),'exec'),ns);_VALIDITY=ns['validity']
    return _BUILDER


def prepare(payload):
    entry,expected,mask_ids=payload;bm=builder()
    with np.load(entry['path'],allow_pickle=False) as z:
        raw={k:z[k] for k in z.files}
    verify_semantic(raw,expected)
    valid,*_= _VALIDITY(raw['loglik'],raw['char_lens'],raw['gold'],raw.get('n_options'))
    keep=~np.isin(raw['item_ids'].astype(str),mask_ids)
    n=int(valid[keep].sum())
    ledger={'repo':entry['repo'],'valid_after_mask':n,'eligible':n>=50}
    if n<50:return entry,ledger,None
    # Original converter checks this actual container hash after semantic verification.
    entry['source_sha256']=sha(entry['path'])
    result=bm._convert_entry((entry,str(ROOT/'original_code/cell_recover.py'),bm.CELL_RECOVER_SHA256_PIN,tuple(mask_ids)))
    return entry,ledger,result


def check_matrix(path,c,bench):
    with np.load(path,allow_pickle=False) as z:
        expected=c['benches'][bench]['arrays']
        if set(z.files)!=set(expected):raise ValueError('MATRIX_KEYS_CHANGED')
        for k in z.files:
            got=array_record(z[k]);want=expected[k]
            if any(got[x]!=want[x] for x in ['shape','dtype','digest']):raise ValueError('MATRIX_IDENTITY_MISMATCH: '+k)


def build(args,c):
    if args.out.exists():raise ValueError('Output already exists; use a new directory')
    bench=args.bench;bc=c['benches'][bench];sem=load_semantic()
    mask=json.loads((ROOT.parent/'upstream_metadata/UPSTREAM_ITEM_IDENTITY_MASK.json').read_text())['mask_ids'][bench]
    axis_path=ROOT/bc['canonical_items_file']
    if sha(axis_path)!=bc['canonical_items_sha256']:raise ValueError('CANONICAL_ORDER_CHANGED')
    axis=[s.strip() for s in axis_path.read_text().splitlines() if s.strip() and not s.startswith('#')]
    jobs=[]
    for repo in bc['acquisition_repos']:
        rel=bench+'/'+repo.split('/',1)[1]+'.npz'
        if args.npz and rel not in args.npz:continue
        path=args.raw_root/rel
        if not path.is_file() and args.supplemental_root:path=args.supplemental_root/rel
        if not path.is_file():raise ValueError('ACQUIRED_FILE_MISSING: '+rel)
        jobs.append(({'repo':repo,'bench':bench,'path':str(path.resolve())},sem['entries'][rel],mask))
    if not jobs:raise ValueError('Empty acquisition selection')
    args.out.mkdir(parents=True)
    entries=[];rows=[];results=[]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i,(entry,row,result) in enumerate(ex.map(prepare,jobs,chunksize=4),1):
            rows.append(row)
            if result is not None:entries.append(entry);results.append(result)
            if i%500==0:print('verified and converted',i,'/',len(jobs),flush=True)
    if not args.npz and [e['repo'] for e in entries]!=bc['analysis_models']:raise ValueError('ELIGIBILITY_DIFFERS_FROM_BOUND_PANEL')
    arrays,stats=builder().assemble(entries,results,axis,coverage_min=.95,min_model_presence=0.0,bench=bench,mask_ids=tuple(mask),min_valid_cells_per_model=50)
    path=args.out/f'matrix_{bench}.npz';np.savez(path,**arrays)
    if not args.npz:check_matrix(path,c,bench)
    report={'status':'BUILT_SAMPLE_NOT_FITTABLE' if args.npz else 'BUILT_EXACT_BOUND_PANEL','bench':bench,'n_input_models':len(jobs),'n_models':len(entries),'shape':list(arrays['BIN'].shape),'contract_sha256':sha(CONTRACT_PATH),'matrix_sha256':sha(path),'array_identities':{k:array_record(v) for k,v in arrays.items()},'eligibility':rows,'stats':stats,'scope':'Executor reconstruction; scientific independent acceptance remains separate','input_mode':'Every raw array and scalar checked against public acquisition identities before masking; original byte-container identity is not required'}
    (args.out/'build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','bench','n_models','shape']}),flush=True)


def fit(args,c):
    check_matrix(args.matrix,c,args.bench)
    path=ROOT/'original_code/joint_runner.py';text=path.read_text()
    mod=types.ModuleType('portable_original_joint_runner');mod.__file__=str(path);exec(compile(text,str(path),'exec'),mod.__dict__)
    def public_identity_gate(unused,*,bench,matrix_sha256,epochs):
        if bench!=args.bench or epochs!=2500 or matrix_sha256!=sha(args.matrix):raise ValueError('PUBLIC_FIT_CONFIG_MISMATCH')
        check_matrix(args.matrix,c,bench)
        return {'validation':'public scientific matrix identity and fixed config','contract_sha256':sha(CONTRACT_PATH),'independent_acceptance_claimed':False}
    mod.check_input_contract=public_identity_gate
    args.out.mkdir(parents=True,exist_ok=False)
    pilot=args.pilot
    suffix='pilot' if pilot else 'full_2500ep'
    result=mod.run(args.bench,str(args.matrix.resolve()),str(args.out/f'joint_vectors_{args.bench}_{suffix}.npz'),str(args.out/f'joint_run_{args.bench}_{suffix}.json'),pilot=pilot,epochs=3 if pilot and args.device=='cpu' else 5 if pilot else 2500,input_contract=None,device=args.device,validate_only=args.validate_only)
    print(json.dumps(result,default=str))


def main():
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    b=sp.add_parser('build');b.add_argument('--bench',choices=['arc','hellaswag'],required=True);b.add_argument('--raw-root',type=Path,required=True);b.add_argument('--supplemental-root',type=Path);b.add_argument('--out',type=Path,required=True);b.add_argument('--workers',type=int,choices=range(1,5),default=4);b.add_argument('--npz',action='append',help='sample-only selection; output cannot pass the complete fit identity gate')
    f=sp.add_parser('fit');f.add_argument('--bench',choices=['arc','hellaswag'],required=True);f.add_argument('--matrix',type=Path,required=True);f.add_argument('--out',type=Path,required=True);f.add_argument('--device',choices=['cpu','cuda'],default='cuda');f.add_argument('--pilot',action='store_true');f.add_argument('--validate-only',action='store_true')
    a=p.parse_args();c=contract();(build if a.command=='build' else fit)(a,c)
if __name__=='__main__':main()
