"""Public corrected-data orchestration, with archived scientific code unchanged.

Uses the accepted correction/cohort helper verbatim; only its public input paths
are rebound. No private source arrays, internal plan, or fixed old qualification
rows are used. The original candidate identities are lightweight metadata.
"""
import gzip, importlib.util, json
from functools import lru_cache
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
DATA=HERE/'corrected_inputs'
_spec=importlib.util.spec_from_file_location('public_corrected_contracts',DATA/'corrected_core_contracts.py')
ccc=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(ccc)
ccc.REFERENCE_ARRAYS=DATA/'FROZEN_CANDIDATE_AXES.npz'
ccc.STAGING_ML_SPLIT=HERE/'original_inputs/confidence_comparison/mainline_v2/split_manifest.json'
ccc.STAGING_CORE_SPLIT=HERE/'original_inputs/confidence_comparison/risk_metrics_v1/split_manifest.json'


@lru_cache(maxsize=1)
def raw_records():
    with gzip.open(DATA/'CORRECTED_RAW_IDENTITIES.json.gz','rt') as f:return json.load(f)['files']


def source_subset(cv):
    return ccc.frozen_core_subset(cv,ccc.REFERENCE_ARRAYS)


def legacy_rebind(driver,cc,rr):
    d=driver;index={(r['bench'],r['filename']):r for r in raw_records()}
    for row in rr:
        for bench,col in [('arc','arc_path'),('hellaswag','HS_path')]:
            rec=index[(bench,Path(row[col]).name)]
            row[col]=str(d.RAW/bench/Path(row[col]).name)
            row[col.replace('_path','_sha256')]=rec['output_sha256']
    arrays=ccc.load_npz(d.CV/'source_arrays.npz')
    split=ccc.runtime_split_manifest(d.OLD/'split_manifest.json',arrays,ccc.STAGING_ML_SPLIT,d.OLD/'split_manifest.json')
    d.put(d.WORK/'SPLIT_REBIND_MAINLINE.json',split)
    d.csvout(d.OLD/'cohort_manifest.csv',rr)
    cc.update(source_arrays_sha256=d.sha(d.CV/'source_arrays.npz'),cohort_manifest_sha256=d.sha(d.OLD/'cohort_manifest.csv'),Q0_nmodels=len(arrays['model_ids']),Q0_norgs=len(set(arrays['org_ids'].astype(str))),n_source_items=len(arrays['item_ids']),n_F_items=int((arrays['source_half']==0).sum()),n_C_items=int((arrays['source_half']==1).sum()),Q0_model_order_sha256=d.hashlib.sha256(('\n'.join(arrays['model_ids'].astype(str))+'\n').encode()).hexdigest(),fixed_metadata_snapshot='corrected view; original hash-rank-mod-2 rule recomputed on retained item union',metadata_exclusions_count=sum(r['eligible'].lower()!='true' for r in d.rows(d.CV/'eligibility.csv')))


def check_qualification(driver):
    d=driver;got=d.js(d.OLD/'A3_qualification.json')
    # Accepted corrected cohort is checked explicitly, rather than forced into the old 6666 rows.
    if (got['n_models'],got['n_orgs'])!=(6701,1330):raise ValueError('CORRECTED_QUALIFICATION_COUNT_MISMATCH')
    d.put(d.WORK/'QUALIFICATION_COMPARISON.json',{'mode':'corrected fresh qualification','n_models':got['n_models'],'n_orgs':got['n_orgs'],'source':'recomputed from original source fits; old qualification equality intentionally inapplicable'})


def core_rebind(driver,cc):
    d=driver;arrays=ccc.load_npz(d.CV/'source_arrays.npz')
    report=ccc.runtime_split_manifest(d.CORE/'split_manifest.json',arrays,ccc.STAGING_CORE_SPLIT,d.CORE/'split_manifest.json');d.put(d.WORK/'SPLIT_REBIND_CORE.json',report)
    fixed=ccc.recompute_fixed_rows(d.CORE,d.OLD,cc)
    split=d.js(d.CORE/'split_manifest.json')
    cc['n_models']=cc['legacy_qualification']['n_models'];cc['n_orgs']=cc['legacy_qualification']['n_orgs'];cc['n_projection_contexts']=split['n_projection_contexts'];cc['n_unique_reference_sets']=split['n_unique_reference_org_sets']
    result=ccc.verify_cohort_contract(d.CORE,d.OLD,cc,split)
    d.put(d.CORE/'CORE_COHORT_VERIFY.json',{'fixed_rows':fixed,**result})
