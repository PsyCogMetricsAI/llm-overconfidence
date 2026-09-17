"""Metadata validation and all-valid-ARC baseline construction from source arrays.
No source reload; no target risk calculation. C-half preliminary file superseded.
"""
from pathlib import Path
import collections,csv,json,hashlib
import numpy as np
from prepare_data import OUT,dump_csv
from score_targets import metrics

def main():
 scan=json.loads((OUT/'input_scan.json').read_text());contract=json.loads((OUT/'data_contract.json').read_text())
 audit={};duplicates={}
 for b in ['arc','hellaswag']:
  rows=[r for r in scan if r['bench']==b];ids=[r['model_id'] for r in rows if r['model_id']];counts=collections.Counter(ids)
  duplicates[b]={k:v for k,v in counts.items() if v>1}
  audit[b]={'n_scanned_files':len(rows),'n_unique_model_ids':len(counts),'n_valid_models_at_least_one_item':sum(r['n_valid']>0 for r in rows),'n_file_excluded':sum(r['status']!='OK' for r in rows),'n_tmp_files_excluded':sum('/'+b+'/' in x for x in contract['tmp_files_excluded'])}
  audit[b]['n_directory_files_at_scan']=len(rows)+audit[b]['n_tmp_files_excluded']
  audit[b]['file_exclusion_reasons']=dict(collections.Counter(r['reason'] for r in rows if r['status']!='OK'))
  opt=collections.Counter()
  for r in rows:opt.update(r.get('real_option_counts',{}))
  audit[b]['real_option_counts_across_rows']=dict(opt)
 if any(duplicates.values()):raise RuntimeError('duplicate model identities detected: cannot accept CV0')
 src={r['model_id'] for r in scan if r['bench']=='arc' and r['model_id']};tgt={r['model_id'] for r in scan if r['bench']=='hellaswag' and r['model_id']}
 audit['n_common_identities_before_thresholds']=len(src&tgt);audit['duplicate_identities']=duplicates
 archive=np.load(OUT/'source_arrays.npz',allow_pickle=False);z={k:archive[k] for k in archive.files};archive.close();feat=[]
 for m,mid in enumerate(z['model_ids']):
  v=z['valid'][m];c=z['c'][m,v];y=z['y'][m,v];mm,_=metrics(c,y,z['item_ids'][v]);a=float(y.mean());cc=float(c.mean())
  feat.append(dict(model_id=str(mid),org_id=str(z['org_ids'][m]),A=a,C=cc,O=cc-a,REL10=mm['Y_cal'],R50=mm['Y_sel'],n_valid_source=int(v.sum())))
 dump_csv(OUT/'source_B_features.csv',feat,['model_id','org_id','A','C','O','REL10','R50','n_valid_source'])
 (OUT/'SOURCE_C_SUPERSEDED.md').write_text('source_C_features.csv is SUPERSEDED and must not enter any predictor. Base statistics use all valid ARC F union C items to equalize source-item information. source_B_features.csv is the sole active baseline file.\n')
 contract['cohort_audit']=audit;contract['base_features']={'file':'source_B_features.csv','item_set':'ALL_VALID_ARC (F union C)','columns':['A','C','O','REL10','R50'],'n_valid_source':'audit only; not a feature','source_C_features.csv':'SUPERSEDED; not used'}
 (OUT/'data_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
 (OUT/'COHORT_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
 print(json.dumps({'status':'CV0_FINALIZED_BASE_ALL_ARC','n_models':len(feat),'n_orgs':len(set(z['org_ids'])),'cohort_audit':audit}))
if __name__=='__main__':main()
