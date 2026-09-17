"""Post-CV0 contract checks: source data only; never computes target outcomes."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
from prepare_data import OUT,sha_order

def main():
 z=np.load(OUT/'source_arrays.npz',allow_pickle=False)
 m,i=z['ell'].shape;v=z['valid'];y=z['y'];c=z['c'];ell=z['ell'];half=z['source_half']
 assert v.shape==y.shape==c.shape==ell.shape
 assert len(z['model_ids'])==len(z['org_ids'])==m and len(z['item_ids'])==i
 assert np.isfinite(c[v]).all() and np.isfinite(ell[v]).all()
 assert np.isnan(c[~v]).all() and np.isnan(ell[~v]).all()
 assert ((c[v]>=0)&(c[v]<=1)).all()
 assert np.all((v&(half==0)).sum(1)>=100) and np.all((v&(half==1)).sum(1)>=100)
 expected=sha_order(z['item_ids'],2);assert np.array_equal(half,[expected[x] for x in z['item_ids']])
 with (OUT/'source_B_features.csv').open() as f:features={r['model_id']:r for r in csv.DictReader(f)}
 maxerr=0.
 for j,mid in enumerate(z['model_ids']):
  row=features[mid];vv=v[j];a=float(y[j,vv].mean());cc=float(c[j,vv].mean())
  for k,val in [('A',a),('C',cc),('O',cc-a)]:maxerr=max(maxerr,abs(float(row[k])-val))
  assert row['org_id']==z['org_ids'][j] and int(row['n_valid_source'])==vv.sum()
 assert maxerr<=1e-12
 split=sha_order(z['org_ids'],5);groups=set(z['org_ids']);ng_train=[sum(f!=h for f in split.values()) for h in range(5)]
 result={'status':'PASS','source_shape':[m,i],'n_orgs':len(groups),'source_mean_feature_max_abs_error':maxerr,'n_outer_training_orgs':ng_train,'sample_thresholds_pass':m>=300 and len(groups)>=30 and min(ng_train)>=20,'source_arrays_sha256':hashlib.sha256((OUT/'source_arrays.npz').read_bytes()).hexdigest(),'data_contract_sha256':hashlib.sha256((OUT/'data_contract.json').read_bytes()).hexdigest(),'target_values_accessed':False}
 (OUT/'SOURCE_CONTRACT_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
