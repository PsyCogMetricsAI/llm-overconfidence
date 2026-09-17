"""Constructed tests for the statistical and nested-training contracts."""
from common import *
from modeling import fit_imputer,impute,select_package,fit_package
from predictors import select_configuration
from legacy_primitives import group_weights,group_mse
from evaluation_math import group_losses,simultaneous
import tempfile,unittest,contextlib,io
from unittest.mock import patch
class PipelineTests(unittest.TestCase):
    def test_fixed_source_rows_are_hash_pinned(self):
        with tempfile.TemporaryDirectory(prefix='risk-metrics-contract-') as td:
            root=Path(td);(root/'reviews').mkdir();plan=root/'plan.md';plan.write_text('fixed plan')
            dump(root/'protocol.json',dict(plan_path=str(plan),plan_sha256=digest(plan)))
            source=root/'fixed.csv';source.write_text('model_id,A\nm,0.5\n')
            dump(root/'E1_BUNDLE.json',dict(files={'protocol.json':digest(root/'protocol.json')},fixed_rows_sha256={'fixed.csv':digest(source)}))
            dump(root/'reviews/DATA_REVIEW.json',dict(verdict='ACCEPT',E1_bundle_sha256=digest(root/'E1_BUNDLE.json')))
            require_data(root)
            source.write_text('model_id,A\nm,0.9\n')
            with self.assertRaises(AssertionError):require_data(root)
    def test_training_imputation(self):
        X=np.array([[1,np.nan,7],[3,np.nan,7],[np.nan,np.nan,7]])
        imp=fit_imputer(X);out=impute([[99,123,np.nan],[np.nan,np.nan,7]],imp)
        np.testing.assert_array_equal(out,[[99,0,7],[2,0,7]])
        self.assertEqual(imp['median'].tolist(),[2,0,7])
    def test_group_weighted_ridge_intercept(self):
        X=np.ones((4,1));y=np.array([0.,0.,0.,1.]);orgs=['large']*3+['small']
        pred,_=fit_package(dict(family='ridge',id='r',**{'lambda':1.}),X,y,orgs,[[1],[20]])
        np.testing.assert_allclose(pred,[.5,.5],atol=1e-12)
        np.testing.assert_allclose(group_weights(orgs),[1/6,1/6,1/6,.5])
    def test_configuration_and_package_ties(self):
        c=protocol()['candidate_configurations'];loss=np.ones(17)
        self.assertEqual(c[select_configuration(c,loss)]['lambda'],10000)
        h=c[select_configuration(c,loss,'hgb')];self.assertEqual((h['max_leaf_nodes'],h['l2_regularization'],h['learning_rate']),(7,10.,.05))
        # Family tie set and global tie set need not choose the same config.
        loss[:]=10.;loss[0]=.0;loss[8]=.9e-12;loss[9]=-.5e-12
        self.assertEqual(select_configuration(c,loss,'ridge'),8)
        self.assertEqual(select_configuration(c,loss),0)
        self.assertEqual(select_package(['B0','P_A1','P_A2'],{'B0':1.,'P_A1':1.-.5e-12,'P_A2':1.}), 'B0')
        self.assertEqual(select_package(['B0','P_A1','P_A2'],{'B0':1.,'P_A1':0.,'P_A2':.5},{'B0':True,'P_A1':False,'P_A2':True}), 'P_A2')
    def test_group_loss_and_simultaneous(self):
        order,L=group_losses([0,0,1],[0,1,0],['a','a','b'])
        self.assertEqual(order.tolist(),['a','b']);np.testing.assert_array_equal(L[:,0],[.5,1.])
        d=np.column_stack([np.linspace(.01,.03,50),np.zeros(50),np.ones(50)*.01,-np.linspace(.01,.03,50)])
        fd=np.tile(d.mean(0),(5,1));r=simultaneous(d,np.ones(4)*.1,fd,n_boot=2000)
        self.assertEqual(r['results'][0]['status'],'EXPLORATORY_CANDIDATE')
        self.assertEqual(r['results'][1]['simultaneous_CI'],[0.,0.])
        self.assertEqual(r['results'][2]['reason'],'DEGENERATE_VARIANCE')
        self.assertEqual(r['results'][3]['status'],'NO_PREDEFINED_INCREMENT')
        se=np.std(d,axis=0,ddof=1)/np.sqrt(50);calc=[0,3]
        bootstrap=np.stack([d[ix].mean(0) for ix in r['bootstrap_indices']])
        crit=np.quantile(np.max(abs((bootstrap[:,calc]-d.mean(0)[calc])/se[calc]),axis=1),.95,method='linear')
        self.assertAlmostEqual(r['critical_value'],crit,places=12)
        fd[:2,0]=-1
        self.assertEqual(simultaneous(d,np.ones(4)*.1,fd,n_boot=50)['results'][0]['status'],'NO_PREDEFINED_INCREMENT')
    def test_full_outer190_and_exact_target_whitelist(self):
        import train,target_admin
        pr=protocol();names=list(dict.fromkeys(n for v in pr['feature_groups'].values() for n in v))
        with tempfile.TemporaryDirectory(prefix='risk-metrics-test-') as td:
            root=Path(td)
            for d in ['features/outer_0','predictions','estimators','logs','cost']:(root/d).mkdir(parents=True)
            dump(root/'protocol.json',pr)
            t=[];h=[]
            for orgnum in range(36):
                for i in range(2):
                    mid=f'g{orgnum:02d}/m{i}';r=dict(model_id=mid,org_id=f'g{orgnum:02d}',context='synthetic')
                    r.update({n:float((orgnum+i+1)/40) for n in names});r['A1']=float('nan');r['missing_A1']=1
                    (t if orgnum<30 else h).append(r)
            inner={f'g{i:02d}':i%4 for i in range(30)}
            split=dict(outer_org_fold={f'g{i:02d}':1 if i<30 else 0 for i in range(36)},outer={'0':dict(inner_org_fold=inner)})
            dump(root/'split_manifest.json',split);fixed={}
            datasets={'train.csv':t,'test.csv':h}
            for j in range(4):
                datasets[f'inner{j}_train.csv']=[r for r in t if inner[r['org_id']]!=j]
                datasets[f'inner{j}_valid.csv']=[r for r in t if inner[r['org_id']]==j]
            for name,rr in datasets.items():
                write_rows(root/'features/outer_0'/name,rr)
                fixed['features/outer_0/'+name]=dict(model_ids=[r['model_id'] for r in rr],org_ids=[r['org_id'] for r in rr])
            dump(root/'cohort_contract.json',dict(fixed_rows=fixed));dump(root/'features/FEATURES_READY.json',{})
            dump(root/'SOURCE_RELEASE.json',dict(verdict='ACCEPT',features_ready_sha256=digest(root/'features/FEATURES_READY.json')))
            dump(root/'CODE_RELEASE.json',{});calls=[];fit_count=[0]
            def yprovider(k,rr,context):
                req=dict(outer_fold=k,context=context,target='hellaswag',model_ids=[r['model_id'] for r in rr],allowed_orgs=sorted({r['org_id'] for r in rr}))
                mids,ident=target_admin.allowed_request(req,root)
                self.assertFalse(set(ident.values())&{r['org_id'] for r in h});calls.append(context)
                yy=np.array([(int(m[1:3])+1)/40 for m in mids]);return {'Y_sel':yy,'Y_cal':yy*.5}
            def lightweight_fit(conf,X,y,orgs,Xn):
                fit_count[0]+=1;pred=np.full(len(Xn),float(group_weights(orgs)@y))
                return pred,{'model':{'timing':{}},'synthetic':True}
            with patch.object(train,'require_design',lambda root:None),patch.object(train,'verify_features',lambda root:None),patch.object(train,'labels',yprovider),patch.object(train,'fit_package',lightweight_fit),contextlib.redirect_stdout(io.StringIO()):train.outer(0,root)
            out=rows(root/'predictions/outer_0.csv');logs=readjson(root/'predictions/outer_0_selection.json')
            self.assertEqual(len(out),12);self.assertEqual(len(out[0])-4,190);self.assertEqual(len(calls),9)
            self.assertEqual(fit_count[0],31*2*4*17+31*2*2)
            self.assertFalse(logs['source_evaluable']['A1'])
            self.assertTrue(all(x['chosen_package'] in ['B0','Bd'] for x in logs['procedures']))
            for bad in [dict(outer_fold=0,context='outer0/test',target='hellaswag',model_ids=[],allowed_orgs=[]),dict(outer_fold=0,context='outer0/final_train',target='hellaswag',model_ids=['g30/m0'],allowed_orgs=['g30'])]:
                with self.assertRaises(PermissionError):target_admin.allowed_request(bad,root)
if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests);r=unittest.TextTestRunner(verbosity=2).run(suite)
    dump(ROOT/'pipeline_selftests.json',dict(tests_run=r.testsRun,failures=len(r.failures),errors=len(r.errors),passed=r.wasSuccessful(),data='constructed only; no real target values',code_sha256={p.name:digest(p) for p in [Path(__file__),ROOT/'code/modeling.py',ROOT/'code/train.py',ROOT/'code/evaluation_math.py',ROOT/'code/target_admin.py']}))
    raise SystemExit(0 if r.wasSuccessful() else 1)
