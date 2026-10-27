"""Expanded grouped nested benchmark, with all model choices made inside training.

The extension is exploratory: the initial 900-fit results were already known
when it was designed. Individual leaderboard winners are descriptive. The
selectors below quantify training-only choice among the declared candidates.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, platform, time, warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'expanded'
ALPHAS=[.001,.01,.1,1.,10.,100.,1000.]


class BoundedRegressor(RegressorMixin,BaseEstimator):
    """Use the known score range in training validation and final predictions."""
    def __init__(self,estimator,lower=0.,upper=8.):
        self.estimator=estimator; self.lower=lower; self.upper=upper
    def fit(self,X,y):
        self.estimator_=clone(self.estimator).fit(X,y)
        return self
    def predict_raw(self,X): return np.asarray(self.estimator_.predict(X)).ravel()
    def predict(self,X): return np.clip(self.predict_raw(X),self.lower,self.upper)


def pipe(model,scale='standard',pca=False):
    steps=[]
    if scale=='standard': steps=[('scale',StandardScaler())]
    elif scale=='variance': steps=[('variance',VarianceThreshold(1e-6)),('scale',StandardScaler())]
    elif scale=='normalize': steps=[('normalize',Normalizer())]
    if pca: steps.append(('pca',PCA(svd_solver='full')))
    return Pipeline(steps+[('model',model)])


def definitions():
    rows=[]
    def add(name,feature,estimator,grid=None,kind='alternative'):
        rows.append({'name':name,'feature':feature,'estimator':estimator,'grid':grid or {},'kind':kind})
    for dim,feature in [(1,'one'),(4,'four'),(18,'eighteen'),(1024,'deep')]:
        add(f'Bounded OLS {dim}',feature,LinearRegression(),kind='baseline')
    add('DenseNet4 Huber','four',pipe(HuberRegressor(max_iter=2000)),{'model__epsilon':[1.1,1.35,1.75],'model__alpha':[.01,1.,10.]})
    add('DenseNet18 ElasticNet','eighteen',pipe(ElasticNet(max_iter=20000,tol=1e-5)),{'model__alpha':[.01,.1,1.],'model__l1_ratio':[.1,.5,.9]})
    add('DenseNet4 kernel ridge','four',pipe(KernelRidge(kernel='rbf')),{'model__alpha':[.01,.1,1.],'model__gamma':[.01,.1,1.]})
    add('DenseNet4 extra trees','four',ExtraTreesRegressor(n_estimators=80,random_state=20261027,n_jobs=1),{'min_samples_leaf':[1,3,5],'max_features':[.5,1.]})
    add('DenseNet4 random forest','four',RandomForestRegressor(n_estimators=80,random_state=20261027,n_jobs=1),{'min_samples_leaf':[1,3,5]})
    add('DenseNet4 gradient boosting','four',HistGradientBoostingRegressor(max_iter=100,min_samples_leaf=5,random_state=20261027),{'max_leaf_nodes':[3,7],'l2_regularization':[0.,1.,10.]})
    add('DenseNet1024 unscaled ridge','deep',pipe(Ridge(solver='svd'),scale=None),{'model__alpha':ALPHAS})
    add('DenseNet1024 filtered ridge','deep',pipe(Ridge(solver='svd'),scale='variance'),{'model__alpha':ALPHAS})
    add('DenseNet1024 PCA ridge','deep',pipe(Ridge(solver='svd'),scale=None,pca=True),{'pca__n_components':[2,5,10,15],'model__alpha':[.01,1.,100.]})
    add('DenseNet1024 PLS','deep',PLSRegression(scale=False,max_iter=1000),{'n_components':[1,2,4,8]})
    add('DenseNet1024 normalized SVR','deep',pipe(SVR(kernel='rbf'),scale='normalize'),{'model__C':[1.,10.,100.],'model__epsilon':[.1,.3],'model__gamma':['scale',1.]})
    for label,prefix in [('BioViL','biovil'),('BioViL-T','biovil_t')]:
        add(f'{label} pooled ridge',prefix+'_pooled',pipe(Ridge(solver='svd'),scale=None),{'model__alpha':ALPHAS})
        add(f'{label} PCA ridge',prefix+'_pooled',pipe(Ridge(solver='svd'),scale=None,pca=True),{'pca__n_components':[2,5,10,15],'model__alpha':[.01,1.,100.]})
        add(f'{label} projected ridge',prefix+'_projected',pipe(Ridge(solver='svd'),scale=None),{'model__alpha':ALPHAS})
        add(f'{label} projected SVR',prefix+'_projected',pipe(SVR(kernel='rbf'),scale=None),{'model__C':[1.,10.,100.],'model__epsilon':[.1,.3],'model__gamma':['scale',1.]})
    add('RAD-DINO ridge','rad_dino',pipe(Ridge(solver='svd'),scale=None),{'model__alpha':ALPHAS})
    add('RAD-DINO PCA ridge','rad_dino',pipe(Ridge(solver='svd'),scale=None,pca=True),{'pca__n_components':[2,5,10,15],'model__alpha':[.01,1.,100.]})
    add('RAD-DINO normalized SVR','rad_dino',pipe(SVR(kernel='rbf'),scale='normalize'),{'model__C':[1.,10.,100.],'model__epsilon':[.1,.3],'model__gamma':['scale',1.]})
    add('RAD-DINO PLS','rad_dino',PLSRegression(scale=False,max_iter=1000),{'n_components':[1,2,4,8]})
    return rows


def protocol():
    candidates=definitions()
    return {'extension_status':'exploratory; initial benchmark outcomes known before extension',
            'n_candidates':len(candidates),'primary_metric':'image-weighted held-out MAE',
            'outer':'same 50 patient-grouped half splits as original benchmark',
            'sensitivity':'5-fold GroupKFold outer CV with every patient held out once',
            'inner':'5-fold GroupKFold on outer training patients only; fold mean MAE',
            'bounded_prediction':'all new pipelines clipped to known target range during inner scoring and test prediction; raw predictions retained',
            'baseline':'original unbounded OLS and bounded OLS both reported',
            'selectors':{'Training selected OLS':['Bounded OLS 1','Bounded OLS 4'],
                         'Training selected modern':'all BioViL, BioViL-T and RAD-DINO pipelines',
                         'Training selected all':'all declared new candidates, including baseline fallback'},
            'selection_rule':'choose smallest inner CV MAE independently for each outer split and target; never outer-test MAE',
            'candidate_specs':[{'name':s['name'],'feature':s['feature'],'kind':s['kind'],'estimator':repr(s['estimator']),'grid':s['grid']} for s in candidates],
            'variance_filter':'native training feature variance > 1e-6, selected after diagnosing the initial pipeline; exploratory sensitivity',
            'uncertainty':'split SD descriptive; conditional patient bootstrap excludes training variation and is not external validation'}


def load_data(legacy_only):
    meta=pd.read_csv(ROOT/'data/scored_metadata.csv')
    data=np.load(ROOT/'data/frozen_features.npz',allow_pickle=False)
    names=data['pathologies'].tolist()
    logits=data['logits'].astype(float)
    X={'one':logits[:,[names.index('Lung Opacity')]],
       'four':logits[:,[names.index(n) for n in ['Lung Opacity','Pneumonia','Infiltration','Consolidation']]],
       'eighteen':logits,'deep':data['features'].astype(float)}
    if not legacy_only:
        for prefix in ['biovil','biovil_t']:
            f=np.load(ROOT/f'data/{prefix}_features.npz',allow_pickle=False)
            assert f['filenames'].tolist()==meta.filename.tolist()
            for key in ['pooled','projected']: X[prefix+'_'+key]=f[key].astype(float)
        f=np.load(ROOT/'data/rad_dino_features.npz',allow_pickle=False)
        assert f['filenames'].tolist()==meta.filename.tolist()
        X['rad_dino']=f['cls'].astype(float)
    assert len(meta)==94 and meta.patientid.nunique()==65
    return meta,X


def evaluate(y,yp,raw,groups,high,upper):
    err=yp-y; mask=y>=high
    patient=pd.DataFrame({'p':groups,'ae':abs(err)}).groupby('p').ae.mean()
    return {'MAE':mean_absolute_error(y,yp),'MSE':mean_squared_error(y,yp),'R2':r2_score(y,yp),
            'Pearson':float(pearsonr(y,yp).statistic) if np.ptp(yp)>1e-12 else None,
            'Patient_MAE':patient.mean(),'Bias':err.mean(),'High_score_MAE':abs(err[mask]).mean() if mask.any() else None,
            'High_score_bias':err[mask].mean() if mask.any() else None,'Within_1':(abs(err)<=1).mean(),
            'Out_of_range':((yp<0)|(yp>upper)).mean(),'Raw_out_of_range':((raw<0)|(raw>upper)).mean()}


def outer_splits(meta,scheme):
    if scheme=='half':
        raw=json.loads((ROOT/'results/outer_splits.json').read_text())
        return [(s['seed'],np.array(s['train_rows']),np.array(s['test_rows'])) for s in raw]
    groups=meta.patientid.to_numpy()
    return [(i,a,b) for i,(a,b) in enumerate(GroupKFold(5).split(meta,groups=groups))]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--legacy-only',action='store_true')
    parser.add_argument('--scheme',choices=['half','fivefold'],default='half')
    parser.add_argument('--lock-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    spec=protocol()
    serialized=json.dumps(spec,indent=2)
    protocol_path=OUT/'protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text())==spec,'Protocol changed; use a new results directory'
    else:
        protocol_path.write_text(serialized,encoding='utf-8')
        (OUT/'protocol_lock.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'sha256':hashlib.sha256(serialized.encode()).hexdigest()},indent=2))
    if args.lock_only:
        print(f'Locked {spec["n_candidates"]} candidates before expanded outcome inspection',flush=True)
        return
    meta,matrices=load_data(args.legacy_only)
    groups=meta.patientid.to_numpy()
    specs=[s for s in definitions() if s['feature'] in matrices]
    folder=OUT/args.scheme
    folder.mkdir(exist_ok=True)
    splits=outer_splits(meta,args.scheme)
    (folder/'outer_splits.json').write_text(json.dumps([{'seed':seed,'train_rows':a.tolist(),'test_rows':b.tolist()} for seed,a,b in splits],indent=2))
    all_results=[]
    with threadpool_limits(limits=1):
        for target,label,upper,high in [('Geographic Extent','geographic_mean',8.,6.),('Opacity','opacity_mean',6.,4.)]:
            y=meta[label].to_numpy(dtype=float)
            for idx,specification in enumerate(specs):
                name=specification['name']; X=matrices[specification['feature']]
                path=folder/f'fits_{label}_{name.replace(" ","_")}.json'
                if path.exists():
                    records=json.loads(path.read_text())
                    assert len(records)==len(splits)
                    all_results.extend(records)
                    continue
                started=time.monotonic(); records=[]
                for seed,train,test in splits:
                    assert not set(groups[train]) & set(groups[test])
                    inner=list(GroupKFold(5).split(X[train],y[train],groups[train]))
                    for a,b in inner: assert not set(groups[train][a]) & set(groups[train][b])
                    estimator=BoundedRegressor(specification['estimator'],upper=upper)
                    grid={'estimator__'+k:v for k,v in specification['grid'].items()}
                    search=GridSearchCV(estimator,grid,scoring='neg_mean_absolute_error',cv=inner,n_jobs=1,error_score='raise',return_train_score=False)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        search.fit(X[train],y[train])
                    yp=search.predict(X[test]); raw=search.best_estimator_.predict_raw(X[test])
                    assert np.isfinite(yp).all() and np.isfinite(raw).all()
                    record={'target':target,'model':name,'seed':seed,'feature':specification['feature'],'n_train_images':len(train),
                            'n_test_images':len(test),'n_train_patients':len(np.unique(groups[train])),
                            'n_test_patients':len(np.unique(groups[test])),'inner_MAE':-search.best_score_,
                            'best_params':search.best_params_,'warnings':sorted(set(str(w.message) for w in caught)),
                            'test_rows':test.tolist(),'truth':y[test].tolist(),'prediction':yp.tolist(),'raw_prediction':raw.tolist()}
                    record.update(evaluate(y[test],yp,raw,groups[test],high,upper))
                    records.append(record)
                path.write_text(json.dumps(records,indent=2,allow_nan=False),encoding='utf-8')
                all_results.extend(records)
                print(f'{args.scheme} {target} {idx+1}/{len(specs)} {name}: completed {len(splits)} fits in {time.monotonic()-started:.1f}s',flush=True)
    if args.legacy_only:
        print('Legacy subset cached. Run without --legacy-only when new features are ready.',flush=True)
        return
    # Selection is evaluated on test predictions only after the chosen family
    # and hyperparameters have been determined using inner training folds.
    pools={'Training selected OLS':['Bounded OLS 1','Bounded OLS 4'],
           'Training selected modern':[s['name'] for s in specs if s['feature'].startswith(('biovil','rad_dino'))],
           'Training selected all':[s['name'] for s in specs]}
    choices=[]
    for target,label,upper,high in [('Geographic Extent','geographic_mean',8.,6.),('Opacity','opacity_mean',6.,4.)]:
        for seed,train,test in splits:
            options=[r for r in all_results if r['target']==target and r['seed']==seed]
            for selector,pool in pools.items():
                chosen=min((r for r in options if r['model'] in pool),key=lambda r:r['inner_MAE'])
                record=dict(chosen); record['model']=selector; record['chosen_model']=chosen['model']
                choices.append(record)
    all_results.extend(choices)
    metrics=[]; predictions=[]
    for r in all_results:
        metrics.append({k:v for k,v in r.items() if k not in ['test_rows','truth','prediction','raw_prediction','warnings','best_params']})
        for row,truth,pred,raw in zip(r['test_rows'],r['truth'],r['prediction'],r['raw_prediction']):
            predictions.append({'target':r['target'],'model':r['model'],'seed':r['seed'],'row':row,
                                'patientid':int(groups[row]),'filename':meta.iloc[row].filename,'truth':truth,
                                'prediction':pred,'raw_prediction':raw,'error':pred-truth})
    pd.DataFrame(metrics).to_csv(folder/'split_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(folder/'test_predictions.csv',index=False)
    pd.DataFrame([{k:r.get(k) for k in ['target','model','seed','chosen_model','inner_MAE','best_params']} for r in choices]).to_csv(folder/'model_selection.csv',index=False)
    frame=pd.DataFrame(metrics)
    columns=['MAE','MSE','R2','Pearson','Patient_MAE','Bias','High_score_MAE','High_score_bias','Within_1','Out_of_range','Raw_out_of_range']
    summary=frame.groupby(['target','model'],sort=False)[columns].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(folder/'summary.csv',index=False)
    print(summary[['MAE_mean','MAE_std','R2_mean']].to_string(),flush=True)


if __name__=='__main__': main()
