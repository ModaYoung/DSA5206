"""Run patient-grouped nested comparisons from the packaged frozen features."""
from pathlib import Path
import json, os, platform, warnings, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
ALPHAS=[0.01,0.1,1.0,10.0,100.0,1000.0,10000.0]
FEATURES4=['Lung Opacity','Pneumonia','Infiltration','Consolidation']

def main():
    RESULTS.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(ROOT/'data/scored_metadata.csv')
    loaded=np.load(ROOT/'data/frozen_features.npz',allow_pickle=False)
    logits=loaded['logits'].astype(np.float64)
    features=loaded['features'].astype(np.float64)
    names=loaded['pathologies'].tolist()
    matrices={
        'mean':np.zeros((len(meta),1)),
        'one':logits[:,[names.index('Lung Opacity')]],
        'four':logits[:,[names.index(s) for s in FEATURES4]],
        'eighteen':logits,
        'deep':features,
    }
    models=[('Mean baseline','mean','mean'),('Published OLS 1','one','ols'),
            ('Published OLS 4','four','ols'),('Published OLS 18','eighteen','ols'),
            ('Published OLS 1024','deep','ols'),('Ridge 4','four','ridge'),
            ('Ridge 18','eighteen','ridge'),('Ridge 1024','deep','ridge'),
            ('RBF SVR 4','four','svr')]
    groups=meta.patientid.to_numpy()
    assert len(meta)==94 and len(np.unique(groups))==65
    splits=[next(GroupShuffleSplit(train_size=.5,test_size=.5,random_state=s).split(logits,groups=groups)) for s in range(50)]
    split_records=[]
    for seed,(tr,te) in enumerate(splits):
        assert not set(groups[tr]) & set(groups[te])
        split_records.append({'seed':seed,'train_rows':tr.tolist(),'test_rows':te.tolist(),
                              'train_patients':np.unique(groups[tr]).tolist(),'test_patients':np.unique(groups[te]).tolist()})
    (RESULTS/'outer_splits.json').write_text(json.dumps(split_records,indent=2),encoding='utf-8')
    protocol={'primary_endpoint':'image-weighted held-out MAE','primary_alternative':'Ridge 1024',
              'secondary_alternatives':['Ridge 4','Ridge 18','RBF SVR 4'],
              'outer':'50 GroupShuffleSplit seeds 0 through 49; 50 percent of patient groups for test',
              'inner':'5-fold GroupKFold on outer training patients only; select by MAE',
              'ridge_alphas':ALPHAS,'svr_grid':{'C':[.1,1,10,100],'epsilon':[.1,.3],'gamma':['scale',.1]},
              'preprocessing':'StandardScaler fitted in each inner training fold for Ridge and SVR',
              'prediction_clipping':False,'high_score_definition':{'geographic_mean':6,'opacity_mean':4},
              'high_score_note':'Exploratory imaging-score strata, not validated clinical severity categories',
              'inference':'Repeated splits are dependent; SD is not a confidence interval. Patient-bootstrap intervals condition on stored out-of-sample predictions.'}
    (RESULTS/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[]; preds=[]; tuning=[]
    with threadpool_limits(limits=2):
        for target,label,max_score,high in [('Geographic Extent','geographic_mean',8,6),('Opacity','opacity_mean',6,4)]:
            y=meta[label].to_numpy(dtype=float)
            for model_name,matrix,kind in models:
                X=matrices[matrix]
                for seed,(train,test) in enumerate(splits):
                    if kind=='mean':
                        est=DummyRegressor(strategy='mean')
                    elif kind=='ols':
                        est=LinearRegression()
                    else:
                        predictor=Ridge(solver='svd') if kind=='ridge' else SVR(kernel='rbf')
                        pipeline=Pipeline([('scale',StandardScaler()),('model',predictor)])
                        params={'model__alpha':ALPHAS} if kind=='ridge' else {'model__C':[.1,1,10,100],'model__epsilon':[.1,.3],'model__gamma':['scale',.1]}
                        # Materialize and verify grouped inner folds before fitting.
                        inner=list(GroupKFold(n_splits=5).split(X[train],y[train],groups[train]))
                        for a,b in inner:
                            assert not set(groups[train][a]) & set(groups[train][b])
                        est=GridSearchCV(pipeline,params,scoring='neg_mean_absolute_error',cv=inner,n_jobs=1,error_score='raise')
                    est.fit(X[train],y[train])
                    yp=est.predict(X[test])
                    assert len(yp)==len(test) and np.isfinite(yp).all()
                    err=yp-y[test]
                    highmask=y[test]>=high
                    patient_errors=pd.DataFrame({'p':groups[test],'ae':abs(err)}).groupby('p').ae.mean()
                    r=float(pearsonr(yp,y[test]).statistic) if np.ptp(yp)>1e-12 else np.nan
                    rows.append({'target':target,'model':model_name,'seed':seed,'n_train_images':len(train),'n_test_images':len(test),
                                 'n_train_patients':len(np.unique(groups[train])),'n_test_patients':len(np.unique(groups[test])),
                                 'MAE':mean_absolute_error(y[test],yp),'MSE':mean_squared_error(y[test],yp),'R2':r2_score(y[test],yp),'Pearson':r,
                                 'Patient_MAE':patient_errors.mean(),'Bias':err.mean(),'High_score_MAE':abs(err[highmask]).mean(),
                                 'High_score_bias':err[highmask].mean(),'Within_1':(abs(err)<=1).mean(),
                                 'Out_of_range':((yp<0)|(yp>max_score)).mean()})
                    for index,prediction in zip(test,yp):
                        preds.append({'target':target,'model':model_name,'seed':seed,'row':int(index),'filename':meta.iloc[index].filename,
                                      'patientid':int(groups[index]),'truth':y[index],'prediction':prediction,'error':prediction-y[index]})
                    if kind in ['ridge','svr']:
                        tuning.append({'target':target,'model':model_name,'seed':seed,'best_inner_MAE':-est.best_score_,**est.best_params_})
                print(f'Finished {target} / {model_name}',flush=True)
    metrics=pd.DataFrame(rows)
    predictions=pd.DataFrame(preds)
    metrics.to_csv(RESULTS/'split_metrics.csv',index=False)
    predictions.to_csv(RESULTS/'test_predictions.csv',index=False)
    pd.DataFrame(tuning).to_csv(RESULTS/'selected_hyperparameters.csv',index=False)
    summary=metrics.groupby(['target','model'],sort=False)[['MAE','MSE','R2','Pearson','Patient_MAE','Bias','High_score_MAE','High_score_bias','Within_1','Out_of_range']].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary=summary.reset_index()
    summary.to_csv(RESULTS/'summary.csv',index=False)
    # Each patient receives equal weight after averaging absolute errors over
    # their images and repeated out-of-sample evaluations. This is descriptive
    # uncertainty conditional on stored predictions, not a full training bootstrap.
    ae=predictions.assign(ae=predictions.error.abs()).groupby(['target','model','patientid'],sort=False).ae.mean().unstack('model')
    rng=np.random.default_rng(20261027)
    contrasts=[]
    for target in ['Geographic Extent','Opacity']:
        t=ae.loc[target]
        for baseline in ['Published OLS 1','Published OLS 4','Published OLS 1024']:
            for alternative in ['Ridge 4','Ridge 18','Ridge 1024','RBF SVR 4']:
                d=(t[alternative]-t[baseline]).to_numpy()
                bootstrap=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
                split_base=metrics[(metrics.target==target)&(metrics.model==baseline)].set_index('seed')
                split_alt=metrics[(metrics.target==target)&(metrics.model==alternative)].set_index('seed')
                split_delta=split_alt.MAE-split_base.MAE
                contrasts.append({'target':target,'baseline':baseline,'alternative':alternative,
                                  'image_MAE_difference':split_delta.mean(),'split_win_fraction':(split_delta<0).mean(),
                                  'patient_MAE_difference':d.mean(),'conditional_bootstrap_low':np.quantile(bootstrap,.025),
                                  'conditional_bootstrap_high':np.quantile(bootstrap,.975),'n_patients':len(d)})
    pd.DataFrame(contrasts).to_csv(RESULTS/'paired_comparisons.csv',index=False)
    pkgs=['numpy','pandas','scipy','scikit-learn','torch','torchvision','torchxrayvision','scikit-image','imageio','matplotlib']
    env={'python':platform.python_version(),'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in pkgs}}
    (RESULTS/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
    print(summary[['target','model','MAE_mean','MAE_std','R2_mean','Pearson_mean']].to_string(index=False),flush=True)

if __name__=='__main__':
    main()
