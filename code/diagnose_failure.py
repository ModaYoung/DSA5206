"""Recheck the retained Ridge 1024 failure with an independent solver."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def main():
    root=Path(__file__).resolve().parents[1]
    f=np.load(root/'data/frozen_features.npz')
    m=pd.read_csv(root/'data/scored_metadata.csv')
    split=json.loads((root/'results/outer_splits.json').read_text())[27]
    train=split['train_rows']
    tuning=pd.read_csv(root/'results/selected_hyperparameters.csv')
    alpha=float(tuning[(tuning.target=='Geographic Extent')&(tuning.model=='Ridge 1024')&(tuning.seed==27)]['model__alpha'].iloc[0])
    x=f['features'].astype(float)
    est=Pipeline([('scale',StandardScaler()),('model',Ridge(alpha=alpha,solver='cholesky'))]).fit(x[train],m.geographic_mean.to_numpy()[train])
    prediction=float(est.predict(x[[50]])[0])
    scaler=est.named_steps['scale']
    z=scaler.transform(x[[50]])[0]
    c=z*est.named_steps['model'].coef_
    j=int(np.argmax(abs(c)))
    p=pd.read_csv(root/'results/test_predictions.csv')
    stored=float(p[(p.target=='Geographic Extent')&(p.model=='Ridge 1024')&(p.seed==27)&(p.row==50)].prediction.iloc[0])
    assert abs(stored-prediction)<1e-5
    output={'seed':27,'alpha':alpha,'filename':m.iloc[50].filename,'stored_svd_prediction':stored,
            'independent_cholesky_prediction':prediction,'absolute_solver_difference':abs(stored-prediction),
            'largest_contribution_feature':j,'train_feature_sd':float(scaler.scale_[j]),
            'test_feature_value':float(x[50,j]),'test_standardized_value':float(z[j]),'contribution':float(c[j])}
    (root/'results/failure_diagnostic.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
