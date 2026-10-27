"""Reproduce the small OLS 1024 refit difference due to BLAS threads."""
from pathlib import Path
import json,numpy as np,pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

def main():
    root=Path(__file__).resolve().parents[1]
    meta=pd.read_csv(root/'data/scored_metadata.csv')
    X=np.load(root/'data/frozen_features.npz')['features'].astype(float)
    split=json.loads((root/'results/outer_splits.json').read_text())[15]
    train,test=split['train_rows'],split['test_rows']
    old=pd.read_csv(root/'results/test_predictions.csv')
    new=pd.read_csv(root/'results/expanded/half/test_predictions.csv')
    a=old.query("target == 'Geographic Extent' and model == 'Published OLS 1024' and seed == 15").set_index('row').loc[test].prediction.to_numpy()
    b=new.query("target == 'Geographic Extent' and model == 'Bounded OLS 1024' and seed == 15").set_index('row').loc[test].raw_prediction.to_numpy()
    result={'seed':15,'target':'Geographic Extent','diagnostic_refits_not_in_benchmark_count':2,'comparisons':[]}
    for threads in [1,2]:
        with threadpool_limits(limits=threads):
            yp=LinearRegression().fit(X[train],meta.geographic_mean.to_numpy()[train]).predict(X[test])
        result['comparisons'].append({'BLAS_threads':threads,'max_difference_from_initial':float(max(abs(yp-a))),
                                     'max_difference_from_extension':float(max(abs(yp-b)))})
    (root/'results/expanded/ols_thread_sensitivity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
