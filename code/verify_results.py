"""Validate cohort, split separation and recorded performance without refitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def main():
    root=Path(__file__).resolve().parents[1]
    meta=pd.read_csv(root/'data/scored_metadata.csv')
    provenance=json.loads((root/'data/provenance.json').read_text())
    assert hashlib.sha256((root/'data/frozen_features.npz').read_bytes()).hexdigest()==provenance['feature_npz_sha256']
    assert len(meta)==94 and meta.patientid.nunique()==65 and meta.filename.is_unique
    splits=json.loads((root/'results/outer_splits.json').read_text())
    metrics=pd.read_csv(root/'results/split_metrics.csv')
    predictions=pd.read_csv(root/'results/test_predictions.csv')
    assert len(metrics)==900
    for split in splits:
        assert not set(split['train_patients']) & set(split['test_patients'])
    for (target,model,seed),p in predictions.groupby(['target','model','seed']):
        r=metrics[(metrics.target==target)&(metrics.model==model)&(metrics.seed==seed)].iloc[0]
        assert set(p.row)==set(splits[int(seed)]['test_rows'])
        assert np.isclose(mean_absolute_error(p.truth,p.prediction),r.MAE)
        assert np.isclose(mean_squared_error(p.truth,p.prediction),r.MSE)
        assert np.isclose(r2_score(p.truth,p.prediction),r.R2)
    print('PASS: feature checksum; 94 images and 65 patients; no train/test patient overlap; all 900 recorded MAE, MSE and R2 values match predictions.')

if __name__=='__main__':main()
