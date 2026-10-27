"""Recompute expanded metrics and audit grouping, selection and provenance."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/expanded'


def main():
    meta=pd.read_csv(ROOT/'data/scored_metadata.csv')
    assert len(meta)==94 and meta.patientid.nunique()==65
    provenance=json.loads((ROOT/'data/extended_feature_provenance.json').read_text())
    for name in ['biovil','biovil_t','rad_dino']:
        path=ROOT/f'data/{name}_features.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==provenance[name]['features_sha256']
        features=np.load(path,allow_pickle=False)
        assert features['filenames'].tolist()==meta.filename.tolist()
    checked=0
    for scheme,count in [('half',50),('fivefold',5)]:
        metrics=pd.read_csv(OUT/scheme/'split_metrics.csv')
        predictions=pd.read_csv(OUT/scheme/'test_predictions.csv')
        assert len(metrics)==30*2*count
        splits=json.loads((OUT/scheme/'outer_splits.json').read_text())
        for split in splits:
            assert not set(meta.iloc[split['train_rows']].patientid)&set(meta.iloc[split['test_rows']].patientid)
        split_lookup={s['seed']:s for s in splits}
        indexed=metrics.set_index(['target','model','seed'])
        for key,frame in predictions.groupby(['target','model','seed'],sort=False):
            row=indexed.loc[key]
            assert set(frame.row)==set(split_lookup[key[2]]['test_rows'])
            expected=meta.iloc[frame.row]
            assert np.array_equal(frame.patientid.to_numpy(),expected.patientid.to_numpy())
            label='geographic_mean' if key[0]=='Geographic Extent' else 'opacity_mean'
            assert np.allclose(frame.truth,expected[label])
            for metric,value in [('MAE',mean_absolute_error(frame.truth,frame.prediction)),
                                 ('MSE',mean_squared_error(frame.truth,frame.prediction)),
                                 ('R2',r2_score(frame.truth,frame.prediction))]:
                assert np.isclose(row[metric],value,rtol=1e-8,atol=1e-8),(key,metric)
            upper=8 if key[0]=='Geographic Extent' else 6
            assert frame.prediction.between(0,upper).all()
            assert np.allclose(frame.prediction,np.clip(frame.raw_prediction,0,upper))
            checked+=1
        selections=pd.read_csv(OUT/scheme/'model_selection.csv')
        for row in selections.itertuples():
            chosen=indexed.loc[(row.target,row.chosen_model,row.seed)]
            selected=indexed.loc[(row.target,row.model,row.seed)]
            assert np.isclose(chosen.inner_MAE,row.inner_MAE)
            assert np.isclose(chosen.MAE,selected.MAE)
            pool=metrics[(metrics.target==row.target)&(metrics.seed==row.seed)&~metrics.model.str.startswith('Training')]
            if row.model=='Training selected modern':pool=pool[pool.model.str.startswith(('BioViL','RAD-DINO'))]
            elif row.model=='Training selected OLS':pool=pool[pool.model.isin(['Bounded OLS 1','Bounded OLS 4'])]
            assert np.isclose(row.inner_MAE,pool.inner_MAE.min())
        if scheme=='fivefold':
            assert predictions.groupby(['target','model']).row.nunique().eq(94).all()
            assert predictions.groupby(['target','model','patientid']).seed.nunique().eq(1).all()
    # Bounded OLS differs from the already-saved ordinary OLS only by clipping.
    initial=pd.read_csv(ROOT/'results/test_predictions.csv')
    extended=pd.read_csv(OUT/'half/test_predictions.csv')
    for dimension in [1,4,18,1024]:
        a=initial[initial.model.eq(f'Published OLS {dimension}')]
        b=extended[extended.model.eq(f'Bounded OLS {dimension}')]
        merged=a.merge(b,on=['target','seed','row'],suffixes=('_a','_b'),validate='one_to_one')
        assert len(merged)==len(a)==len(b)
        if dimension<1024:
            assert np.allclose(merged.prediction_a,merged.raw_prediction,rtol=1e-7,atol=1e-7)
        else:
            # The initial run used two BLAS threads, the extension one.
            # An independent refit of seed 15 reproduced this small difference.
            assert np.max(abs(merged.prediction_a-merged.raw_prediction))<.001
    message=f'PASS: {checked} held-out metric rows recomputed; patient separation, target alignment, bounds, training-only selection and feature hashes verified. OLS 1/4/18 raw fits match the initial run; OLS 1024 differs by at most 0.001 under the audited BLAS thread change.'
    (OUT/'validation_log.txt').write_text(message+'\n')
    print(message)


if __name__=='__main__': main()
