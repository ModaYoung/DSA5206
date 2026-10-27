"""Verify the extension, preserve all outcomes, and make report figures."""
from pathlib import Path
import gzip, hashlib, json, importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'expanded'
FIG=OUT/'figures'
BLUE='#236B8E'; ORANGE='#BA6425'; GRAY='#626971'


def main():
    FIG.mkdir(exist_ok=True)
    old_metrics=pd.read_csv(ROOT/'results/split_metrics.csv')
    old_predictions=pd.read_csv(ROOT/'results/test_predictions.csv')
    new_metrics=pd.read_csv(OUT/'half/split_metrics.csv')
    new_predictions=pd.read_csv(OUT/'half/test_predictions.csv')
    metrics=pd.concat([old_metrics.assign(stage='initial'),new_metrics.assign(stage='expanded')],ignore_index=True)
    predictions=pd.concat([old_predictions.assign(stage='initial'),new_predictions.assign(stage='expanded')],ignore_index=True)
    assert len(old_metrics)==900 and len(new_metrics)==3000
    assert np.isfinite(predictions[['truth','prediction','error']]).all().all()
    assert np.allclose(predictions.prediction-predictions.truth,predictions.error)
    group=metrics.groupby(['target','model'],sort=False)
    summary=group[['MAE','MSE','R2','Pearson','Patient_MAE','Bias','High_score_MAE','High_score_bias','Within_1','Out_of_range']].agg(['mean','std'])
    summary.columns=['_'.join(x) for x in summary.columns]
    summary=summary.reset_index()
    summary.to_csv(OUT/'all_models_summary.csv',index=False)
    # Aggregate repeated out-of-sample errors within each patient. Bootstrap
    # intervals describe this saved prediction set, not a new clinical cohort.
    per_patient=predictions.assign(ae=predictions.error.abs()).groupby(['target','model','patientid']).ae.mean().unstack('model')
    rng=np.random.default_rng(20261027)
    contrasts=[]; best={}
    for target,baseline in [('Geographic Extent','Published OLS 1'),('Opacity','Published OLS 4')]:
        t=summary[summary.target==target].sort_values('MAE_mean')
        modern=t[t.model.str.startswith(('BioViL','RAD-DINO'))]
        alternatives=t[~t.model.str.startswith(('Published','Bounded','Training')) & t.model.ne('Mean baseline')]
        best[target]={'original_reference':baseline,'bounded_reference':baseline.replace('Published','Bounded'),
                      'best_modern':modern.iloc[0].model,'best_alternative':alternatives.iloc[0].model,
                      'lowest_observed_model':t.iloc[0].model}
        for reference in [baseline,baseline.replace('Published','Bounded'),'Training selected OLS']:
            base=metrics[(metrics.target==target)&(metrics.model==reference)].set_index('seed')
            for model in t.model:
                alt=metrics[(metrics.target==target)&(metrics.model==model)].set_index('seed')
                assert list(base.index)==list(alt.index)
                delta=alt.MAE-base.MAE
                pp=per_patient.loc[target]
                d=(pp[model]-pp[reference]).to_numpy()
                draws=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(axis=1)
                contrasts.append({'target':target,'reference':reference,'model':model,'MAE_difference':delta.mean(),
                                  'relative_MAE_reduction':1-alt.MAE.mean()/base.MAE.mean(),'split_win_fraction':(delta<0).mean(),
                                  'patient_error_difference':d.mean(),'conditional_bootstrap_low':np.quantile(draws,.025),
                                  'conditional_bootstrap_high':np.quantile(draws,.975)})
    pd.DataFrame(contrasts).to_csv(OUT/'paired_comparisons.csv',index=False)
    # Keep fold parameters and warnings without duplicating large text JSONs.
    fit_details=[]
    for scheme in ['half','fivefold']:
        for path in sorted((OUT/scheme).glob('fits_*.json')):
            for record in json.loads(path.read_text()):
                fit_details.append({'scheme':scheme,**{k:v for k,v in record.items() if k not in ['test_rows','truth','prediction','raw_prediction']}})
    with gzip.open(OUT/'fit_details.jsonl.gz','wt',encoding='utf-8') as stream:
        for record in fit_details: stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    pd.DataFrame([{k:r.get(k) for k in ['scheme','target','model','seed','inner_MAE','best_params','warnings']} for r in fit_details]).to_csv(OUT/'selected_hyperparameters.csv',index=False)
    # Every patient appears in exactly one fivefold test partition per model.
    five=pd.read_csv(OUT/'fivefold/test_predictions.csv')
    sensitivity=[]
    for (target,model),f in five.groupby(['target','model'],sort=False):
        assert len(f)==94 and f.row.nunique()==94
        assert f.groupby('patientid').seed.nunique().eq(1).all()
        sensitivity.append({'target':target,'model':model,'MAE':f.error.abs().mean(),'MSE':np.mean(f.error**2),
                            'R2':r2_score(f.truth,f.prediction),'Patient_MAE':f.assign(ae=f.error.abs()).groupby('patientid').ae.mean().mean(),
                            'Bias':f.error.mean()})
    pd.DataFrame(sensitivity).to_csv(OUT/'fivefold_pooled_summary.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(2,1,figsize=(8,8))
    for ax,target in zip(axes,['Geographic Extent','Opacity']):
        b=best[target]
        top=summary[(summary.target==target)&~summary.model.str.startswith(('Published','Bounded','Training'))&summary.model.ne('Mean baseline')].nsmallest(6,'MAE_mean').model.tolist()
        names=[b['original_reference'],b['bounded_reference']]+top+['Training selected modern','Training selected all']
        frame=summary[summary.target.eq(target)].set_index('model').loc[names]
        colors=[GRAY if n.startswith(('Published','Bounded')) else ORANGE if n.startswith('Training') else BLUE for n in names]
        for j,(name,row) in enumerate(frame.iterrows()):
            ax.errorbar(row.MAE_mean,j,xerr=row.MAE_std,fmt='o',color=colors[j],capsize=3)
            ax.annotate(f'{row.MAE_mean:.3f}',(row.MAE_mean+row.MAE_std+.012,j),va='center',fontsize=10)
        ax.set_yticks(range(len(names)),names); ax.invert_yaxis()
        ax.set_title(target); ax.set_xlabel('Held-out MAE  (mean and split SD)')
        ax.grid(axis='x',alpha=.18)
        ax.set_xlim(left=0,right=frame.MAE_mean.max()+frame.MAE_std.max()+.25)
    fig.suptitle('Expanded comparison on the same 50 patient splits',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(FIG/'main_model_comparison.png',bbox_inches='tight'); plt.close(fig)
    # Paired differences use the same patients in each corresponding split.
    fig,axes=plt.subplots(2,1,figsize=(8,5.5))
    for ax,target in zip(axes,['Geographic Extent','Opacity']):
        b=best[target]; reference=b['bounded_reference']
        names=[b['best_modern'],'Training selected modern','Training selected all']
        base=metrics[(metrics.target==target)&(metrics.model==reference)].set_index('seed').MAE
        values=[(metrics[(metrics.target==target)&(metrics.model==n)].set_index('seed').MAE-base).to_numpy() for n in names]
        ax.boxplot(values,vert=False,tick_labels=names,showfliers=True,patch_artist=True,boxprops={'facecolor':'#DCE9EE'})
        ax.axvline(0,color=GRAY,linewidth=1); ax.set_title(target)
        ax.set_xlabel('MAE change from '+reference+'\nNegative values favor the alternative')
        ax.grid(axis='x',alpha=.18)
    fig.tight_layout(); fig.savefig(FIG/'paired_mae_changes.png',bbox_inches='tight');plt.close(fig)
    # Averaging a scan's repeated test predictions is a visualization of this
    # resampling experiment, not the output of a single fitted deployed model.
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for ax,target,upper in zip(axes,['Geographic Extent','Opacity'],[8,6]):
        name=best[target]['best_modern']
        frame=predictions[(predictions.target==target)&(predictions.model==name)].groupby('row').agg(truth=('truth','first'),prediction=('prediction','mean'))
        ax.scatter(frame.truth,frame.prediction,s=22,alpha=.6,color=BLUE)
        ax.plot([0,upper],[0,upper],color=GRAY,linestyle='--')
        ax.set(xlabel='Expert mean score',ylabel='Mean out-of-sample prediction',title=target+'\n'+name,xlim=(-.2,upper+.2),ylim=(-.2,upper+.2))
        ax.set_aspect('equal')
    fig.tight_layout();fig.savefig(FIG/'prediction_scatter.png',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8,5.5))
    for ax,target in zip(axes,['Geographic Extent','Opacity']):
        b=best[target]
        names=[b['bounded_reference'],b['best_modern'],'Training selected modern','Training selected all']
        frame=summary[summary.target==target].set_index('model').loc[names]
        ax.barh(np.arange(len(names))-.15,frame.Bias_mean,height=.28,label='All scans',color='#92B8C9')
        ax.barh(np.arange(len(names))+.15,frame.High_score_bias_mean,height=.28,label='High score scans',color=ORANGE)
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis();ax.axvline(0,color=GRAY,lw=1)
        ax.set_title(target);ax.set_xlabel('Mean signed error  (negative = underestimation)')
        ax.legend(loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(FIG/'high_score_bias.png',bbox_inches='tight');plt.close(fig)
    selection=pd.read_csv(OUT/'half/model_selection.csv')
    fig,axes=plt.subplots(2,1,figsize=(8,6.5))
    for ax,target in zip(axes,['Geographic Extent','Opacity']):
        counts=selection[(selection.target==target)&(selection.model=='Training selected all')].chosen_model.value_counts()
        counts=counts.head(8)
        ax.barh(counts.index,counts.values,color=BLUE);ax.invert_yaxis()
        ax.set_title(target);ax.set_xlabel('Selections among 50 outer training sets')
    fig.tight_layout();fig.savefig(FIG/'selection_frequency.png',bbox_inches='tight');plt.close(fig)
    # Isolate the effect of physically bounded scores and the diagnosed scaling
    # failure rather than attributing every numerical gain to a new encoder.
    ablations=[]
    for target in ['Geographic Extent','Opacity']:
        names=['Published OLS 1','Bounded OLS 1','Published OLS 4','Bounded OLS 4','Ridge 1024',
               'DenseNet1024 unscaled ridge','DenseNet1024 filtered ridge','DenseNet1024 PCA ridge']
        ablations.append(summary[(summary.target==target)&summary.model.isin(names)])
    pd.concat(ablations).to_csv(OUT/'ablation_summary.csv',index=False)
    packages=['numpy','pandas','scipy','scikit-learn','torch','torchvision','torchxrayvision','transformers','huggingface-hub','hi-ml-multimodal','safetensors']
    (OUT/'environment.json').write_text(json.dumps({p:importlib.metadata.version(p) for p in packages},indent=2))
    verification={'n_images':94,'n_patients':65,'initial_outer_fits':900,'expanded_half_outer_fits':2700,
                  'expanded_fivefold_outer_fits':270,'selector_half_evaluations':300,'selector_fivefold_evaluations':30,
                  'total_actual_outer_fits':3870,'all_half_metric_rows':len(metrics),'best':best,
                  'protocol_sha256':hashlib.sha256((OUT/'protocol.json').read_bytes()).hexdigest(),
                  'fivefold_predictions_unique':True,'all_predictions_finite':True,
                  'model_selection_uses':'inner training CV only; individual leaderboard rankings remain exploratory'}
    (OUT/'verification.json').write_text(json.dumps(verification,indent=2))
    print(json.dumps(verification,indent=2))
    print(summary.sort_values(['target','MAE_mean'])[['target','model','MAE_mean','MAE_std']].to_string(index=False))


if __name__=='__main__': main()
