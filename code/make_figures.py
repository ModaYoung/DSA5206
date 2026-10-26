from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
MODELS=['Mean baseline','Published OLS 1','Published OLS 4','Published OLS 18','Published OLS 1024','Ridge 4','Ridge 18','Ridge 1024','RBF SVR 4']
def main():
    r=ROOT/'results'
    s=pd.read_csv(r/'summary.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(11,5.5),sharey=True,layout='constrained')
    colors=['#aaa','#215c87','#215c87','#7191aa','#7191aa','#c07732','#c07732','#a34235','#674f8d']
    for ax,target in zip(axes,['Geographic Extent','Opacity']):
        d=s[s.target==target].set_index('model').loc[MODELS]
        y=np.arange(len(d))
        for i,(model,row) in enumerate(d.iterrows()):
            ax.errorbar(row.MAE_mean,i,xerr=row.MAE_std,fmt='o',color=colors[i],capsize=3,markersize=6)
            ax.annotate(f'{row.MAE_mean:.3f}',(row.MAE_mean,i),xytext=(0,9),textcoords='offset points',fontsize=8,ha='center')
        ax.set_title(target,fontweight='bold')
        ax.set_xlabel('Held-out MAE  |  lower is better')
        ax.set_yticks(y,MODELS)
        ax.grid(axis='x',alpha=.2)
        ax.set_xlim(left=0)
    axes[0].invert_yaxis()
    fig.suptitle('Alternatives did not beat the strongest published low-dimensional model',fontsize=12,fontweight='bold')
    fig.savefig(r/'model_comparison.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    m=pd.read_csv(r/'split_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
    selected=['Published OLS 1','Published OLS 4','Ridge 4','Ridge 18','Ridge 1024','RBF SVR 4']
    for ax,target in zip(axes,['Geographic Extent','Opacity']):
        d=s[s.target==target].set_index('model').loc[selected]
        x=np.arange(len(d))
        ax.bar(x,d.High_score_bias_mean,color=['#215c87','#397497','#c07732','#ce935a','#a34235','#674f8d'])
        ax.axhline(0,color='black',linewidth=.8)
        ax.set_xticks(x,[x.replace('Published ','') for x in selected],rotation=35,ha='right')
        ax.set_ylabel('Mean prediction minus expert score')
        ax.set_title(target+('  |  score >= 6' if target=='Geographic Extent' else '  |  score >= 4'))
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('High-score images were systematically underestimated',fontsize=12,fontweight='bold')
    fig.savefig(r/'high_score_bias.png',dpi=220,bbox_inches='tight')
    plt.close(fig)
    # Audit completeness and compute machine-readable reporting facts.
    p=pd.read_csv(r/'test_predictions.csv')
    assert len(m)==900
    assert m.groupby(['target','model']).size().eq(50).all()
    assert p.groupby(['target','model']).row.nunique().eq(94).all()
    assert not m[['MAE','MSE','R2']].isna().any().any()
    print('Verified 900 outer fits, 50 splits per model and target, and OOS coverage of all 94 images.')

if __name__=='__main__':main()
