"""Group cross-fit temperature scaling; report quality without tuning on test."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedGroupKFold
from src.straw_global_local.deployment import softmax

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'artifacts/b1_deployment'


def nll(z,y,t):
    values=z/t
    return float(np.mean(logsumexp(values,axis=1)-values[np.arange(len(y)),y]))


def fit(z,y):
    result=minimize_scalar(lambda logt:nll(z,y,np.exp(logt)),bounds=(-3,3),method='bounded')
    return float(np.exp(result.x))


def quality(prob,y):
    predicted=prob.argmax(1);confidence=prob.max(1);correct=predicted==y
    ece=0.
    for low in np.linspace(0,.9,10):
        selected=(confidence>=low)&(confidence<(low+.1) if low<.9 else confidence<=1.)
        if selected.any(): ece+=float(selected.mean()*abs(confidence[selected].mean()-correct[selected].mean()))
    return {'nll':float(-np.log(np.maximum(prob[np.arange(len(y)),y],1e-300)).mean()),
        'brier':float(np.square(prob-np.eye(5)[y]).sum(1).mean()),'ece_10_equal_width':ece}


def main():
    saved=np.load(OUT/'torch_fp32_jpeg512/val_predictions.npz')
    frame=pd.read_csv(ROOT/'artifacts/straw_global_local/data_v1/val.csv').set_index('sample_id').loc[saved['sample_ids']]
    z=saved['logits'].astype(np.float64); y=saved['labels']; groups=frame.group_id.to_numpy()
    folds=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026)
    oof=np.zeros_like(z);records=[]
    for fold,(train,holdout) in enumerate(folds.split(z,y,groups)):
        assert not set(groups[train])&set(groups[holdout])
        temperature=fit(z[train],y[train]);oof[holdout]=softmax(z[holdout],temperature)
        records.append({'fold':fold,'temperature':temperature,'n_fit':len(train),'n_holdout':len(holdout),
            'holdout_ids':saved['sample_ids'][holdout].tolist()})
    baseline=quality(softmax(z),y);crossfit=quality(oof,y)
    report={'candidate':'torch_fp32_jpeg512','role':'validation_group_crossfit','n':len(y),'folds':records,
        'before':baseline,'crossfit':crossfit,'full_validation_fit_temperature':fit(z,y),
        'enable_suggested':crossfit['nll']<baseline['nll'] and crossfit['brier']<baseline['brier'],
        'note':'Exploratory cross-fitting on 300 images. No test fit. Positive temperature does not alter top-1 class. Not applied automatically.'}
    (OUT/'temperature_crossfit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['before','crossfit','full_validation_fit_temperature','enable_suggested']},indent=2))


if __name__=='__main__': main()
