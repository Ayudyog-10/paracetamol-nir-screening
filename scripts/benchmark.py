"""Benchmark models with random-scan vs sample-grouped CV.

Example: python scripts/benchmark.py c09 C09 label_claim_pct random,grouped SNV,SNV+SG1
"""
import sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, warnings, json, os; warnings.filterwarnings('ignore')
from paracetamol_nir.preprocess import apply
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, train_test_split
A=pd.read_parquet(os.environ.get('MASTER', 'data/paracetamol_nir_master.parquet'))
ac=[c for c in A if c.startswith('A_')]; wl=np.array([float(c[2:]) for c in ac])
M={
 **{f'PLSR-{k}':(lambda k=k: PLSRegression(n_components=k,scale=False)) for k in (4,6,8,10,12,15)},
 'Ridge':lambda: make_pipeline(StandardScaler(),Ridge(alpha=10)),
 'SVR':lambda: make_pipeline(StandardScaler(),SVR(C=100,gamma='scale',epsilon=0.5)),
 'KNN-3':lambda: KNeighborsRegressor(n_neighbors=3,weights='distance'),
 'RF':lambda: RandomForestRegressor(n_estimators=100,max_features='sqrt',n_jobs=-1,random_state=0),
 'MLP':lambda: make_pipeline(StandardScaler(),PCA(n_components=30),MLPRegressor(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=800,random_state=0)),
}
def met(y,p):
    r=y-p; rm=np.sqrt(np.mean(r**2)); return dict(R2=1-np.sum(r**2)/np.sum((y-y.mean())**2),RMSE=rm,RPD=np.std(y,ddof=1)/rm,Bias=np.mean(p-y))
def cv(D,target,pre,scheme,name,k=5):
    X=apply(pre,D[ac].values); y=D[target].values; g=D.sample_id.values
    if scheme=='random':
        tr,te=train_test_split(np.arange(len(y)),test_size=0.2,random_state=42)
        m=M[name](); m.fit(X[tr],y[tr]); p=np.ravel(m.predict(X[te])); return y[te],p,g[te]
    p=np.zeros(len(y))
    for tr,te in GroupKFold(k).split(X,y,g):
        m=M[name](); m.fit(X[tr],y[tr]); p[te]=np.ravel(m.predict(X[te]))
    return y,p,g
if __name__=='__main__':
    tag,camp,target=sys.argv[1],sys.argv[2],sys.argv[3]; schemes=sys.argv[4].split(','); pres=sys.argv[5].split(',')
    D=A[A.campaign.str.match(camp)]
    out=f'reports/tables/benchmark_{tag}.csv'
    done=pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=['pre','scheme','model'])
    for pre in pres:
      for sch in schemes:
        for name in M:
          if ((done.pre==pre)&(done.scheme==sch)&(done.model==name)).any(): continue
          y,p,g=cv(D,target,pre,sch,name)
          d=pd.DataFrame(dict(y=y,p=p,g=g)).groupby('g').mean()
          row=dict(pre=pre,scheme=sch,model=name,**{f'scan_{k}':v for k,v in met(y,p).items()},**{f'sample_{k}':v for k,v in met(d.y.values,d.p.values).items()})
          done=pd.concat([done,pd.DataFrame([row])]); done.to_csv(out,index=False); print(pre,sch,name,round(row['sample_R2'],3),round(row['sample_RMSE'],2),flush=True)
