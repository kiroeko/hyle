from pathlib import Path
import json
import numpy as np
source=Path('.scratch/d5-f40-first-analyze.py').read_text(encoding='utf-8').split('\nreport=[]')[0]
exec(compile(source,'first-helpers','exec'))
def activation(n,seed):
    i=np.arange(n)
    return (np.cos((i+31*seed)*.017)*.0625+(i+seed)%7*.002-.006).astype(np.float32)
past=activation(1536,31).reshape(3,512)
capture=Path('.scratch/validation/d5-f40-trace')
result=[]
for r in range(4):
    x=activation(4096,3+14*r)
    nx=norm(x)
    qa=float_accum(nx,1536,4096,197,True)
    qb=float_accum(norm(qa),16384,1536,199,True)
    kv=float_accum(nx,512,4096,211,True)
    latent=norm(kv)
    absorbed=head(qb,512,256,223)
    cache=np.concatenate([past,latent.reshape(1,512)],axis=0)
    score=absorbed.reshape(64,512).astype(np.float64)@cache.T.astype(np.float64)/16
    score=np.exp(score-score.max(axis=1,keepdims=True)); score/=score.sum(axis=1,keepdims=True)
    attended=(score@cache.astype(np.float64)).astype(np.float32).ravel()
    vb=head(attended,256,512,227)
    out=float_accum(vb,4096,16384,229,True)
    past=cache[1:]
    expected=[past.ravel(),qb,absorbed,attended,vb,out]
    actual=[np.fromfile(capture/f'r{r}p{p}-actual.f32',dtype=np.float32) for p in range(6)]
    row={'replay':r,'chain_with_f32_accum':[stats(a,e) for a,e in zip(actual,expected)]}
    result.append(row)
    print(json.dumps(row))
(root/'chain-prediction.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
