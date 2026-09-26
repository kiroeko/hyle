from pathlib import Path
import json
import numpy as np

helpers=Path('.scratch/d5-f40-trace-analyze.py').read_text(encoding='utf-8').split('\nhistory=')[0]
exec(compile(helpers,'trace-helpers','exec'))
root=Path('.scratch/validation/d5-f40-first-split')
def get(r,v): return np.fromfile(root/f'r{r}v{v}.f32',dtype=np.float32)
def norm(x):
    scale=(.875+(np.arange(x.size)*13+5)%29/128).astype(np.float32)
    return (x.astype(np.float64)/np.sqrt(np.mean(x.astype(np.float64)**2)+float(np.float32(1e-5)))*scale).astype(np.float32)
def float_accum(x,n,k,seed,fma):
    rounded=bf16(x).reshape(-1,128)
    scales=np.maximum(np.max(np.abs(rounded),axis=1),np.float32(1e-10))/np.float32(448)
    codes=quantize(x).reshape(-1,128)/scales[:,None]
    codes=codes.ravel()
    row=np.arange(n)
    cols=[(row*17+seed+term*(k//2+13))%k for term in range(2)]
    weights=[np.array([.5,1,1.5,2])[(row+seed+term)%4] for term in range(2)]
    result=np.zeros(n,dtype=np.float32)
    for block in range(k//128):
        dot=np.zeros(n,dtype=np.float64)
        for col,w in zip(cols,weights): dot+=np.where(col//128==block,codes[col]*w,0)
        scaled=(dot*scales[block]).astype(np.float32)
        weight_scale=(17+(row//128*3+block+seed)%11)/1024
        if fma: result=(scaled.astype(np.float64)*weight_scale+result.astype(np.float64)).astype(np.float32)
        else: result+=(scaled.astype(np.float64)*weight_scale).astype(np.float32)
    return bf16(result)

report=[]
for r in range(4):
    index=np.arange(4096)
    seed=3+14*r
    x=(np.cos((index+31*seed)*.017)*.0625+(index+seed)%7*.002-.006).astype(np.float32)
    local={}
    for label,inp,out,n,k,seed in [('qa',5,7,1536,4096,197),('qb',12,14,16384,1536,199),('kva',16,18,512,4096,211),('out',58,60,4096,16384,229)]:
        a=get(r,out)
        expected=linear(get(r,inp),n,k,seed)
        local[label]={'double':stats(a,expected),'f32_fma':stats(a,float_accum(get(r,inp),n,k,seed,True)),
                      'f32_separate':stats(a,float_accum(get(r,inp),n,k,seed,False)),
                      'different_indices':np.flatnonzero(a!=expected).tolist()[:30]}
    row={'replay':r,'input_norm':stats(get(r,4),norm(x)),
         'input_norm_bf16':stats(get(r,5),bf16(norm(x))),
         'qa_norm_local':stats(get(r,11),norm(get(r,8))),
         'kv_norm_local':stats(get(r,26),norm(get(r,19))),
         'local_projections':local}
    print(json.dumps(row))
    report.append(row)
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
