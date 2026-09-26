from pathlib import Path
import json
import numpy as np

root=Path('.scratch/validation/d5-f40-trace')
def read(r,p,which):
    return np.fromfile(root/f'r{r}p{p}-{which}.f32',dtype=np.float32)
def bf16(x):
    bits=np.asarray(x,dtype=np.float32).view(np.uint32).copy()
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
values=np.array([np.ldexp(float(c&7),-9) if c<8 else np.ldexp(1.+float(c&7)/8,(c>>3)-7) for c in range(127)],dtype=np.float32)
def quantize(x):
    x=bf16(x).reshape(-1,128)
    scales=np.maximum(np.max(np.abs(x),axis=1),np.float32(1e-10))/np.float32(448)
    z=x/scales[:,None]
    az=np.abs(z)
    right=np.minimum(np.searchsorted(values,az),126)
    left=np.maximum(right-1,0)
    dl=az-values[left]; dr=values[right]-az
    codes=np.where((dl<dr)|((dl==dr)&((left&1)==0)),left,right)
    q=np.copysign(values[codes],z)
    return (q.astype(np.float64)*scales[:,None]).ravel()
def linear(x,n,k,seed):
    quant=quantize(x)
    row=np.arange(n)
    result=np.zeros(n,dtype=np.float64)
    for term in range(2):
        col=(row*17+seed+term*(k//2+13))%k
        w=np.array([.5,1,1.5,2])[(row+seed+term)%4]
        scale=(17+(row//128*3+col//128+seed)%11)/1024
        result+=quant[col]*w*scale
    return bf16(result)
def head(x,n,k,seed):
    x=bf16(x).reshape(64,k)
    row=np.arange(64*n)
    col=(17*row+seed)%k
    coeff=.5+(row+seed)%13/32
    return bf16(x[row//n,col].astype(np.float64)*coeff)
def stats(a,b):
    delta=np.abs(a.astype(np.float64)-b)
    return {'nonidentical':int(np.count_nonzero(delta)),
            'max_abs':float(delta.max()),
            'gate_fail':int(np.count_nonzero(delta>2e-5+.025*np.abs(b))),
            'rms_relative':float(np.sqrt(np.mean(delta**2))/max(1e-30,np.sqrt(np.mean(b.astype(np.float64)**2))))}

history=(np.cos((np.arange(1536)+31*31)*.017)*.0625+(np.arange(1536)+31)%7*.002-.006).astype(np.float32).reshape(3,512)
report=[]
for r in range(4):
    a=[read(r,p,'actual') for p in range(6)]
    e=[read(r,p,'expected') for p in range(6)]
    full=np.concatenate([history,a[0].reshape(3,512)[-1:]],axis=0)
    query=a[2].reshape(64,512).astype(np.float64)
    score=query@full.T.astype(np.float64)/16
    score=np.exp(score-score.max(axis=1,keepdims=True)); score/=score.sum(axis=1,keepdims=True)
    attended=(score@full.astype(np.float64)).astype(np.float32).ravel()
    row={'replay':r,'whole_chain':[stats(x,y) for x,y in zip(a,e)],
         'local_attention':stats(a[3],attended),
         'local_kb':stats(a[2],head(a[1],512,256,223)),
         'local_vb':stats(a[4],head(a[3],256,512,227)),
         'local_out':stats(a[5],linear(a[4],4096,16384,229)),
         'oracle_reproduction':stats(e[5],linear(e[4],4096,16384,229))}
    print(json.dumps(row))
    report.append(row)
    history=a[0].reshape(3,512)
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
