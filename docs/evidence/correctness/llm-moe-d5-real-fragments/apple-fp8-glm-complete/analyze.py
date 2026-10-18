from pathlib import Path
import json
import numpy as np

from pathlib import Path
import json
import numpy as np

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

root = Path(__file__).resolve().parent / 'arrays'

def read(model, r, suffix):
    return np.fromfile(root / f'{model}-r{r}{suffix}.f32', dtype=np.float32)

def bounds(x,n,k,seed):
    q=quantize(x)
    row=np.arange(n)
    total=np.zeros(n,dtype=np.float64); magnitude=total.copy()
    for term in range(2):
        col=(row*17+seed+term*(k//2+13))%k
        value=q[col]*np.array([.5,1,1.5,2])[(row+seed+term)%4]*((17+(row//128*3+col//128+seed)%11)/1024)
        total+=value; magnitude+=np.abs(value)
    u=2**-24; gamma=3*u/(1-3*u)
    return bf16(np.nextafter((total-gamma*magnitude).astype(np.float32),-np.inf)),bf16(np.nextafter((total+gamma*magnitude).astype(np.float32),np.inf))

def prepared(x,key):
    x=bf16(x).reshape(64,-1)
    output=np.zeros((64,512 if key else 256),dtype=np.float64)
    vals=np.array([.5,1,1.5,2,-.5,-1,-1.5,-2])
    for row in range(64*448):
        h=row//448; within=row%448
        if (within<192)!=key: continue
        for term in range(2):
            col=(row*17+223+term*269)%512
            coefficient=bf16(vals[(row+term*3)%8]*(17+(row//128*3+col//128+223)%11)/1024)
            if key: output[h,col]+=float(x[h,within])*float(coefficient)
            else: output[h,within-192]+=float(x[h,col])*float(coefficient)
    return bf16(output.ravel())

report=[]
for model in ['flash','official']:
    projections = [('qa',5,7,1536,4096,197),('qb',12,14,16384,1536,199),('kva',16,18,512,4096,211),('out',58,60,4096,16384,229)] if model=='flash' else [('qa',7,9,2048,6144,197),('qb',14,16,16384,2048,199),('kva',18,20,576,6144,211),('out',77,79,6144,16384,229)]
    for r in range(4):
        get=lambda v:read(model,r,'v'+str(v))
        local={}
        for label,inp,out,n,k,seed in projections:
            x=get(inp);a=get(out);lo,hi=bounds(x,n,k,seed)
            local[label]={'fma':stats(a,float_accum(x,n,k,seed,True)),
                          'separate':stats(a,float_accum(x,n,k,seed,False)),
                          'outside_f32_bound':int(np.count_nonzero((a<lo)|(a>hi)))}
        if model=='flash':
            local['head_key']=stats(get(37),head(get(34),512,256,223))
            local['head_value']=stats(get(56),head(get(53),256,512,227))
            cache=get(38).reshape(4,512);query=get(37).reshape(64,512);attention=get(48)
            norms=[('query',8,11),('latent',22,26)]
        else:
            local['head_key']=stats(get(40),prepared(get(37),True))
            local['head_value']=stats(get(75),prepared(get(72),False))
            cache=get(58).reshape(4,576);query=get(55).reshape(64,576);attention=get(68)
            norms=[('query',10,13),('latent',24,28)]
        for label,i,o in norms:
            local[label+'_norm']=stats(get(o),norm(get(i)))
            local[label+'_norm_bf16']=stats(bf16(get(o)),bf16(norm(get(i))))
        scores=query.astype(np.float64)@cache.T.astype(np.float64)/16
        scores=np.exp(scores-scores.max(axis=1,keepdims=True));scores/=scores.sum(axis=1,keepdims=True)
        expected=(scores@cache[:,:512].astype(np.float64)).astype(np.float32).ravel()
        local['attention']=stats(attention,expected)
        whole=[stats(read(model,r,f'p{p}-actual'),read(model,r,f'p{p}-expected')) for p in range(6)]
        row={'model':model,'replay':r,'local':local,'whole':whole}
        report.append(row)
        print(json.dumps(row))
(root.parent/'reproduced-analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
