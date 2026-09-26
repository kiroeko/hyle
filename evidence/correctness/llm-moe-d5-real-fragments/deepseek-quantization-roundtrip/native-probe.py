from pathlib import Path
import ctypes as ct
import hashlib
import json
import numpy as np

ct.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
import argparse
parser=argparse.ArgumentParser(); parser.add_argument('--artifacts',type=Path,required=True)
out=parser.parse_args().artifacts
compiled = {}
driver = ct.WinDLL('nvcuda.dll')
ptr, u64, i32, u32, f32 = ct.c_void_p, ct.c_uint64, ct.c_int32, ct.c_uint32, ct.c_float
def bind(name, types):
    function = getattr(driver, name)
    function.argtypes = types
    function.restype = i32
    def call(*args):
        code = function(*args)
        if code:
            raise RuntimeError(f'{name}: CUDA driver error {code}')
    return call
init = bind('cuInit', [u32])
device_get = bind('cuDeviceGet', [ct.POINTER(i32), i32])
device_name = bind('cuDeviceGetName', [ptr, i32, i32])
device_attribute = bind('cuDeviceGetAttribute', [ct.POINTER(i32), i32, i32])
driver_version = bind('cuDriverGetVersion', [ct.POINTER(i32)])
retain = bind('cuDevicePrimaryCtxRetain', [ct.POINTER(ptr), i32])
release = bind('cuDevicePrimaryCtxRelease_v2', [i32])
set_current = bind('cuCtxSetCurrent', [ptr])
sync = bind('cuCtxSynchronize', [])
load = bind('cuModuleLoadData', [ct.POINTER(ptr), ptr])
unload = bind('cuModuleUnload', [ptr])
get_function = bind('cuModuleGetFunction', [ct.POINTER(ptr), ptr, ct.c_char_p])
alloc = bind('cuMemAlloc_v2', [ct.POINTER(u64), ct.c_size_t])
free = bind('cuMemFree_v2', [u64])
upload = bind('cuMemcpyHtoD_v2', [u64, ptr, ct.c_size_t])
download = bind('cuMemcpyDtoH_v2', [ptr, u64, ct.c_size_t])
launch = bind('cuLaunchKernel', [ptr, u32,u32,u32,u32,u32,u32,u32,ptr,ct.POINTER(ptr),ptr])
init(0)
dev, major, minor, version = i32(), i32(), i32(), i32()
device_get(ct.byref(dev), 0)
device_attribute(ct.byref(major), 75, dev)
device_attribute(ct.byref(minor), 76, dev)
driver_version(ct.byref(version))
name = ct.create_string_buffer(256)
device_name(name, len(name), dev)
assert major.value * 10 + minor.value == 89
context = ptr()
retain(ct.byref(context), dev)
set_current(context)
modules, kernels = [], {}
receipt = {'scope':'Original SGLang cubin via native Windows CUDA Driver; not HYLE ordinary-entry qualification',
           'device':name.value.decode(), 'sm':89, 'driver_version':version.value, 'cases':[],
           'first_launch_failure': {'api':'cuLaunchKernel', 'code':1,
               'cause':'Probe omitted Triton 3.8 global_scratch/profile_scratch ABI arguments',
               'correction':'Original tools/compile.py appends both pointers, null when metadata sizes are zero; math and cubin unchanged'}}
def save():
    (out / 'native-receipt.json').write_text(json.dumps(receipt,indent=2) + '\n')
receipt.pop('first_launch_failure')
receipt['scope']='Original DeepSeek V4 act_quant/fp4_quant in-place semantics; TileLang0.1.14 + CUDA13.3 AOT; native qualification only'
def bf16(x):
    bits=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
def reference(bits,group):
    x=(bits.astype(np.uint32)<<16).view(np.float32).reshape(-1,group)
    maximum,floor=(448.,1e-4) if group==64 else (6.,6*2.**-126)
    scaled=np.maximum(np.abs(x).max(axis=1),np.float32(floor))*np.float32(1/maximum)
    scales=np.exp2(np.ceil(np.log2(scaled.astype(np.float64)))).astype(np.float32)
    positive=(np.array([np.ldexp(float(c&7),-9) if c<8 else np.ldexp(1.+float(c&7)/8,(c>>3)-7) for c in range(127)],dtype=np.float32)
              if group==64 else np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32))
    v=x/scales[:,None]
    mag=np.minimum(np.abs(v),maximum)
    hi=np.searchsorted(positive,mag).clip(0,len(positive)-1)
    lo=np.maximum(hi-1,0)
    dl,du=mag-positive[lo],positive[hi]-mag
    which=np.where((dl<du)|((dl==du)&((lo&1)==0)),lo,hi)
    decoded=np.copysign(positive[which],v)
    return bf16(decoded*scales[:,None]).reshape(-1),scales
try:
    for label,group in [('act_quant_kernel',64),('fp4_quant_kernel',32)]:
        blob=(out/(label+'.cubin')).read_bytes()
        image=ct.create_string_buffer(blob);module=ptr();load(ct.byref(module),image);modules.append(module)
        function=ptr();get_function(ct.byref(function),module,(label+'__kernel').encode())
        fixtures=[]
        for rows,width in [(1,group),(1,448 if group==64 else 128),(3,448 if group==64 else 128),(33,group),(128,448 if group==64 else 128)]:
            for replay in [0,1]:
                ids=np.arange(rows*width)
                x=(np.sin((ids+replay*7)*.13)*(1+ids%19)/7).astype(np.float32)
                fixtures.append((f'ordinary-{rows}x{width}-r{replay}',bf16(x)))
        raw=np.arange(65536,dtype=np.uint16)
        finite=raw[(raw&0x7f80)!=0x7f80]
        fixtures.append(('all-finite-bf16',finite))
        # Every rounding midpoint in a block with an exact power-of-two scale.
        positive=np.array([np.ldexp(float(c&7),-9) if c<8 else np.ldexp(1.+float(c&7)/8,(c>>3)-7) for c in range(127)] if group==64 else [0,.5,1,1.5,2,3,4,6],dtype=np.float32)
        mids=(positive[:-1]+positive[1:])/2
        a=np.zeros((mids.size*2+2,group),dtype=np.float32)
        a[:mids.size,0]=mids;a[mids.size:2*mids.size,0]=-mids;a[:-2,-1]=positive[-1]
        a[-2,0]=-0.;a[-1,:]=np.float32(2.**-126)
        fixtures.append(('rounding-ties-zero-tiny',bf16(a.reshape(-1))))
        for title,bits in fixtures:
            assert bits.size%group==0
            expected,scales=reference(bits,group);groups=bits.size//group
            arrays=[np.full(groups+16,-9876.,np.float32),bits,np.full(bits.size+16,0x5a5a,np.uint16)]
            buffers=[]
            try:
                for a in arrays:
                    p=u64();alloc(ct.byref(p),a.nbytes);buffers.append(p);upload(p,a.ctypes.data,a.nbytes)
                args=[*buffers,i32(groups)]
                params=(ptr*len(args))(*(ct.cast(ct.byref(a),ptr) for a in args))
                launch(function,(groups+31)//32,1,1,128,1,1,32*group*2,None,params,None)
                sync()
                for index in [0,2]:download(arrays[index].ctypes.data,buffers[index],arrays[index].nbytes)
                mismatch=int(np.count_nonzero(arrays[2][:bits.size]!=expected))
                scale_mismatch=int(np.count_nonzero(arrays[0][:groups]!=scales))
                guard=bool(np.all(arrays[2][bits.size:]==0x5a5a) and np.all(arrays[0][groups:]==-9876.))
                row={'kernel':label,'fixture':title,'elements':int(bits.size),'mismatches':mismatch,'scale_mismatches':scale_mismatch,'guards':guard,'cubin_sha256':hashlib.sha256(blob).hexdigest()}
                if mismatch:
                    positions=np.flatnonzero(arrays[2][:bits.size]!=expected)[:12]
                    row['examples']=[{'i':int(i),'input':int(bits[i]),'actual':int(arrays[2][i]),'expected':int(expected[i])} for i in positions]
                receipt['cases'].append(row);save()
                assert mismatch==0 and scale_mismatch==0 and guard,row
            finally:
                for p in buffers:free(p)
    receipt['verdict']='PASS_NATIVE_QUANTIZATION_ROUNDTRIP_ONLY';save()
    print(len(receipt['cases']),'native cases PASS; all finite BF16 values, ties, tails and guards.')
finally:
    for m in modules:unload(m)
    set_current(None);release(dev)
