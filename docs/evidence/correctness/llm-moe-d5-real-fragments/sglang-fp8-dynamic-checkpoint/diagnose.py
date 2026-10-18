from pathlib import Path
import ctypes as ct
import hashlib
import json
import numpy as np

ct.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
here = Path(__file__).resolve().parent
root = here.parents[3]
out = root / '.scratch/validation/d5-f38-diagnosis-reproduce'
out.mkdir(parents=True, exist_ok=True)
compiled = json.loads((here / 'compilation.json').read_text())
import re
header = (root/'third_party/sglang_activation/fp8/sglang_fp8_dynamic_sm89.h').read_text()
names = {'quantize_bf16':'kSglangFp8QuantizeBf16',
         'quantize_bf16_pow2':'kSglangFp8QuantizeBf16Pow2', 'matmul_bf16':'kSglangFp8MatmulBf16'}
for label, symbol in names.items():
    if label == 'quantize_bf16':
        # Preserve the failed artifact after removing it from the product.
        (out/(label+'.cubin')).write_bytes((here/'failed-quantize-bf16.cubin').read_bytes())
        continue
    body = header.split(symbol+'[] = {',1)[1].split('};',1)[0]
    blob = bytes(int(x,16) for x in re.findall(r'0x([0-9a-f]{2})',body))
    (out/(label+'.cubin')).write_bytes(blob)
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
def bf16(value):
    raw = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16).astype(np.uint16)
def from_bf16(value):
    return (np.asarray(value, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
positive = np.zeros(127, dtype=np.float32)
for code in range(127):
    exponent, mantissa = code >> 3, code & 7
    positive[code] = np.ldexp(float(mantissa), -9) if exponent == 0 else np.ldexp(1.0+mantissa/8.0, exponent-7)
decode = np.concatenate([positive,[np.nan],-positive,[np.nan]]).astype(np.float32)
def encode(values):
    mag = np.minimum(np.abs(values),448.0)
    upper = np.searchsorted(positive,mag,side='left').clip(0,126)
    lower = np.maximum(upper-1,0)
    dl,du = mag-positive[lower],positive[upper]-mag
    chosen = np.where((dl<du)|((dl==du)&((lower&1)==0)),lower,upper)
    return (chosen|(np.signbit(values).astype(np.int64)<<7)).astype(np.uint8)

results = []
dump = here
try:
    for kernel in compiled['kernels']:
        image = ct.create_string_buffer((out/(kernel['label']+'.cubin')).read_bytes())
        module = ptr(); load(ct.byref(module),image); modules.append(module)
        function = ptr(); get_function(ct.byref(function),module,kernel['metadata']['name'].encode())
        kernels[kernel['label']] = (function,kernel['metadata']['shared'],kernel['metadata']['num_warps']*32)
    for m in [1,3]:
        for replay in [0,1]:
            ports = [np.fromfile(dump/f'ffn-{m}-{replay}-{port}.bin',dtype=np.float32).reshape(2,m,-1) for port in range(3)]
            values = ports[1][0].copy()
            k,n = 2048,6144
            buffers = []
            def buffer(a):
                a = np.ascontiguousarray(a); p=u64(); alloc(ct.byref(p),a.nbytes)
                buffers.append(p); upload(p,a.ctypes.data,a.nbytes); return p
            def run(label,grid,args):
                args = [*args,u64(0),u64(0)]
                params = (ptr*len(args))(*(ct.cast(ct.byref(x),ptr) for x in args))
                function,shared,threads=kernels[label]
                launch(function,grid,1,1,threads,1,1,shared,None,params,None); sync()
            try:
                qa=np.zeros((m,k),dtype=np.uint8); scales=np.zeros((m,k//128),dtype=np.float32)
                da,dqa,ds=buffer(bf16(values)),buffer(qa),buffer(scales)
                run('quantize_bf16',m*(k//128),[da,dqa,ds,i32(128),i32(128),f32(1e-10),f32(-448),f32(448)])
                download(qa.ctypes.data,dqa,qa.nbytes); download(scales.ctypes.data,ds,scales.nbytes)
                expected_scale=np.maximum(np.max(np.abs(values.reshape(m,k//128,128)),axis=2),np.float32(1e-10))/np.float32(448)
                scaled_divide=values/np.repeat(expected_scale,128,axis=1)
                scaled_multiply=values*np.repeat(np.float32(1)/expected_scale,128,axis=1)
                half=scaled_multiply.astype(np.float16)
                halfbits=half.view(np.uint16).copy()
                halfbits[np.abs(half.astype(np.float32))>np.abs(scaled_multiply)]-=1
                half_rtz=halfbits.view(np.float16).astype(np.float32)
                divide_codes=encode(scaled_divide); multiply_codes=encode(scaled_multiply); half_codes=encode(half_rtz)
                rows=np.arange(n); terms=np.arange(2); col=(rows[:,None]*17+31+terms*(k//2+13))%k
                wc=decode[np.array([0x30,0x38,0x3c,0x40,0xb0,0xb8,0xbc,0xc0],np.uint8)[(rows[:,None]+terms*3)%8]]
                ws=(17+(rows[:,None]//128*3+col//128+13)%11).astype(np.float32)/np.float32(1024)
                expected=np.zeros((m,n),np.float64)
                for r in range(m):
                    expected[r]=np.sum(decode[qa[r,col]].astype(np.float64)*scales[r,col//128]*wc*ws,axis=1)
                rounded=from_bf16(bf16(expected.astype(np.float32)))
                local_ulps=np.abs(bf16(rounded).astype(np.int32)-bf16(ports[2][0]).astype(np.int32))
                local_ulps[rounded==ports[2][0]]=0
                details=[]
                for idx in np.argwhere(qa!=divide_codes)[:10]:
                    r,c=map(int,idx)
                    details.append({'row':r,'col':c,'value':float(values[r,c]),'scale':float(expected_scale[r,c//128]),'normalized':float(scaled_divide[r,c]),'half_rtz':float(half_rtz[r,c]),'expected_code':int(divide_codes[r,c]),'actual_code':int(qa[r,c])})
                item={'m':m,'replay':replay,'ports':[],'scale_max_relative':float(np.max(np.abs(scales-expected_scale)/expected_scale)),
                      'codes_divide_mismatch':int(np.count_nonzero(qa!=divide_codes)),
                      'codes_reciprocal_mismatch':int(np.count_nonzero(qa!=multiply_codes)),
                      'codes_half_rtz_mismatch':int(np.count_nonzero(qa!=half_codes)),
                      'down_from_actual_codes_max_bf16_ulps':int(np.max(local_ulps)), 'examples':details}
                for actual,expected_port in ports:
                    error=np.abs(actual-expected_port)
                    item['ports'].append({'different':int(np.count_nonzero(actual!=expected_port)),
                        'max_absolute_error':float(error.max()),'fails_shared_gate':int(np.count_nonzero(error>2e-5+0.025*np.abs(expected_port)))})
                results.append(item)
            finally:
                for p in buffers: free(p)
finally:
    for module in modules: unload(module)
    set_current(None); release(dev)
(out/'ffn-diagnosis.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
