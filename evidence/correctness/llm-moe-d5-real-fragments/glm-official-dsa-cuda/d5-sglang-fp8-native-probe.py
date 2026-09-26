from pathlib import Path
import ctypes as ct
import hashlib
import json
import numpy as np

ct.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
out = Path('.scratch/validation/d5-sglang-fp8-aot')
compiled = json.loads((out / 'compilation.json').read_text())
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
try:
    for kernel in compiled['kernels']:
        assert kernel['metadata']['global_scratch_size'] == 0
        assert kernel['metadata']['profile_scratch_size'] == 0
        image = ct.create_string_buffer((out / (kernel['label']+'.cubin')).read_bytes())
        module = ptr()
        load(ct.byref(module), image)
        modules.append(module)
        function = ptr()
        get_function(ct.byref(function), module, kernel['metadata']['name'].encode())
        kernels[kernel['label']] = (function, kernel['metadata']['shared'])
    def run(label, grid, args):
        args = [*args, u64(0), u64(0)]
        params = (ptr * len(args))(*(ct.cast(ct.byref(x), ptr) for x in args))
        function, shared = kernels[label]
        launch(function, grid,1,1,128,1,1,shared,None,params,None)
        sync()
    positive = np.zeros(127, dtype=np.float32)
    for code in range(127):
        exponent, mantissa = code >> 3, code & 7
        positive[code] = np.ldexp(float(mantissa), -9) if exponent == 0 else np.ldexp(1.0+mantissa/8.0, exponent-7)
    decode = np.concatenate([positive, [np.nan], -positive, [np.nan]]).astype(np.float32)
    def encode(values):
        mag = np.minimum(np.abs(values),448.0)
        upper = np.searchsorted(positive,mag,side='left').clip(0,126)
        lower = np.maximum(upper-1,0)
        dl, du = mag-positive[lower],positive[upper]-mag
        chosen = np.where((dl < du) | ((dl == du) & ((lower & 1) == 0)),lower,upper)
        return (chosen | (np.signbit(values).astype(np.int64)<<7)).astype(np.uint8)
    shapes = [(3072,7168,True),(7168,3072,True),(2048,4096,True),(4096,2048,True),
              (2048,6144,False),(6144,2048,False),(2048,4096,False),(4096,2048,False),
              (576,6144,False)]
    codes = np.array([0x30,0x38,0x3c,0x40,0xb0,0xb8,0xbc,0xc0],dtype=np.uint8)
    for n,k,ue8 in shapes:
        blocks = k//128
        rows = np.arange(n)
        cols = np.stack([(rows*17+31+t*(k//2+13))%k for t in range(2)],axis=1)
        wc = codes[(rows[:,None]+np.arange(2)*3)%8]
        w = np.zeros((n,k),dtype=np.uint8)
        w[rows[:,None],cols] = wc
        scale_rows = np.arange((n+127)//128)[:,None]
        scale_cols = np.arange(blocks)[None,:]
        ws = (np.exp2(-6+(scale_rows+scale_cols+13)%3) if ue8 else
              (17+(scale_rows*3+scale_cols+13)%11)/1024.0).astype(np.float32)
        for m in [1,3]:
            buffers = []
            def buffer(array):
                result = u64()
                alloc(ct.byref(result),array.nbytes)
                buffers.append(result)
                upload(result, array.ctypes.data,array.nbytes)
                return result
            a = np.zeros((m,k),dtype=np.float32)
            aq = np.zeros((m,k),dtype=np.uint8)
            asc = np.zeros((m,blocks),dtype=np.float32)
            result = np.zeros((m,n),dtype=np.float32)
            try:
                da,dq,das,dw,dws,dc = [buffer(x) for x in [a,aq,asc,w,ws,result]]
                for replay in range(2):
                    indices = np.arange(m*k).reshape(m,blocks,128)
                    scales = np.exp2(-5+(np.arange(m*blocks).reshape(m,blocks)+replay)%8).astype(np.float32)
                    values = (((indices*13+replay*7)%511)-255).astype(np.float32)*0.5
                    values[:,:,0]=448.0
                    values[:,:,1]=-448.0
                    # Exactly representable ties check the independent RNE encoder.
                    values[:,:,2]=1.0625
                    values[:,:,3]=-1.1875
                    values[:,0,:]=0.0
                    a[:] = (values*scales[:,:,None]).reshape(m,k)
                    expected_scale = np.maximum(np.max(np.abs(a.reshape(m,blocks,128)),axis=2),np.float32(1e-10))/np.float32(448.0)
                    scaled = a.reshape(m,blocks,128)*(np.float32(1.0)/expected_scale[:,:,None])
                    expected_codes = encode(scaled).reshape(m,k)
                    upload(da,a.ctypes.data,a.nbytes)
                    run('quantize',m*blocks,[da,dq,das,i32(128),i32(128),f32(1e-10),f32(-448),f32(448)])
                    download(aq.ctypes.data,dq,aq.nbytes)
                    download(asc.ctypes.data,das,asc.nbytes)
                    mismatches = int(np.count_nonzero(aq != expected_codes))
                    scale_error = float(np.max(np.abs((asc-expected_scale)/expected_scale)))
                    assert mismatches == 0, ('quantization',n,k,m,replay,mismatches)
                    assert scale_error <= 2e-6
                    args=[dq,dw,dc,das,dws,*[i32(x) for x in [m,n,k,k,1,1,k,n,1,blocks,1,1,blocks]]]
                    run('matmul',((m+63)//64)*((n+127)//128),args)
                    download(result.ctypes.data,dc,result.nbytes)
                    ref = np.zeros((m,n),dtype=np.float64)
                    for t in range(2):
                        col=cols[:,t]
                        ref += (decode[expected_codes[:,col]].astype(np.float64)*expected_scale[:,col//128].astype(np.float64)*
                                decode[wc[:,t]].astype(np.float64)*ws[rows//128,col//128].astype(np.float64))
                    ref=ref.astype(np.float32)
                    error=np.abs(result.astype(np.float64)-ref)
                    limit=2e-6+2e-6*np.abs(ref)
                    assert np.isfinite(result).all() and np.all(error <= limit), ('matmul',n,k,m,replay,float(error.max()))
                    receipt['cases'].append({'m':m,'n':n,'k':k,'weight_scale':'UE8M0 widened losslessly' if ue8 else 'F32',
                        'replay':replay,'code_mismatches':mismatches,'scale_relative_error':scale_error,
                        'output_max_absolute_error':float(error.max()),'output_sha256':hashlib.sha256(result.tobytes()).hexdigest()})
                    save()
                print('PASS',m,n,k,ue8,flush=True)
            finally:
                for b in reversed(buffers):
                    free(b)
    receipt['verdict']='PASS_FINITE_NATIVE_QUALIFICATION'
    save()
finally:
    for module in reversed(modules):
        unload(module)
    release(dev)
