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
    maximum,floor=(448.,1e-4) if group!=32 else (6.,6*2.**-126)
    scaled=np.maximum(np.abs(x).max(axis=1),np.float32(floor))*np.float32(1/maximum)
    scales=np.exp2(np.ceil(np.log2(scaled.astype(np.float64)))).astype(np.float32)
    positive=(np.array([np.ldexp(float(c&7),-9) if c<8 else np.ldexp(1.+float(c&7)/8,(c>>3)-7) for c in range(127)],dtype=np.float32)
              if group!=32 else np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32))
    v=x/scales[:,None]
    mag=np.minimum(np.abs(v),maximum)
    hi=np.searchsorted(positive,mag).clip(0,len(positive)-1)
    lo=np.maximum(hi-1,0)
    dl,du=mag-positive[lo],positive[hi]-mag
    which=np.where((dl<du)|((dl==du)&((lo&1)==0)),lo,hi)
    decoded=np.copysign(positive[which],v)
    return bf16(decoded*scales[:,None]).reshape(-1),scales
receipt['scope'] = 'Original DeepSeek encoded FP8 plus index-only FP4 entry; four shapes, native-only'
receipt['cases'] = []
e, slots = 7, 6
quant_module, quant_fn = ptr(), ptr()
quant_blob = ct.create_string_buffer((out/'act_quant_kernel_encoded.cubin').read_bytes())
load(ct.byref(quant_module), quant_blob)
modules.append(quant_module)
get_function(ct.byref(quant_fn), quant_module, b'act_quant_kernel__kernel')
try:
    for n, k in [(3072, 7168), (7168, 3072), (2048, 4096), (4096, 2048)]:
        label = f'fp4_{n}_{k}'
        blob = (out/(label+'.cubin')).read_bytes()
        image = ct.create_string_buffer(blob)
        module, function = ptr(), ptr()
        load(ct.byref(module), image)
        modules.append(module)
        get_function(ct.byref(function), module, b'deepseek_indexed_fp4')
        row = np.arange(e * n)
        packed = np.zeros((e * n, k // 2), np.uint8)
        scales_b = np.exp2(-5 + (row[:, None] + np.arange(k // 32)[None, :] + 83) % 3).astype(np.float32)
        terms = []
        for term in [0, 1]:
            col = (17 * row + 83 + term * (k // 2 + 17)) % k
            code = 1 + (row + 83 + term * 3) % 7 + ((row + term) % 2) * 8
            packed[row, col // 2] |= (code << ((col % 2) * 4)).astype(np.uint8)
            values = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])[code & 7] * np.where(code & 8, -1, 1)
            terms.append((col, values * scales_b[row, col // 32]))
        cases = [(t, a, seed, None) for t in [1, 3] for a in [1, slots] for seed in [0, 1]]
        cases += [(1, 1, 0, -1), (1, slots, 1, e)]
        for tokens, activation_slots, seed, bad in cases:
            count = tokens * activation_slots
            index = np.arange(count * k)
            x = bf16(np.sin((index + seed * 17) * .13) * (1 + index % 19) / 7)
            dequant, scales = reference(x, 128)
            decoded = (dequant.astype(np.uint32) << 16).view(np.float32).reshape(count, k)
            ids = ((np.arange(tokens * slots) * (2 if seed else 1) + seed) % e).astype(np.int32)
            if seed: ids[1] = ids[0]  # repeated expert addresses are legal
            if bad is not None: ids[-1] = bad
            expected = np.full(tokens * slots * n + 32, 0x5a5a, np.uint16)
            for slot, expert in enumerate(ids):
                if expert < 0 or expert >= e: continue
                source_row = slot // slots if activation_slots == 1 else slot
                result = np.zeros(n, np.float64)
                for col, coefficient in terms:
                    part = slice(expert * n, (expert + 1) * n)
                    result += decoded[source_row, col[part]].astype(np.float64) * coefficient[part]
                expected[slot * n:(slot + 1) * n] = bf16(result)
            arrays = [x, np.full(x.size + 32, 0x5a, np.uint8),
                      np.full(x.size // 128 + 16, -9876, np.float32), packed, scales_b, ids,
                      np.full(expected.size, 0x5a5a, np.uint16), np.array([0], np.uint32)]
            buffers = []
            try:
                for array in arrays:
                    pointer = u64()
                    alloc(ct.byref(pointer), array.nbytes)
                    buffers.append(pointer)
                    upload(pointer, array.ctypes.data, array.nbytes)
                qargs = [buffers[2], buffers[0], buffers[1], i32(x.size // 128)]
                params = (ptr * len(qargs))(*(ct.cast(ct.byref(arg), ptr) for arg in qargs))
                launch(quant_fn, (x.size // 128 + 31) // 32, 1, 1, 128, 1, 1, 8192, None, params, None)
                args = [buffers[1], buffers[3], buffers[6], buffers[2], buffers[4], buffers[5], buffers[7],
                        i32(e), i32(slots), i32(activation_slots)]
                params = (ptr * len(args))(*(ct.cast(ct.byref(arg), ptr) for arg in args))
                launch(function, n // 128, 1, tokens * slots, 128, 1, 1, 10240, None, params, None)
                sync()
                for idx in [1, 2, 6, 7]:
                    download(arrays[idx].ctypes.data, buffers[idx], arrays[idx].nbytes)
                mismatch = int(np.count_nonzero(arrays[6] != expected))
                guards = bool(np.all(arrays[1][x.size:] == 0x5a) and np.all(arrays[2][scales.size:] == -9876))
                scale_mismatches = int(np.count_nonzero(arrays[2][:scales.size] != scales))
                status = int(arrays[7][0])
                result = {'n': n, 'k': k, 'tokens': tokens, 'activation_slots': activation_slots,
                          'seed': seed, 'invalid_id': bad, 'mismatches': mismatch,
                          'scale_mismatches': scale_mismatches, 'guards': guards, 'status': status,
                          'cubin_sha256': hashlib.sha256(blob).hexdigest()}
                receipt['cases'].append(result)
                save()
                assert mismatch == 0 and guards and scale_mismatches == 0 and status == int(bad is not None), result
            finally:
                for buffer in buffers: free(buffer)
        print(label, '10 cases exact PASS', flush=True)
    receipt['verdict'] = 'PASS_NATIVE_INDEXED_FP4_ONLY'
    save()
finally:
    for module in modules: unload(module)
    set_current(None)
    release(dev)
