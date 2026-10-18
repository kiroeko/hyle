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
def bf16(x):')], __file__, 'exec'))

def bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

receipt['scope'] = 'One original DeepSeek FP8 x FP4 GEMM, N4096 K2048; native candidate only'
receipt['cases'] = []
n, k = 4096, 2048
codes = np.arange(256, dtype=np.uint32)
exponent, mantissa = (codes >> 3) & 15, codes & 7
table = np.where(exponent == 0, np.ldexp(mantissa.astype(float), -9),
                 np.ldexp(1 + mantissa.astype(float) / 8, exponent.astype(np.int32) - 7))
table *= np.where(codes & 128, -1, 1)
e2m1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float64)
blob = (out/'fp4_gemm_kernel.cubin').read_bytes()
image = ct.create_string_buffer(blob)
module = ptr()
load(ct.byref(module), image)
modules.append(module)
function = ptr()
get_function(ct.byref(function), module, b'fp4_gemm_kernel__kernel')
try:
    for seed in [0, 1]:
        packed = np.zeros((n, k // 2), np.uint8)
        scales_b = np.exp2(-5 + (np.arange(n)[:, None] + np.arange(k // 32)[None, :] + seed) % 3).astype(np.float32)
        terms = []
        rows = np.arange(n)
        for term in [0, 1]:
            col = (17 * rows + 83 + term * (k // 2 + 17)) % k
            code = 1 + (rows + 83 + term * 3) % 7 + ((rows + term) % 2) * 8
            packed[rows, col // 2] |= (code << ((col % 2) * 4)).astype(np.uint8)
            coefficient = e2m1[code & 7] * np.where(code & 8, -1, 1) * scales_b[rows, col // 32]
            terms.append((col, coefficient))
        for m in [1, 3, 33]:
            index = np.arange(m * k).reshape(m, k)
            a = ((index * 29 + seed * 37) % 256).astype(np.uint8)
            a[(a & 127) == 127] -= 1
            scales_a = np.exp2(-10 + (np.arange(m)[:, None] + np.arange(k // 128)[None, :] + seed) % 4).astype(np.float32)
            reference = np.zeros((m, n), np.float64)
            for col, coefficient in terms:
                reference += table[a[:, col]] * scales_a[:, col // 128] * coefficient[None, :]
            expected = bf16(reference).reshape(-1)
            arrays = [a, packed, np.full(m * n + 32, 0x5a5a, np.uint16), scales_a, scales_b]
            buffers = []
            try:
                for array in arrays:
                    pointer = u64()
                    alloc(ct.byref(pointer), array.nbytes)
                    buffers.append(pointer)
                    upload(pointer, array.ctypes.data, array.nbytes)
                args = [*buffers, i32(m)]
                params = (ptr * len(args))(*(ct.cast(ct.byref(arg), ptr) for arg in args))
                launch(function, n // 128, (m + 31) // 32, 1, 128, 1, 1, 10240, None, params, None)
                sync()
                download(arrays[2].ctypes.data, buffers[2], arrays[2].nbytes)
                mismatch = int(np.count_nonzero(arrays[2][:m * n] != expected))
                guards = bool(np.all(arrays[2][m * n:] == 0x5a5a))
                row = {'m': m, 'n': n, 'k': k, 'seed': seed, 'bf16_mismatches': mismatch,
                       'guards': guards, 'cubin_sha256': hashlib.sha256(blob).hexdigest()}
                receipt['cases'].append(row)
                save()
                assert mismatch == 0 and guards, row
            finally:
                for buffer in buffers:
                    free(buffer)
    receipt['verdict'] = 'PASS_NATIVE_ONE_FP4_SIGNATURE_ONLY'
    save()
    print('6 original FP8 x FP4 native cases exact PASS; not an HYLE entry qualification')
finally:
    for module in modules:
        unload(module)
    set_current(None)
    release(dev)
