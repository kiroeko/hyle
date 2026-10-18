from pathlib import Path

prefix=Path('.scratch/d5-sglang-fp8-native-probe.py').read_text(encoding='utf-8').split('try:\n    for kernel in compiled')[0]
prefix=prefix.replace('.scratch/validation/d5-sglang-fp8-aot','.scratch/validation/d5-f41-fp8-weight-decode')
exec(compile(prefix,'f41-driver-abi','exec'))
receipt.pop('first_launch_failure')
receipt['scope']='Original pinned DeepSeek block128 weight decode, native Windows Driver; not HYLE entry qualification'
positive=np.array([np.ldexp(float(c&7),-9) if c<8 else np.ldexp(1.+float(c&7)/8,(c>>3)-7) for c in range(127)],dtype=np.float32)
decode=np.concatenate([positive,[np.nan],-positive,[np.nan]]).astype(np.float32)
def bf16(x):
    bits=np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))>>16).astype(np.uint16)
try:
    for kernel in compiled['kernels']:
        image=ct.create_string_buffer((out/(kernel['label']+'.cubin')).read_bytes())
        module=ptr(); load(ct.byref(module),image); modules.append(module)
        function=ptr(); get_function(ct.byref(function),module,kernel['metadata']['name'].encode())
        kernels[kernel['label']]=(function,kernel['metadata']['shared'],kernel['metadata']['num_warps']*32)
    for n,k in [(1,128),(129,256),(576,6144),(28672,512)]:
        rows=np.arange(n,dtype=np.int64)[:,None]
        cols=np.arange(k,dtype=np.int64)[None,:]
        for replay in [0,1]:
            codes=((rows*17+cols*11+replay*3)%256).astype(np.uint8)
            scales=((17+(np.arange((n+127)//128)[:,None]*3+np.arange(k//128)[None,:]+replay)%11)/1024).astype(np.float32)
            scale_indices=(np.arange(scales.size).reshape(scales.shape)+replay)%9
            scales[scale_indices==0]=0
            scales[scale_indices==1]*=-1
            expected=(decode[codes]*scales[rows//128,cols//128]).astype(np.float32)
            for dtype in ['bf16','fp32']:
                arrays=[codes,scales,np.full(n*k+16,0x5a5a,dtype=np.uint16 if dtype=='bf16' else np.float32)]
                buffers=[]
                try:
                    for array in arrays:
                        address=u64(); alloc(ct.byref(address),array.nbytes); buffers.append(address)
                        upload(address,array.ctypes.data,array.nbytes)
                    args=[*buffers,i32(n),i32(k),u64(0),u64(0)]
                    params=(ptr*len(args))(*(ct.cast(ct.byref(x),ptr) for x in args))
                    function,shared,threads=kernels['decode_'+dtype]
                    launch(function,(n+127)//128,k//128,1,threads,1,1,shared,None,params,None)
                    sync(); download(arrays[2].ctypes.data,buffers[2],arrays[2].nbytes)
                    actual=arrays[2][:n*k].reshape(n,k)
                    wanted=bf16(expected) if dtype=='bf16' else expected
                    finite=np.isfinite(expected)
                    mismatch=int(np.count_nonzero(actual[finite]!=wanted[finite]))
                    actual_float=(actual.astype(np.uint32)<<16).view(np.float32) if dtype=='bf16' else actual
                    nan_mismatch=int(np.count_nonzero(np.isnan(actual_float)!=np.isnan(expected)))
                    guards=bool(np.all(arrays[2][n*k:]==0x5a5a))
                    result={'n':n,'k':k,'dtype':dtype,'replay':replay,'mismatches':mismatch,
                            'nan_mismatches':nan_mismatch,'output_guard_unchanged':guards}
                    receipt['cases'].append(result); save()
                    assert mismatch==0 and nan_mismatch==0 and guards,result
                finally:
                    for address in buffers: free(address)
    receipt['verdict']='PASS_NATIVE_DECODE_ONLY'; save()
    print('Native decode:',len(receipt['cases']),'cases PASS; all 256 codes, partial rows, real GLM KV shape, output guards.')
finally:
    for module in modules: unload(module)
    set_current(None); release(dev)
