from pathlib import Path
import argparse,hashlib,io,json,re,subprocess,zipfile

PIN='85bcfa4ebc29cc4d2ed5fb122074204aca675d58'
p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--route',choices=['cpu','cuda','apple','vulkan'],required=True);p.add_argument('--output',required=True)
a=p.parse_args();repo=Path(a.repo).resolve();out=Path(a.output).resolve()
assert subprocess.check_output(['git','-C',str(repo),'rev-parse',PIN],text=True).strip()==PIN
assert not out.exists();out.mkdir(parents=True)
owners={'cpu':'compute_provider_ggml','cuda':'compute_provider_cuda','apple':'compute_route_apple','vulkan':'compute_route_vulkan'}
owner=owners[a.route]
# Whole original public declarations plus private ABI/bridge headers. Only the
# selected original implementation TUs are compiled by the future workload.
roots=['include','src/'+owner,'src/compute','src/compute_backend_cuda','src/compute_backend_cpu','src/compute_ggml_runtime','src/compute_provider_ggml_vulkan']
if a.route=='vulkan':roots.extend(['third_party/ggml_vulkan_shaders',
    'src/compute_ops/view_alias.h','src/compute_provider_iree_vulkan/native_device.h',
    'src/compute_provider_iree_vulkan/native_program.h'])
payload=subprocess.check_output(['git','-C',str(repo),'archive','--format=zip',PIN,*roots])
records=[];overlay=[];shared_private=[]
with zipfile.ZipFile(io.BytesIO(payload)) as z:
    for info in z.infolist():
        if info.is_dir():continue
        relative=Path(info.filename);assert not relative.is_absolute() and '..' not in relative.parts
        data=z.read(info.filename)
        if relative.parts[0]=='src' and relative.parts[1]!=owner and relative.suffix not in ['.h','.cuh','.in']:continue
        transformed=data
        # A distinct native route retains its complete original state classes,
        # dispatch/retirement/transfer code. Shared Hyle declarations stay intact.
        owned_header=info.filename.startswith('include/hyle/compute/') and 'vulkan' in relative.name
        if a.route=='vulkan' and (info.filename.startswith('src/'+owner+'/') or owned_header):
            assert b'hyle::compute::g08_original_vulkan' not in data
            transformed=data.replace(b'hyle::compute::vulkan',b'hyle::compute::g08_original_vulkan')
            transformed=re.sub(rb'\bvulkan(?=::)',b'g08_original_vulkan',transformed)
            assert transformed.replace(b'g08_original_vulkan',b'vulkan')==data
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(transformed)
        records.append(dict(path=info.filename,original_sha256=hashlib.sha256(data).hexdigest(),materialized_sha256=hashlib.sha256(transformed).hexdigest(),namespace_only=data!=transformed))
        if a.route=='vulkan' and owned_header:
            target=out/'public-overlay'/relative.relative_to('include');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(transformed)
            overlay.append(dict(path=target.relative_to(out).as_posix(),sha256=hashlib.sha256(transformed).hexdigest()))
        if relative.parts[0]=='src' and relative.parts[1]!=owner:
            assert (repo/relative).read_bytes()==data,('common private bridge changed',info.filename)
            shared_private.append(dict(path=info.filename,sha256=hashlib.sha256(data).hexdigest()))
sources={
 'cpu':['ggml_cpu_provider.cc','standard_index.cc','native_batch.cc'],
 'cuda':['collectives_provider.cc','collectives.cu'],
 'apple':['apple_tensor.cc','apple_mature_indexing.cc'],
 'vulkan':[Path(x['path']).name for x in records if x['path'].startswith('src/compute_route_vulkan/') and Path(x['path']).suffix=='.cc']
}[a.route]
assert all((out/'src'/owner/x).is_file() for x in sources)
definitions={'cpu':['ggml_cpu=g08_original_ggml_cpu'],'cuda':['cuda_native=g08_original_cuda_native'],'apple':['lower_tensor=g08_original_lower_tensor','lower_mature_indexing=g08_original_lower_mature_indexing','is_mlx_tensor_custom=g08_original_is_mlx_tensor_custom'],'vulkan':[]}[a.route]
receipt=dict(schema='hyle.inference-g08-original-source-materialization/v1',state='ORIGINAL_SOURCE_MATERIALIZED_NOT_BUILD_OR_RUNTIME_QUALIFICATION',original_revision=PIN,route=a.route,owner=owner,
    compiled_original_sources=['src/'+owner+'/'+x for x in sources],compile_definitions=definitions,files=records,owned_public_overlay=overlay,shared_private_headers=shared_private,
    public_header_policy='Current qualified SDK supplies all common public declarations, including the same fixed Mem implementation to both arms. The archived full include tree is provenance only and is never an include directory. Vulkan reference TUs alone prepend the owned namespace-only overlay.',
    scope='Original implementation bodies and public/private declarations from immutable Git objects. CPU/CUDA/Apple symbol isolation is by the listed compile definitions. Vulkan only changes its owned namespace and relative references to that namespace; common headers are not rewritten. Generated shader/compiler configuration and dependency identities still require qualification. No numerical policy, output, completion, allocation or sorting rewrite.')
(out/'source-identity.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
(out/'sources.cmake').write_text('set(G08_ORIGINAL_SOURCES\n'+''.join('  "${HYLE_G08_REFERENCE_DIR}/'+x+'"\n' for x in receipt['compiled_original_sources'])+')\n',encoding='utf-8')
print(json.dumps(dict(state=receipt['state'],route=a.route,files=len(records),original_translation_units=len(sources))))
