from pathlib import Path
import argparse,hashlib,io,json,subprocess,zipfile

p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--route',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--original',type=Path)
a=p.parse_args();r=json.loads((a.reference/'source-identity.json').read_text());sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
assert r['schema']=='hyle.inference-g08-original-source-materialization/v1' and r['original_revision']=='85bcfa4ebc29cc4d2ed5fb122074204aca675d58' and r['route']==a.route
for item in r['files']:
    data=(a.reference/item['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==item['materialized_sha256'],item['path']
    original=data.replace(b'g08_original_vulkan',b'vulkan') if item['namespace_only'] else data
    assert hashlib.sha256(original).hexdigest()==item['original_sha256'],('original body changed',item['path'])
for item in r['owned_public_overlay']:assert sha(a.reference/item['path'])==item['sha256'],item['path']
for item in r['shared_private_headers']:assert sha(a.repo/item['path'])==item['sha256'],('common private bridge changed',item['path'])
assert all((a.reference/x).is_file() for x in r['compiled_original_sources'])
assert (a.reference/'sources.cmake').read_text()=='set(G08_ORIGINAL_SOURCES\n'+''.join('  "${HYLE_G08_REFERENCE_DIR}/'+x+'"\n' for x in r['compiled_original_sources'])+')\n'
peer={}
if a.original:
    peer['revision']=subprocess.check_output(['git','-C',str(a.original),'rev-parse','HEAD'],text=True).strip();assert peer['revision']==r['original_revision']
    assert not subprocess.check_output(['git','-C',str(a.original),'status','--porcelain','--untracked-files=all'],text=True).strip()
    peer['tree']=subprocess.check_output(['git','-C',str(a.original),'rev-parse','HEAD^{tree}'],text=True).strip()
    roots=['include','src/'+r['owner'],'src/compute','src/compute_backend_cuda','src/compute_backend_cpu','src/compute_ggml_runtime','src/compute_provider_ggml_vulkan']
    if a.route=='vulkan':roots.append('third_party/ggml_vulkan_shaders')
    payload=subprocess.check_output(['git','-C',str(a.original),'archive','--format=zip',r['original_revision'],*roots])
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for item in r['files']:assert hashlib.sha256(z.read(item['path'])).hexdigest()==item['original_sha256'],('pinned peer blob differs',item['path'])
a.output.write_text(json.dumps(dict(state='PINNED_REFERENCE_AND_SHARED_PRIVATE_BRIDGES_VERIFIED_NOT_RUNTIME_QUALIFICATION',route=a.route,original_revision=r['original_revision'],source_identity_sha256=sha(a.reference/'source-identity.json'),files=len(r['files']),shared_private_headers=len(r['shared_private_headers']),compiled_original_sources=r['compiled_original_sources'],original_checkout_verified=bool(peer),original_checkout=peer),indent=2)+'\n')
print('Pinned original implementation, owned overlay and unchanged common private bridges: PASS')
