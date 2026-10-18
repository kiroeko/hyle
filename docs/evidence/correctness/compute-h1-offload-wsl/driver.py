import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

base = Path('/root/hyle-scratch/compute-unified')
source = base/'source-7a7b4c3b'
producer = base/'l-cpu-cuda/selected-graph/source-build'
parent = producer.parent/'source-parent'
record = base/'h1-offload-wsl'
record.mkdir(exist_ok=True)
candidate = '7eaf67b8a496ee19b609533980cd69e7efb5aa49'
os.environ['PATH'] = '/usr/local/cuda-13.3/bin:'+os.environ['PATH']
os.environ['HYLE_DEP_SOURCE_CACHE'] = '/root/.cache/hyle/dependencies/v1'
os.environ['CMAKE_BUILD_PARALLEL_LEVEL'] = '1'
mode = sys.argv[1] if len(sys.argv)>1 else 'source'
assert mode in ['sync','timestamp-fix','source','shared','static','seams']
transport = Path('/mnt/c/h7/transport')
manifest = json.loads((transport/'h1-product-7eaf67b8.json').read_bytes())

def verify_source():
    mismatch = []
    for name, expected in manifest['git_blob_sha1'].items():
        path = source/name
        if not path.is_file():
            mismatch.append(name+':missing')
            continue
        content = path.read_bytes()
        actual = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        if actual != expected:
            mismatch.append(name)
    assert not mismatch, mismatch[:20]

if mode == 'sync':
    delta = json.loads((transport/'h1-offload-7eaf67b8.delta.json').read_bytes())
    archive = transport/'h1-offload-7eaf67b8.delta.tar'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == delta['sha256']
    assert delta['candidate'] == candidate and not delta['deleted']
    marker = json.loads((source/'.hyle-h1-candidate.json').read_bytes())
    assert marker['candidate'] in [delta['base'],candidate]
    changed = 0
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name = PurePosixPath(member.name)
            assert not name.is_absolute() and '..' not in name.parts
            if not member.isfile():
                assert member.isdir()
                continue
            content = tar.extractfile(member).read()
            target = source/member.name
            assert target.resolve().is_relative_to(source.resolve())
            if target.is_file() and target.read_bytes() == content:
                continue
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(content)
            changed += 1
    # The old validation archive carried a generated packaging manifest, not
    # the tracked Git version. Replace only this metadata with the candidate.
    expected_manifest = (transport/'h1-source-manifest-7eaf67b8.txt').read_bytes()
    if (source/'SOURCE-MANIFEST.txt').read_bytes() != expected_manifest:
        (source/'SOURCE-MANIFEST.txt').write_bytes(expected_manifest)
        changed += 1
    verify_source()
    marker.update({'candidate':candidate,'delta_sha256':delta['sha256'],
                   'verified_product_and_consumer_files':len(manifest['git_blob_sha1'])})
    (source/'.hyle-h1-candidate.json').write_text(json.dumps(marker,indent=2)+'\n')
    (record/'sync.json').write_text(json.dumps({'candidate':candidate,'changed_files':changed,
        'verified_files':len(manifest['git_blob_sha1']),'preserved':'unchanged file timestamps and producer objects'},indent=2)+'\n')
    print('Exact candidate synchronized and all product/consumer identities verified.',flush=True)
    sys.exit(0)

assert json.loads((source/'.hyle-h1-candidate.json').read_bytes())['candidate'] == candidate
verify_source()
if mode == 'timestamp-fix':
    header = source/'src/compute_provider_ggml_cuda/native_bridge.h'
    before = header.stat().st_mtime
    content = header.read_bytes()
    assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == 'ad950e08997cb82368eecfd9f2f9c36b58f1fce8'
    os.utime(header,(header.stat().st_atime,1789869443))
    (record/'timestamp-fix.json').write_text(json.dumps({'candidate':candidate,
        'file':str(header),'blob':'ad950e08997cb82368eecfd9f2f9c36b58f1fce8',
        'also_matches_c3_baseline':'061b46700','old_mtime':before,'restored_mtime':1789869443,
        'scope':'Only restore unchanged source dependency mtime to its last Git content change; no object or Ninja log edits'},indent=2)+'\n')
    print('Restored verified unchanged native bridge timestamp.',flush=True)
    sys.exit(0)
group_name = next(line.split('::',1)[1] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
group = Path('/sys/fs/cgroup')/group_name.lstrip('/')
assert int((group/'memory.max').read_text()) == 12*1024**3
assert int((group/'memory.swap.max').read_text()) == 0
print('Verified Linux process-group hard memory cap 12GiB and swap cap zero.',flush=True)
rows = []
if (record/(mode+'-receipt.json')).is_file():
    previous = json.loads((record/(mode+'-receipt.json')).read_bytes())
    assert previous['candidate'] == candidate
    rows = previous['rows']

def run(stage,args,force=False):
    if not force and any(row['stage']==stage and row['exit']==0 for row in rows):
        print('REUSE completed unchanged stage '+stage,flush=True)
        return
    log = record/(stage+'.log')
    print(stage+' -> '+str(log),flush=True)
    begin = time.monotonic()
    with log.open('wb') as output:
        process = subprocess.run(args,cwd=source,stdout=output,stderr=subprocess.STDOUT)
    row = {'stage':stage,'exit':process.returncode,'seconds':round(time.monotonic()-begin,2),'log':str(log)}
    row['unit_memory_peak_bytes'] = int((group/'memory.peak').read_text())
    row['unit_memory_events'] = (group/'memory.events').read_text().strip()
    rows.append(row)
    (record/(mode+'-receipt.json')).write_text(json.dumps({'candidate':candidate,'rows':rows,
        'build_parallel':1,'test_parallel':1,'linux_unit_memory_max_gib':12},indent=2)+'\n')
    if process.returncode:
        print('\n'.join(line[:190] for line in log.read_text(errors='replace').splitlines()[-22:]),flush=True)
        sys.exit(process.returncode)
    print('PASS '+stage,flush=True)

if mode == 'source':
    cmake = parent/'CMakeLists.txt'
    text = cmake.read_text()
    for target,location,product in [('hyle_cpu_cuda_sample','samples/package/graph/cpu_cuda/main.cc','graph'),
        ('hyle_cuda_weights_sample','samples/package/compute/cuda_weights/main.cc','compute'),
        ('hyle_cpu_cuda_contracts','tests/graph/cpu_cuda_contracts.cc','graph')]:
        if 'add_executable('+target not in text:
            text += '\nadd_executable('+target+' "'+str(source/location)+'")\ntarget_link_libraries('+target+' PRIVATE hyle::'+product+')\nhyle_stage_runtime('+target+')\n'
        if 'add_test(NAME '+target+' ' not in text:
            text += '\nadd_test(NAME '+target+' COMMAND '+target+')\nset_tests_properties('+target+' PROPERTIES TIMEOUT 180)\n'
    if cmake.read_text() != text:
        cmake.write_text(text)
    run('source-configure',['cmake','-S',str(parent),'-B',str(producer),'-DHYLE_STATIC=OFF'])
    run('source-build',['cmake','--build',str(producer),'--target','hyle_cpu_cuda_sample','hyle_cuda_weights_sample',
        'hyle_cpu_cuda_contracts','hyle_selected_source_consumer','hyle_selected_source_compute','--parallel','1'])
    run('source-test',['ctest','--test-dir',str(producer),'--output-on-failure','-j','1',
        '-R','^(hyle_cpu_cuda_sample|hyle_cuda_weights_sample|hyle_cpu_cuda_contracts|hyle_selected_source_consumer|hyle_selected_source_compute)$'])
elif mode == 'seams':
    headers=['compute/resource_budget.h','compute/cuda_weights.h','graph/cpu_cuda.h','graph/executable.h']
    for flavor in ['shared','static']:
        prefix=record/(flavor+'-sdk')
        for index,header in enumerate(headers):
            unit=record/(flavor+'-header-'+str(index)+'.cc')
            unit.write_text('#include <hyle/'+header+'>\nint main() { return 0; }\n')
            args=['/usr/bin/g++-14','-std=c++23','-fsyntax-only','-I'+str(prefix/'include'),
                  '-I/usr/local/cuda-13.3/include']
            if flavor=='static':
                args.append('-DHYLE_STATIC')
            run(flavor+'-installed-header-'+str(index),args+[str(unit)])
        moved=record/(flavor+'-sdk-relocated')
        assert prefix.resolve().parent==record.resolve() and moved.resolve().parent==record.resolve()
        assert prefix.is_dir() and not moved.exists()
        prefix.rename(moved)
        try:
            build=record/(flavor+'-relocated-joint')
            run(flavor+'-relocated-configure',['cmake','-S',str(source/'samples/package/graph/cpu_cuda'),
                '-B',str(build),'-G','Ninja','-DCMAKE_CXX_COMPILER=/usr/bin/g++-14',
                '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH='+str(moved),
                '-DHyle_DIR='+str(moved/'lib/cmake/Hyle')])
            run(flavor+'-relocated-build',['cmake','--build',str(build),'--parallel','1'])
            run(flavor+'-relocated-test',['ctest','--test-dir',str(build),'--output-on-failure','-j','1','--no-tests=error'])
        finally:
            assert not prefix.exists()
            moved.rename(prefix)
else:
    prefix = record/(mode+'-sdk')
    if mode == 'shared' and not (prefix/'lib/cmake/Hyle/HyleConfig.cmake').is_file():
        old_receipt = record/'shared-receipt.json'
        if old_receipt.is_file() and not (record/'shared-initial-no-install-receipt.json').exists():
            (record/'shared-initial-no-install-receipt.json').write_bytes(old_receipt.read_bytes())
        cmake = parent/'CMakeLists.txt'
        text = cmake.read_text().replace('set(HYLE_INSTALL OFF)','set(HYLE_INSTALL ON)')
        if text != cmake.read_text():
            cmake.write_text(text)
        run('shared-producer-configure',['cmake','-S',str(parent),'-B',str(producer),'-DHYLE_STATIC=OFF'])
        run('shared-producer-ready',['cmake','--build',str(producer),'--target','hyle_cpu_cuda_sample',
            'hyle_cuda_weights_sample','hyle_cpu_cuda_contracts','--parallel','1'])
    if mode == 'static':
        run('static-configure',['cmake','-S',str(parent),'-B',str(producer),'-DHYLE_STATIC=ON'])
        run('static-producer',['cmake','--build',str(producer),'--target','hyle_cpu_cuda_sample',
            'hyle_cuda_weights_sample','hyle_cpu_cuda_contracts','--parallel','1'])
    run(mode+'-install',['cmake','--install',str(producer),'--prefix',str(prefix)],
        force=not (prefix/'lib/cmake/Hyle/HyleConfig.cmake').is_file())
    assert (prefix/'lib/cmake/Hyle/HyleConfig.cmake').is_file()
    for name,location in [('joint','samples/package/graph/cpu_cuda'),('direct','samples/package/compute/cuda_weights'),('contracts','tests/graph/cpu_cuda_contracts')]:
        build = record/(mode+'-'+name)
        run(mode+'-'+name+'-configure',['cmake','-S',str(source/location),'-B',str(build),'-G','Ninja',
            '-DCMAKE_CXX_COMPILER=/usr/bin/g++-14','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH='+str(prefix)])
        run(mode+'-'+name+'-build',['cmake','--build',str(build),'--parallel','1'])
        run(mode+'-'+name+'-test',['ctest','--test-dir',str(build),'--output-on-failure','-j','1','--no-tests=error'])
print('Completed only listed actual '+mode+' stages.',flush=True)
