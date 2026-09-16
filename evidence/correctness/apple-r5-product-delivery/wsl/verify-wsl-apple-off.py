from pathlib import Path
import hashlib, json, os, re, subprocess, zipfile
r=Path('/root/hyle-verify/r5-apple-20260917')
logs=r/'logs/apple-off'
logs.mkdir(parents=True,exist_ok=True)
archive=Path('/mnt/c/hyle-scratch/r5-apple-20260917/hyle-r5-delivery-source.zip')
assert hashlib.sha256(archive.read_bytes()).hexdigest()=='1aecfb41b9af31881152bdf78a652f7a6ab0f9ec67a4bfbad8993dc1f097d8ce'
unpack=r/'source-delivery-cbedb837'
if not unpack.exists():
    unpack.mkdir()
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            assert (unpack/n).resolve().is_relative_to(unpack)
        z.extractall(unpack)
s=unpack/'hyle-0.1.0-source'
assert s.is_dir()
env=os.environ.copy()
env['PATH']='/usr/bin:/bin'
def run(name,args,success=True):
    p=subprocess.run([str(x) for x in args],cwd=s,env=env,stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT,text=True,timeout=120)
    (logs/(name+'.log')).write_text(p.stdout)
    print(name+': rc='+str(p.returncode),flush=True)
    if success: p.check_returncode()
    return p
base=['/usr/bin/cmake','-S',s,'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
      '-DCMAKE_C_COMPILER=/usr/bin/gcc-14','-DCMAKE_CXX_COMPILER=/usr/bin/g++-14',
      '-DHYLE_COMPUTE_CUDA=OFF','-DHYLE_COMPUTE_ONEDNN=OFF','-DHYLE_COMPUTE_APPLE=OFF',
      '-DHYLE_BUILD_TESTS=OFF','-DHYLE_BUILD_BENCH=OFF','-DHYLE_INSTALL=ON',
      '-DFETCHCONTENT_FULLY_DISCONNECTED=ON']
results={}
for label,component,cpu in (('core','core','OFF'),('no-backend','compute;graph','OFF'),
                            ('cpu-only','compute;graph','ON')):
    b=r/('off-'+label)
    run(label,base+['-B',b,'-DHYLE_COMPONENTS='+component,'-DHYLE_COMPUTE_CPU='+cpu])
    cache=(b/'CMakeCache.txt').read_text()
    assert not re.search(r'^(mlx|metal_cpp|json|fmt)_SOURCE_DIR:',cache,re.M)
    assert not re.search(r'^CMAKE_(CUDA|OBJCXX)_COMPILER:',cache,re.M)
    results[label]={'result':'PASS','scope':'configure isolation, not native execution',
                    'cache_sha256':hashlib.sha256(cache.encode()).hexdigest()}
negative=run('unsupported-on',base+['-B',r/'on-unsupported','-DHYLE_COMPONENTS=compute;graph',
    '-DHYLE_COMPUTE_CPU=OFF','-DHYLE_COMPUTE_APPLE=ON'],success=False)
assert negative.returncode and 'requires macOS Apple Silicon' in negative.stdout
results['unsupported-on']={'result':'PASS','scope':'expected platform rejection before native SDK admission'}
header=r/'apple-off-headers.cc'
header.write_text('#include <hyle/compute/apple.h>\n#include <hyle/graph/apple.h>\nint main() {}\n')
run('public-headers-off',['/usr/bin/g++-14','-std=c++23','-Wall','-Wextra','-Werror',
    '-fsyntax-only','-I'+str(s/'include'),header])
results['public-headers-off']={'result':'PASS','scope':'actual GCC syntax check without MLX or Apple SDK'}
record={'source_commit':'cbedb837791da15392019299f2652d9d3d70ec9d',
        'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'machine':subprocess.check_output(['uname','-a'],text=True).strip(),
        'compiler':subprocess.check_output(['/usr/bin/g++-14','--version'],text=True).splitlines()[0],
        'results':results,'logs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in logs.glob('*.log')}}
(logs/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS: finite Apple OFF/source isolation on WSL ext4; no GPU correctness or performance claim',flush=True)
