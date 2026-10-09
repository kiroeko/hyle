"""Fixed V4 public consumer batch on a disposable POSIX validation clone."""
import argparse,hashlib,json,os,platform,re,shutil,subprocess,time
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True,type=Path)
parser.add_argument('--candidate',required=True)
parser.add_argument('--cache',required=True,type=Path)
parser.add_argument('--resume-source-build',action='store_true')
args=parser.parse_args()
repo=Path.cwd().resolve();root=args.root.expanduser().resolve();cache=args.cache.expanduser().resolve()
if root==repo or not root.is_relative_to(Path.home()/'hyle-scratch'):
    raise SystemExit('Use an owned validation directory below ~/hyle-scratch')
root.mkdir(parents=True,exist_ok=True)
out=root/'receipts'
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
if head!=args.candidate or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
    raise SystemExit('Require the clean, exact Windows-origin candidate')
if args.resume_source_build:
    receipt=json.loads((out/'receipt.json').read_text())
    if len(receipt['steps'])!=2 or receipt['steps'][0]['returncode']!=0 or receipt['steps'][1]['returncode']==0:
        raise SystemExit('Resume is limited to the recorded first source build failure')
    changed=subprocess.check_output(['git','diff','--name-only',receipt['candidate'],head,'--','include','src','cmake','third_party','CMakeLists.txt'],text=True).splitlines()
    if changed!=['src/compute_provider_iree_vulkan/native_program.cc']:
        raise SystemExit('Resume requires only the bounded native-program warning correction')
    receipt['continued_candidate']=head
else:
    if out.exists():raise SystemExit('Retain the previous batch receipt; do not silently restart')
    out.mkdir()
    receipt={'candidate':head,'product_candidate':'2f78714c8','system':platform.platform(),'steps':[]}
def run(label,command,env=None):
    command=list(map(str,command));start=time.monotonic()
    with (out/(label+'.log')).open('w') as stream:
        p=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
    receipt['steps'].append({'label':label,'argv':command,'returncode':p.returncode,'seconds':round(time.monotonic()-start,3)})
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(label,p.returncode,receipt['steps'][-1]['seconds'],flush=True)
    if p.returncode:raise SystemExit('Failed gate: '+str(out/(label+'.log')))
consumer=root/'consumer';consumer.mkdir()
todo=[repo/f'tests/{entry}_route_vulkan/d5_qwen35_execution_test.cc' for entry in ('compute','graph')]
copied={}
while todo:
    source=todo.pop().resolve();relative=source.relative_to(repo).as_posix()
    if relative in copied:continue
    if not relative.startswith('tests/'):raise SystemExit('Fixture reaches private product source: '+relative)
    data=source.read_bytes();target=consumer/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    copied[relative]=hashlib.sha256(data).hexdigest()
    for include in re.findall(rb'^\s*#include\s+"([^"]+)"',data,re.M):
        name=include.decode()
        if not name.startswith('hyle/') and name!='vulkan_program_internal.h':todo.append(source.parent/name)
iq1_source=repo/'tests/compute_route_vulkan/iq1_test_support.h'
iq1=iq1_source.read_text().split('template <class Prepare> void domains(Prepare prepare) {',1)[0]
iq1=iq1.replace('#include "../../src/compute_route_vulkan/vulkan_iq1.h"\n','')+'} // namespace vulkan_iq1_test\n'
(consumer/'tests/compute_route_vulkan/iq1_public_support.h').write_text(iq1)
for name,source in [('tests/compute_route_vulkan/iq1_original_grid.h',repo/'tests/compute_route_vulkan/iq1_original_grid.h'),
                    ('iq1_delta.cc',Path(__file__).parent/'consumer/iq1_delta.cc'),
                    ('CMakeLists.txt',Path(__file__).parent/'consumer/CMakeLists.txt')]:
    shutil.copyfile(source,consumer/name);copied[name]=hashlib.sha256(source.read_bytes()).hexdigest()
copied['iq1_public_prefix']={'origin_sha256':hashlib.sha256(iq1_source.read_bytes()).hexdigest(),'sha256':hashlib.sha256(iq1.encode()).hexdigest()}
(out/'consumer-inputs.json').write_text(json.dumps(copied,indent=2)+'\n')
cmake=shutil.which('cmake');ctest=shutil.which('ctest');ninja=shutil.which('ninja')
if not all((cmake,ctest,ninja)):raise SystemExit('Prepare CMake, CTest and Ninja first')
common=['-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
deps=['-DD5_GGML_SOURCE='+str(cache/'sources/ggml-456172ec733a-41b7e10fd424'),'-DD5_GTEST_SOURCE='+str(cache/'sources/googletest-063de7e9578f-162765170ae6')]
if platform.system()=='Darwin':common+=['-DCMAKE_OSX_DEPLOYMENT_TARGET=14.4']
source_options=['-DHYLE_DEP_SOURCE_CACHE='+str(cache),'-DHYLE_WERROR=ON','-DHYLE_VULKAN_GLSLANG='+str(shutil.which('glslangValidator'))]
producer=root/'producer';runtime_env=dict(os.environ);runtime_env.pop('GTEST_FILTER',None)
runtime_env['HYLE_D5_QWEN35_FIXTURE']=str(Path.home()/'hyle-scratch/d5-qwen35-six-profile-fixture')
runtime_env['HYLE_VULKAN_VALIDATION']='0'
if platform.system()=='Darwin':runtime_env['DYLD_FALLBACK_LIBRARY_PATH']='/opt/homebrew/lib:/usr/lib'
for stage in ('source','static','shared'):
    staging=root/(stage+'-staging');prefix=root/(stage+'-relocated')
    if stage in ('source','shared'):
        if not (stage=='source' and args.resume_source_build):
            run(stage+'-producer-configure',[cmake,'-S',consumer,'-B',producer,*common,*deps,*source_options,
                '-DD5_SOURCE_ROOT='+str(repo),'-DV4_SOURCE_STATIC='+('OFF' if stage=='shared' else 'ON'),'-DCMAKE_INSTALL_PREFIX='+str(staging)])
        run(stage+'-producer-build'+('-w2' if stage=='source' and args.resume_source_build else ''),[cmake,'--build',producer,'-j','2'])
    if stage=='source':build=producer
    else:
        run(stage+'-install',[cmake,'--install',producer,'--prefix',staging])
        if prefix.exists():raise SystemExit('Refuse existing SDK move destination')
        staging.rename(prefix);build=root/(stage+'-consumer')
        run(stage+'-consumer-configure',[cmake,'-S',consumer,'-B',build,*common,*deps,'-DCMAKE_PREFIX_PATH='+str(prefix)])
        run(stage+'-consumer-build',[cmake,'--build',build,'--target','d5_vulkan_delivery','-j','2'])
        commands=subprocess.check_output([ninja,'-C',str(build),'-t','commands','d5_vulkan_delivery']).decode()
        (out/(stage+'-compile-link-commands.log')).write_text(commands)
        for forbidden in (str(repo/'src'),str(repo/'include'),str(producer)):
            if forbidden in commands:raise SystemExit('Installed consumer uses producer inputs: '+forbidden)
        licenses={str(x.relative_to(prefix)):hashlib.sha256(x.read_bytes()).hexdigest() for x in prefix.rglob('*') if x.is_file() and '/licenses/' in str(x)}
        if not any('iree' in x for x in licenses) or not any('ggml' in x for x in licenses):raise SystemExit('Required carrier notices absent')
        (out/(stage+'-licenses.json')).write_text(json.dumps(licenses,indent=2)+'\n')
    hidden=[]
    try:
        if stage!='source':
            for directory in (producer/'bin',producer/'hyle/bin'):
                if directory.exists():
                    target=directory.with_name(directory.name+'.v4-hidden')
                    if target.exists() or not target.resolve().is_relative_to(producer):raise SystemExit('Invalid producer hide target')
                    directory.rename(target);hidden.append((target,directory))
        run(stage+'-consumer-test',[ctest,'--test-dir',build,'--output-on-failure','--no-tests=error','-j','1','--output-junit',out/(stage+'-tests.xml')],runtime_env)
    finally:
        for source,target in reversed(hidden):source.rename(target)
        log=build/'Testing/Temporary/LastTest.log'
        if log.exists():shutil.copyfile(log,out/(stage+'-native-tests.log'))
    raw=log.read_bytes();passed=sum(map(int,re.findall(rb'\[  PASSED  \]\s+(\d+) tests?\.',raw)))
    if b'[  SKIPPED ]' in raw or passed!=14:raise SystemExit('Expected fourteen actual passing tests without skips')
    receipt.setdefault('stages',{})[stage]={'verdict':'PASS','tests':passed}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
receipt['verdict']='PASS';(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('source/static/shared PASS, 14 tests each',flush=True)
