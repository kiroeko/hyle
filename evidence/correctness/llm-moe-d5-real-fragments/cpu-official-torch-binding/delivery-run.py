"""Reuse CPU producers for original official low precision consumer deliveries."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

repo=Path.cwd().resolve()
stage_arg=sys.argv[1]
stage=stage_arg.split('-')[0]
assert stage in ('source','static','shared','prepare')
root=Path('C:/hv/d5cp').resolve()
source=Path('C:/hv/d5cd/source-tree').resolve()
consumer=root/'consumer'
flags={'creationflags':subprocess.CREATE_NO_WINDOW}
candidate=subprocess.check_output(['git','rev-parse','HEAD'],text=True,**flags).strip()
out=repo/'.scratch/validation/d5-cpu-apple-official-binding'/(candidate[:12]+'-delivery-'+stage_arg)
assert not out.exists()
out.mkdir(parents=True)
flags={'creationflags':subprocess.CREATE_NO_WINDOW}
ctypes.windll.kernel32.SetErrorMode(1|2|0x8000)
sdk=repo/'.scratch/validation/d5-cpu-apple-official-sources/cpu-sdk-full/libtorch'
env=dict(os.environ,VSLANG='1033',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
batch=out/'env.cmd'
batch.write_text('@echo off\ncall "C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat" -no_logo -arch=x64\nif errorlevel 1 exit /b 1\nset VSLANG=1033\nset\n')
r=subprocess.run(['cmd','/d','/c',str(batch)],capture_output=True,text=True,env=env,**flags);r.check_returncode()
for line in r.stdout.splitlines():
 if '=' in line and not line.startswith('='):
  key,value=line.split('=',1);env[key]=value
candidate=subprocess.check_output(['git','rev-parse','HEAD'],text=True,**flags).strip()
assert not subprocess.check_output(['git','status','--porcelain'],**flags).strip()
receipt={'candidate':candidate,'stage':stage,'steps':[],'scope':'official GLM/DeepSeek CPU full fragments, independent Direct/Graph C++ delivery; no P1 verdict'}
def save(): (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
def run(name,args,runtime_env=None):
 args=list(map(str,args));start=time.monotonic()
 with (out/(name+'.log')).open('w',encoding='utf-8') as log:
  r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=runtime_env or env,**flags)
 receipt['steps'].append({'name':name,'argv':args,'returncode':r.returncode,'seconds':round(time.monotonic()-start,3)});save()
 print(stage,name,r.returncode,receipt['steps'][-1]['seconds'],flush=True)
 if r.returncode:
  print((out/(name+'.log')).read_text(encoding='utf-8',errors='replace')[-10000:],flush=True);raise SystemExit(r.returncode)

if stage=='prepare':
 paths=['include','src','cmake','third_party','CMakeLists.txt','LICENSE','README.md','SOURCE-MANIFEST.txt','scripts']
 archive=out/'product.tar'
 run('archive',['git','archive','--format=tar','--output='+str(archive),candidate,'--',*paths])
 assert source==Path('C:/hv/d5cd/source-tree').resolve() and source.is_dir()
 written=0
 with tarfile.open(archive) as bundle:
  for member in bundle:
   if not member.isfile():continue
   destination=(source/member.name).resolve();assert destination.is_relative_to(source)
   data=bundle.extractfile(member).read()
   if not destination.exists() or destination.read_bytes()!=data:
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data);written+=1
 # Only obsolete files previously tracked in this explicitly named disposable source tree.
 deleted=subprocess.check_output(['git','diff','--name-only','--diff-filter=D','9273bc71017a1febb69db240b6ea8970b88dd77c',candidate,'--',*paths],text=True,**flags)
 for name in deleted.splitlines():
  target=(source/name).resolve();assert target.is_relative_to(source)
  if target.is_file():target.unlink()
 listing=subprocess.check_output(['git','ls-tree','-r',candidate,'--',*paths],text=True,**flags)
 verified={}
 for line in listing.splitlines():
  meta,name=line.split('\t',1);mode,kind,oid=meta.split()
  assert kind=='blob' and mode in ('100644','100755')
  data=(source/name).read_bytes()
  if name!='SOURCE-MANIFEST.txt':assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==oid,name
  verified[name]=hashlib.sha256(data).hexdigest()
 (out/'source-identity.json').write_text(json.dumps({'candidate':candidate,'source':str(source),'sha256':verified},indent=2)+'\n')
 copied={};todo=[repo/('tests/'+route+'/'+name) for route in ['compute_route_cpu','graph_route_cpu'] for name in ['d5_qwen38_official_test.cc','d5_qwen_gguf_vocabulary_test.cc']]
 while todo:
  path=todo.pop().resolve();relative=path.relative_to(repo).as_posix()
  if relative in copied:continue
  assert relative.startswith('tests/')
  data=path.read_bytes();destination=consumer/relative;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
  copied[relative]=hashlib.sha256(data).hexdigest()
  for name in re.findall(rb'^\s*#include\s+"([^"]+)"',data,re.M):
   if not name.startswith(b'hyle/'):todo.append(path.parent/name.decode())
 cmake=(repo/'evidence/correctness/llm-moe-d5-real-fragments/cpu-delivery/consumer/CMakeLists.txt').read_text()
 for target in ('direct','graph'):
  cmake,count=re.subn(r'add_executable\(d5_cpu_'+target+r'\s+[^)]+\)', 'add_executable(d5_cpu_'+target+' tests/'+('compute_route_cpu' if target=='direct' else 'graph_route_cpu')+'/d5_qwen38_official_test.cc tests/'+('compute_route_cpu' if target=='direct' else 'graph_route_cpu')+'/d5_qwen_gguf_vocabulary_test.cc)',cmake);assert count==1
 cmake=cmake.replace('target_link_libraries(d5_cpu_graph PRIVATE hyle::graph)','target_link_libraries(d5_cpu_graph PRIVATE hyle::graph)\ntarget_compile_definitions(d5_cpu_graph PRIVATE HYLE_TEST_GRAPH_ROUTE_CPU=1)')
 cmake,count=re.subn(r'"--gtest_filter=[^"]+"','"--gtest_filter=*TorchOfficial*"',cmake);assert count==1
 cmake=cmake.replace('set(HYLE_COMPUTE_CPU ON CACHE BOOL "" FORCE)', 'set(HYLE_COMPUTE_CPU ON CACHE BOOL "" FORCE)\n  set(HYLE_COMPUTE_TORCH_CPU ON CACHE STRING "" FORCE)')
 cmake=cmake.replace('target_compile_features(${_target} PRIVATE cxx_std_23)', 'target_compile_features(${_target} PRIVATE cxx_std_23)\n  target_compile_definitions(${_target} PRIVATE HYLE_TEST_TORCH_CPU=1)')
 (consumer/'CMakeLists.txt').write_text(cmake)
 copied['CMakeLists.txt']=hashlib.sha256((consumer/'CMakeLists.txt').read_bytes()).hexdigest()
 (out/'consumer-inputs.json').write_text(json.dumps(copied,indent=2)+'\n')
 receipt.update(changed_source_files=written,verified_source_files=len(verified),archive_bytes=archive.stat().st_size);save()
 print('Prepared immutable source and both external consumers',written,len(verified),flush=True)
 raise SystemExit(0)

cache=Path('C:/Users/kiro/AppData/Local/hyle/dependencies/v1/sources')
dep_args=['-DD5_GGML_SOURCE='+str(cache/'ggml-456172ec733a-41b7e10fd424'),'-DD5_GTEST_SOURCE='+str(cache/'googletest-063de7e9578f-162765170ae6')]
common=['-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']
prefix=root/('official-'+stage+'-relocated')
staging=root/('official-'+stage+'-staging')
producer=root/'source-build'
if stage=='source':
 run('source-configure',['cmake','-S',consumer,'-B',producer,*common,*dep_args,'-DD5_SOURCE_ROOT='+str(source),'-DHYLE_COMPUTE_TORCH_CPU_ROOT='+str(sdk)])
 run('source-build',['cmake','--build',producer,'--target','d5_cpu_delivery','-j','2'])
 (root/'official-source-candidate.txt').write_text(candidate+'\n');build=producer
else:
 if stage=='shared':
  producer=repo/'.scratch/build-compute-v2-dev'
  run('producer-configure',['cmake','-S',repo,'-B',producer,'-DHYLE_BUILD_TESTS=OFF','-DHYLE_INSTALL=ON','-DHYLE_STATIC=OFF','-DCMAKE_INSTALL_PREFIX='+str(staging),'-DHYLE_COMPUTE_TORCH_CPU=ON','-DHYLE_COMPUTE_TORCH_CPU_ROOT='+str(sdk)])
  run('producer-build',['cmake','--build',producer,'-j','2'])
 else:assert (root/'official-source-candidate.txt').read_text().strip()==candidate
 assert not prefix.exists() and not staging.exists()
 run('install',['cmake','--install',producer,'--prefix',staging])
 assert staging.resolve().is_relative_to(root) and prefix.resolve().is_relative_to(root)
 staging.rename(prefix)
 license_path=prefix/'share/licenses/Hyle/ds4-quantization-LICENSE'
 assert license_path.read_bytes()==(repo/'third_party/ds4_quantization/LICENSE').read_bytes()
 for original,installed in [('LICENSE','TorchAO-CPU-LICENSE'),('NOTICE.md','TorchAO-CPU-NOTICE.md')]:
  assert (prefix/'share/licenses/Hyle'/installed).read_bytes()==(repo/'third_party/torchao_cpu'/original).read_bytes()
 build=root/('official-'+stage+'-consumer')
 run('consumer-configure',['cmake','-S',consumer,'-B',build,*common,*dep_args,'-DCMAKE_PREFIX_PATH='+str(prefix)+';'+str(sdk)])
 run('consumer-build',['cmake','--build',build,'--target','d5_cpu_delivery','-j','2'])
 commands=subprocess.check_output(['ninja','-C',str(build),'-t','commands','d5_cpu_delivery'],env=env,**flags)
 (out/'compile-link-commands.log').write_bytes(commands)
 for forbidden in (str(repo/'include'),str(repo/'src'),str(producer),str(source)):
  assert forbidden.replace('\\','/').lower() not in commands.decode().replace('\\','/').lower(),forbidden
 dlls={}
 for dll in build.glob('*.dll'):
  matches=[*prefix.rglob(dll.name),*sdk.glob('lib/'+dll.name)];assert len(matches)==1,(dll.name,matches)
  data=dll.read_bytes();assert data==matches[0].read_bytes()
  dlls[dll.name]={'sha256':hashlib.sha256(data).hexdigest(),'origin':str(matches[0])}
 (out/'dll-origins.json').write_text(json.dumps(dlls,indent=2)+'\n')
cpu_runtime={}
for dll in (sdk/'lib').glob('*.dll'):
 shutil.copyfile(dll,build/dll.name)
 cpu_runtime[dll.name]={'sha256':hashlib.file_digest(dll.open('rb'),'sha256').hexdigest(),'origin':str(dll)}
(out/'cpu-runtime-origins.json').write_text(json.dumps(cpu_runtime,indent=2)+'\n')
system_root=os.environ['SystemRoot']
runtime_env=dict(env);runtime_env['PATH']=os.pathsep.join([system_root+'/System32',system_root]);runtime_env.pop('GTEST_FILTER',None)
run('tests',['ctest','--test-dir',build,'--output-on-failure','--no-tests=error','-j','1','--output-junit',out/'tests.xml'],runtime_env)
native=(build/'Testing/Temporary/LastTest.log').read_bytes();(out/'native-tests.log').write_bytes(native)
passed=sum(map(int,re.findall(rb'\[  PASSED  \]\s+(\d+) tests?\.',native)))
assert b'[  SKIPPED ]' not in native and passed==26,passed
receipt.update(verdict='PASS',passed_tests=passed);save();print(stage,'PASS',passed,flush=True)
