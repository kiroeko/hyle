import ctypes
import json
from pathlib import Path
import subprocess

root=Path.cwd().resolve();out=root/'.scratch/validation/d5-cpu-apple-official-binding/wsl-official-consumers';assert not out.exists();out.mkdir(parents=True)
ctypes.windll.kernel32.SetErrorMode(1|2|0x8000)
flags={'creationflags':subprocess.CREATE_NO_WINDOW}
candidate=subprocess.check_output(['git','rev-parse','HEAD'],text=True,**flags).strip()
assert not subprocess.check_output(['git','status','--porcelain'],**flags).strip()
archive=out/'source.tar'
paths=['include','src','cmake','third_party','tests','scripts','CMakeLists.txt','LICENSE','README.md','SOURCE-MANIFEST.txt']
subprocess.run(['git','archive','--format=tar','--output='+str(archive),candidate,'--',*paths],check=True,**flags)
remote_out='/mnt/c/public/project-hyle/hyle/.scratch/validation/d5-cpu-apple-official-binding/wsl-official-consumers'
program='''import hashlib,json,os,subprocess,tarfile,time
from pathlib import Path
out=Path(REMOTE_OUT)
source=Path('/root/hyle-scratch/active-cpu-z/source').resolve()
boundary=Path('/root/hyle-scratch/active-cpu-z').resolve()
assert source.is_relative_to(boundary) and source.is_dir() and not (source/'.git').exists()
receipt={'candidate':CANDIDATE,'system':os.uname()._asdict() if hasattr(os.uname(),'_asdict') else list(os.uname()),'scope':'CPU official GLM/DeepSeek full fragments, Direct/Graph; Linux source correctness only','steps':[]}
members=set();updated=0
with tarfile.open(out/'source.tar') as bundle:
 for entry in bundle:
  if not entry.isfile():continue
  path=(source/entry.name).resolve();assert path.is_relative_to(source)
  data=bundle.extractfile(entry).read();members.add(entry.name)
  if not path.exists() or path.read_bytes()!=data:
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data);updated+=1
removed=0
for name in ('include','src','cmake','third_party','tests','scripts'):
 for p in (source/name).rglob('*'):
  if p.is_file() and p.relative_to(source).as_posix() not in members:
   assert p.resolve().is_relative_to(source);p.unlink();removed+=1
receipt.update(synced_files=len(members),updated=updated,removed=removed,archive_sha256=hashlib.sha256((out/'source.tar').read_bytes()).hexdigest())
def run(label,args):
 start=time.monotonic()
 with (out/(label+'.log')).open('w') as stream:
  r=subprocess.run(list(map(str,args)),stdout=stream,stderr=subprocess.STDOUT)
 receipt['steps'].append({'label':label,'argv':list(map(str,args)),'returncode':r.returncode,'seconds':round(time.monotonic()-start,3)})
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
 print(label,r.returncode,receipt['steps'][-1]['seconds'],flush=True)
 if r.returncode:
  print((out/(label+'.log')).read_text(errors='replace')[-8000:],flush=True);raise SystemExit(r.returncode)
build=boundary/'build'
overlay=out/'fixture-warnings.cmake'
overlay.write_text("""function(d5_ds4_consumers)
  add_executable(d5_ds4_cpu_direct "${CMAKE_SOURCE_DIR}/tests/compute_route_cpu/d5_qwen38_official_test.cc" "${CMAKE_SOURCE_DIR}/tests/compute_route_cpu/d5_qwen_gguf_vocabulary_test.cc")
  add_executable(d5_ds4_cpu_graph "${CMAKE_SOURCE_DIR}/tests/graph_route_cpu/d5_qwen38_official_test.cc" "${CMAKE_SOURCE_DIR}/tests/graph_route_cpu/d5_qwen_gguf_vocabulary_test.cc")
  target_link_libraries(d5_ds4_cpu_direct PRIVATE hyle_compute_route_cpu)
  target_link_libraries(d5_ds4_cpu_graph PRIVATE hyle_graph_route_cpu)
  target_compile_definitions(d5_ds4_cpu_graph PRIVATE HYLE_TEST_GRAPH_ROUTE_CPU=1)
  foreach(t IN ITEMS d5_ds4_cpu_direct d5_ds4_cpu_graph)
    target_compile_features(${t} PRIVATE cxx_std_23)
    target_compile_definitions(${t} PRIVATE HYLE_TEST_TORCH_CPU=1)
    target_link_libraries(${t} PRIVATE hyle_simd GTest::gtest_main ggml-base)
    target_compile_options(${t} PRIVATE -Wall -Wextra -Wpedantic -Werror -Wno-missing-field-initializers -Wno-dangling-else -Wno-misleading-indentation)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endfunction()
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  cmake_language(DEFER CALL d5_ds4_consumers)
endif()
""")
receipt['test_only_warning_overlay']={'path':str(overlay),'sha256':hashlib.sha256(overlay.read_bytes()).hexdigest(),'targets':['hyle_compute_route_cpu_test','hyle_graph_route_cpu_test'],'product_werror':True,'reason':'Existing fixtures intentionally default remaining aggregate fields; gtest macros under an if trigger dangling-else warnings. No product warning flags changed.'}
os.environ['GTEST_FILTER']='*TorchOfficial*'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
run('configure',['cmake','-S',source,'-B',build,'-DCMAKE_PROJECT_INCLUDE='+str(overlay),'-DCMAKE_BUILD_TYPE=Release','-DHYLE_WERROR=ON','-DHYLE_STATIC=ON','-DHYLE_BUILD_TESTS=ON','-DHYLE_INSTALL=OFF','-DHYLE_BUILD_BENCH=OFF','-DHYLE_TEST_INSTRUMENTATION=OFF','-DHYLE_DEV_FACTORS=compute_provider_native;compute_route_cpu;graph_route_cpu','-DHYLE_COMPUTE_CPU=ON','-DHYLE_COMPUTE_ONEDNN=ON','-DHYLE_COMPUTE_TORCH_CPU=ON','-DHYLE_COMPUTE_TORCH_CPU_ROOT=/root/hyle-scratch/libtorch-cpu-2.8.0/libtorch','-DHYLE_COMPUTE_CUDA=OFF','-DHYLE_COMPUTE_APPLE=OFF','-DHYLE_COMPUTE_VULKAN=OFF','-DFETCHCONTENT_FULLY_DISCONNECTED=ON'])
run('build',['cmake','--build',build,'--target','d5_ds4_cpu_direct','d5_ds4_cpu_graph','-j','2'])
run('tests',['ctest','--test-dir',build,'--output-on-failure','--no-tests=error','-R','^d5_ds4_cpu_(direct|graph)$','-j','1','--output-junit',out/'tests.xml'])
data=(build/'Testing/Temporary/LastTest.log').read_bytes();(out/'native-tests.log').write_bytes(data)
import re
assert b'[  SKIPPED ]' not in data and sum(map(int,re.findall(rb'\[  PASSED  \]\s+(\d+) tests?\.',data)))==26
receipt['verdict']='PASS';(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print('WSL native correctness PASS',flush=True)
'''.replace('REMOTE_OUT',repr(remote_out)).replace('CANDIDATE',repr(candidate))
script=out/'run.py';script.write_text(program,encoding='utf-8')
if '--prepare-only' in __import__('sys').argv:
 print('Prepared immutable Linux source archive and existing consumer overlay',candidate);raise SystemExit(0)
r=subprocess.run(['wsl','--','python3',remote_out+'/run.py'],**flags)
raise SystemExit(r.returncode)
