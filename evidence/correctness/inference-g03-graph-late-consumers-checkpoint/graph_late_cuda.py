from pathlib import Path
import hashlib,json,os,re,shutil,subprocess,sys,time,zipfile
canonical=Path('/mnt/c/public/project-hyle/hyle');base=Path('/root/hyle-scratch/compute-unified')
source=base/'source-7a7b4c3b';parent=base/'l-cpu-cuda/selected-graph/source-parent';build=parent.parent/'source-build'
out=canonical/'.scratch/inference-gaps/g03/graph-late-cuda';out.mkdir(exist_ok=True)
assert source.is_dir() and not (source/'.git').exists()
with zipfile.ZipFile(sys.argv[1]) as package:
 inputs=json.loads(package.read('source-manifest.json'))
 for item in inputs:
  path=Path(item['path']);assert not path.is_absolute() and '..' not in path.parts
  data=package.read(item['path']);assert hashlib.sha256(data).hexdigest()==item['sha256']
  target=source/path
  if not target.exists() or target.read_bytes()!=data:
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
digest=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
identity={name:subprocess.check_output(args,text=True).strip() for name,args in [
 ('host',['uname','-a']),('compiler',['c++','--version']),
 ('gpu',['nvidia-smi','--query-gpu=name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'])]}
env=os.environ.copy();env['PATH']='/usr/local/cuda-13.3/bin:'+env['PATH']
env['LD_LIBRARY_PATH']='/usr/local/cuda-13.3/targets/x86_64-linux/lib:/usr/lib/wsl/lib:'+env.get('LD_LIBRARY_PATH','')
env['LD_LIBRARY_PATH']=str(build/'hyle/src/compute_ggml_runtime')+':'+env['LD_LIBRARY_PATH']
joint=len(sys.argv)>2 and sys.argv[2]=='joint'
target='g03_pending_cpucuda_graph64' if joint else 'g03_pending_cuda_graph64';cmake=parent/'CMakeLists.txt';body=cmake.read_text()
def write_parent(text):
 if cmake.read_text()!=text:cmake.write_text(text)
if f'add_executable({target} ' not in body:
 body+=f'\nadd_executable({target} {source}/tests/compute_batch/pending_paged_receiver_cpu_consumer.cc'+(f' {source}/src/graph_route_cpu_cuda/compile.cc' if joint else '')+f')\ntarget_compile_features({target} PRIVATE cxx_std_23)\ntarget_compile_definitions({target} PRIVATE HYLE_PENDING_GRAPH_RECEIVER=1 HYLE_PENDING_GRAPH_CUDA=1'+(' HYLE_PENDING_GRAPH_CPU_CUDA=1' if joint else '')+f')\ntarget_compile_options({target} PRIVATE -Wall -Wextra -Wpedantic -Werror)\ntarget_link_libraries({target} PRIVATE hyle_graph_batch hyle_compute_backend_cuda hyle_compute_provider_flashinfer'+(' hyle_compute_backend_cpu hyle_compute_provider_native' if joint else '')+f')\nhyle_stage_runtime({target})\n'
 write_parent(body)
old=f'target_link_libraries({target} PRIVATE hyle::graph hyle_compute_provider_flashinfer)'
if old in body:
 body=body.replace(old,f'target_link_libraries({target} PRIVATE hyle_graph_batch hyle_compute_provider_flashinfer)');write_parent(body)
old=f'target_link_libraries({target} PRIVATE hyle_graph_batch hyle_compute_provider_flashinfer)'
if old in body:
 body=body.replace(old,f'target_link_libraries({target} PRIVATE hyle_graph_batch hyle_compute_backend_cuda hyle_compute_provider_flashinfer)');write_parent(body)
if joint and 'g03_cached_cpu_math' not in body:
 vendor=build/'hyle/src/compute_ggml_runtime/vendor/src'
 libraries=[build/'hyle/src/compute_provider_ggml/libhyle_compute_provider_ggml.a',build/'hyle/src/compute_ggml_runtime/libhyle_compute_ggml_runtime.a',vendor/'libggml.a',vendor/'libggml-cpu.a',vendor/'libggml-base.a']
 assert all(p.is_file() for p in libraries)
 body+='\nadd_library(g03_cached_cpu_math INTERFACE)\n'
 body+='target_link_libraries(g03_cached_cpu_math INTERFACE '+ ' '.join(str(p) for p in libraries)+' OpenMP::OpenMP_CXX)\n'
 body+=f'target_link_libraries({target} PRIVATE g03_cached_cpu_math)\ntarget_compile_definitions({target} PRIVATE HYLE_COMPUTE_GGML_CPU=1)\n'
 write_parent(body)
if joint:
 body=body.replace('find_package(OpenMP REQUIRED COMPONENTS CXX)\n','')
 body=body.replace('add_library(g03_cached_cpu_math INTERFACE)','find_package(OpenMP REQUIRED COMPONENTS CXX)\nadd_library(g03_cached_cpu_math INTERFACE)')
 vendor=build/'hyle/src/compute_ggml_runtime/vendor/src'
 libraries=[build/'hyle/src/compute_provider_ggml/libhyle_compute_provider_ggml.a',build/'hyle/src/compute_ggml_runtime/libhyle_compute_ggml_runtime.a',vendor/'libggml.a',vendor/'libggml-cpu.a',vendor/'libggml-base.a']
 borrowed=[]
 for library in libraries:
  data=library.read_bytes();sha=hashlib.sha256(data).hexdigest()
  frozen=base/'g03-graph-late-cached-cpu'/sha/library.name
  frozen.parent.mkdir(parents=True,exist_ok=True)
  if not frozen.exists():shutil.copy2(library,frozen)
  assert hashlib.sha256(frozen.read_bytes()).hexdigest()==sha
  borrowed.append(dict(original=str(library),frozen=str(frozen),bytes=len(data),sha256=sha))
  body=body.replace(str(library),str(frozen))
 body=re.sub(r'target_link_libraries\(g03_cached_cpu_math INTERFACE[^\n]*',
     'target_link_libraries(g03_cached_cpu_math INTERFACE "$<LINK_GROUP:RESCAN,'+
     ','.join([i['frozen'] for i in borrowed]+['hyle_compute_ops','hyle_compute','hyle_mem',str(build/'hyle/src/mem/libmimalloc_static.a'),'hyle_simd','hyle_error'])+'>" OpenMP::OpenMP_CXX)',body)
 (out/'cached-cpu-artifacts.json').write_text(json.dumps(dict(scope='Immutable copies of existing original CPU GGML artifacts for actual joint source diagnostic; whole-factor rebuild, relocation and complete artifact input qualification remain unproven',libraries=borrowed,source_inputs=[i for i in inputs if i['path'].startswith(('src/compute_provider_ggml/','src/compute_ggml_runtime/','include/hyle/compute/providers/ggml_cpu.h'))]),indent=2)+'\n')
 write_parent(body)
rows=json.loads((out/'result.json').read_bytes())['stages'] if (out/'result.json').exists() else []
def record(stage,args,code,seconds,log):
 rows.append(dict(stage=stage,command=args,exit_code=code,elapsed_seconds=seconds,input_manifest_sha256=digest,log=str(log)))
 (out/'result.json').write_text(json.dumps(dict(scope='G03 new common CUDA Graph pending64, original kernel artifacts only; whole G03 unproven',identity=identity,inputs=inputs,stages=rows,cumulative_seconds=sum(r['elapsed_seconds'] for r in rows)),indent=2)+'\n')
 (out/'job.json').write_text(json.dumps(dict(state='TERMINAL',exit_code=code)))
 print(json.dumps(rows[-1]),flush=True)
def run(stage,args):
 log=out/f'{stage}-{1+sum(r["stage"]==stage for r in rows)}.log';start=time.monotonic()
 with log.open('w') as output:
  process=subprocess.Popen(args,env=env,stdout=output,stderr=subprocess.STDOUT)
  (out/'job.json').write_text(json.dumps(dict(state='RUNNING',stage=stage,pid=process.pid,log=str(log))))
  print('RUN',stage,process.pid,flush=True);code=process.wait()
 record(stage,args,code,time.monotonic()-start,log)
 if code:print(log.read_text(errors='replace')[-6500:],flush=True);raise SystemExit(code)
configuration=hashlib.sha256(body.encode()+json.dumps([i for i in inputs if i['path'].startswith('cmake/') or i['path'].endswith('CMakeLists.txt')],sort_keys=True).encode()).hexdigest()
cached=out/'configured-inputs-sha256.txt'
if not cached.exists() or cached.read_text()!=configuration:
 run('configure',['cmake','-S',str(parent),'-B',str(build)]);cached.write_text(configuration)
snapshot=build/'g03-graph-late-preflight.ninja';original=(build/'build.ninja').read_bytes();snapshot.write_bytes(original)
args=['ninja','-C',str(build),'-f',str(snapshot),'-n',target];start=time.monotonic()
dry=subprocess.run(args,env=env,text=True,capture_output=True);log=out/f'dry-run-{1+sum(r["stage"]=="no-kernel-build-preflight" for r in rows)}.log';log.write_text(dry.stdout+dry.stderr)
kernel=[line for line in dry.stdout.splitlines() if '.cu.o' in line or 'nvcc' in line]
assert 'Re-running CMake' not in dry.stdout
artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (build/'hyle/src/compute_provider_flashinfer').glob('libhyle_compute_provider_flashinfer*') if p.is_file()]
(out/(log.stem+'-artifacts.json')).write_text(json.dumps(dict(artifacts=artifacts,kernel_lines=kernel,original_manifest_sha256=hashlib.sha256(original).hexdigest(),dry_manifest_identical=snapshot.read_bytes()==original,actual_build_uses_original_manifest=True),indent=2)+'\n')
record('no-kernel-build-preflight',args,90 if kernel else dry.returncode,time.monotonic()-start,log)
if kernel:print('UNPROVEN: requires new kernel compilation; no build/runtime launched',flush=True);raise SystemExit(90)
if dry.returncode:raise SystemExit(dry.returncode)
run('joint-graph-compile' if joint else 'graph-compile',['cmake','--build',str(build),'--target',target,'--parallel','4'])
run('joint-graph64' if joint else 'graph64',[str(build/target)])
