from pathlib import Path
import hashlib,json,os,subprocess,sys,time,zipfile
base=Path('/Users/kiro/hyle-scratch/compute-unified');source=base/'source-c8874688';out=base/'g03-graph-late-batch';out.mkdir(exist_ok=True)
identity={key:subprocess.check_output(command,text=True).strip() for key,command in [
 ('model',['/usr/sbin/sysctl','-n','hw.model']),('arch',['uname','-m']),('os',['sw_vers']),
 ('compiler',['xcrun','clang++','--version']),('memory',['/usr/sbin/sysctl','-n','hw.memsize'])]}
assert identity['model']=='Mac16,12' and identity['arch']=='arm64'
assert source.is_dir() and not (source/'.git').exists()
with zipfile.ZipFile(out/sys.argv[1]) as package:
 manifest=json.loads(package.read('source-manifest.json'))
 for item in manifest:
  relative=Path(item['path']);assert not relative.is_absolute() and '..' not in relative.parts
  data=package.read(item['path']);assert hashlib.sha256(data).hexdigest()==item['sha256']
  target=source/relative
  if not target.exists() or target.read_bytes()!=data:
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  assert hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']
digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
env=os.environ.copy();env['PATH']='/Users/kiro/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:'+env['PATH']
record=out/'result.json';rows=json.loads(record.read_bytes())['stages'] if record.exists() else []
def run(route,stage,args):
 attempt=1+sum(row['route']==route and row['stage']==stage for row in rows)
 log=out/f'{route}-{stage}-{attempt}.log';start=time.monotonic()
 with log.open('w') as output:
  child=subprocess.Popen(args,env=env,stdout=output,stderr=subprocess.STDOUT)
  (out/'job.json').write_text(json.dumps(dict(state='RUNNING',route=route,stage=stage,pid=child.pid,log=str(log))))
  print('RUN',route,stage,child.pid,flush=True);code=child.wait()
 row=dict(route=route,stage=stage,attempt=attempt,command=args,exit_code=code,elapsed_seconds=time.monotonic()-start,input_manifest_sha256=digest,log=str(log));rows.append(row)
 record.write_text(json.dumps(dict(identity=identity,scope='G03 GL01-04 directed native source batch; public late producer -> original F2 -> pending direct receiver64. Graph pending numeric journey still required.',inputs=manifest,stages=rows,cumulative_seconds=sum(row['elapsed_seconds'] for row in rows)),indent=2)+'\n')
 (out/'job.json').write_text(json.dumps(dict(state='TERMINAL',exit_code=code)))
 print(json.dumps(row),flush=True)
 if code:
  print(log.read_text(errors='replace')[-5500:],flush=True);raise SystemExit(code)
for route in ('apple','vulkan'):
 if len(sys.argv)>2 and route!=sys.argv[2]:continue
 parent=base/f'a-cpu-{route}/selected-graph/source-parent';build=parent.parent/'source-build'
 cmake=parent/'CMakeLists.txt';text=cmake.read_text()
 target='g03_late_result64'
 if f'add_executable({target} ' not in text:
  text+=f'\nadd_executable({target} {source}/tests/compute_batch/pending_paged_receiver_native_consumer.cc)\ntarget_compile_features({target} PRIVATE cxx_std_23)\ntarget_compile_definitions({target} PRIVATE HYLE_PENDING_NATIVE_RESULT_SOURCE=1'+(' HYLE_PENDING_RECEIVER_APPLE=1' if route=='apple' else '')+f')\ntarget_compile_options({target} PRIVATE -Wall -Wextra -Wpedantic -Werror)\ntarget_link_libraries({target} PRIVATE hyle::graph)\nhyle_stage_runtime({target})\n'
  cmake.write_text(text)
 run(route,'batch-compile',['/opt/homebrew/bin/cmake','--build',str(build),'--target',f'hyle_graph_route_cpu_{route}','hyle_graph_batch',target,'--parallel','4'])
 run(route,'late-result64',[str(build/target)])
