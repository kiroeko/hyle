from pathlib import Path
import hashlib,json,os,subprocess,sys,time
root=Path.cwd();out=root/'.scratch/inference-gaps/g03/graph-late-windows';out.mkdir(exist_ok=True)
env=os.environ.copy()
for line in subprocess.check_output(['cmd','/d','/c',str(root/'.scratch/inference-gaps/g01/cpu-ragged-env.cmd')],text=True).splitlines():
 if '=' in line and not line.startswith('='):
  key,value=line.split('=',1);env[key]=value
build=Path('C:/h7/inference-g01-native-vulkan/build')
names=subprocess.check_output(['git','diff','--name-only','2bc71f56a'],text=True).splitlines()
inputs=[dict(path=n,sha256=hashlib.sha256((root/n).read_bytes()).hexdigest()) for n in names if n.startswith(('include/','src/','tests/'))]
rows=json.loads((out/'result.json').read_bytes())['stages'] if (out/'result.json').exists() else []
stages=[('batch-compile',['cmake','--build',str(build),'--target','hyle_graph_batch','hyle_graph_route_cpu_vulkan','g03_common_receiver_risks','g02_joint_public_compile','g03_owned_outcome','g03_late_result64','g03_pending_graph64','g03_pending_joint64','--parallel','4']),
 ('pending-graph64',[str(build/'g03_pending_graph64.exe')]),
 ('common-receiver-risks',[str(build/'g03_common_receiver_risks.exe')]),
 ('original-outcome-regression',[str(build/'g03_owned_outcome.exe')])]
if len(sys.argv)>1:
 assert sys.argv[1]=='ranges'
 stages=[('copy-range-compile',['cmake','--build',str(build),'--target','g03_copy_ranges','--parallel','4']),('copy-range-tests',[str(build/'g03_copy_ranges.exe')])]
for stage,args in stages:
 attempt=1+sum(r['stage']==stage for r in rows);log=out/f'{stage}-{attempt}.log';started=time.monotonic()
 with log.open('w') as output:
  process=subprocess.Popen(args,env=env,stdout=output,stderr=subprocess.STDOUT)
  (out/'job.json').write_text(json.dumps(dict(state='RUNNING',pid=process.pid,stage=stage,log=str(log))))
  print('RUN',stage,process.pid,flush=True);code=process.wait()
 rows.append(dict(stage=stage,command=args,exit_code=code,elapsed_seconds=time.monotonic()-started,inputs=inputs,log=str(log)))
 (out/'result.json').write_text(json.dumps(dict(scope='G03 GL01-04 batch Windows directed compile and original common/outcome regressions; no native GPU execution, new Graph public numeric journey still required',stages=rows,cumulative_seconds=sum(r['elapsed_seconds'] for r in rows)),indent=2)+'\n',newline='\n')
 (out/'job.json').write_text(json.dumps(dict(state='TERMINAL',exit_code=code)))
 print(json.dumps({k:v for k,v in rows[-1].items() if k!='inputs'}),flush=True)
 if code:
  print(log.read_text(errors='replace')[-5500:],flush=True);raise SystemExit(code)
