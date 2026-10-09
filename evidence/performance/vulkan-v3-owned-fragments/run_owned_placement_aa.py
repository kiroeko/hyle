"""One bounded paired A/A for the owned native-plan/Graph contract."""
from pathlib import Path
import ctypes, hashlib, importlib.util, json, math, os, re, statistics, subprocess, sys, time, uuid
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from hyle_perf.environment import PERF_ENVIRONMENT_BLOCKLIST,PERF_ENVIRONMENT_FIXED,PERF_ENVIRONMENT_PREFIXES
from hyle_perf.machine import get_machine_profile,load_registry,probe_machine,evaluate_machine_profile,expected_child_policy
from hyle_perf.runner import _apply_sampling_policy,_required_telemetry
OUT=Path(__file__).resolve().parent
EXE=Path('C:/hv/vp1/bin/hyle_graph_route_vulkan_owned_placement.exe')
TITLES=('Qwen3.5-2B official BF16 FFN M1 H2048 N6144','DeepSeek V4 Flash UD-Q4_K_XL hash-MoE H4096 N2048 top6')
ctypes.windll.kernel32.SetErrorMode(1|2|0x8000)
FLAGS=subprocess.CREATE_NO_WINDOW|subprocess.ABOVE_NORMAL_PRIORITY_CLASS
receipt=OUT/'owned-placement-aa-receipt.json'
assert not receipt.exists(),'one-shot owned contract A/A budget already used'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True,creationflags=subprocess.CREATE_NO_WINDOW).strip(),'clean immutable candidate required'
candidate=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,creationflags=subprocess.CREATE_NO_WINDOW).strip()
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
identity={'candidate':candidate,'binary':digest(EXE),'runner':digest(__file__),'dlls':{p.name:digest(p) for p in EXE.parent.glob('hyle_*.dll')}}
profile=get_machine_profile(load_registry(),'W1-WIN');probe=probe_machine()
gate=evaluate_machine_profile(profile,probe,role='engineering',sampling_topology='native-runtime-pair')
if gate['gate']!='PASS':raise RuntimeError(gate)
policy=expected_child_policy(profile,sampling_topology='native-runtime-pair',probe=probe)
raw=OUT/'owned-placement-aa-raw';raw.mkdir(exist_ok=False)
runs=[];errors=[];start=time.monotonic()
try:
 with _apply_sampling_policy(policy):
  for arm in ('selected','peer'):
   for order in (0,1):
    for allocation in (0,1):
     env={k:v for k,v in os.environ.items() if k not in PERF_ENVIRONMENT_BLOCKLIST and not any(k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
     env.update(PERF_ENVIRONMENT_FIXED);nonce=uuid.uuid4().hex
     env.update(HYLE_D5_QWEN35_FIXTURE=str(ROOT/'.scratch/d5-qwen35-six-profile-fixture'),HYLE_BENCH_AA_ROUTE=arm,HYLE_PERF_RUN_NONCE=nonce,HYLE_PERF_CALLER_CPU=str(policy['caller_logical_cpu']),HYLE_PERF_SAMPLING_TOPOLOGY='native-runtime-pair')
     if order:env['HYLE_BENCH_RIVAL_FIRST']='1'
     if allocation:env['HYLE_BENCH_PEER_ALLOCATION_FIRST']='1'
     log=raw/f'{arm}-{order}-{allocation}.log'
     with log.open('wb') as stream:
      code=subprocess.run([str(EXE)],cwd=EXE.parent,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=FLAGS,timeout=180).returncode
     text=log.read_text(encoding='utf-8',errors='replace')
     if code:raise RuntimeError(str(log)+': child exit '+str(code))
     _required_telemetry(text,['# HYLE_D5_FRAGMENT_RUN '],{'execution_order':'peer-first' if order else 'candidate-first','allocation_order':'peer-first' if allocation else 'candidate-first','aa_route':arm,'paired_mode':'nanobench-compare','run_nonce':nonce,'verify_only':'0','affinity_topology':'native-runtime-pair','caller_cpu':str(policy['caller_logical_cpu']),'affinity_ok':'1','parent_mask':policy['affinity_mask'],'priority':policy['priority']})
     records=re.findall(r'# D5_PAIRED_ROUND profile="([^"]+)" arm="([^"]+)" round=(\d+) iterations=(\d+) seconds_per_call=(\S+) paired_relative=(\S+) slots=(\d+)',text)
     if len(records)!=96:raise RuntimeError('incomplete paired records')
     pairs={}
     for title,name,round_,iterations,seconds,relative,slots in records:
      if title not in TITLES or name not in ('Graph complete fragment','Direct complete fragment') or int(iterations)!=6 or int(slots)!=(2 if title==TITLES[0] else 3):raise RuntimeError('wrong original cycle or subject')
      key=(title,name,int(round_));value=float(seconds)
      if key in pairs or not math.isfinite(value) or value<=0:raise RuntimeError('duplicate/invalid round')
      pairs[key]=value
     ratios={title:statistics.median(pairs[(title,'Direct complete fragment',i)]/pairs[(title,'Graph complete fragment',i)] for i in range(24)) for title in TITLES}
     runs.append({'arm':arm,'order':order,'allocation':allocation,'ratios':ratios,'log':log.relative_to(OUT).as_posix(),'sha256':digest(log),'nonce':nonce})
     print(json.dumps({'done':len(runs),'total':8,'ratios':ratios}),flush=True)
except Exception as error:
 errors.append(str(error))
summaries={}
for arm in ('selected','peer'):
 group=[r for r in runs if r['arm']==arm]
 if len(group)!=4:errors.append(arm+': incomplete four-factor A/A');continue
 for title in TITLES:
  values=[r['ratios'][title] for r in group];medians={'all':statistics.median(values)}
  medians.update({f'{f}={l}':statistics.median(r['ratios'][title] for r in group if r[f]==l) for f in ('order','allocation') for l in (0,1)})
  if any(not 1/1.10<=v<=1.10 for v in values):errors.append(arm+'/'+title+': gross10%')
  if any(not 1/1.05<=v<=1.05 for v in medians.values()):errors.append(arm+'/'+title+': factor5%')
  summaries[arm+'/'+title]={'ratios':values,'factor_medians':medians}
if digest(EXE)!=identity['binary']:errors.append('binary changed')
result={'schema':'vulkan-owned-paired-aa/v1','identity':identity,'machine':gate,'policy':policy,'gate':'FAIL' if errors else 'PASS','runs':runs,'summaries':summaries,'errors':errors,'seconds':time.monotonic()-start,'product_samples':0,'comparison':'same owned native plan service; old Recorder AA not reused'}
receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps({'gate':result['gate'],'errors':errors,'seconds':result['seconds']}),flush=True)
raise SystemExit(bool(errors))
