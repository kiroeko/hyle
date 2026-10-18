import ctypes,json,os,subprocess,time,shutil,psutil,sys
from pathlib import Path
ctypes.windll.kernel32.SetErrorMode(1|2|0x8000)
flags={'creationflags':subprocess.CREATE_NO_WINDOW}
repo=Path.cwd().resolve();candidate=subprocess.check_output(['git','rev-parse','HEAD'],text=True,**flags).strip()
assert not subprocess.check_output(['git','status','--porcelain'],**flags).strip()
out=repo/'.scratch/validation/d5-cpu-apple-official-binding'/(candidate[:12]+'-glm-carry-supplement')
assert not out.exists();out.mkdir(parents=True)
consumer=Path('C:/hv/d5cp/consumer');header='tests/compute_models/d5_cpu_torch_official_tests.h'
shutil.copyfile(repo/header,consumer/header)
product='3c22b7b9f230abc86e749225174d86b59c79881e'
assert not subprocess.check_output(['git','diff','--name-only',product,candidate,'--','include','src','cmake','third_party','CMakeLists.txt'],**flags).strip()
receipt={'candidate':candidate,'product_candidate':product,'scope':'added non-Flash GLM full cold KV/absorbed MLA/NextN/head consumer; reuse unchanged 13-case product evidence','steps':[]}
env=dict(os.environ,VSLANG='1033',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
batch=out/'environment.cmd';batch.write_text('@echo off\ncall "C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat" -no_logo -arch=x64\nif errorlevel 1 exit /b 1\nset\n')
r=subprocess.run(['cmd','/d','/c',str(batch)],capture_output=True,text=True,env=env,**flags);r.check_returncode()
for line in r.stdout.splitlines():
 if '=' in line and not line.startswith('='):
  key,value=line.split('=',1);env[key]=value
sdk=repo/'.scratch/validation/d5-cpu-apple-official-sources/cpu-sdk-full/libtorch'
for stage,build in [('source',Path('C:/hv/d5cp/source-build')),('static',Path('C:/hv/d5cp/official-static-consumer')),('shared',Path('C:/hv/d5cp/official-shared-consumer'))]:
 args=['cmake','--build',str(build),'--target','d5_cpu_delivery','-j','2']
 with (out/(stage+'-build.log')).open('w') as log:
  start=time.monotonic();r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env,**flags)
 receipt['steps'].append({'stage':stage,'name':'build','argv':args,'returncode':r.returncode,'seconds':round(time.monotonic()-start,3)})
 assert r.returncode==0,(out/(stage+'-build.log')).read_text()[-3000:]
 for route in ['direct','graph']:
  for dll in (sdk/'lib').glob('*.dll'):
   target=build/dll.name
   assert target.read_bytes()==dll.read_bytes()
  args=[str(build/('d5_cpu_'+route+'.exe')),'--gtest_filter=*D5GlmOfficialAbsorbedMlaAndNextN','--gtest_output=xml:'+str(out/(stage+'-'+route+'.xml'))]
  peak=0;start=time.monotonic()
  with (out/(stage+'-'+route+'.log')).open('w') as log:
   process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,**flags)
   while process.poll() is None:
    try:peak=max(peak,psutil.Process(process.pid).memory_info().rss)
    except psutil.Error:pass
    if peak>16*1024**3 or time.monotonic()-start>600:process.kill()
    time.sleep(.5)
   process.wait()
  receipt['steps'].append({'stage':stage,'name':route,'argv':args,'returncode':process.returncode,'seconds':round(time.monotonic()-start,3),'observed_process_peak_rss':peak})
  (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
  print(stage,route,process.returncode,receipt['steps'][-1]['seconds'],flush=True)
  if process.returncode:print((out/(stage+'-'+route+'.log')).read_text()[-6000:]);raise SystemExit(1)
receipt['verdict']='PASS';(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
