import ctypes,json,os,subprocess,time,sys,shutil,psutil
from pathlib import Path
root=Path.cwd().resolve()
flags={'creationflags':subprocess.CREATE_NO_WINDOW}
ctypes.windll.kernel32.SetErrorMode(1|2|0x8000)
candidate=subprocess.check_output(['git','rev-parse','HEAD'],text=True,**flags).strip()
assert not subprocess.check_output(['git','status','--porcelain'],**flags).strip()
out=root/'.scratch/validation/d5-cpu-apple-official-binding'/(candidate[:12]+'-tests')
assert not out.exists();out.mkdir(parents=True)
build=root/'.scratch/build-compute-v2-dev'
sdk=root/'.scratch/validation/d5-cpu-apple-official-sources/cpu-sdk-full/libtorch'
env=dict(os.environ,VSLANG='1033',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HYLE_D5_QWEN35_FIXTURE=str(root/'.scratch/d5-qwen35-six-profile-fixture'))
batch=out/'environment.cmd'
batch.write_text('@echo off\ncall "C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat" -no_logo -arch=x64\nif errorlevel 1 exit /b 1\nset VSLANG=1033\nset\n')
r=subprocess.run(['cmd','/d','/c',str(batch)],capture_output=True,text=True,env=env,**flags);r.check_returncode()
for line in r.stdout.splitlines():
 if '=' in line and not line.startswith('='):
  key,value=line.split('=',1);env[key]=value
receipt={'candidate':candidate,'steps':[],'threads':{'OMP_NUM_THREADS':env['OMP_NUM_THREADS'],'MKL_NUM_THREADS':env['MKL_NUM_THREADS']},'scope':'Native CPU mature low precision plus ordinary CPU Direct/Graph; no Apple or P1 verdict'}
def run(name,args):
 start=time.monotonic();args=list(map(str,args))
 peak=0;peak_executable='';stop_reason=None
 with (out/(name+'.log')).open('w',encoding='utf-8') as log:
  r=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,**flags)
  while r.poll() is None:
   try:
    processes=[psutil.Process(r.pid),*psutil.Process(r.pid).children(recursive=True)]
    for proc in processes:
     try:
      memory=proc.memory_info().rss
      if memory>peak:peak=memory;peak_executable=proc.name()
     except psutil.Error:pass
    if name=='tests' and (peak>16*1024**3 or time.monotonic()-start>900):
     stop_reason='owned test process exceeded 16 GiB working set or 900 second batch bound'
     for proc in reversed(processes):
      try:proc.kill()
      except psutil.Error:pass
   except psutil.Error:pass
   time.sleep(0.5)
  r.wait()
 receipt['steps'].append({'name':name,'argv':args,'returncode':r.returncode,'seconds':round(time.monotonic()-start,3),'observed_process_peak_rss':peak,'peak_executable':peak_executable,'stop_reason':stop_reason})
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(name,r.returncode,receipt['steps'][-1]['seconds'],flush=True)
 if r.returncode:
  print((out/(name+'.log')).read_text(errors='replace')[-3500:],flush=True);raise SystemExit(r.returncode)
if '--build-only' in sys.argv:
 prior=root/'.scratch/validation/d5-cpu-apple-official-binding/8860c1186b78-tests/receipt.json'
 earlier=json.loads(prior.read_text());assert earlier['steps'][0]['returncode']==0
 unchanged=['CMakeLists.txt','cmake','src/compute_provider_native/CMakeLists.txt','src/compute_provider_native/torch_cpu_sdk.cmake','tests/compute_provider_native/CMakeLists.txt','tests/compute_route_cpu/CMakeLists.txt','tests/graph_route_cpu/CMakeLists.txt']
 assert not subprocess.check_output(['git','diff','--name-only',earlier['candidate'],candidate,'--',*unchanged],**flags).strip()
 receipt['reused_configuration']={'receipt':str(prior),'candidate':earlier['candidate'],'sdk':str(sdk)}
else:
 run('configure',['cmake','-S',root,'-B',build,'-DHYLE_BUILD_TESTS=ON','-DHYLE_INSTALL=OFF','-DHYLE_STATIC=OFF','-DHYLE_COMPUTE_TORCH_CPU=ON','-DHYLE_COMPUTE_TORCH_CPU_ROOT='+str(sdk)])
run('build',['cmake','--build',build,'--target','hyle_compute_provider_native_tests','hyle_compute_route_cpu_tests','hyle_graph_route_cpu_tests','-j','2'])
# Copy CPU DLLs beside every target executable; never depend on Python/CUDA PATH.
dlls={}
for exe in build.rglob('*_test.exe'):
 for dll in (sdk/'lib').glob('*.dll'):
  destination=exe.parent/dll.name
  if not destination.exists() or destination.stat().st_size!=dll.stat().st_size:shutil.copyfile(dll,destination)
  dlls[str(destination.relative_to(build))]=str(dll.relative_to(sdk))
(out/'dll-origins.json').write_text(json.dumps(dlls,indent=2)+'\n')
run('tests',['ctest','--test-dir',build,'--output-on-failure','--no-tests=error','-R','^hyle_(compute_provider_native|compute_route_cpu|graph_route_cpu)(_.*)?_test$','-j','1','--output-junit',out/'tests.xml'])
(out/'native-tests.log').write_bytes((build/'Testing/Temporary/LastTest.log').read_bytes())
print('CPU batch finished; inspect all full-profile assertions before any ledger verdict.',flush=True)
