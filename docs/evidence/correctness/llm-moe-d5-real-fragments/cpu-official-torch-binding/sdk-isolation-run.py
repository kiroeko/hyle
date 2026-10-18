import ctypes,json,os,subprocess,time
from pathlib import Path
ctypes.windll.kernel32.SetErrorMode(1|2|0x8000)
flags={'creationflags':subprocess.CREATE_NO_WINDOW}
repo=Path.cwd().resolve()
candidate=subprocess.check_output(['git','rev-parse','HEAD'],text=True,**flags).strip()
out=repo/'.scratch/validation/d5-cpu-apple-official-binding'/(candidate[:12]+'-sdk-isolation-r2')
assert not out.exists();out.mkdir(parents=True)
env=dict(os.environ,VSLANG='1033')
batch=out/'environment.cmd';batch.write_text('@echo off\ncall "C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat" -no_logo -arch=x64\nif errorlevel 1 exit /b 1\nset\n')
r=subprocess.run(['cmd','/d','/c',str(batch)],capture_output=True,text=True,env=env,**flags);r.check_returncode()
for line in r.stdout.splitlines():
 if '=' in line and not line.startswith('='):
  key,value=line.split('=',1);env[key]=value
common=['cmake','-S',str(repo),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DHYLE_BUILD_TESTS=OFF','-DHYLE_INSTALL=OFF','-DHYLE_DEV_FACTORS=compute_provider_native','-DHYLE_COMPUTE_CPU=ON','-DHYLE_COMPUTE_ONEDNN=OFF','-DHYLE_COMPUTE_CUDA=OFF','-DHYLE_COMPUTE_APPLE=OFF','-DHYLE_COMPUTE_VULKAN=OFF']
receipts=[]
for name,option,expected in [('off-invalid-root','OFF',0),('on-invalid-root','ON',1)]:
 build=out/(name+'-build');args=common+['-B',str(build),'-DHYLE_COMPUTE_TORCH_CPU='+option,'-DHYLE_COMPUTE_TORCH_CPU_ROOT='+str(out/'missing-sdk')]
 start=time.monotonic();r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env,**flags)
 (out/(name+'.log')).write_text(r.stdout)
 assert (r.returncode==0)==(expected==0),(name,r.returncode)
 if expected:
  assert 'CUDA Torch is not accepted' in ' '.join(r.stdout.split())
 else:
  assert 'Torch_DIR' not in (build/'CMakeCache.txt').read_text()
 receipts.append({'name':name,'argv':args,'returncode':r.returncode,'seconds':round(time.monotonic()-start,3),'verdict':'PASS'})
 print(name,r.returncode,flush=True)
(out/'receipt.json').write_text(json.dumps({'candidate':candidate,'scope':'owning CPU SDK ON/OFF discovery; configuration only, no correctness/P1 verdict','steps':receipts},indent=2)+'\n')
