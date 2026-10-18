import hashlib,json,subprocess,time,os
from pathlib import Path
out=Path('/mnt/c/public/project-hyle/hyle/.scratch/validation/d5-cpu-apple-official-binding/e15dc754c52b-glm-wsl-supplement')
build=Path('/root/hyle-scratch/active-cpu-z/build')
source=Path('/root/hyle-scratch/active-cpu-z/source')
header='tests/compute_models/d5_cpu_torch_official_tests.h'
original=Path('/mnt/c/public/project-hyle/hyle')/header
assert source.resolve().is_relative_to(Path('/root/hyle-scratch')) and not (source/'.git').exists()
(source/header).write_bytes(original.read_bytes())
receipt={'consumer_candidate':'e15dc754c52bff02a228b77a03f56ae9c34ef1d5','product_candidate':'3c22b7b9f230abc86e749225174d86b59c79881e','source_header_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'scope':'added non-Flash GLM full MLA/NextN/head; product and previous 13 consumers unchanged','steps':[]}
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['GTEST_FILTER']='*D5GlmOfficialAbsorbedMlaAndNextN'
for name,args in [('build',['cmake','--build',str(build),'--target','d5_ds4_cpu_direct','d5_ds4_cpu_graph','-j','2']),('tests',['ctest','--test-dir',str(build),'--output-on-failure','--no-tests=error','-R','^d5_ds4_cpu_(direct|graph)$','-j','1','--output-junit',str(out/'tests.xml')])]:
 start=time.monotonic()
 with (out/(name+'.log')).open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
 receipt['steps'].append({'name':name,'argv':args,'returncode':r.returncode,'seconds':round(time.monotonic()-start,3)})
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'
')
 print(name,r.returncode,flush=True)
 if r.returncode:print((out/(name+'.log')).read_text()[-7000:]);raise SystemExit(1)
(out/'native-tests.log').write_bytes((build/'Testing/Temporary/LastTest.log').read_bytes())
receipt['verdict']='PASS';(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'
')
