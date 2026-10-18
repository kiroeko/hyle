from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path.cwd();out=root/'.scratch/inference-gaps/g03/graph-late-cuda';out.mkdir(exist_ok=True)
attempt=1
while (out/f'source-{attempt}.zip').exists():attempt+=1
archive=out/f'source-{attempt}.zip';inputs=[]
paths=[p for p in subprocess.check_output(['git','ls-files'],text=True).splitlines() if not p.startswith(('docs/evidence/','.scratch/','.git')) and (root/p).is_file()]
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as package:
 for path in paths:
  data=(root/path).read_bytes();inputs.append(dict(path=path,sha256=hashlib.sha256(data).hexdigest()));package.writestr(path,data)
 package.writestr('source-manifest.json',json.dumps(inputs,indent=2)+'\n')
(out/f'source-{attempt}-manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
remote='/mnt/c/public/project-hyle/hyle/.scratch/inference-gaps/g03/'
extra=sys.argv[1:];assert extra in ([],['joint'])
raise SystemExit(subprocess.run(['wsl','-d','Ubuntu-24.04','-u','root','--','python3',remote+'graph_late_cuda.py',remote+'graph-late-cuda/'+archive.name,*extra]).returncode)
