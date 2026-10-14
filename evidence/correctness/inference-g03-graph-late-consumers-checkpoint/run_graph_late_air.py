from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
root=Path.cwd();out=root/'.scratch/inference-gaps/g03/graph-late-air';out.mkdir(exist_ok=True)
attempt=1
while (out/f'source-{attempt}.zip').exists():attempt+=1
archive=out/f'source-{attempt}.zip'
# Complete canonical tracked source snapshot, including current worktree bytes.
# Evidence/git metadata are not build inputs; no hand-picked header overlay.
paths=[p for p in subprocess.check_output(['git','ls-files'],text=True).splitlines() if not p.startswith(('evidence/','.scratch/','.git')) and (root/p).is_file()]
manifest=[]
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as package:
 for path in paths:
  data=(root/path).read_bytes();manifest.append(dict(path=path,sha256=hashlib.sha256(data).hexdigest()));package.writestr(path,data)
 package.writestr('source-manifest.json',json.dumps(manifest,indent=2)+'\n')
(out/f'source-{attempt}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
host='kiro@172.22.10.198';remote='/Users/kiro/hyle-scratch/compute-unified/g03-graph-late-batch';script=root/'.scratch/inference-gaps/g03/graph_late_air.py'
subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,'mkdir -p '+remote],check=True)
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=12',str(archive),str(script),f'{host}:{remote}/'],check=True)
route=sys.argv[1] if len(sys.argv)>1 else None
assert route in (None,'all','apple','vulkan')
mode=sys.argv[2] if len(sys.argv)>2 else None
assert mode in (None,'joint','reduction')
command=f'python3 {remote}/{script.name} {archive.name}'+(f' {route}' if route else '')
if mode:command+=' '+mode
code=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,command]).returncode
for name in ('result.json','job.json'):
 subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=12',f'{host}:{remote}/{name}',str(out/name)],check=True)
raise SystemExit(code)
