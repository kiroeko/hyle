import json
from pathlib import Path
import subprocess
old = json.loads(Path('evidence/correctness/llm-moe-d5-real-fragments/qwen-nine-profile-complete-gate/loc.json').read_text(encoding='utf-8'))
args = ['python', 'scripts/loc-report.py', '--module', 'compute-graph', '--json']
for path in old['module']['paths']:
    args += ['--module-path', path]
out = Path('.scratch/validation/d5-moe-semantic/loc.json')
with out.open('w', encoding='utf-8') as stream:
    result = subprocess.run(args, stdout=stream)
if result.returncode:
    raise SystemExit(result.returncode)
report = json.loads(out.read_text(encoding='utf-8'))
print(report['repository'])
for item in report['metrics']:
    if item['id'].startswith('module'):
        print(item)
