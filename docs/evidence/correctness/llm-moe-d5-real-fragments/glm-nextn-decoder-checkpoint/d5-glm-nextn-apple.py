from pathlib import Path
import ctypes
import json
import subprocess
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
out = Path('.scratch/validation/d5-glm-nextn-corrected/apple')
out.mkdir(parents=True, exist_ok=True)
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
candidate = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, **flags).strip()
(out / 'candidate.json').write_text(json.dumps({'candidate': candidate}, indent=2) + '\n', encoding='utf-8')
host = 'kiro@172.22.10.198'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host]
bundle = out / 'candidate.bundle'
subprocess.run(['git', 'bundle', 'create', str(bundle), '9d8d90e4..HEAD'], check=True, **flags)
subprocess.run(['scp', '-q', str(bundle), host + ':hyle-scratch/d5-glm-nextn.bundle'], check=True, **flags)
with (out / 'sync.log').open('w', encoding='utf-8') as log:
    subprocess.run(ssh + ['cd "$HOME/hyle" && git fetch "$HOME/hyle-scratch/d5-glm-nextn.bundle" HEAD && git merge --ff-only FETCH_HEAD && git status --short && git rev-parse HEAD'], stdout=log, stderr=subprocess.STDOUT, check=True, **flags)
tree = '$HOME/hyle-scratch/d5-graph-825c41f8'
command = 'export PATH="$HOME/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:$PATH"; '
command += f'cmake --build "{tree}" --target hyle_compute_route_apple_tests hyle_graph_route_apple_tests -j 2'
filter_ = '*MoeFragments.D5Glm*:*MoeFragments.D5EightGgufProfilesFullMoe:*MoeFragments.*OfficialKimi*'
for entry in ('compute', 'graph'):
    command += f' && "{tree}/bin/hyle_{entry}_route_apple_test" --gtest_filter="{filter_}"'
path = out / 'build-and-test.log'
with path.open('w', encoding='utf-8') as log:
    result = subprocess.run(ssh + [command], stdout=log, stderr=subprocess.STDOUT, **flags)
print('apple batch', result.returncode, flush=True)
output = path.read_text(encoding='utf-8', errors='replace')
print('\n'.join(line for line in output.splitlines() if '[  PASSED  ]' in line or '[  FAILED  ]' in line or '[  SKIPPED ]' in line), flush=True)
if result.returncode:
    print(output[-4000:], flush=True)
raise SystemExit(result.returncode)
