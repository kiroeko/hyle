import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
root = Path.cwd().resolve()
out = root / '.scratch/validation/d5-moe-semantic/apple'
out.mkdir(parents=True, exist_ok=True)
host = 'kiro@172.22.10.198'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host]
action = sys.argv[1]
if action == 'sync':
    bundle = out / 'candidate.bundle'
    subprocess.run(['git', 'bundle', 'create', str(bundle), '877c9f5b..HEAD'], check=True)
    subprocess.run(['scp', '-q', str(bundle), host + ':hyle-scratch/d5-moe-semantic.bundle'], check=True)
    subprocess.run(ssh + ['cd "$HOME/hyle" && git fetch "$HOME/hyle-scratch/d5-moe-semantic.bundle" HEAD && git merge --ff-only FETCH_HEAD && git log -1 --oneline'], check=True)
else:
    command = 'export PATH="$HOME/hyle-verify-tools/metal-17F109/shim:/opt/homebrew/bin:$PATH"; '
    tree = '$HOME/hyle-scratch/d5-graph-825c41f8'
    command += f'cmake --build "{tree}" --target hyle_compute_route_apple_tests hyle_graph_route_apple_tests -j 2'
    for entry in ('compute', 'graph'):
        command += f' && "{tree}/bin/hyle_{entry}_route_apple_test" --gtest_filter="*MoeFragments.*:D5ExactMoeVocabulary/*"'
    path = out / 'build-and-test.log'
    with path.open('w', encoding='utf-8') as log:
        result = subprocess.run(ssh + [command], stdout=log, stderr=subprocess.STDOUT)
    print('apple batch', result.returncode, flush=True)
    print(path.read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)
    raise SystemExit(result.returncode)
