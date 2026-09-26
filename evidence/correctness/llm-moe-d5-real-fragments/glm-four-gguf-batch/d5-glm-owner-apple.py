from pathlib import Path
import ctypes
import subprocess

ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
host = 'kiro@172.22.10.198'
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host]
out = Path('.scratch/validation/d5-glm-owner/apple')
out.mkdir(parents=True, exist_ok=True)
bundle = out / 'candidate.bundle'
subprocess.run(['git', 'bundle', 'create', str(bundle), '3fb5e818..HEAD'], check=True, **flags)
subprocess.run(['scp', '-q', str(bundle), host + ':hyle-scratch/d5-glm-owner.bundle'], check=True, **flags)
with (out / 'sync.log').open('w', encoding='utf-8') as log:
    subprocess.run(ssh + ['cd "$HOME/hyle" && git fetch "$HOME/hyle-scratch/d5-glm-owner.bundle" HEAD && git merge --ff-only FETCH_HEAD && git status --short && git rev-parse HEAD'], check=True, stdout=log, stderr=subprocess.STDOUT, **flags)
source = Path('.scratch/d5-q24-apple-owner.py').read_text(encoding='utf-8')
source = source.replace('d5-q24-semantic-batch', 'd5-glm-owner')
source = source.replace("'--no-tests=error', '-L'", "'--no-tests=error', '-j', '1', '-L'")
source = source.replace('receipt = {', 'out.mkdir(parents=True, exist_ok=True)\nreceipt = {')
# Existing Release/WERROR tree is already configured; CMake's build regeneration
# handles changed sources. Retain and verify the actual cache identity instead.
source = '\n'.join(line for line in source.splitlines() if "('owner-configure'" not in line) + '\n'
source = source.replace("receipt = {", "assert 'HYLE_WERROR:BOOL=ON' in (build / 'CMakeCache.txt').read_text()\nreceipt = {")
driver = out / 'owner.py'
driver.write_text(source, encoding='utf-8')
subprocess.run(['scp', '-q', str(driver), host + ':hyle-scratch/d5-glm-owner.py'], check=True, **flags)
with (out / 'driver.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(ssh + ['python3 "$HOME/hyle-scratch/d5-glm-owner.py"'], stdout=log, stderr=subprocess.STDOUT, **flags)
subprocess.run(['scp', '-q', host + ':hyle-scratch/d5-glm-owner/*', str(out)], check=True, **flags)
print((out / 'driver.log').read_text(encoding='utf-8', errors='replace')[-3500:])
native = out / 'owner-native.log'
if native.exists() and '[  SKIPPED ]' in native.read_text(encoding='utf-8', errors='replace'):
    raise SystemExit('SKIP detected; this is not a passing receipt')
raise SystemExit(result.returncode)
