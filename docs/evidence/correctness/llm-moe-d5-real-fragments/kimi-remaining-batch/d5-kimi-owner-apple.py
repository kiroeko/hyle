import subprocess
import sys
from pathlib import Path

source = Path('.scratch/d5-q24-apple-owner.py').read_text(encoding='utf-8')
source = source.replace('d5-q24-semantic-batch', 'd5-kimi-owner')
source = source.replace("'--no-tests=error', '-L'", "'--no-tests=error', '-j', '1', '-L'")
source = source.replace("receipt = {", "out.mkdir(parents=True, exist_ok=True)\nreceipt = {")
out = Path('.scratch/validation/d5-kimi-owner/apple')
out.mkdir(parents=True, exist_ok=True)
driver = out / 'owner.py'
driver.write_text(source, encoding='utf-8')
host = 'kiro@172.22.10.198'
subprocess.run(['scp', '-q', str(driver), host + ':hyle-scratch/d5-kimi-owner.py'], check=True)
with (out / 'driver.log').open('w', encoding='utf-8') as log:
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
        'python3 "$HOME/hyle-scratch/d5-kimi-owner.py"'], stdout=log, stderr=subprocess.STDOUT)
subprocess.run(['scp', '-q', host + ':hyle-scratch/d5-kimi-owner/*', str(out)], check=True)
print((out / 'driver.log').read_text(encoding='utf-8', errors='replace')[-3500:])
native = out / 'owner-native.log'
if native.exists() and '[  SKIPPED ]' in native.read_text(encoding='utf-8', errors='replace'):
    print('SKIP detected; inspect before accepting')
    sys.exit(2)
sys.exit(result.returncode)
