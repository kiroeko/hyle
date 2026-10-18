# One-time exact cleanup for the completed HTTP2 matched-execution probe.
# Preserve source, logs, raw samples, maps, archives and shared caches.
$ErrorActionPreference = 'Stop'
$archive = Join-Path $PSScriptRoot 'evidence/performance/library-review-2026-10-18/http2/matched-execution/source-results.zip'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'd36e92e4c495a8019438be09e057b26a9e8fd80fa5982d57d3eb854b7163fbef') { throw 'Frozen evidence archive mismatch' }
$pythonSource = @'
import hashlib, json, os, stat
from pathlib import Path

root = Path('/root/hyle-scratch/http2-current-20261018/http2-matched-execution')
targets = [
    (root / 'main.o', 143608, '0fd1578f63b371181f43f0bbd40cb39cc06ddd571fb3b79412700f15ee3a151e'),
    (root / 'probe', 892096, '7058d3f30a9a2e07ef9c49f4afaae63208a0c47b89bea3a36fe37594cae60191'),
]
retained = [
    (Path('/root/hyle-scratch/http2-current-20261018/current-holdout/libhyle_io_core.so.0.1.0'), 387104, '9ead235b1bbd72f56d44eab9c8ac135b8c4b0ce0ec0a6fb72660aa917d3b4779'),
    (Path('/root/hyle-scratch/http2-current-20261018/discard-frame/http2.o'), 223280, '0de3047e82522fddeaf24b57d576f4c7393901d9609217577ec2d5fc566303b6'),
]
assert root.resolve(strict=True) == root
assert len({x[0] for x in targets + retained}) == 4
assert sum(x[1] for x in targets) == 1035704

def inspect(item):
    path, size, expected_sha = item
    assert path.is_absolute() and path.is_relative_to(Path('/root/hyle-scratch/http2-current-20261018'))
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        assert not stat.S_ISLNK(current.lstat().st_mode), str(current)
    assert path.resolve(strict=True) == path
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_size == size
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as stream:
        opened = os.fstat(stream.fileno())
        assert (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino)
        digest = hashlib.sha256(stream.read()).hexdigest()
        after = os.fstat(stream.fileno())
    current = path.lstat()
    assert digest == expected_sha and after.st_size == size
    assert (current.st_dev, current.st_ino, current.st_size) == (after.st_dev, after.st_ino, after.st_size)
    return current.st_dev, current.st_ino

identities = {x[0]: inspect(x) for x in targets + retained}
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit():
        continue
    try:
        executable = os.readlink(proc / 'exe')
    except (OSError, PermissionError):
        continue
    assert executable not in {str(x[0]) for x in targets}, 'Target executable still running: ' + proc.name
print(json.dumps({'preflight_files': 4, 'delete_files': 2, 'delete_bytes': 1035704, 'retain_files': 2}), flush=True)
for item in targets:
    assert item[0].parent == root
    assert inspect(item) == identities[item[0]]
    item[0].unlink()
    print('Deleted exact generated file: ' + str(item[0]), flush=True)
for item in retained:
    assert inspect(item) == identities[item[0]]
print(json.dumps({'status': 'completed', 'deleted_files': 2, 'deleted_bytes': 1035704, 'retained_cache_files_verified': 2, 'directories_removed': 0}), flush=True)
'@
$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pythonSource))
$code = 'import base64;exec(base64.b64decode("' + $payload + '"))'
& wsl.exe -d Ubuntu-24.04 -u root -- python3 -c $code
if ($LASTEXITCODE -ne 0) { throw "Exact WSL cleanup failed: $LASTEXITCODE" }

