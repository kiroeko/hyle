# TEMPLATE ONLY: review and copy into repository-root test.ps1 before execution.
# Five exact negative-candidate artifacts only. No directory enumeration or recursive removal.
$ErrorActionPreference = 'Stop'
if ([IO.Path]::GetFullPath($PSCommandPath) -ne 'C:\public\project-hyle\hyle\test.ps1') { throw 'Template must be reviewed and placed in repository-root test.ps1' }
$plan = 'C:\public\project-hyle\hyle\evidence\performance\library-review-2026-10-18\http2\lazy-header-idle-flush\wsl-cleanup-plan.json'
if ((Get-FileHash -LiteralPath $plan -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'd302130d5e5e2b25174f16e903abb364a59b0258f3e08c5c12b0d9279e38ed6a') { throw 'Exact cleanup plan SHA changed' }
$pythonSource = @'
DELETE=True
import hashlib
import json
import os
from pathlib import Path
import stat

manifest = json.loads(r'''{
  "root": "/root/hyle-scratch/http2-current-20261018",
  "targets": [
    {
      "path": "/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/tests",
      "bytes": 1559608,
      "sha256": "b2441db6f04a67a93a4e270bfbbc38063b4a8a7fb6d76b5909e70b24393a8df2"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/holdout.o",
      "bytes": 110800,
      "sha256": "604f70345247c6f2b84d206f998cd9b40721c4ebe087cd7c3846dd760183729a"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/holdout-old",
      "bytes": 869040,
      "sha256": "5c7637ec02398f91865bd537945578bb8a4702a968bd6be42110b95e06339e69"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/holdout-new",
      "bytes": 869040,
      "sha256": "70bb782345bb71cc93477c0225cbfe8a6d6ddd36e3346c55bab27a4aa38fb681"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/http2.o",
      "bytes": 223480,
      "sha256": "6fe0e7ccfe1828f5dd496372c09f905942be8e950da6ead4aa62ac0b16ab0d9c"
    }
  ],
  "retain": [
    {
      "path": "/root/hyle-scratch/http2-current-20261018/discard-frame/http2.o",
      "bytes": 223280,
      "sha256": "0de3047e82522fddeaf24b57d576f4c7393901d9609217577ec2d5fc566303b6"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/header-lifetime/http2_test.o",
      "bytes": 654624,
      "sha256": "debb22bed71cd36bbda594158905dc547ca59655ed62c81f8b4a42c0431f75d3"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/current-holdout/libhyle_io_core.so.0.1.0",
      "bytes": 387104,
      "sha256": "9ead235b1bbd72f56d44eab9c8ac135b8c4b0ce0ec0a6fb72660aa917d3b4779"
    }
  ],
  "delete_count": 5,
  "delete_logical_bytes": 3631968,
  "retain_count": 3,
  "retain_logical_bytes": 1265008,
  "duplicate_or_overlap_count": 0,
  "disposition": "Plan only: negative-candidate artifacts may be removed after root review. Accepted discard object, protocol tests, current core and every source/log/raw/map/archive/SDK/foundation remain. Logical byte count is not a physical disk recovery promise."
}''')
root = Path(manifest['root'])
targets = manifest['targets']
retained = manifest['retain']
expected_targets = {'/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/http2.o', '/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/tests', '/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/holdout-new', '/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/holdout-old', '/root/hyle-scratch/http2-current-20261018/lazy-header-idle-flush/holdout.o'}
expected_retained = {'/root/hyle-scratch/http2-current-20261018/current-holdout/libhyle_io_core.so.0.1.0', '/root/hyle-scratch/http2-current-20261018/header-lifetime/http2_test.o', '/root/hyle-scratch/http2-current-20261018/discard-frame/http2.o'}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

require(str(root) == '/root/hyle-scratch/http2-current-20261018', 'Wrong literal root')
require(root.resolve(strict=True) == root, 'Root resolves outside literal path')
require(len(targets) == 5 and len(retained) == 3, 'Wrong exact counts')
require({x['path'] for x in targets} == expected_targets, 'Wrong target allowlist')
require({x['path'] for x in retained} == expected_retained, 'Wrong retained allowlist')
all_items = targets + retained
require(len({x['path'] for x in all_items}) == 8, 'Repeated or overlapping path')
require(sum(x['bytes'] for x in targets) == 3631968, 'Wrong delete byte sum')
require(sum(x['bytes'] for x in retained) == 1265008, 'Wrong retained byte sum')

def inspect(item):
    path = Path(item['path'])
    require(path.is_absolute() and str(path) == item['path'], 'Non-canonical absolute path')
    require(path.parent in {root / 'lazy-header-idle-flush', root / 'discard-frame', root / 'header-lifetime', root / 'current-holdout'},
            'Path outside exact batch directories: ' + str(path))
    cursor = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        cursor = cursor / part
        status = cursor.lstat()
        require(not stat.S_ISLNK(status.st_mode), 'Symlink forbidden: ' + str(cursor))
        if index + 1 < len(path.parts) - 1:
            require(stat.S_ISDIR(status.st_mode), 'Non-directory ancestor: ' + str(cursor))
    require(path.resolve(strict=True) == path, 'Resolved path changed: ' + str(path))
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), 'Not a regular file: ' + str(path))
    require(before.st_size == item['bytes'], 'Size mismatch: ' + str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        opened = os.fstat(stream.fileno())
        require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                'File changed while opening: ' + str(path))
        hasher = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
        after = os.fstat(stream.fileno())
    require(after.st_size == item['bytes'] and hasher.hexdigest() == item['sha256'],
            'Size/SHA mismatch: ' + str(path))
    current = path.lstat()
    require(not stat.S_ISLNK(current.st_mode) and
            (current.st_dev, current.st_ino, current.st_size) ==
            (after.st_dev, after.st_ino, after.st_size),
            'File changed while hashing: ' + str(path))
    return current.st_dev, current.st_ino


# Full preflight of all eight paths precedes any deletion.
identities = {item['path']: inspect(item) for item in all_items}
print(json.dumps({'phase': 'preflight', 'files_verified': len(all_items),
                  'delete_count': len(targets), 'delete_logical_bytes': 3631968,
                  'retain_count': len(retained), 'retain_logical_bytes': 1265008,
                  'targets': targets, 'retained': retained}), flush=True)
if DELETE:
    deleted = []
    for item in targets:
        require(inspect(item) == identities[item['path']],
                'Identity changed after preflight: ' + item['path'])
        Path(item['path']).unlink()
        deleted.append(item['path'])
    for item in retained:
        require(inspect(item) == identities[item['path']],
                'Retained file changed: ' + item['path'])
    print(json.dumps({'phase': 'complete', 'deleted': deleted,
                      'deleted_logical_bytes': 3631968,
                      'retained_files_verified': len(retained),
                      'retained_logical_bytes': 1265008}), flush=True)

'@
$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pythonSource))
$decode = "import base64;exec(base64.b64decode('$payload'))"
& wsl -d Ubuntu-24.04 -u root -- python3 -B -c $decode
if ($LASTEXITCODE -ne 0) { throw "WSL exact cleanup failed with exit $LASTEXITCODE" }
