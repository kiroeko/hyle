# TEMPLATE ONLY: parent reviews and copies into repository-root test.ps1 before execution.
# Exact current NO_COPY batch; no directory removal, scans, SDK or source deletion.
$ErrorActionPreference = 'Stop'
$pythonSource = @'
import hashlib
import json
import os
from pathlib import Path
import stat

manifest = json.loads(r'''{
  "root": "/root/hyle-scratch/http2-current-20261018",
  "targets": [
    {
      "path": "/root/hyle-scratch/http2-current-20261018/owned-wire-no-copy/tests",
      "bytes": 1559360,
      "sha256": "b6837ba3c6fcf31158abfc4c8a011d7809e2dbb90349b9bc0250591a37ae3eea"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/owned-wire-no-copy/holdout-old",
      "bytes": 864552,
      "sha256": "52f6aeb4d679a8b2f4ed2295b0f211a55e37f0732e3c49eb5f87793876f5b88c"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/owned-wire-no-copy/holdout-new",
      "bytes": 868784,
      "sha256": "390947e7513d4d33adb4912fe8497ede5a1cc33257f027647918875bd6f578a4"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/owned-wire-no-copy/holdout.o",
      "bytes": 110800,
      "sha256": "604f70345247c6f2b84d206f998cd9b40721c4ebe087cd7c3846dd760183729a"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/header-lifetime/http2.o",
      "bytes": 222272,
      "sha256": "9ec083687a2fdb420b1ea177d6158d97ceb6ec77dbf6f138b1f80ac82790754b"
    }
  ],
  "retain": [
    {
      "path": "/root/hyle-scratch/http2-current-20261018/owned-wire-no-copy/http2.o",
      "bytes": 223976,
      "sha256": "4f9960ea6a93c0cd2cc7ced5f946c68e3b3cc6fdb8b045eb3201819f99a2c3a9"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/current-holdout/libhyle_io_core.so.0.1.0",
      "bytes": 387104,
      "sha256": "9ead235b1bbd72f56d44eab9c8ac135b8c4b0ce0ec0a6fb72660aa917d3b4779"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/header-lifetime/http2_test.o",
      "bytes": 654624,
      "sha256": "debb22bed71cd36bbda594158905dc547ca59655ed62c81f8b4a42c0431f75d3"
    }
  ]
}''')
root = Path(manifest['root'])
targets = manifest['targets']
retained = manifest['retain']
expected_targets = {
    str(root / 'owned-wire-no-copy/tests'),
    str(root / 'owned-wire-no-copy/holdout-old'),
    str(root / 'owned-wire-no-copy/holdout-new'),
    str(root / 'owned-wire-no-copy/holdout.o'),
    str(root / 'header-lifetime/http2.o'),
}
expected_retained = {
    str(root / 'owned-wire-no-copy/http2.o'),
    str(root / 'header-lifetime/http2_test.o'),
    str(root / 'current-holdout/libhyle_io_core.so.0.1.0'),
}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

require(str(root) == '/root/hyle-scratch/http2-current-20261018', 'Wrong literal batch root')
require(root.resolve(strict=True) == root, 'Root resolves outside literal path')
require(len(targets) == 5 and len(retained) == 3, 'Wrong exact manifest counts')
require({item['path'] for item in targets} == expected_targets, 'Wrong target allowlist')
require({item['path'] for item in retained} == expected_retained, 'Wrong retained allowlist')
all_items = targets + retained
require(len({item['path'] for item in all_items}) == 8, 'Repeated or overlapping manifest path')

def inspect(item):
    path = Path(item['path'])
    require(path.is_absolute() and str(path) == item['path'], 'Non-canonical absolute path')
    require(path.parent in {root / 'owned-wire-no-copy', root / 'header-lifetime', root / 'current-holdout'},
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

# Every target and retained cache must pass before the first unlink.
identities = {item['path']: inspect(item) for item in all_items}
print(json.dumps({'preflight_files': len(all_items), 'delete_files': len(targets),
                  'delete_bytes': sum(item['bytes'] for item in targets),
                  'retain_files': len(retained),
                  'retain_bytes': sum(item['bytes'] for item in retained)}), flush=True)
deleted = []
for item in targets:
    require(inspect(item) == identities[item['path']],
            'Identity changed after preflight: ' + item['path'])
    Path(item['path']).unlink()
    deleted.append(item['path'])
    print('Deleted exact file: ' + item['path'], flush=True)
for item in retained:
    require(inspect(item) == identities[item['path']],
            'Retained file changed: ' + item['path'])
print(json.dumps({'deleted': deleted, 'retained_files_verified': len(retained),
                  'deleted_bytes': sum(item['bytes'] for item in targets)}), flush=True)

'@
$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($pythonSource))
$decode = "import base64;exec(base64.b64decode('$payload'))"
& wsl -u root -- python3 -c $decode
if ($LASTEXITCODE -ne 0) { throw "WSL exact cleanup failed with exit $LASTEXITCODE" }
