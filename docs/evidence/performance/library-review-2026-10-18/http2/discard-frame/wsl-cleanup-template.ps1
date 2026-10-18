# TEMPLATE ONLY: parent reviews and copies into repository-root test.ps1 before execution.
# Exact discard-frame replacement cleanup; no directory/source/SDK removal.
$ErrorActionPreference = 'Stop'
if ([IO.Path]::GetFullPath($PSCommandPath) -ne 'C:\public\project-hyle\hyle\test.ps1') { throw 'Template must be reviewed and placed in repository-root test.ps1' }
$plan = 'C:\public\project-hyle\hyle\.scratch\perf-review-2026-10-18\http2\discard-frame\wsl-cleanup-plan.json'
if ((Get-FileHash -LiteralPath $plan -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'ac018a05b9d81c1abc4a96a56798daaa2a73e86bd222dc4dad281b21f0ae5089') { throw 'Exact cleanup plan SHA changed' }
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
      "path": "/root/hyle-scratch/http2-current-20261018/discard-frame/tests",
      "bytes": 1559608,
      "sha256": "2674a1c8b06f51a79bb4b5882ba55bff9f0da568247ac5b93ad273df55270722"
    },
    {
      "path": "/root/hyle-scratch/http2-current-20261018/owned-wire-no-copy/http2.o",
      "bytes": 223976,
      "sha256": "4f9960ea6a93c0cd2cc7ced5f946c68e3b3cc6fdb8b045eb3201819f99a2c3a9"
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
  ]
}''')
root = Path(manifest['root'])
targets = manifest['targets']
retained = manifest['retain']
expected_targets = {'/root/hyle-scratch/http2-current-20261018/owned-wire-no-copy/http2.o', '/root/hyle-scratch/http2-current-20261018/discard-frame/tests'}
expected_retained = {'/root/hyle-scratch/http2-current-20261018/current-holdout/libhyle_io_core.so.0.1.0', '/root/hyle-scratch/http2-current-20261018/header-lifetime/http2_test.o', '/root/hyle-scratch/http2-current-20261018/discard-frame/http2.o'}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

require(str(root) == '/root/hyle-scratch/http2-current-20261018', 'Wrong literal batch root')
require(root.resolve(strict=True) == root, 'Root resolves outside literal path')
require(len(targets) == 2 and len(retained) == 3, 'Wrong exact manifest counts')
require({item['path'] for item in targets} == expected_targets, 'Wrong target allowlist')
require({item['path'] for item in retained} == expected_retained, 'Wrong retained allowlist')
all_items = targets + retained
require(len({item['path'] for item in all_items}) == 5, 'Repeated or overlapping manifest path')

def inspect(item):
    path = Path(item['path'])
    require(path.is_absolute() and str(path) == item['path'], 'Non-canonical absolute path')
    require(path.parent in {root / 'discard-frame', root / 'owned-wire-no-copy', root / 'header-lifetime', root / 'current-holdout'},
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

def require_unused(item):
    # An executable may be running, or a linker may still hold/mmap the old obj.
    wanted = item['path']
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            executable = process / 'exe'
            if executable.exists():
                require(str(executable.resolve()) != wanted,
                        'Target is a running process executable: ' + wanted)
            for descriptor in (process / 'fd').iterdir():
                try:
                    require(os.readlink(descriptor) != wanted,
                            'Target still open by process' + process.name + ': ' + wanted)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            for line in (process / 'maps').read_text().splitlines():
                parts = line.split(maxsplit=5)
                require(len(parts) < 6 or parts[5] != wanted,
                        'Target still mapped by process' + process.name + ': ' + wanted)
        except (FileNotFoundError, ProcessLookupError):
            continue

# Every target and retained cache must pass before the first unlink.
identities = {item['path']: inspect(item) for item in all_items}
for item in targets:
    require_unused(item)
print(json.dumps({'preflight_files': len(all_items), 'delete_files': len(targets),
                  'delete_bytes': sum(item['bytes'] for item in targets),
                  'retain_files': len(retained),
                  'retain_bytes': sum(item['bytes'] for item in retained)}), flush=True)
deleted = []
for item in targets:
    require_unused(item)
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
& wsl -d Ubuntu-24.04 -u root -- python3 -B -c $decode
if ($LASTEXITCODE -ne 0) { throw "WSL exact cleanup failed with exit $LASTEXITCODE" }
