# TEMPLATE ONLY: root reviews and copies to test.ps1; not executed here.
# Apple ordinary math: exact5rebuildable files; retain fresh5host/testobjects and graph/Sinkcache.
$ErrorActionPreference = 'Stop'
$pythonSource = @'
import hashlib
import json
import os
from pathlib import Path
import stat

manifest = json.loads(r'''{
  "root": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math",
  "targets": [
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/baseline-tests",
      "bytes": 10419728,
      "sha256": "5f60bf3a6fed4e68fecdf4404c07ae4465307d3022e345bea7e31328824a9051"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/candidate-tests",
      "bytes": 10421056,
      "sha256": "e5e3da4b942f116c4abd329eda460ea5c0a0e1f1bb8577718e44fb049570cd93"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/consumer.o",
      "bytes": 36400,
      "sha256": "2ac0e24081a73700b7fafbfc7b3d0ae8173044a8327d40a4a66a650a52b53737"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/baseline-space",
      "bytes": 9574528,
      "sha256": "736905bcf7ec2703ccbe0273cd25bc63f82cfd9203adecdf7e2a7ff354102643"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/candidate-space",
      "bytes": 9575856,
      "sha256": "d3797a40bf4d35e83cfe5262eb17404b8d2232a18f30b23275786624e5b35892"
    }
  ],
  "retain": [
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/context.o",
      "bytes": 187280,
      "sha256": "5c0c5c2e324f6667ec3ec18598b63330784d6977361cd97b8aef990a7102e896"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/operations.o",
      "bytes": 73616,
      "sha256": "1f43f11529f2582ec1103576ab700710669f80221a706aa2a7e4135815fe0f8c"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/tensor.o",
      "bytes": 131640,
      "sha256": "25e905b914d0c28740c1e3721ba277f9b8506c1fbbc6ab5939d629c895665b40"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/attention-tests.o",
      "bytes": 268256,
      "sha256": "09e1194f1504e010ec3a756df61e3bbf591fc14308cc3cd7084498a568e6852c"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/graph-tests.o",
      "bytes": 243480,
      "sha256": "aae725cda26cd1ccde9becf170463eb224a90d50478c7200872034e198f9e022"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-short-q/operations.o",
      "bytes": 72048,
      "sha256": "52cc70b09f6291b946215fd1d5b6b6ebff8154db11f3c71e898096e3d4c13f83"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-graph.o",
      "bytes": 87824,
      "sha256": "13b405c34bb446496563f51db9b91439ea6004d34b00d833edcba6e25d197de6"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-short-q/sink-tests.o",
      "bytes": 258920,
      "sha256": "1f7f88bb886dc897082422be2585d775c58dc0ba115a82d0bd3f2eef2d62e051"
    }
  ],
  "delete_files": 5,
  "delete_bytes": 40027568,
  "retain_files": 8,
  "retain_bytes": 1323064,
  "executed": false,
  "note": "Only newconsumerobj+fournewexe deleted after review; all older caches, source/maps/log/data/archives/SDK unchanged. Old baseline context/tensor are outside targets and remain untouched.",
  "retain_fresh_objects": [
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/context.o",
      "bytes": 187280,
      "sha256": "5c0c5c2e324f6667ec3ec18598b63330784d6977361cd97b8aef990a7102e896"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/operations.o",
      "bytes": 73616,
      "sha256": "1f43f11529f2582ec1103576ab700710669f80221a706aa2a7e4135815fe0f8c"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/tensor.o",
      "bytes": 131640,
      "sha256": "25e905b914d0c28740c1e3721ba277f9b8506c1fbbc6ab5939d629c895665b40"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/attention-tests.o",
      "bytes": 268256,
      "sha256": "09e1194f1504e010ec3a756df61e3bbf591fc14308cc3cd7084498a568e6852c"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/graph-tests.o",
      "bytes": 243480,
      "sha256": "aae725cda26cd1ccde9becf170463eb224a90d50478c7200872034e198f9e022"
    }
  ],
  "retain_fresh_bytes": 904272,
  "retain_cached_Graph_Sink_objects": [
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-short-q/operations.o",
      "bytes": 72048,
      "sha256": "52cc70b09f6291b946215fd1d5b6b6ebff8154db11f3c71e898096e3d4c13f83"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-graph.o",
      "bytes": 87824,
      "sha256": "13b405c34bb446496563f51db9b91439ea6004d34b00d833edcba6e25d197de6"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-short-q/sink-tests.o",
      "bytes": 258920,
      "sha256": "1f7f88bb886dc897082422be2585d775c58dc0ba115a82d0bd3f2eef2d62e051"
    }
  ],
  "retain_cached_Graph_Sink_bytes": 418792
}''')
root = Path(manifest['root'])
targets = manifest['targets']
retained = manifest['retain']
expected_targets = set(['/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/baseline-tests', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/candidate-tests', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/consumer.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/baseline-space', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/candidate-space'])
expected_retained = set(['/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/context.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/operations.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/tensor.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/attention-tests.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math/graph-tests.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-short-q/operations.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-graph.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-short-q/sink-tests.o'])
allowed_target_parents = {root}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

require(str(root) == '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-math',
        'Wrong literal batch root')
require(root.resolve(strict=True) == root, 'Root resolves outside literal path')
require(len(targets) == 5 and len(retained) == 8, 'Wrong exact manifest counts')
require({item['path'] for item in targets} == expected_targets, 'Wrong target allowlist')
require({item['path'] for item in retained} == expected_retained, 'Wrong retained allowlist')
all_items = targets + retained
require(len({item['path'] for item in all_items}) == 13, 'Repeated or overlapping path')
for item in targets:
    path = Path(item['path'])
    require(path.parent in allowed_target_parents, 'Target outside exact batch directories')

def inspect(item):
    path = Path(item['path'])
    require(path.is_absolute() and str(path) == item['path'], 'Non-canonical absolute path')
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

# Preflight every target and all 8 retained cache files before any deletion.
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
$remoteCode = 'import base64;exec(base64.b64decode("' + $payload + '"))'
$remoteCommand = "python3 -c '" + $remoteCode + "'"
& ssh 'kiro@172.22.10.198' $remoteCommand
if ($LASTEXITCODE -ne 0) { throw "Mac exact cleanup failed with SSH exit $LASTEXITCODE" }
