# TEMPLATE ONLY: parent reviews and copies to root test.ps1 before execution.
# Current Apple short-Q cleanup; do not rerun the already-executed old WSL test.ps1.
$ErrorActionPreference = 'Stop'
$pythonSource = @'
import hashlib
import json
import os
from pathlib import Path
import stat

manifest = json.loads(r'''{
  "root": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q",
  "targets": [
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/measurement/consumer.o",
      "bytes": 44400,
      "sha256": "0352a37a834e8a9726bc6a1774324a4cf1bd91bcd39e356c201306e63db774f6"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/measurement/baseline",
      "bytes": 9574136,
      "sha256": "7c03a1bf61fde8b5184b499bd037f8093008407c82e5afd8899975f03ce48c59"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/measurement/candidate",
      "bytes": 9574568,
      "sha256": "fd99ddf0e6e3eb55e0ded40b81b2c0798b23a9bdf9ea8e3284e413dc81dd8d61"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/diagnostic/full-output-probe.o",
      "bytes": 81112,
      "sha256": "bdd05528047ff6b1e9b7a75f18ba29c034e64a2a30404c32797644de70419d57"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/diagnostic/probe-baseline",
      "bytes": 9593392,
      "sha256": "c56edfd82235aa647be22278ff4806a5f54b74027f152cc15060df23883693a3"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/diagnostic/probe-candidate",
      "bytes": 9593856,
      "sha256": "9c870b8c2e9b3d8f7a000bd1754dd8fcf8469f0f3a5c2ba17e2a151ae6da63f3"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/precision-replay/replay-probe.o",
      "bytes": 82288,
      "sha256": "5adc84c1c9b3c0a136211ef9ba2a760d8853be79344b831ed8ed5857d0ec1c4f"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/precision-replay/replay-probe",
      "bytes": 9594976,
      "sha256": "21c27699fd0f43ff1cda4999f375b64cb8d1e63c802d4486e9ecc534d760176a"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/centered-functional/baseline-centered-tests",
      "bytes": 10401688,
      "sha256": "da293b73f0e37ac17ebd1b0a8dff6aef6b27c4065b7f7e751fdad15a88d1e48a"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/centered-functional/candidate-centered-tests",
      "bytes": 10402136,
      "sha256": "caddc554a6711feebaee0f1e88ffc3907c0f0354f351a6069b1737bcedc37f5f"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/attention-tests.o",
      "bytes": 240736,
      "sha256": "327e48535ef84e971dc705f66cb8413e67838151db3e438b621e6a4bb3fb7639"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/graph-tests.o",
      "bytes": 225040,
      "sha256": "6f66303ce495214b50642e1dd1e25b97e63ef546714530f528be0a164e640641"
    }
  ],
  "retain": [
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/centered-functional/centered-attention.o",
      "bytes": 241328,
      "sha256": "12b17edda6e216a8fef9fcf8cf3bc2470d23e94b1466aca9d7026a1e0bd238f7"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/centered-functional/centered-graph.o",
      "bytes": 225672,
      "sha256": "27d346e1915d0daa123a0c8f96ebdf99d6783d6708424a6d3d09757dd7c98fe9"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-q128/candidate-context.o",
      "bytes": 181672,
      "sha256": "8d40802ad526811d9955c09809524f3e1e01cf1a9db97fd84d1eb0b27e29c937"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-q128/candidate-graph.o",
      "bytes": 89144,
      "sha256": "ae0de519f4375f4899ccce885ff0b5e5fe40e82025259629152ef01723f03e03"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-q128/candidate-operations.o",
      "bytes": 70592,
      "sha256": "46f9513654a2b2a52caeace1df7cf0846bee378bd5eeb43e955427ff94858ca9"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-force-scale/apple_tensor.o",
      "bytes": 126016,
      "sha256": "cb31c6ec728dfb5fd090468c6baf94e50698d62a364af4d8e5fe330e71c6c8d0"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-context.o",
      "bytes": 182048,
      "sha256": "0ad035588062f078880e81421e75e3715233d3c7f325b3c3d368557da7bd2be2"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-operations.o",
      "bytes": 72016,
      "sha256": "2fa2f5bd746a5db5fbc0b19749397efd7fbc0d257435e8083bdd33dc96f439e3"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-tensor.o",
      "bytes": 129432,
      "sha256": "62e4a8b2668fe35811f5992d0ed48d29cc8ffbbc2b24d1fcc4b4a53fb4f40715"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-graph.o",
      "bytes": 87824,
      "sha256": "13b405c34bb446496563f51db9b91439ea6004d34b00d833edcba6e25d197de6"
    },
    {
      "path": "/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-q128/applied-direct.o",
      "bytes": 258080,
      "sha256": "2526c62aac5c655cdaddbe08bfae22a32da85b67ba3683eeb01398d8f7d38616"
    }
  ]
}''')
root = Path(manifest['root'])
targets = manifest['targets']
retained = manifest['retain']
expected_targets = {str(root / name) for name in ['measurement/consumer.o', 'measurement/baseline', 'measurement/candidate', 'diagnostic/full-output-probe.o', 'diagnostic/probe-baseline', 'diagnostic/probe-candidate', 'precision-replay/replay-probe.o', 'precision-replay/replay-probe', 'centered-functional/baseline-centered-tests', 'centered-functional/candidate-centered-tests', 'attention-tests.o', 'graph-tests.o']}
expected_retained = set(['/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/centered-functional/centered-attention.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/centered-functional/centered-graph.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-q128/candidate-context.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-q128/candidate-graph.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-q128/candidate-operations.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-force-scale/apple_tensor.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-context.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-operations.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-tensor.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q/candidate-graph.o', '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-sink-q128/applied-direct.o'])
allowed_target_parents = {root, root / 'measurement', root / 'diagnostic',
                          root / 'precision-replay', root / 'centered-functional'}

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

require(str(root) == '/Users/kiro/hyle-scratch/compute-unified/library-review-20261018-apple-ordinary-short-q',
        'Wrong literal batch root')
require(root.resolve(strict=True) == root, 'Root resolves outside literal path')
require(len(targets) == 12 and len(retained) == 11, 'Wrong exact manifest counts')
require({item['path'] for item in targets} == expected_targets, 'Wrong target allowlist')
require({item['path'] for item in retained} == expected_retained, 'Wrong retained allowlist')
all_items = targets + retained
require(len({item['path'] for item in all_items}) == 23, 'Repeated or overlapping path')
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

# Preflight every target and all 11 retained cache files before any deletion.
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
