"""Bind the reused qualified product image, SDK closure and frozen request subset."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    declared = json.loads(Path(__file__).with_name('image.json').read_text())
    aa = json.loads(Path(__file__).with_name('aa-receipt.json').read_text())
    qualified = {row['id'] for row in aa['decisions'] if row['qualification'] == 'QUALIFIED'}
    actual = {line.split('\t', 1)[0] for line in Path(declared['pair_file']).read_text().splitlines()}
    if actual != qualified or aa['binary_sha256'] != declared['files']['hyle_cuda_matrix_bench.exe']:
        raise RuntimeError('Qualified row/image identity does not match actual AA receipt')
    for name, value in declared['files'].items():
        if digest(args.build / name) != value:
            raise RuntimeError('Qualified image/runtime changed: ' + name)
    if digest(Path(declared['pair_file'])) != declared['pair_file_sha256']:
        raise RuntimeError('Frozen qualified request input changed')
    if declared['qualified_pairs'] != 47:
        raise RuntimeError('Unexpected qualified row set')
    (args.build / 'matrix-aa.ready').write_text('PASS\n')
    print('PASS exact AA-qualified product image/runtime and same-request qualified subset')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
