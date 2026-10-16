"""Materialize only authenticated original lowering; rename link symbols only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

PIN = '669121418669b31f6849c74f64362aaf8d6eb919'
SOURCES = {
    'cuda': {
        'src/compute_provider_torch_cuda/artifact_provider.cc': 'c4f6e71cdd8135b409c0fef020fe77fc2eb4528e06fd5163846b27a631477d49',
        'third_party/mlx_cuda_quant/mlx_cuda_quant.h': '672189b4cc387811845eb30ddceaaa5f40d1943723608740ec23fd57885233af',
    },
    'apple': {
        'src/compute_route_apple/apple_mature_float.cc': '20ffaec7840532a910596268546b5679e228a07e1a1a8613fcd8ce27aeebb962',
    },
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--route', choices=SOURCES, required=True)
    a = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(a.original), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != PIN:
        raise RuntimeError('wrong original product revision')
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, expected in SOURCES[a.route].items():
        raw = subprocess.check_output(['git', '-C', str(a.original), 'show', PIN + ':' + name])
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RuntimeError('original source hash mismatch: ' + name)
        text = raw.decode().replace('\r', '')
        replacements = []
        if name.endswith('artifact_provider.cc'):
            replacements = [('Provider provider() noexcept', 'Provider g07_original_provider() noexcept')]
        elif a.route == 'apple':
            replacements = [(n, 'g07_original_' + n) for n in
                            ['is_mlx_float_operation', 'mlx_float_input_spec', 'lower_mlx_float']]
        for before, after in replacements:
            if text.count(before) != 1:
                raise RuntimeError('unexpected exported symbol count: ' + before)
            text = text.replace(before, after)
        target = a.output / Path(name).name
        target.write_text(text, encoding='utf-8')
        rows.append(dict(path=name, original_sha256=expected,
                         generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                         changes=replacements, newline_normalization='CR bytes removed, no C++ token change'))
    (a.output/'source-provenance.json').write_text(json.dumps(dict(revision=revision, route=a.route, sources=rows), indent=2)+'\n')

if __name__ == '__main__':
    main()
