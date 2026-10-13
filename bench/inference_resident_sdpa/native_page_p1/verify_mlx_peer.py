"""Bind the original fused peer and the explicit frozen page source adaptations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--peer',type=Path,required=True)
parser.add_argument('--linked',type=Path,required=True)
parser.add_argument('--manifest',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
pin='1f8e74e3f12f31365464a6867c6579f0e9b29d85'
assert subprocess.check_output(['git','-C',str(args.peer),'rev-parse','HEAD'],text=True).strip()==pin
assert not subprocess.check_output(['git','-C',str(args.peer),'status','--porcelain'],text=True).strip()
patches={row['path']:row for row in json.loads(args.manifest.read_bytes())['files']}
names={'mlx/fast.cpp','mlx/fast.h','mlx/fast_primitives.h',
       'mlx/backend/metal/scaled_dot_product_attention.cpp',
       'mlx/backend/metal/kernels/scaled_dot_product_attention.metal'}
names.update(patches)
for directory,pattern in [('mlx/backend/metal/kernels','*sdpa*'),('mlx/backend/metal','*sdpa*')]:
    names.update(str(p.relative_to(args.peer)) for p in (args.peer/directory).glob(pattern) if p.is_file())
names.update(str(p.relative_to(args.peer)) for p in (args.peer/'mlx/backend/metal/kernels/steel/attn').rglob('*') if p.is_file())
records=[]
for name in sorted(names):
    original=subprocess.check_output(['git','-C',str(args.peer),'show',pin+':'+name])
    expected=original
    if name in patches:
        row=patches[name]
        assert hashlib.sha256(original).hexdigest()==row['before_sha256'],name
        for edit in row['edits']:
            before,after=edit['before'].encode(),edit['after'].encode()
            assert expected.count(before)==1,(name,'ambiguous adaptation')
            expected=expected.replace(before,after,1)
        assert hashlib.sha256(expected).hexdigest()==row['after_sha256'],name
    assert (args.linked/name).read_bytes()==expected,name
    records.append({'path':name,'original_sha256':hashlib.sha256(original).hexdigest(),
                    'linked_sha256':hashlib.sha256(expected).hexdigest(),'adapted':name in patches})
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps({'revision':pin,'linked_source':str(args.linked),'files':records,
 'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
 'scope':'Independent pristine pin plus exactly replayed frozen HYLE adaptations. Peer calls original fused nonpaged SDPA (force_fused), not the page primitive; vector/NAX formulas and reduction borrowed, optional page address mode disabled. This does not claim every linked file is pristine.'},indent=2)+'\n')
print('original fused peer and explicit frozen source adaptations authenticated')
