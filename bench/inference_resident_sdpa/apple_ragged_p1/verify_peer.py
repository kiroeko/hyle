"""Private comparator identity check; no product or performance-core changes."""
import argparse,hashlib,json,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--peer',type=Path,required=True);parser.add_argument('--linked',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
pin='1f8e74e3f12f31365464a6867c6579f0e9b29d85'
assert subprocess.check_output(['git','-C',str(args.peer),'rev-parse','HEAD'],text=True).strip()==pin
assert not subprocess.check_output(['git','-C',str(args.peer),'status','--porcelain'],text=True).strip()
names=['mlx/fast.cpp','mlx/fast.h','mlx/fast_primitives.h',
       'mlx/backend/metal/scaled_dot_product_attention.cpp',
       'mlx/backend/metal/kernels/scaled_dot_product_attention.metal']
names += [str(p.relative_to(args.peer)) for p in (args.peer/'mlx/backend/metal/kernels').glob('*sdpa*') if p.is_file()]
names += [str(p.relative_to(args.peer)) for p in (args.peer/'mlx/backend/metal/kernels/steel/attn').rglob('*') if p.is_file()]
names += [str(p.relative_to(args.peer)) for p in (args.peer/'mlx/backend/metal').glob('*sdpa*') if p.is_file()]
assert len(names)>3
records=[]
for name in sorted(names):
 original=subprocess.check_output(['git','-C',str(args.peer),'show',pin+':'+name])
 linked=(args.linked/name).read_bytes();assert linked==original,name
 records.append({'path':name,'sha256':hashlib.sha256(original).hexdigest()})
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({'revision':pin,'linked_source':str(args.linked),'files':records,'scope':'original fast SDPA and its Metal shader files identical; unrelated HYLE BF16 copy/compiler patches separately retained'},indent=2)+'\n')
print('original MLX fast SDPA linked source identities matched')
