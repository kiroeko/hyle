"""Bind the exact filtered AA image, SDK, and actual qualified subset; no timing."""
import argparse,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--build',type=Path,required=True);args=parser.parse_args()
    sdk=json.loads((HERE/'sdk-identity.json').read_text());prefix=Path(sdk['prefix']);build=args.build
    for name,expected in sdk['files'].items():assert digest(prefix/name)==expected,name
    for name,expected in sdk['runtime_files'].items():assert digest(build/name)==expected,name
    cache=(build/'CMakeCache.txt').read_text()
    peer=Path(next(x.split('=',1)[1] for x in cache.splitlines() if x.startswith('HYLE_CPU_CURRENT_PEER_ROOT:')))
    assert subprocess.check_output(['git','-C',str(peer),'rev-parse','HEAD'],text=True).strip()==sdk['product_candidate']
    for name,expected in sdk['source_files'].items():assert digest(peer/name)==expected,name
    record=json.loads((HERE/'filtered-aa-identity.json').read_text())
    archive=ROOT/record['archive'];assert digest(archive)==record['archive_sha256']
    with zipfile.ZipFile(archive) as z:aa=json.loads(z.read('raw/receipt.json'))
    assert digest(build/'hyle_cpu_best_of_sealed.exe')==aa['artifact']['executable_sha256']
    assert digest(HERE/'cpu_best_of_bench.cc')==aa['artifact']['consumer_sha256']
    assert digest(HERE/'qualified_pairs.h')==aa['artifact']['pair_header_sha256']
    selected=(HERE/'p1-qualified-pairs.txt').read_text().splitlines()
    assert selected and len(selected)==len(set(selected)) and set(selected)<=set(aa['qualified'])
    asset=json.loads((HERE/'contract.json').read_text())['extensions']['cpu_pair_asset']
    assert 'HYLE_CPU_BESTOF_AA' not in os.environ,'A/A substitution must not reach A/B sampling'
    assert Path(os.environ.get(asset['environment_name'],'')).resolve()==Path(asset['path']).resolve()
    assert digest(Path(asset['path']))==asset['sha256']
    print(json.dumps({'gate':'PASS','sdk_files':len(sdk['files']),'runtime_files':len(sdk['runtime_files']),'pairs':len(selected),'AA':'exact filtered image and only individually qualified rows; no rerun or inherited old version'},sort_keys=True))
if __name__=='__main__':
    try:main()
    except Exception as error:print(str(error),file=sys.stderr);raise SystemExit(2)
