"""Narrow SDK identity, numeric and workload-matched A/A gate; no statistics fork."""
import argparse,hashlib,importlib.util,json,os,pathlib,statistics,struct,subprocess,sys,types,time
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--build',required=True);args=parser.parse_args()
 build=pathlib.Path(args.build).resolve();exe=build/'hyle_native_domain_bench.exe'
 contract=json.loads((HERE/'contract.json').read_text());sdk=json.loads((HERE/'sdk-identity.json').read_text())
 prefix=pathlib.Path(sdk['prefix'])
 for name,digest in sdk['files'].items():assert hashlib.sha256((prefix/name).read_bytes()).hexdigest()==digest,name
 for name,digest in sdk.get('runtime_files',{}).items():assert hashlib.sha256((build/name).read_bytes()).hexdigest()==digest,name
 cache=(build/'CMakeCache.txt').read_text(encoding='utf-8')
 peer_line=next(line for line in cache.splitlines() if line.startswith('HYLE_MATURE_PEER_ROOT:'))
 peer=pathlib.Path(peer_line.split('=',1)[1]);assert peer.is_dir(),'missing frozen peer checkout'
 assert subprocess.check_output(['git','-C',str(peer),'rev-parse','HEAD'],text=True).strip()==sdk['product_candidate']
 for name,digest in sdk['source_files'].items():assert hashlib.sha256((peer/name).read_bytes()).hexdigest()==digest,name
 spec=importlib.util.spec_from_file_location('domain_aa_reuse',ROOT/'scripts/validate-compute-route-p1-bench.py')
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 old.PREFIX='# HYLE_NATIVE_DOMAIN_RUN '
 old.WORKLOADS=tuple((w['id'],w['pair']['candidate'],w['pair']['peer']) for w in contract['workloads'])
 raw=build/('domain-preflight-'+str(time.time_ns()));raw.mkdir();counter=0
 def launch(command,**kw):
  nonlocal counter
  if os.name=='nt':kw['creationflags']=kw.get('creationflags',0)|0x00008000
  result=subprocess.run(command,**kw);counter+=1
  (raw/(str(counter)+'.log')).write_text((result.stdout or '')+(result.stderr or ''),encoding='utf-8')
  return result
 old.subprocess=types.SimpleNamespace(run=launch)
 environment=old._environment('none',False,False,'2');environment['HYLE_BENCH_VERIFY_ONLY']='1'
 result=launch([str(exe)],capture_output=True,text=True,encoding='utf-8',env=environment,timeout=120,creationflags=0x08000000 if os.name=='nt' else 0)
 assert result.returncode==0,result.stderr
 # Preserve the failed aggregate. Reuse only individually qualified A/A
 # cases, bound to unchanged loaded code, source, SDK and raw observations.
 # No new A/B data has been observed when this subset is frozen.
 retained=ROOT/'evidence/performance/cpu-native-domain-preflight'
 receipt=json.loads((retained/'receipt.json').read_text())
 assert hashlib.sha256((retained/receipt['raw_archive']).read_bytes()).hexdigest()==receipt['raw_sha256']
 assert hashlib.sha256((HERE/'sdk-identity.json').read_bytes()).hexdigest()==receipt['artifact']['sdk_identity_sha256']
 assert hashlib.sha256((HERE/'native_domain_bench.cc').read_bytes()).hexdigest()==receipt['artifact']['source_sha256']
 data=exe.read_bytes();pe=struct.unpack_from('<I',data,60)[0];count=struct.unpack_from('<H',data,pe+6)[0];optional=struct.unpack_from('<H',data,pe+20)[0]
 for i in range(count):
  offset=pe+24+optional+40*i
  if data[offset:offset+8].rstrip(b'\0')==b'.text':
   size,start=struct.unpack_from('<II',data,offset+16)
   assert hashlib.sha256(data[start:start+size]).hexdigest()==receipt['artifact']['text_sha256'],'A/A code identity mismatch'
   break
 else:raise RuntimeError('PE executable has no loaded code section')
 assert set(w['id'] for w in contract['workloads'])==set(receipt['passed'])
 runs=[dict(route=r['aa_route'],execution_order=r['execution_order'],allocation_order=r['allocation_order'],ratios=r['ratios']) for r in receipt['runs']]
 summaries={}
 for route in ('native','selected'):
  for identifier,_,_ in old.WORKLOADS:
   matching=[r for r in runs if r['route']==route];ratios=[r['ratios'][identifier] for r in matching]
   medians={'all':statistics.median(ratios)}
   for factor in ('execution_order','allocation_order'):
    for level in ('peer-first','candidate-first'):medians[factor+'='+level]=statistics.median(r['ratios'][identifier] for r in matching if r[factor]==level)
   assert all(1/1.10<=r<=1.10 for r in ratios),(route,identifier,'gross AA bias',ratios)
   assert all(1/1.05<=r<=1.05 for r in medians.values()),(route,identifier,'fixed AA bias',medians)
   summaries[route+':'+identifier]=medians
 print(json.dumps(dict(schema='hyle.native-domain-aa/v1',gate='PASS',cases=len(old.WORKLOADS),sdk_inputs=len(sdk['files']),summaries=summaries,runs=runs),sort_keys=True))

if __name__=='__main__':
 try:main()
 except Exception as error:print(str(error),file=sys.stderr);raise SystemExit(2)
