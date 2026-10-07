"""Bounded workload-matched preflight; common parser owns measurement decoding."""
import argparse,hashlib,json,os,re,statistics,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from hyle_perf.nanobench import parse_nanobench,select_pair

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    args=parser.parse_args();build=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    exe=build/('hyle_cpu_best_of_batch.exe' if os.name=='nt' else 'hyle_cpu_best_of_batch')
    assert exe.is_file()
    family={'Native':'native','OneDnn':'onednn','Ggml':'ggml','Aten':'aten'}
    pairs=[]
    for case,left,right in re.findall(r'\{"([^"\n]+)", Family::(\w+), Family::(\w+)\}',(HERE/'qualified_pairs.h').read_text()):
        pairs.append(case+'-'+family[left]+'-vs-'+family[right])
    assert len(pairs)==767 and len(set(pairs))==767
    if args.selection:
        selected=args.selection.read_text().splitlines()
        assert selected and len(set(selected))==len(selected) and set(selected)<=set(pairs)
        pairs=selected
    artifact={'executable_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'consumer_sha256':hashlib.sha256((HERE/'cpu_best_of_bench.cc').read_bytes()).hexdigest(),'pair_header_sha256':hashlib.sha256((HERE/'qualified_pairs.h').read_bytes()).hexdigest()}
    runs=[]
    for side in ('candidate','peer'):
        for execution in ('candidate-first','peer-first'):
            for allocation in ('candidate-first','peer-first'):
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HYLE_CPU_BESTOF_AA=side,HYLE_PERF_CALLER_CPU='2')
                if args.selection:env['HYLE_CPU_BESTOF_PAIR_FILE']=str(args.selection.resolve())
                else:env.pop('HYLE_CPU_BESTOF_PAIR_FILE',None)
                for key in ('HYLE_BENCH_RIVAL_FIRST','HYLE_BENCH_PEER_ALLOCATION_FIRST'):env.pop(key,None)
                if execution=='peer-first':env['HYLE_BENCH_RIVAL_FIRST']='1'
                if allocation=='peer-first':env['HYLE_BENCH_PEER_ALLOCATION_FIRST']='1'
                number=len(runs)+1;start=time.monotonic()
                flags={}
                if os.name=='nt':flags['creationflags']=subprocess.CREATE_NO_WINDOW|subprocess.ABOVE_NORMAL_PRIORITY_CLASS
                r=subprocess.run([str(exe)],env=env,capture_output=True,encoding='utf-8',errors='replace',**flags)
                (out/(str(number)+'.stdout.log')).write_text(r.stdout,encoding='utf-8')
                (out/(str(number)+'.stderr.log')).write_text(r.stderr,encoding='utf-8')
                assert r.returncode==0,r.stderr[-4000:]
                markers=[x for x in r.stderr.splitlines() if x.startswith('# CPU_BEST_OF ')]
                assert len(markers)==1 and 'caller_cpu=2 ' in markers[0] and 'affinity_ok=1 ' in markers[0],markers
                rows=parse_nanobench(r.stdout);ratios={}
                for title in pairs:
                    a,b=select_pair(rows,'Candidate '+title,'Peer '+title,title=title)
                    ratios[title]=a.operations_per_second/b.operations_per_second
                runs.append({'side':side,'execution_order':execution,'allocation_order':allocation,'seconds':round(time.monotonic()-start,3),'ratios':ratios})
                (out/'runs.json').write_text(json.dumps({'artifact':artifact,'runs':runs},indent=2)+'\n')
                print({'run':number,'side':side,'execution':execution,'allocation':allocation,'seconds':runs[-1]['seconds']},flush=True)
    qualified=[];failed={};summaries={}
    for title in pairs:
        reasons=[];facts={}
        for side in ('candidate','peer'):
            relevant=[r for r in runs if r['side']==side];values=[r['ratios'][title] for r in relevant]
            medians={'all':statistics.median(values)}
            for factor in ('execution_order','allocation_order'):
                for level in ('candidate-first','peer-first'):
                    medians[factor+'='+level]=statistics.median(r['ratios'][title] for r in relevant if r[factor]==level)
            facts[side]=medians
            if any(x<1/1.10 or x>1.10 for x in values):reasons.append(side+':gross-instability')
            if any(x<1/1.05 or x>1.05 for x in medians.values()):reasons.append(side+':fixed-bias')
        summaries[title]=facts
        if reasons:failed[title]=reasons
        else:qualified.append(title)
    receipt={'schema':'hyle.cpu-final-aa/v1','artifact':artifact,'runs':len(runs),'pairs':len(pairs),'qualified':qualified,'failed':failed,'summaries':summaries,'gate':'PASS' if not failed else 'INCONCLUSIVE','scope':'workload-matched AA, fixed eight runs/four order strata, shared nanobench parser, caller CPU2; not P1, no winner or loss','limits':['machine/profile/controls and immutable full P1 still required','failed AA domains use engineering-default decision or named blocker; no increased sample budget']}
    if args.selection:receipt['selection_sha256']=hashlib.sha256(args.selection.read_bytes()).hexdigest()
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print({'pairs':len(pairs),'qualified':len(qualified),'inconclusive':len(failed),'gate':receipt['gate']},flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
