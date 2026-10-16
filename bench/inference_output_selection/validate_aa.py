"""G08 workload qualification, using the existing machine/parser/bias policy."""
import argparse,hashlib,json,os,platform,re,statistics,subprocess,sys,uuid,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from hyle_perf.environment import PERF_ENVIRONMENT_BLOCKLIST,PERF_ENVIRONMENT_FIXED,PERF_ENVIRONMENT_PREFIXES
from hyle_perf.machine import applied_macos_qos,evaluate_machine_profile,get_machine_profile,load_registry,probe_machine
from hyle_perf.nanobench import parse_nanobench,select_pair
from hyle_perf.controls import AA_MAXIMUM_BIAS_PERCENT
from hyle_perf.stats import equivalence_band

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cache=dict(x.split('=',1) for x in (a.build/'CMakeCache.txt').read_text().splitlines() if '=' in x and not x.startswith(('#','//')))
    route=cache['HYLE_G08_ROUTE:STRING'];assert route in ['cpu','cuda','apple','vulkan']
    machine={'cpu':'W1-WIN','cuda':'W1-WSL','apple':'M1-MAC','vulkan':'M1-MAC'}[route]
    air=platform.system()=='Darwin';assert air==(route in ['apple','vulkan'])
    if air:assert subprocess.check_output(['/usr/sbin/sysctl','-n','hw.model'],text=True).strip()=='Mac16,12'
    gate=evaluate_machine_profile(get_machine_profile(load_registry(),machine),probe_machine(paths=[('build',a.build)],expected_qos='interactive' if air else None),role='engineering')
    if gate['gate']!='PASS':raise RuntimeError('A/A machine gate failed: '+str(gate['errors']))
    suffix='.exe' if platform.system()=='Windows' else ''
    executables={v:a.build/'bin'/('g08_topk_'+v+suffix) for v in ['primary','placement']}
    artifacts={v:digest(f) for v,f in executables.items()};assert len(set(artifacts.values()))==2,'placement artifacts identical'
    specification=Path(__file__).parent/('qualified-sdk-'+route+'.json');sdk_identity=json.loads(specification.read_text())
    receipt_path=ROOT/sdk_identity['receipt'];receipt=json.loads(receipt_path.read_text());archive=receipt_path.parent/sdk_identity['archive']
    recorded_archive=receipt.get('raw_archive',receipt.get('raw'))
    assert recorded_archive['path']==sdk_identity['archive'] and recorded_archive['sha256']==sdk_identity['archive_sha256']
    assert digest(archive)==sdk_identity['archive_sha256']
    with zipfile.ZipFile(archive) as z:sdk=json.loads(z.read(sdk_identity['result_archive_entry']))
    assert sdk['candidate']==sdk_identity['candidate']=='5103c13d8ed119b06aa64e799f9768aea45ab913' and sdk['mode']=='static'
    assert sdk['state'] in ['RELOCATED_SDK_PUBLIC_CONSUMERS_HEADERS_LICENSE_PASS','RELOCATED_SDK_PUBLIC_CONSUMERS_HEADERS_LICENSE_RUNTIME_PASS']
    assert Path(cache['Hyle_DIR:PATH']).resolve().parents[2]==Path(sdk_identity['prefix']).resolve(),'wrong installed SDK'
    def verify_sdk():
        for row in sdk_identity['owner_artifacts']:assert digest(Path(row['path']))==row['sha256'],('qualified SDK changed',row['path'])
    verify_sdk()
    reference=Path(cache['HYLE_G08_REFERENCE_DIR:PATH']);original=json.loads((reference/'source-identity.json').read_text())
    build_reference=json.loads((a.build/'reference-identity.json').read_text())
    assert build_reference['original_checkout_verified'] and build_reference['original_checkout']['revision']=='85bcfa4ebc29cc4d2ed5fb122074204aca675d58','unbound original peer checkout'
    assert original['original_revision']=='85bcfa4ebc29cc4d2ed5fb122074204aca675d58' and original['route']==route
    for row in original['files']:assert digest(reference/row['path'])==row['materialized_sha256']
    for row in original['owned_public_overlay']:assert digest(reference/row['path'])==row['sha256']
    for row in original['shared_private_headers']:assert digest(ROOT/row['path'])==row['sha256']
    commands=json.loads((a.build/'compile_commands.json').read_text());compiled=[]
    expected=[(reference/x).resolve() for x in original['compiled_original_sources']]
    if route in ['cpu','cuda']:expected.append((Path(__file__).parent/'reference_provider.cc').resolve())
    for path in expected:
        matching=[x for x in commands if Path(x['file']).resolve()==path];assert len(matching)==1,('missing/duplicate original TU',str(path))
        command=matching[0]['command']
        for definition in original['compile_definitions']:
            assert '-D'+definition in command or '/D'+definition in command,('original symbol isolation missing',str(path),definition)
        compiled.append(dict(path=str(path),command_sha256=hashlib.sha256(command.encode()).hexdigest()))
    codegen=dict(original_translation_units=compiled)
    if route=='apple':
        symbols=subprocess.check_output(['nm','-C',str(a.build/'libg08_original85.a')],text=True)
        for name in ['g08_original_lower_tensor','g08_original_lower_mature_indexing']:
            assert re.search(r'\b[Tt]\s+.*\b'+name+r'\(',symbols),'original Apple definition missing: '+name
        assert re.search(r'\bU\s+.*\bg08_original_lower_mature_indexing\(',symbols),'original Apple tensor caller is not bound to original indexing'
        assert 'hyle::compute::apple::detail::lower_mature_indexing(' not in symbols,'original Apple reference leaks current indexing'
        symbol_log=a.output/'original-apple-symbols.log';symbol_log.write_text(symbols)
        codegen['apple_symbol_binding']=dict(state='ORIGINAL_TENSOR_CALLS_ORIGINAL_INDEXING',sha256=digest(symbol_log),log=str(symbol_log.resolve()))
    sources=['topk_bench.h','topk_bench.cc','generic_arm.cc','apple_arm.cc','vulkan_arm.cc','reference_provider.cc','placement.cc','CMakeLists.txt','verify_reference.py','validate_aa.py']
    identity=dict(artifacts=artifacts,machine=gate['physical_host_identity'],sdk_record=digest(receipt_path),sdk_identity=digest(specification),
        original_source=digest(reference/'source-identity.json'),reference_build_check=digest(a.build/'reference-identity.json'),
        original_codegen=codegen,benchmark_sources={x:digest(Path(__file__).parent/x) for x in sources})
    ready=a.build/'bin/g08_topk_aa.ready';ready.write_text('FAIL\n');target=a.output/'topk-aa.json'
    if target.exists():
        old=json.loads(target.read_text())
        assert old.get('gate')=='PASS' and old.get('identity')==identity,'retained A/A failed or changed; no blind repeat'
        assert all(digest(Path(row['log']))==row['sha256'] for row in old['runs'])
        ready.write_text('PASS\n');return 0
    raw=a.output/'topk-aa-raw';raw.mkdir(parents=True)  # Preserve any partial qualification.
    shapes=[(1,1),(16,1)]+([(16,3)] if route=='apple' else [])
    def environment():
        e={k:v for k,v in os.environ.items() if k not in PERF_ENVIRONMENT_BLOCKLIST and not any(k.startswith(x) for x in PERF_ENVIRONMENT_PREFIXES)}
        e.update(PERF_ENVIRONMENT_FIXED);e.update(HYLE_PERF_RUN_NONCE=uuid.uuid4().hex,HYLE_PERF_APPLE_QOS='interactive');return e
    def execute(exe,log,e):
        with log.open('w') as stream:result=subprocess.run([str(exe)],env=e,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('numerical/A-A process failed: '+str(log))
        text=log.read_text();records=[x for x in text.splitlines() if x.startswith('# G08_ORACLE ')]
        assert len(records)==len(shapes) and all(sum(f'route={route} R={rows} V=248320 K={count} ' in x and 'stored_bits_and_first_index=PASS cycles=3 warm_cycles=8' in x for x in records)==1 for rows,count in shapes),'missing independent numerical preflight'
        return text
    for variant,exe in executables.items():
        e=environment();e['HYLE_BENCH_VERIFY_ONLY']='1';execute(exe,raw/('preflight-'+variant+'.log'),e)
    runs=[]
    for arm in ['native','selected']:
        for variant,exe in executables.items():
            for execution in [False,True]:
                for allocation in [False,True]:
                    e=environment();e['HYLE_BENCH_AA_ROUTE']=arm
                    if execution:e['HYLE_BENCH_RIVAL_FIRST']='1'
                    if allocation:e['HYLE_BENCH_PEER_ALLOCATION_FIRST']='1'
                    log=raw/f'{arm}-{variant}-{int(execution)}-{int(allocation)}.log';text=execute(exe,log,e)
                    markers=[x for x in text.splitlines() if x.startswith('# HYLE_TOPK_RUN ')];assert len(markers)==1,'missing/duplicate telemetry'
                    actual=dict(re.findall(r'(\w+)=(.*?)(?= \w+=|$)',markers[0].split(' ',2)[2]))
                    expected=dict(run_nonce=e['HYLE_PERF_RUN_NONCE'],route=route,verify_only='0',aa_route=arm,execution_order='peer-first' if execution else 'candidate-first',allocation_order='peer-first' if allocation else 'candidate-first',placement='1' if variant=='placement' else '0',paired_rounds='104' if route=='cuda' else '16',apple_qos='user-interactive' if air else 'not-applicable',qos_ok='1' if air else 'not-applicable')
                    assert all(actual.get(k)==v for k,v in expected.items()),'A/A policy mismatch'
                    pairs=parse_nanobench(text);ratios={}
                    for rows,count in shapes:
                        title=f'{route} TopK R={rows} V=248320 K={count}'
                        c,p=select_pair(pairs,'HYLE TopK','Original85 TopK',title=title);ratios[title]=c.operations_per_second/p.operations_per_second
                    runs.append(dict(arm=arm,variant=variant,execution_order=expected['execution_order'],allocation_order=expected['allocation_order'],ratios=ratios,log=str(log.resolve()),sha256=digest(log)))
                    print(json.dumps(dict(completed=len(runs),total=16)),flush=True)
    summaries={};errors=[];narrow=equivalence_band(AA_MAXIMUM_BIAS_PERCENT);gross=equivalence_band(10)
    for arm in ['native','selected']:
        matching=[r for r in runs if r['arm']==arm]
        for title in matching[0]['ratios']:
            values=[r['ratios'][title] for r in matching];medians={'all':statistics.median(values)}
            for factor in ['variant','execution_order','allocation_order']:
                for level in sorted({r[factor] for r in matching}):medians[factor+'='+level]=statistics.median(r['ratios'][title] for r in matching if r[factor]==level)
            if any(not gross[0]<=x<=gross[1] for x in values):errors.append(arm+title+': gross instability')
            if any(not narrow[0]<=x<=narrow[1] for x in medians.values()):errors.append(arm+title+': factor median bias')
            summaries[arm+'/'+title]=dict(ratios=values,factor_medians=medians)
    verify_sdk();result=dict(gate='FAIL' if errors else 'PASS',identity=identity,runs=runs,summaries=summaries,errors=errors);target.write_text(json.dumps(result,indent=2)+'\n')
    if not errors:ready.write_text('PASS\n')
    print(json.dumps(dict(gate=result['gate'],errors=errors)));return 2 if errors else 0

if __name__=='__main__':
    with applied_macos_qos('interactive' if platform.system()=='Darwin' else None):raise SystemExit(main())
