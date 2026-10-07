"""Task-local immutable batch receipts; no changes to public validation tools."""
import argparse, hashlib, json, os, platform, re, subprocess, time
import xml.etree.ElementTree as ET
from pathlib import Path

GROUPS={
    'artifacts':'*N1*:*N2*',
    'fixed':'*C5ActualGguf*',
    'current':'*C5Parameters*:*C5NewDeepSeek*:*CurrentRegions.*',
    'families':':'.join(['*MoeFragments.D5Glm*','*MoeFragments.D5Kimi*',
        '*MoeFragments.D5Kda*','*MoeFragments.D5SixProfiles*',
        '*MoeFragments.D5EightGguf*','*MoeFragments.D5OfficialKimi*',
        '*MoeFragments.D5Thirteen*','*QwenRecurrent.D5NineProfiles*',
        '*Default*.D5Qwen35Official*']),
    'lifecycle':':'.join(['*CudaArtifactLifecycle*','CudaArtifactAdmission.*',
        'CudaDefaultCompute.DenseFfnUsesFixedRulesAndIndependentSlots']),
}

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def atomic_json(path,value):
    temporary=path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)

def names_from_listing(value):
    names=set();suite=''
    for line in value.splitlines():
        body=line.split(' #')[0].strip()
        if line and not line.startswith(' ') and body.endswith('.'):suite=body
        elif line.startswith('  ') and suite:names.add(suite+body)
    return names

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--journey',required=True)
    parser.add_argument('--fixture',required=True,type=Path)
    parser.add_argument('--minimal-path',action='store_true')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--import-fixed',type=Path)
    parser.add_argument('--import-batch',type=Path)
    parser.add_argument('--rerun-group',action='append',default=[])
    parser.add_argument('--reason')
    args=parser.parse_args()
    if args.rerun_group and not args.reason:parser.error('reruns require a concrete impact reason')
    if set(args.rerun_group)-GROUPS.keys():parser.error('unknown rerun group')
    flags={};environment=dict(os.environ)
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(32771)
        flags={'creationflags':subprocess.CREATE_NO_WINDOW}
    environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                       HYLE_D5_QWEN35_FIXTURE=str(args.fixture))
    if args.minimal_path:
        assert os.name=='nt','This reduced PATH journey is Windows specific'
        for name in list(environment):
            if name.lower()=='path':del environment[name]
        environment['PATH']=os.environ['SystemRoot']+'/System32;'+os.environ['SystemRoot']
    args.output.mkdir(parents=True,exist_ok=True)
    machine=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version',
        '--format=csv,noheader'],capture_output=True,text=True,check=True,**flags).stdout.strip()
    runtime={p.name:digest(p) for p in sorted(args.exe.parent.glob('*.dll'))}
    runtime.update({p.name:digest(p) for p in sorted(args.exe.parent.glob('*.so*')) if p.is_file()})
    identity={'schema':1,'journey':args.journey,'exe':str(args.exe.resolve()),
        'exe_sha256':digest(args.exe),'runtime_sha256':runtime,
        'machine':machine,'platform':platform.platform(),
        'fixture':str(args.fixture.resolve()),
        'fixture_sha256':{p.relative_to(args.fixture).as_posix():digest(p)
            for p in sorted(args.fixture.rglob('*')) if p.is_file()},
        'environment':{key:environment.get(key) for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
            'GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DISABLE_FUSION','GGML_CUDA_GRAPH_OPT',
            'NVIDIA_TF32_OVERRIDE','CUBLAS_WORKSPACE_CONFIG')},
        'full_filter':':'.join(GROUPS.values())}
    if args.minimal_path:identity['minimal_path']=True
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    contract=args.output/'identity.json'
    if contract.exists():
        assert json.loads(contract.read_text(encoding='utf-8'))==identity, \
            'Immutable batch identity changed: create a new output and state its impact; do not overwrite receipts'
    else:atomic_json(contract,identity)
    listing=subprocess.run([str(args.exe),'--gtest_filter='+identity['full_filter'],
        '--gtest_list_tests'],env=environment,capture_output=True,text=True,check=True,**flags)
    expected=names_from_listing(listing.stdout);assert expected,'Empty requested test union'
    if args.import_batch is not None:
        # Keep the failed aggregate intact; import only wholly passing component groups.
        original=args.import_batch
        assert args.exe.stat().st_mtime_ns<=original.stat().st_mtime_ns
        source_tree=ET.parse(original).getroot()
        source_tests={test.attrib['classname']+'.'+test.attrib['name']:test
                      for test in source_tree.iter('testcase')}
        assert set(source_tests)==expected,'Different consumer union cannot be imported'
        import fnmatch
        for name,flt in GROUPS.items():
            receipt=args.output/(name+'.json')
            if receipt.exists():continue
            selected=[test for test_name,test in source_tests.items()
                      if any(fnmatch.fnmatchcase(test_name,pattern) for pattern in flt.split(':'))]
            if not selected:continue
            if any(test.find('failure') is not None or test.find('skipped') is not None
                   or test.attrib.get('status')!='run' for test in selected):continue
            xml=args.output/(name+'-import.xml');root=ET.Element('testsuites')
            for test in selected:root.append(test)
            ET.ElementTree(root).write(xml,encoding='utf-8',xml_declaration=True)
            log=args.output/(name+'-import.json')
            atomic_json(log,{'original_xml':str(original.resolve()),'sha256':digest(original),
                'failed_aggregate_preserved':True,'selected_wholly_passing_group':name})
            value={'identity':key,'status':'PASS','tests':sorted(
                    test.attrib['classname']+'.'+test.attrib['name'] for test in selected),
                'xml':xml.name,'xml_sha256':digest(xml),'log':log.name,'log_sha256':digest(log),
                'returncode':0,'seconds':round(sum(float(test.attrib['time']) for test in selected),3),
                'reason':'Wholly passing group from unchanged consumer; original failed aggregate remains open',
                'imported':True}
            atomic_json(receipt,value)
        if args.import_fixed is not None and not (args.output/'fixed.json').exists():
            original=args.import_fixed/'case.xml'
            source=json.loads((args.import_fixed/'result.json').read_text(encoding='utf-8'))
            assert source['returncode']==0 and Path(source['args'][0]).resolve()==args.exe.resolve()
            assert args.exe.stat().st_mtime_ns<=original.stat().st_mtime_ns
            tree=ET.parse(original).getroot();tests=list(tree.iter('testcase'))
            selected={test.attrib['classname']+'.'+test.attrib['name'] for test in tests}
            assert selected=={name for name in expected if fnmatch.fnmatchcase(name,GROUPS['fixed'])}
            assert all(test.find('failure') is None and test.find('skipped') is None for test in tests)
            xml=args.output/'fixed-import.xml';xml.write_bytes(original.read_bytes())
            log=args.output/'fixed-import.log';log.write_bytes((args.import_fixed/'case.log').read_bytes())
            atomic_json(args.output/'fixed.json',{'identity':key,'status':'PASS','tests':sorted(selected),
                'xml':xml.name,'xml_sha256':digest(xml),'log':log.name,'log_sha256':digest(log),
                'returncode':0,'seconds':source['seconds'],'reason':'Exact unchanged consumer physical-codec group',
                'imported':True})
    plan=[]
    for name,flt in GROUPS.items():
        existing=args.output/(name+'.json')
        prior=json.loads(existing.read_text(encoding='utf-8')) if existing.exists() else None
        if prior:
            assert prior['identity']==key
            if prior['xml'] is not None:
                assert digest(args.output/prior['xml'])==prior['xml_sha256']
            assert digest(args.output/prior['log'])==prior['log_sha256']
            if prior['status']!='PASS' and name not in args.rerun_group:
                action='STOP';reason='Prior failure requires diagnosis and an explicit bounded rerun reason'
            elif name in args.rerun_group:action='RUN';reason=args.reason
            else:action='REUSE';reason='Exact executable, runtime, machine, contract and preserved receipt match'
        else:action='RUN';reason='This delivery group has no matching receipt'
        plan.append({'group':name,'filter':flt,'action':action,'reason':reason})
    atomic_json(args.output/'plan.json',{'identity':key,'requested_tests':len(expected),'groups':plan})
    print(json.dumps({'identity':key,'requested_tests':len(expected),'groups':plan}),flush=True)
    if args.plan_only:return
    observed=set();failed=False
    for row in plan:
        name=row['group'];receipt=args.output/(name+'.json')
        if row['action']=='STOP':failed=True;continue
        if row['action']=='REUSE':value=json.loads(receipt.read_text(encoding='utf-8'))
        else:
            epoch=time.time_ns();xml=args.output/(name+'-'+str(epoch)+'.xml')
            log=args.output/(name+'-'+str(epoch)+'.log');start=time.monotonic()
            imported=False
            if name=='fixed' and args.import_fixed is not None:
                # Import only a completed, unchanged executable result; never a failed monolithic batch.
                source=json.loads((args.import_fixed/'result.json').read_text(encoding='utf-8'))
                assert source['returncode']==0
                assert Path(source['args'][0]).resolve()==args.exe.resolve()
                original=args.import_fixed/'case.xml'
                assert args.exe.stat().st_mtime_ns<=original.stat().st_mtime_ns
                xml.write_bytes(original.read_bytes());log.write_bytes((args.import_fixed/'case.log').read_bytes())
                rc=0;seconds=source['seconds'];imported=True
            else:
                with log.open('wb') as stream:
                    try:
                        result=subprocess.run([str(args.exe),'--gtest_filter='+row['filter'],
                            '--gtest_output=xml:'+str(xml)],env=environment,stdout=stream,
                            stderr=subprocess.STDOUT,timeout=600,**flags)
                        rc=result.returncode
                    except subprocess.TimeoutExpired:rc=124
                seconds=round(time.monotonic()-start,3)
            if not xml.exists():
                value={'identity':key,'status':'FAIL','tests':[],
                    'xml':None,'xml_sha256':None,'log':log.name,'log_sha256':digest(log),
                    'returncode':rc,'seconds':seconds,'reason':row['reason'],'imported':imported}
                atomic_json(receipt,value);failed=True
                print(json.dumps({'group':name,'status':'FAIL','reason':'No completed XML',
                    'returncode':rc,'seconds':seconds}),flush=True)
                continue
            tree=ET.parse(xml).getroot();tests=list(tree.iter('testcase'))
            names={test.attrib['classname']+'.'+test.attrib['name'] for test in tests}
            valid=rc==0 and all(test.find('failure') is None and test.find('skipped') is None
                               and test.attrib.get('status')=='run' for test in tests)
            assert names,'No tests executed'
            value={'identity':key,'status':'PASS' if valid else 'FAIL','tests':sorted(names),
                'xml':xml.name,'xml_sha256':digest(xml),'log':log.name,'log_sha256':digest(log),
                'returncode':rc,'seconds':seconds,'reason':row['reason'],'imported':imported}
            atomic_json(receipt,value)
        assert not observed.intersection(value['tests']),'Overlapping groups'
        observed.update(value['tests']);failed|=value['status']!='PASS'
        print(json.dumps({'group':name,'action':row['action'],'status':value['status'],
            'tests':len(value['tests']),'seconds':value['seconds']}),flush=True)
    exact=observed==expected
    atomic_json(args.output/'coverage.json',{'identity':key,'status':'PASS' if exact and not failed else 'FAIL',
        'exact_test_union':exact,'expected':len(expected),'observed':len(observed),
        'missing':sorted(expected-observed),'extra':sorted(observed-expected)})
    assert exact and not failed,'Batch did not qualify; failed receipts remain intact'

if __name__=='__main__':main()
