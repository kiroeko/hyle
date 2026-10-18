import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil

sys.stdout.reconfigure(encoding='utf-8')
repo = Path('C:/public/project-hyle/hyle')
root = Path('C:/h7/w-cpu-cuda/selected-graph')
mode = sys.argv[1] if len(sys.argv) > 1 else 'source'
assert mode in ['source', 'shared', 'static-configure', 'static', 'static-consumers', 'installed-seams', 'native-overlap', 'final-seams', 'memory-snapshot']
record = Path('C:/h7/h1-offload-windows' + ('' if mode == 'source' else '-' + mode))
candidate = '7eaf67b8a496ee19b609533980cd69e7efb5aa49'
assert (Path('C:/h7') / '.hyle-owner').is_file()
assert root.resolve().is_relative_to(Path('C:/h7').resolve())
class Performance(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in [
        'commit_total', 'commit_limit', 'commit_peak', 'physical_total', 'physical_available',
        'system_cache', 'kernel_total', 'kernel_paged', 'kernel_nonpaged', 'page_size']] + [
        (name, ctypes.c_ulong) for name in ['handles', 'processes', 'threads']]

performance_info = ctypes.WinDLL('psapi', use_last_error=True).GetPerformanceInfo
performance_info.argtypes = [ctypes.POINTER(Performance), ctypes.c_ulong]
performance_info.restype = ctypes.c_int
guards = {'minimum_available_gib': 16, 'minimum_commit_headroom_gib': 16,
          'maximum_owned_private_gib': 12}
owned_root = psutil.Process(os.getpid())

def memory_snapshot():
    m = Performance()
    m.cb = ctypes.sizeof(m)
    if not performance_info(ctypes.byref(m), m.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    private = 0
    count = 0
    for process in [owned_root, *owned_root.children(recursive=True)]:
        try:
            private += process.memory_info().private
            count += 1
        except psutil.NoSuchProcess:
            continue
    page_gib = m.page_size / (1024 ** 3)
    return {'available_gib': m.physical_available * page_gib,
            'committed_gib': m.commit_total * page_gib,
            'commit_limit_gib': m.commit_limit * page_gib,
            'commit_headroom_gib': (m.commit_limit - m.commit_total) * page_gib,
            'commit_peak_since_boot_gib': m.commit_peak * page_gib,
            'owned_private_gib': private / (1024 ** 3), 'owned_processes': count}

def guard_reason(snapshot):
    if snapshot['available_gib'] < guards['minimum_available_gib']:
        return 'available physical memory'
    if snapshot['commit_headroom_gib'] < guards['minimum_commit_headroom_gib']:
        return 'system commit headroom'
    if snapshot['owned_private_gib'] > guards['maximum_owned_private_gib']:
        return 'owned process-tree private memory'
    return None

if mode == 'memory-snapshot':
    print(json.dumps({'snapshot': memory_snapshot(), 'guards': guards,
                      'claim': 'read-only snapshot; no build or stop-trigger qualification'}, indent=2))
    sys.exit(0)

if '--resume-after-memory-stop' not in sys.argv[2:]:
    sys.exit('Heavy validation remains held after user memory alarm; no child started.')
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
for name in subprocess.check_output(['git', 'diff', '--name-only', candidate, 'HEAD'], cwd=repo, text=True).splitlines():
    assert name.startswith(('docs/', 'evidence/')), name

ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
flags = {'creationflags': subprocess.CREATE_NO_WINDOW}
vs = 'C:/Program Files/Microsoft Visual Studio/18/Enterprise/Common7/Tools/VsDevCmd.bat'
cmd = '"' + os.environ['ComSpec'] + '" /d /s /c "call "' + vs + '" -no_logo -arch=x64 -host_arch=x64 >nul && set"'
for line in subprocess.check_output(cmd, text=True, **flags).splitlines():
    key, sep, value = line.partition('=')
    if sep and key:
        os.environ[key] = value
os.environ['HYLE_DEP_SOURCE_CACHE'] = 'C:/public/tools/hyle-dependencies/v1'
os.environ['VSLANG'] = '1033'
record.mkdir(exist_ok=True)
rows = []
if mode == 'installed-seams' and (record/'receipt.json').is_file():
    saved = json.loads((record/'receipt.json').read_bytes())
    assert saved['candidate'] == candidate
    rows = saved['rows']

def run(stage, args):
    if mode == 'installed-seams' and any(row['stage'] == stage and row['exit'] == 0 for row in rows):
        print('REUSE completed unchanged installed-seam stage ' + stage,flush=True)
        return
    snapshot = memory_snapshot()
    free = snapshot['available_gib']
    print(f'{stage}: free={free:.2f} GiB; log={record / (stage + ".log")}', flush=True)
    reason = guard_reason(snapshot)
    if reason:
        raise RuntimeError('memory guard before stage: ' + reason + '; no build started')
    log = record / (stage + '.log')
    started = time.monotonic()
    minimum = free
    minimum_commit_headroom = snapshot['commit_headroom_gib']
    maximum_owned_private = snapshot['owned_private_gib']
    with log.open('wb') as out:
        p = subprocess.Popen(args, cwd=repo, stdout=out, stderr=subprocess.STDOUT, **flags)
        while p.poll() is None:
            try:
                snapshot = memory_snapshot()
                reason = guard_reason(snapshot)
            except Exception as error:
                reason = 'memory monitor failed: ' + str(error)
            minimum = min(minimum, snapshot['available_gib'])
            minimum_commit_headroom = min(minimum_commit_headroom, snapshot['commit_headroom_gib'])
            maximum_owned_private = max(maximum_owned_private, snapshot['owned_private_gib'])
            if reason:
                # This PID is the still-running child started above. Terminate
                # only that owned process tree and preserve its objects/logs.
                if p.poll() is None:
                    subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **flags)
                p.wait()
                (record / (stage + '-interrupted.json')).write_bytes((json.dumps({
                    'candidate': candidate, 'stage': stage, 'verdict': 'INTERRUPTED',
                    'reason': reason, 'snapshot': snapshot, 'guards': guards,
                    'exit': p.returncode}, indent=2) + '\n').encode('utf-8'))
                raise RuntimeError(f'memory guard stopped owned stage {stage}: {reason}')
            time.sleep(1)
    row = {'stage': stage, 'exit': p.returncode, 'seconds': round(time.monotonic() - started, 2),
           'minimum_available_gib': round(minimum, 2), 'log': str(log)}
    row.update({'minimum_commit_headroom_gib': round(minimum_commit_headroom, 2),
                'maximum_owned_private_gib': round(maximum_owned_private, 2), 'guards': guards})
    rows.append(row)
    (record / 'receipt.json').write_text(json.dumps({'candidate': candidate, 'rows': rows,
        'build_parallel': 1, 'test_parallel': 1, 'claim': 'only listed actual stages'}, indent=2) + '\n')
    if p.returncode:
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines()[-28:]:
            print(line[:190], flush=True)
        sys.exit(p.returncode)
    print('PASS ' + stage, flush=True)

if mode == 'source':
    parent = root / 'source-parent'
    cmake = parent / 'CMakeLists.txt'
    body = cmake.read_text(encoding='utf-8')
    if 'add_executable(hyle_cpu_cuda_contracts' not in body:
        body += '\nadd_executable(hyle_cpu_cuda_contracts "' + (repo / 'tests/graph/cpu_cuda_contracts.cc').as_posix() + '")\n'
        body += 'target_link_libraries(hyle_cpu_cuda_contracts PRIVATE hyle::graph)\nhyle_stage_runtime(hyle_cpu_cuda_contracts)\n'
    for target in ['hyle_cpu_cuda_sample', 'hyle_cuda_weights_sample', 'hyle_cpu_cuda_contracts']:
        if 'add_test(NAME ' + target + ' ' not in body:
            body += f'add_test(NAME {target} COMMAND {target})\nset_tests_properties({target} PROPERTIES TIMEOUT 180)\n'
    cmake.write_bytes(body.encode('utf-8'))
    run('source-configure', ['cmake', '-S', str(parent), '-B', str(root / 'source-build')])
    run('source-build', ['cmake', '--build', str(root / 'source-build'), '--target',
        'hyle_cpu_cuda_sample', 'hyle_cuda_weights_sample', 'hyle_cpu_cuda_contracts',
        'hyle_selected_source_consumer', 'hyle_selected_source_compute', '--parallel', '1'])
    run('source-test', ['ctest', '--test-dir', str(root / 'source-build'), '--output-on-failure', '-j', '1',
        '-R', '^(hyle_cpu_cuda_sample|hyle_cuda_weights_sample|hyle_cpu_cuda_contracts|hyle_selected_source_consumer|hyle_selected_source_compute)$'])
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
    print('Actual Windows source batch completed.', flush=True)
elif mode == 'shared':
    delivery = Path('C:/h7/h1-windows-delivery')
    prefix = delivery / 'shared-sdk'
    run('shared-install', ['cmake', '--install', str(root / 'source-build'), '--prefix', str(prefix)])
    for name, source in [('joint','samples/package/graph/cpu_cuda'), ('direct','samples/package/compute/cuda_weights'), ('contracts','tests/graph/cpu_cuda_contracts')]:
        build = delivery / ('shared-' + name)
        run(name + '-configure', ['cmake','-S',str(repo/source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH='+str(prefix),'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY='+str(delivery/'shared-apps')])
        run(name + '-build', ['cmake','--build',str(build),'--parallel','1'])
        run(name + '-test', ['ctest','--test-dir',str(build),'--output-on-failure','-j','1'])
    print('Actual Windows shared SDK batch completed.', flush=True)
elif mode == 'static-configure':
    run('static-configure', ['cmake','-S',str(root/'source-parent'),'-B',str(root/'source-build'),'-DHYLE_STATIC=ON','-DCMAKE_INSTALL_PREFIX=C:/h7/h1-windows-delivery/static-sdk'])
    run('static-build-plan', ['ninja','-C',str(root/'source-build'),'-n','hyle_cpu_cuda_sample','hyle_cuda_weights_sample','hyle_cpu_cuda_contracts'])
elif mode == 'installed-seams':
    delivery = Path('C:/h7/h1-windows-delivery')
    headers = ['compute/resource_budget.h', 'compute/cuda_weights.h', 'graph/cpu_cuda.h', 'graph/executable.h']
    for flavor in ['shared', 'static']:
        prefix = delivery / (flavor + '-sdk')
        for index, header in enumerate(headers):
            expected = subprocess.check_output(['git','show',candidate+':include/hyle/'+header],cwd=repo,**flags)
            assert (prefix/'include/hyle'/header).read_bytes() == expected, header
            unit = record / (flavor + '-header-' + str(index) + '.cc')
            unit.write_bytes(('#include <hyle/'+header+'>\nint main() { return 0; }\n').encode('utf-8'))
            args = ['cl','/nologo','/Zs','/std:c++latest','/EHsc','/permissive-','/I'+str(prefix/'include'),
                    '/I'+str(Path(os.environ['CUDA_PATH'])/'include')]
            if flavor == 'static':
                args.append('/DHYLE_STATIC')
            run(flavor+'-header-'+str(index),args+[str(unit)])
        relocated = delivery / (flavor + '-sdk-relocated')
        assert prefix.resolve().parent == delivery.resolve()
        assert relocated.resolve().parent == delivery.resolve()
        assert prefix.is_dir() and not relocated.exists()
        prefix.rename(relocated)
        try:
            build = delivery / (flavor + '-relocated-joint')
            run(flavor+'-relocated-configure',['cmake','-S',str(repo/'samples/package/graph/cpu_cuda'),
                '-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_PREFIX_PATH='+str(relocated),'-DHyle_DIR='+str(relocated/'lib/cmake/Hyle'),
                '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY='+str(delivery/(flavor+'-apps'))])
            run(flavor+'-relocated-build',['cmake','--build',str(build),'--parallel','1'])
            run(flavor+'-relocated-test',['ctest','--test-dir',str(build),'--output-on-failure','-j','1'])
        finally:
            assert not prefix.exists()
            relocated.rename(prefix)
    print('Installed headers and actual relocated SDK joint consumers completed.',flush=True)
elif mode == 'final-seams':
    run('native-probe-math-check',['C:/h7/h1-windows-delivery/shared-apps/hyle_offload_overlap.exe'])
    run('missing-composition-check',[sys.executable,str(repo/'.scratch/compute-unified/h1-missing-composition.py')])
elif mode == 'native-overlap':
    import hashlib
    import sqlite3
    delivery = Path('C:/h7/h1-windows-delivery')
    build = Path('C:/h7/h1-offload-overlap-probe')
    nsys = Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe')
    assert nsys.is_file()
    run('probe-configure',['cmake','-S',str(repo/'.scratch/compute-unified/h1-offload-overlap'),
        '-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_PREFIX_PATH='+str(delivery/'shared-sdk'),
        '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY='+str(delivery/'shared-apps')])
    run('probe-build',['cmake','--build',str(build),'--parallel','1'])
    trace = record / 'overlap-cuda'
    run('native-trace',[str(nsys),'profile','--trace=cuda','--sample=none','--cpuctxsw=none',
        '--cuda-memory-usage=false','--force-overwrite=true','-o',str(trace),
        str(delivery/'shared-apps/hyle_offload_overlap.exe')])
    database = record/'overlap-cuda.sqlite'
    run('trace-export',[str(nsys),'export','--type=sqlite','--force-overwrite=true',
        '--output',str(database),str(trace)+'.nsys-rep'])
    connection = sqlite3.connect('file:'+database.as_posix()+'?mode=ro',uri=True)
    intersections = connection.execute('''SELECT m.deviceId,m.contextId,m.streamId,k.streamId,
        m.start,m.end,k.start,k.end,n.value,
        min(m.end,k.end)-max(m.start,k.start)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN CUPTI_ACTIVITY_KIND_KERNEL k
        ON m.deviceId=k.deviceId AND m.contextId=k.contextId AND m.streamId!=k.streamId
        AND m.start<k.end AND k.start<m.end JOIN StringIds n ON n.id=k.shortName
        WHERE m.bytes=16777216 AND m.copyKind=1''').fetchall()
    kernels = connection.execute('SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
    connection.close()
    identity = {}
    for path in [database,Path(str(trace)+'.nsys-rep')]:
        identity[path.name] = {'path':str(path),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    analysis = {'candidate':candidate,'verdict':'PASS' if intersections else 'INCONCLUSIVE',
        'scope':'Physical native overlap only; no P1 or winner/speedup claim',
        'copy_bytes':16777216,'kernel_count':kernels,'intersections':intersections,
        'columns':['device','context','copy_stream','kernel_stream','copy_start_ns','copy_end_ns',
                   'kernel_start_ns','kernel_end_ns','kernel_name','intersection_ns'],
        'trace_identity':identity,'probe_sha256':hashlib.sha256((repo/'.scratch/compute-unified/h1-overlap.cc').read_bytes()).hexdigest()}
    (record/'overlap-analysis.json').write_bytes((json.dumps(analysis,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({'native_overlap':analysis['verdict'],'kernel_count':kernels,
                      'intersection_count':len(intersections)}),flush=True)
else:
    delivery = Path('C:/h7/h1-windows-delivery')
    prefix = delivery/'static-sdk'
    if mode != 'static-consumers':
        run('static-build', ['cmake','--build',str(root/'source-build'),'--target','hyle_cpu_cuda_sample','hyle_cuda_weights_sample','hyle_cpu_cuda_contracts','--parallel','1'])
        run('static-install', ['cmake','--install',str(root/'source-build'),'--prefix',str(prefix)])
    else:
        for header in ['compute/resource_budget.h', 'compute/cuda_weights.h', 'graph/cpu_cuda.h', 'graph/executable.h']:
            assert (prefix / 'include/hyle' / header).read_bytes() == (repo / 'include/hyle' / header).read_bytes(), header
    for name, source in [('joint','samples/package/graph/cpu_cuda'), ('direct','samples/package/compute/cuda_weights'), ('contracts','tests/graph/cpu_cuda_contracts')]:
        build = delivery/('static-'+name)
        run(name+'-configure', ['cmake','-S',str(repo/source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH='+str(prefix),'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY='+str(delivery/'static-apps')])
        run(name+'-build', ['cmake','--build',str(build),'--parallel','1'])
        run(name+'-test', ['ctest','--test-dir',str(build),'--output-on-failure','-j','1'])
    print('Actual Windows static SDK batch completed.', flush=True)
