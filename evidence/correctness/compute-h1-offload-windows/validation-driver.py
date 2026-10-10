import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
repo = Path('C:/public/project-hyle/hyle')
root = Path('C:/h7/w-cpu-cuda/selected-graph')
mode = sys.argv[1] if len(sys.argv) > 1 else 'source'
assert mode in ['source', 'shared', 'static-configure', 'static']
record = Path('C:/h7/h1-offload-windows' + ('' if mode == 'source' else '-' + mode))
candidate = '7eaf67b8a496ee19b609533980cd69e7efb5aa49'
assert (Path('C:/h7') / '.hyle-owner').is_file()
assert root.resolve().is_relative_to(Path('C:/h7').resolve())
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip()
for name in subprocess.check_output(['git', 'diff', '--name-only', candidate, 'HEAD'], cwd=repo, text=True).splitlines():
    assert name.startswith('docs/'), name

class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
        (n, ctypes.c_ulonglong) for n in ['total', 'available', 'page_total', 'page_available',
                                        'virtual_total', 'virtual_available', 'extended']]

def available():
    m = Memory()
    m.length = ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.available / (1024 ** 3)

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

def run(stage, args):
    free = available()
    print(f'{stage}: free={free:.2f} GiB; log={record / (stage + ".log")}', flush=True)
    if free < 12:
        raise RuntimeError('memory guard before stage; no build started')
    log = record / (stage + '.log')
    started = time.monotonic()
    minimum = free
    with log.open('wb') as out:
        p = subprocess.Popen(args, cwd=repo, stdout=out, stderr=subprocess.STDOUT, **flags)
        while p.poll() is None:
            minimum = min(minimum, available())
            if minimum < 12:
                # This PID is the still-running child started above. Terminate
                # only that owned process tree and preserve its objects/logs.
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **flags)
                p.wait()
                raise RuntimeError(f'memory guard stopped owned stage {stage}')
            time.sleep(1)
    row = {'stage': stage, 'exit': p.returncode, 'seconds': round(time.monotonic() - started, 2),
           'minimum_available_gib': round(minimum, 2), 'log': str(log)}
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
else:
    delivery = Path('C:/h7/h1-windows-delivery')
    prefix = delivery/'static-sdk'
    run('static-build', ['cmake','--build',str(root/'source-build'),'--target','hyle_cpu_cuda_sample','hyle_cuda_weights_sample','hyle_cpu_cuda_contracts','--parallel','1'])
    run('static-install', ['cmake','--install',str(root/'source-build'),'--prefix',str(prefix)])
    for name, source in [('joint','samples/package/graph/cpu_cuda'), ('direct','samples/package/compute/cuda_weights'), ('contracts','tests/graph/cpu_cuda_contracts')]:
        build = delivery/('static-'+name)
        run(name+'-configure', ['cmake','-S',str(repo/source),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_PREFIX_PATH='+str(prefix),'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY='+str(delivery/'static-apps')])
        run(name+'-build', ['cmake','--build',str(build),'--parallel','1'])
        run(name+'-test', ['ctest','--test-dir',str(build),'--output-on-failure','-j','1'])
    print('Actual Windows static SDK batch completed.', flush=True)
