from pathlib import Path
import ctypes
import os
import subprocess

ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
os.environ['HYLE_D5_QWEN35_FIXTURE'] = str(Path('.scratch/d5-qwen35-six-profile-fixture').resolve())
tree = Path('.scratch/build-compute-v2-dev/bin').resolve()
out = Path('.scratch/validation/d5-kda-full/cpu')
for entry in ('compute', 'graph'):
    log = out / (entry + '-single-state.log')
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run([str(tree / f'hyle_{entry}_route_cpu_test.exe'),
            '--gtest_filter=*QwenRecurrent.D5NineProfilesUseCompleteGateAndActualStateFeedback'],
            cwd=tree, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW)
    contents = log.read_text(encoding='utf-8', errors='replace')
    print(entry, result.returncode, contents[-1800:], flush=True)
    if result.returncode or '[  SKIPPED ]' in contents:
        raise SystemExit(result.returncode or 1)
