#!/usr/bin/env python3
"""D5 native batch transport; the canonical runner owns every verdict."""
import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--route', choices=('cpu', 'apple', 'vulkan'), required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    windows = os.name == 'nt'
    if windows:
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        original_popen = subprocess.Popen

        class HiddenPopen(original_popen):
            def __init__(self, *positional, **keywords):
                keywords['creationflags'] = keywords.get('creationflags', 0) | subprocess.CREATE_NO_WINDOW
                super().__init__(*positional, **keywords)

        subprocess.Popen = HiddenPopen
        batch = ROOT / '.scratch/d5-native-p1-env.cmd'
        batch.write_text('@echo off\ncall "C:\\Program Files\\Microsoft Visual Studio\\18\\Enterprise\\Common7\\Tools\\VsDevCmd.bat" -no_logo -arch=x64\nif errorlevel 1 exit /b 1\nset\n')
        environment = subprocess.run(['cmd', '/c', str(batch)], capture_output=True, text=True, check=True)
        for line in environment.stdout.splitlines():
            if '=' in line and not line.startswith('='):
                key, value = line.split('=', 1)
                os.environ[key] = value
        os.environ['VSLANG'] = '1033'
        fixture = ROOT / '.scratch/d5-qwen35-six-profile-fixture'
    else:
        os.environ['PATH'] = (str(Path.home() / 'hyle-verify-tools/metal-17F109/shim') +
                             ':/opt/homebrew/bin:' + os.environ['PATH'])
        fixture = Path.home() / 'hyle-scratch/d5-qwen35-six-profile-fixture'
    os.environ['HYLE_D5_QWEN35_FIXTURE'] = str(fixture)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from hyle_perf.runner import run_contract
    result = run_contract(source=Path(__file__).resolve().parent,
        contract_path=ROOT / ('bench/contracts/graph-' + args.route + '-d5-fragments-p1-v1.json'),
        machine_id='W1-WIN' if windows else 'M1-MAC', peer_paths={'mature-direct': args.reference},
        build=args.build, output=args.output, mode='qualify-platform')
    print({key: result[key] for key in ('manifest_path', 'qualification_gate', 'performance_gate')}, flush=True)
    return 0 if result['qualification_gate'] == result['performance_gate'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
