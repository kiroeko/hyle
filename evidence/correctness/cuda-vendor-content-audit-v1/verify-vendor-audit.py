from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

source = Path(sys.argv[1]).resolve()
scratch = Path(sys.argv[2]).resolve()
scratch.mkdir(parents=True, exist_ok=True)
lock = json.loads((source / 'cmake/HyleDependencyLock.json').read_text(encoding='utf-8'))
receipt = {'source': str(source), 'candidate': 'cbedb837791da15392019299f2652d9d3d70ec9d', 'checks': []}
audit = subprocess.run(['cmake', '-P', str(source / 'cmake/HyleDependencyVersions.cmake')],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
(scratch / 'all-vendor-audit.log').write_text(audit.stdout, encoding='utf-8')
audit.check_returncode()
for name, factor, header in [('marlin', 'compute_provider_marlin', 'kernel.h'),
                             ('sglang_activation', 'compute_provider_cuda', 'activation.cuh')]:
    vendor = source / 'third_party' / name
    files = [p for p in vendor.rglob('*') if p.is_file()]
    def manifest_hash(paths):
        rows = []
        for p in paths:
            data = p.read_bytes().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
            rows.append(p.relative_to(vendor).as_posix() + '|' + hashlib.sha256(data).hexdigest() + '\n')
        return hashlib.sha256(''.join(rows).encode()).hexdigest()
    canonical = manifest_hash(sorted(files, key=lambda p: p.relative_to(vendor).as_posix()))
    casefolded = manifest_hash(sorted(files, key=lambda p: p.relative_to(vendor).as_posix().lower()))
    expected = next(d for d in lock['dependencies'] if d['name'] == name)['vendored']['tree_sha256']
    assert canonical == expected and canonical != casefolded
    fixture = scratch / (name + '-mutated')
    assert not fixture.exists(), fixture
    copied = fixture / 'third_party' / name
    shutil.copytree(vendor, copied)
    mutated = copied / header
    mutated.write_bytes(mutated.read_bytes() + b'\n// mutation must fail before CUDA configure\n')
    driver = fixture / 'consume.cmake'
    driver.write_text('cmake_minimum_required(VERSION 3.25)\n'
        + 'include("' + (source / 'cmake/HyleDependencyVersions.cmake').as_posix() + '")\n'
        + 'set(HYLE_DEPENDENCY_SOURCE_ROOT "' + fixture.as_posix() + '")\n'
        + 'set(HYLE_COMPUTE_CUDA_ENABLED ON)\n'
        + 'include("' + (source / 'src' / factor / 'CMakeLists.txt').as_posix() + '")\n', encoding='utf-8')
    result = subprocess.run(['cmake', '-P', str(driver)], text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (scratch / (name + '-negative.log')).write_text(result.stdout, encoding='utf-8')
    normalized = ' '.join(result.stdout.split())
    assert result.returncode != 0 and ("Vendored dependency '" + name + "' content does not match the lock.") in normalized, result.stdout
    receipt['checks'].append({'vendor': name, 'canonical_sha256': canonical,
        'old_casefolded_manifest_sha256': casefolded,
        'mutated_consumer_refused_before_cuda_language': True})
receipt['result'] = 'PASS'
(scratch / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
