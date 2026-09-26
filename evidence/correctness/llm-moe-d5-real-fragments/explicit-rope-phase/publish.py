import gzip
import hashlib
import json
from pathlib import Path
import shutil
import struct

root = Path('C:/public/project-hyle/hyle')
out = root / 'evidence/correctness/llm-moe-d5-real-fragments/explicit-rope-phase'
out.mkdir(parents=True, exist_ok=True)
records = []
for label, folder, names in [
    ('before', 'd5-deepseek-focused-r2', ['receipt.json', 'tests.log', 'tests.xml']),
    ('after', 'd5-deepseek-focused-r3', ['receipt.json', 'build.log', 'tests.log', 'tests.xml',
                                      'native-tests.log', 'cache-identity.txt']),
    ('cpu-owner', 'd5-deepseek-owner-r3', ['receipt.json', 'build.cmd', 'build.log',
                                         'tests.log', 'tests.xml', 'native-tests.log',
                                         'cache-identity.txt']),
]:
    for name in names:
        source = root / '.scratch/validation' / folder / 'cpu' / name
        destination = out / (label + '-' + name + '.gz')
        assert source.is_file()
        with source.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if not destination.exists():
            with source.open('rb') as stream, gzip.open(destination, 'wb', compresslevel=6) as target:
                shutil.copyfileobj(stream, target)
        with gzip.open(destination, 'rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest
        records.append({'file': destination.name, 'source': str(source.relative_to(root)),
                        'uncompressed_sha256': digest, 'bytes': source.stat().st_size})
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
frequency = f32(0.4728708267211914)
phase = []
for position in (2380, 65536):
    direct = f32(f32(position) * frequency)
    reciprocal = f32(f32(position) / f32(1.0 / frequency))
    phase.append({'position': position, 'frequency_f32': frequency,
                  'multiply_f32': direct, 'reciprocal_then_divide_f32': reciprocal,
                  'difference': reciprocal - direct})
receipt = {'candidate_before': '4f18811a45d6cb378388e975986f3b6d111c1565',
           'candidate_after': json.loads((root / '.scratch/validation/d5-deepseek-focused-r3/cpu/receipt.json').read_text())['candidate'],
           'scope': 'CPU focused and complete owner correctness; not a delivery or performance receipt',
           'actual_gtest_cases_passed': 7,
           'cpu_owner': {'ctest_targets_passed': 9, 'gtest_cases_passed': 404,
                         'ggml_provider': 43, 'compute_ops': 57,
                         'direct': 141, 'graph': 156, 'runtime': 7,
                         'other_checks': ['cpu executor', 'runtime source contract',
                                          'cpu allocation faults', 'context allocation faults']},
           'empty_filtered_cpu_executor_is_not_coverage': True,
           'phase_reproduction': phase, 'files': records}
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
loc = json.loads((root / '.scratch/validation/d5-rope-phase-loc.json').read_text(encoding='utf-8-sig'))
(out / 'loc.json').write_text(json.dumps(loc, indent=2) + '\n', encoding='utf-8')
shutil.copyfile(__file__, out / 'publish.py')
print(json.dumps({'phase': phase, 'files': len(records)}))
