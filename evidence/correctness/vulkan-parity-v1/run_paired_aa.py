"""One prespecified eight-process original Vulkan workload qualification."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import uuid

from paired_rounds import read
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from hyle_perf.environment import PERF_ENVIRONMENT_BLOCKLIST, PERF_ENVIRONMENT_FIXED, PERF_ENVIRONMENT_PREFIXES
from hyle_perf.machine import evaluate_machine_profile, expected_child_policy, get_machine_profile, load_registry, probe_machine
from hyle_perf.runner import _apply_sampling_policy, _required_telemetry
from hyle_perf.nanobench import parse_nanobench, select_pair

BUILD = Path('C:/hv/d5pf/vulkan')
EXE = BUILD / 'hyle_graph_route_vulkan_d5_bench.exe'
OUT = ROOT / 'evidence/performance/vulkan-parity-paired-aa'
TITLES = ('Qwen3.5-2B official BF16 FFN M1 H2048 N6144',
    'DeepSeek V4 Flash UD-Q4_K_XL hash-MoE H4096 N2048 top6')
PREFIX = '# HYLE_D5_FRAGMENT_RUN '
ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
FLAGS = subprocess.CREATE_NO_WINDOW | subprocess.ABOVE_NORMAL_PRIORITY_CLASS
identity = {'binary': hashlib.sha256(EXE.read_bytes()).hexdigest(),
    'reader': hashlib.sha256(Path(__file__).with_name('paired_rounds.py').read_bytes()).hexdigest(),
    'candidate': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW).strip()}
OUT.mkdir(parents=True, exist_ok=True)
receipt = OUT / 'receipt.json'
if receipt.exists(): raise RuntimeError('this finite qualification budget has already been used')
raw = OUT / 'raw'; raw.mkdir()
profile = get_machine_profile(load_registry(), 'W1-WIN')
probe = probe_machine()
gate = evaluate_machine_profile(profile, probe, role='engineering', sampling_topology='native-runtime-pair')
if gate['gate'] != 'PASS': raise RuntimeError('machine profile: ' + str(gate['errors']))
policy = expected_child_policy(profile, sampling_topology='native-runtime-pair', probe=probe)
runs, environments = [], set()
with _apply_sampling_policy(policy):
    for arm in ('selected', 'peer'):
        for first in (False, True):
            for allocation in (False, True):
                env = {k: v for k, v in os.environ.items() if k not in PERF_ENVIRONMENT_BLOCKLIST
                    and not any(k.startswith(p) for p in PERF_ENVIRONMENT_PREFIXES)}
                env.update(PERF_ENVIRONMENT_FIXED)
                env.update(HYLE_BENCH_AA_ROUTE=arm, HYLE_PERF_RUN_NONCE=uuid.uuid4().hex,
                    HYLE_PERF_CALLER_CPU=str(policy['caller_logical_cpu']),
                    HYLE_PERF_SAMPLING_TOPOLOGY='native-runtime-pair',
                    HYLE_D5_QWEN35_FIXTURE=str(ROOT / '.scratch/d5-qwen35-six-profile-fixture'))
                if first: env['HYLE_BENCH_RIVAL_FIRST'] = '1'
                if allocation: env['HYLE_BENCH_PEER_ALLOCATION_FIRST'] = '1'
                print('START %s %s %s' % (arm, first, allocation), flush=True)
                result = subprocess.run([str(EXE)], env=env, cwd=EXE.parent, capture_output=True,
                    text=True, encoding='utf-8', errors='replace', timeout=180, creationflags=FLAGS)
                text = result.stdout + result.stderr
                log = raw / ('%s-%d-%d.log' % (arm, first, allocation)); log.write_text(text, encoding='utf-8')
                if result.returncode: raise RuntimeError('native A/A child failed: ' + str(log))
                expected = {'execution_order': 'peer-first' if first else 'candidate-first',
                    'allocation_order': 'peer-first' if allocation else 'candidate-first',
                    'aa_route': arm, 'paired_mode': 'nanobench-compare', 'run_nonce': env['HYLE_PERF_RUN_NONCE'],
                    'verify_only': '0', 'affinity_topology': 'native-runtime-pair',
                    'caller_cpu': str(policy['caller_logical_cpu']), 'affinity_ok': '1',
                    'parent_mask': policy['affinity_mask'], 'priority': policy['priority']}
                _required_telemetry(text, [PREFIX], expected)
                facts = [s for s in text.splitlines() if s.startswith('# D5_VULKAN_ENV ')]
                if len(facts) != 4 or len(set(facts)) != 1: raise RuntimeError('native environment identity differs')
                environments.update(facts)
                paired = read(text, TITLES)
                original = parse_nanobench(text)
                diagnostic = {}
                for title in TITLES:
                    c, p = select_pair(original, 'Graph complete fragment', 'Direct complete fragment', title=title)
                    diagnostic[title] = c.operations_per_second / p.operations_per_second
                runs.append({'arm': arm, 'first': first, 'allocation': allocation,
                    'paired': paired, 'marginal_diagnostic': diagnostic, 'log': str(log),
                    'sha256': hashlib.sha256(log.read_bytes()).hexdigest()})
                print('END %d/8 %s' % (len(runs), {t: paired[t]['paired_process_ratio'] for t in TITLES}), flush=True)
errors, summaries = [], {}
if len(environments) != 1: errors.append('native environment changed within the batch')
for arm in ('selected', 'peer'):
    group = [r for r in runs if r['arm'] == arm]
    for title in TITLES:
        values = [r['paired'][title]['paired_process_ratio'] for r in group]
        medians = {'all': statistics.median(values)}
        for factor in ('first', 'allocation'):
            for level in (False, True):
                medians[f'{factor}={level}'] = statistics.median(r['paired'][title]['paired_process_ratio']
                    for r in group if r[factor] == level)
        if any(not 1 / 1.10 <= v <= 1.10 for v in values): errors.append(arm + ':' + title + ': gross >10%')
        if any(not 1 / 1.05 <= v <= 1.05 for v in medians.values()): errors.append(arm + ':' + title + ': factor >5%')
        summaries[arm + ':' + title] = {'ratios': values, 'medians': medians}
record = {'schema': 'vulkan-parity-paired-aa/v1', 'identity': identity, 'machine': gate,
    'native_environments': sorted(environments), 'runs': runs, 'summaries': summaries,
    'gate': 'FAIL' if errors else 'PASS', 'errors': errors,
    'limits': 'A/A only; no retrospective old PASS, product WIN, or formal n15 qualification'}
receipt.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'gate': record['gate'], 'errors': errors}), flush=True)
raise SystemExit(1 if errors else 0)
