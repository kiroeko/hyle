from pathlib import Path
import hashlib
import json
import os
import statistics
import subprocess

work = Path('/root/hyle/.scratch/perf-io-cqe')
common = Path('/root/hyle-scratch/io-watch-v2-dev/bin')
records = []
for pair in range(5):
    order = ('baseline', 'candidate') if pair % 2 == 0 else ('candidate', 'baseline')
    for label in order:
        directory = work / 'baseline-lib' if label == 'baseline' else common
        env = dict(os.environ, LD_LIBRARY_PATH=f'{directory}:{common}')
        result = subprocess.run(
            ['taskset', '-c', '2', str(work / 'batch'), label, str(work / 'fixture.bin')],
            env=env, capture_output=True, text=True, timeout=60)
        (work / f'pair-{pair}-{label}.jsonl').write_text(result.stdout)
        (work / f'pair-{pair}-{label}.stderr').write_text(result.stderr)
        record = {'pair': pair, 'order': list(order), 'label': label,
                  'exit_code': result.returncode,
                  'rows': [json.loads(line) for line in result.stdout.splitlines()]}
        records.append(record)
        (work / 'all-pairs.json').write_text(json.dumps(records, indent=2))
        if result.returncode:
            raise SystemExit(f'{label} pair {pair} failed: {result.stderr}')

summary = []
for kind, batch in ((row['kind'], row['batch']) for row in records[0]['rows']):
    values = {'baseline': [], 'candidate': []}
    ratios = []
    for pair in range(5):
        for label in values:
            run = next(run for run in records if run['pair'] == pair and run['label'] == label)
            row = next(row for row in run['rows'] if (row['kind'], row['batch']) == (kind, batch))
            values[label].append(row['ns_per_op'])
        ratios.append(values['baseline'][-1] / values['candidate'][-1])
    summary.append({'kind': kind, 'batch': batch,
                    'baseline_median_ns_per_op': statistics.median(values['baseline']),
                    'candidate_median_ns_per_op': statistics.median(values['candidate']),
                    'paired_speedup_ratios': ratios,
                    'paired_speedup_median': statistics.median(ratios)})
(work / 'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
