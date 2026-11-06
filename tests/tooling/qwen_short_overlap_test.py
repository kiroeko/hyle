"""Compare short-context overlap with the original finite request route."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hyle', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-only', action='store_true')
    parser.add_argument('--case', help='run one capacity:mode pair before the continuation checks')
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    inputs = args.output/'inputs';inputs.mkdir()
    for name in ('weights.tsv','mtp-draft-head.tsv','mtp-draft-vocabulary-k3.ids','draft-head-q4_k.bin'):
        if (args.inputs/name).is_file():os.link(args.inputs/name,inputs/name)
    if not (inputs/'weights.tsv').is_file():raise ValueError('prepared weights are required')
    (inputs/'prompt.ids').write_text('1\n',encoding='utf-8')
    cases=((2,'off'),(4,'off'),(9,'off'),(16,'off'),(128,'off'),(512,'off'),(513,'off'),
           (2,'K1'),(4,'K1'),(9,'K1'),(4,'K3'),(9,'K3'))
    selected=() if args.chunk_only else cases
    if args.case:
        selected=tuple(case for case in cases if f'{case[0]}:{case[1]}'==args.case)
        if not selected:raise ValueError('unknown short overlap case')
    def published(ledger):
        # Device-chain roles and auxiliary work may differ; published state,
        # token, quota and sampling records must still match exactly.
        return [{k:v for k,v in entry.items() if k not in ('role','auxiliary_graph_calls','draft_rows')}
                for entry in ledger]
    for capacity,mode in selected:
        flags=['--mtp','off'] if mode=='off' else ['--mtp','on','--mtp-max-drafts',mode[1:]]
        reports={}
        for arm,exe,extra in (('baseline',args.baseline,('--decode-overlap','off')),('candidate',args.hyle,()),
                              ('candidate-serial',args.hyle,('--decode-overlap','off'))):
            folder=args.output/f'{capacity}-{mode}-{arm}';folder.mkdir()
            command=[str(exe),'--inputs',str(inputs),'--backend','cuda','--capacity',str(capacity),
                     '--steps',str(min(7,capacity-1)),*flags,*extra]
            result=subprocess.run(command,capture_output=True,timeout=180)
            (folder/'stdout.log').write_bytes(result.stdout);(folder/'stderr.log').write_bytes(result.stderr)
            if result.returncode:
                raise RuntimeError(f'{capacity}/{mode}/{arm} failed; see {folder}')
            raw=json.loads(result.stdout.decode('utf-8-sig'))
            assert raw['live_committed_versions']==0,(capacity,mode,arm)
            reports[arm]=raw
        old=reports['baseline']['sessions'][0]
        for arm in ('candidate','candidate-serial'):
            observed=reports[arm]['sessions'][0]
            assert observed['result']['generated_ids']==old['result']['generated_ids'],(capacity,mode,arm)
            assert published(observed['ledger'])==published(old['ledger']),(capacity,mode,arm,'published ledger')
        print('PASS',capacity,mode,'baseline/overlap/serial',
              'enabled',reports['candidate']['decode_overlap'],
              'successors',reports['candidate']['overlap_successor_submissions'],flush=True)
    chunked=args.output/'chunked-input';chunked.mkdir()
    for name in ('weights.tsv','mtp-draft-head.tsv','mtp-draft-vocabulary-k3.ids','draft-head-q4_k.bin'):
        if (inputs/name).is_file():os.link(inputs/name,chunked/name)
    (chunked/'prompt.ids').write_text('1 2 3\n',encoding='utf-8')
    reports={}
    for arm,exe,extra in (('baseline',args.baseline,('--decode-overlap','off')),('candidate',args.hyle,())):
        folder=args.output/f'chunked-{arm}';folder.mkdir()
        command=[str(exe),'--inputs',str(chunked),'--backend','cuda','--capacity','4','--steps','1',
                 '--prefill-chunk-tokens','2','--mtp','off',*extra]
        result=subprocess.run(command,capture_output=True,timeout=180)
        (folder/'stdout.log').write_bytes(result.stdout);(folder/'stderr.log').write_bytes(result.stderr)
        if result.returncode:raise RuntimeError(f'chunked/{arm} failed; see {folder}')
        reports[arm]=json.loads(result.stdout.decode('utf-8-sig'))
    assert reports['candidate']['sessions'][0]['result']['generated_ids']==reports['baseline']['sessions'][0]['result']['generated_ids']
    assert reports['candidate']['sessions'][0]['ledger']==reports['baseline']['sessions'][0]['ledger']
    assert reports['candidate']['prefill_target_rows']==3 and reports['candidate']['live_committed_versions']==0
    print('PASS explicit short prefill chunk and original state-only path',flush=True)
    (inputs/'empty.ids').write_text('',encoding='utf-8')
    schedule=inputs/'continuation.txt'
    schedule.write_text('prompt prompt.ids\ngenerate 4\nprompt empty.ids\ngenerate 4\n',encoding='utf-8')
    reports={}
    for arm,exe,extra in (('baseline',args.baseline,('--decode-overlap','off')),('candidate',args.hyle,())):
        folder=args.output/f'continuation-{arm}';folder.mkdir()
        command=[str(exe),'--inputs',str(inputs),'--backend','cuda','--capacity','9','--schedule',str(schedule),
                 '--mtp','on','--mtp-max-drafts','3',*extra]
        result=subprocess.run(command,capture_output=True,timeout=180)
        (folder/'stdout.log').write_bytes(result.stdout);(folder/'stderr.log').write_bytes(result.stderr)
        if result.returncode:raise RuntimeError(f'continuation/{arm} failed; see {folder}')
        reports[arm]=json.loads(result.stdout.decode('utf-8-sig'))
    before=reports['baseline']['sessions'][0];after=reports['candidate']['sessions'][0]
    assert after['result']['generated_ids']==before['result']['generated_ids']
    assert published(after['ledger'])==published(before['ledger'])
    assert reports['candidate']['turn_count']==2 and reports['candidate']['live_committed_versions']==0
    print('PASS short K3 cross-turn pending state and retirement',flush=True)


if __name__=='__main__':main()
