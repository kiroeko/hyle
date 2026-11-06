"""CUDA defaults/override checks using supplied local prepared BF16 2B assets.

No implicit weights, quantizer or tokenizer download. Performance is measured
separately; these runs check the actual CLI, metadata refusal and retirement.
"""
from pathlib import Path
import argparse,json,shlex,subprocess


def check(args):
    args.output.mkdir(parents=True)
    (args.output/'prompt.ids').write_text('1 2 3\n')
    schedule=args.output/'schedule.txt';schedule.write_text('prompt prompt.ids\ngenerate 17\n')
    original_text=(args.inputs/'weights.tsv').read_text(encoding='utf-8')
    original_map=(args.inputs/'mtp-draft-vocabulary-k3.ids').read_text(encoding='utf-8')
    def reference(path):
        head=path.read_text(encoding='utf-8').splitlines()
        payload,count=shlex.split(head[5]);payload=Path(payload)
        if not payload.is_absolute():payload=path.parent/payload
        head[5]='"'+payload.resolve().as_posix()+'" '+count
        return head
    head=reference(args.inputs/'mtp-draft-head.tsv')
    for name in ('legacy','missing-head','stale','wrong-rows','small-explicit'):
        folder=args.output/name;folder.mkdir();(folder/'weights.tsv').write_text(original_text,encoding='utf-8')
        selected=reference(args.full_head) if name in ('legacy','small-explicit') else head
        if name!='missing-head':(folder/'mtp-draft-head.tsv').write_text('\n'.join(selected)+'\n',encoding='utf-8')
        if name!='legacy':
            text=original_map
            if name=='stale':text=text.replace(text.splitlines()[1],'0'*64,1)
            if name in ('wrong-rows','small-explicit'):text='\n'.join(text.splitlines()[:3])+'\n0 1 2\n'
            (folder/'mtp-draft-vocabulary-k3.ids').write_text(text,encoding='utf-8')

    def run(name,extra=(),inputs=None,mtp=True,width=3,subset=True,error=None,rows=114688):
        folder=args.output/name;folder.mkdir()
        command=[str(args.hyle),'--inputs',str(inputs or args.inputs),'--backend','cuda','--capacity','2048',
                 '--schedule',str(schedule),*extra]
        (folder/'command.txt').write_text(subprocess.list2cmdline(command)+'\n',encoding='utf-8')
        with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
            result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=180)
        stderr=(folder/'stderr.log').read_text(encoding='utf-8')
        if error:
            assert result.returncode and error in stderr,(name,stderr)
            assert 'startup_to_ready_seconds=' not in stderr,'invalid defaults reached model readiness'
            return
        assert result.returncode==0,(name,stderr)
        raw=json.loads((folder/'stdout.log').read_text(encoding='utf-8'))
        assert raw['mtp']==mtp and raw['mtp_max_drafts']==(width if mtp else 0),name
        assert raw['draft_vocabulary_rows']==(rows if subset else 0),name
        assert raw['draft_head_rows']==(rows if subset else 248320 if mtp else 0),name
        assert raw['live_committed_versions']==0,name
        if raw['sampling']=='greedy':assert raw['future_shape_hints']==0,name
        ledger=raw['sessions'][0]['ledger'];generations=[r for r in ledger if r['generation']]
        assert sum(r['accepted'] for r in generations)==16,name
        assert raw['actual_target_rows']==raw['prefill_target_rows']+sum(r['verified_rows'] for r in generations),name
        if mtp:assert raw['actual_draft_rows']==raw['prefill_draft_rows']+sum(r['draft_rows'] for r in generations),name
        if raw['sampling']=='greedy':assert all(r['sampling_draw_before']==r['sampling_draw_after']==0 for r in ledger),name
        ids=raw['sessions'][0]['result']['generated_ids'];assert len(ids)==17,name
        print('PASS',name,'mtp',mtp,'K',raw['mtp_max_drafts'],'rows',raw['draft_vocabulary_rows'],flush=True)
        return ids

    automatic=run('default-auto')
    assert automatic==run('explicit-auto',['--mtp','auto'])
    assert automatic==run('enabled-default',['--mtp','on'])
    assert automatic==run('explicit-k3',['--mtp','on','--mtp-max-drafts','3'])
    assert automatic==run('k3-without-overlap',['--mtp','on','--mtp-max-drafts','3','--decode-overlap','off'])
    one=run('override-k1',['--mtp-max-drafts','1'],width=1)
    assert one==run('explicit-k1',['--mtp','on','--mtp-max-drafts','1'],width=1)
    ordinary=run('disable',['--mtp','off'],mtp=False,subset=False)
    assert ordinary==run('outside-default-context',['--capacity','65536'],mtp=False,subset=False)
    assert ordinary==run('legacy-default',inputs=args.output/'legacy',mtp=False,subset=False)
    run('legacy-enabled',['--mtp','on'],inputs=args.output/'legacy',width=1,subset=False)
    small=args.output/'small-explicit'
    small_one=run('small-explicit-on',['--mtp','on'],inputs=small,width=1,rows=3)
    assert small_one==run('small-explicit-k1',['--mtp','on','--mtp-max-drafts','1'],inputs=small,width=1,rows=3)
    small_three=run('small-explicit-k3',['--mtp','on','--mtp-max-drafts','3'],inputs=small,rows=3)
    assert small_three==run('small-manual-k3',['--mtp','on','--mtp-max-drafts','3',
        '--mtp-draft-head',str(small/'mtp-draft-head.tsv'),
        '--mtp-draft-vocabulary',str(small/'mtp-draft-vocabulary-k3.ids')],inputs=small,rows=3)
    run('small-auto-still-refused',inputs=small,error='qualified BF16 Qwen3.5-2B')
    row_error='bound draft vocabulary' if 'SUBSET' in head[0] else 'qualified BF16 Qwen3.5-2B'
    for fixture,message in [('missing-head','requires its complete source-bound'),('stale','source identity differs'),
                            ('wrong-rows',row_error)]:
        run('reject-'+fixture,inputs=args.output/fixture,error=message)
    assert ordinary==run('ignore-stale-disabled',['--mtp','off'],inputs=args.output/'stale',mtp=False,subset=False)
    random=['--sampling','random','--temperature','0.8','--top-k','40','--top-p','0.95','--seed','7']
    assert run('random-auto',random,mtp=False,subset=False)==run('random-off',[*random,'--mtp','off'],mtp=False,subset=False)
    print('PASS automatic K3,explicit source-bound subsets,unchanged legacy and random defaults,stale/incomplete/row refusal,work/RNG/retirement',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for option in ('hyle','inputs','output','full-head'):p.add_argument('--'+option,type=Path,required=True)
    args=p.parse_args()
    for option in ('hyle','inputs','output','full_head'):setattr(args,option,getattr(args,option).resolve())
    check(args)
