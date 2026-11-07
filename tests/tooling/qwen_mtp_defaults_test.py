"""CUDA defaults/override checks using supplied local prepared BF16 2B assets.

No implicit weights, quantizer or tokenizer download. Performance is measured
separately; these runs check the actual CLI, metadata refusal and retirement.
"""
from pathlib import Path
import argparse,json,re,shlex,subprocess


def check(args):
    args.output.mkdir(parents=True)
    (args.output/'prompt.ids').write_text('1 2 3\n')
    schedule=args.output/'schedule.txt';schedule.write_text('prompt prompt.ids\ngenerate 17\n')
    original_text=(args.inputs/'weights.tsv').read_text(encoding='utf-8')
    original_map=(args.inputs/'mtp-draft-vocabulary.ids').read_text(encoding='utf-8')
    def reference(path):
        head=path.read_text(encoding='utf-8').splitlines()
        payload,count=shlex.split(head[5]);payload=Path(payload)
        if not payload.is_absolute():payload=path.parent/payload
        head[5]='"'+payload.resolve().as_posix()+'" '+count
        return head
    head=reference(args.inputs/'mtp-draft-head.tsv')
    for name in ('legacy','missing-head','stale','wrong-rows','small-explicit','generic','generic-head','generic-stale'):
        folder=args.output/name;folder.mkdir();(folder/'weights.tsv').write_text(original_text,encoding='utf-8')
        selected=reference(args.full_head) if name in ('legacy','small-explicit') else head
        if name not in ('missing-head','generic','generic-stale'):
            (folder/'mtp-draft-head.tsv').write_text('\n'.join(selected)+'\n',encoding='utf-8')
        if name not in ('legacy','generic','generic-head','generic-stale'):
            text=original_map
            if name=='stale':text=text.replace(text.splitlines()[1],'0'*64,1)
            if name in ('wrong-rows','small-explicit'):text='\n'.join(text.splitlines()[:5])+'\n0 1 2\n'
            (folder/'mtp-draft-vocabulary.ids').write_text(text,encoding='utf-8')
        if name in ('generic','generic-head','generic-stale'):
            text='\n'.join(original_map.splitlines()[:3])+'\nrecommended_drafts none\nhead original\n0 1 2\n'
            if name=='generic-stale':text=text.replace(text.splitlines()[1],'0'*64,1)
            (folder/'mtp-draft-vocabulary.ids').write_text(text,encoding='utf-8')

    def run(name,extra=(),inputs=None,mtp=True,width=3,subset=True,error=None,rows=114688,head_rows=None):
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
        reasons={'default-auto':('prepared_vocabulary','artifact_recommendation'),
                 'enabled-default':('explicit_on','artifact_recommendation'),
                 'enabled-commands':('explicit_on','artifact_recommendation'),
                 'explicit-k3':('explicit_on','explicit'),
                 'auto-explicit-k4':('prepared_vocabulary','explicit'),
                 'explicit-k4':('explicit_on','explicit'),
                 'auto-k4-commands':('prepared_vocabulary','explicit'),
                 'explicit-k4-commands':('explicit_on','explicit'),
                 'override-k1':('prepared_vocabulary','explicit'),
                 'disable':('explicit_off','disabled'),
                 'auto-commands':('prepared_vocabulary','artifact_recommendation'),
                 'legacy-default':('no_eligible_source','disabled'),
                 'missing-head-auto':('prepared_vocabulary','artifact_recommendation'),
                 'small-auto-manual-k3':('explicit_draft_asset','explicit'),
                 'small-auto-vocabulary-file':('explicit_draft_asset','artifact_recommendation'),
                 'small-auto-head-file':('explicit_draft_asset','artifact_recommendation'),
                 'full-auto-original':('explicit_draft_asset','configured_default'),
                 'full-auto-head':('explicit_draft_asset','configured_default'),
                 'generic-auto':('prepared_vocabulary','configured_default'),
                 'random-auto':('sampling_or_draft_threshold','disabled')}
        if name in reasons:
            assert (raw['mtp_selection_reason'],raw['mtp_depth_source'])==reasons[name],name
        assert raw['mtp']==mtp and raw['mtp_max_drafts']==(width if mtp else 0),name
        assert raw['draft_vocabulary_rows']==(rows if subset else 0),name
        expected_head_rows = (rows if subset else 248320 if mtp else 0) if head_rows is None else head_rows
        assert raw['draft_head_rows']==expected_head_rows,name
        assert raw['live_committed_versions']==0,name
        planned='--warmup-shapes' in extra and extra[extra.index('--warmup-shapes')+1]=='planned'
        if raw['sampling']=='greedy':
            assert (raw['future_shape_hints']>0 if planned else raw['future_shape_hints']==0),name
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
    serial=run('k3-without-overlap',['--mtp','on','--mtp-max-drafts','3','--decode-overlap','off'])
    assert automatic==serial
    assert serial==run('auto-without-overlap',['--decode-overlap','off'])
    captured=run('explicit-prefer-capture',['--mtp','on','--mtp-max-drafts','3','--replay','prefer-capture'])
    assert captured==run('auto-prefer-capture',['--replay','prefer-capture'])
    planned=run('explicit-planned',['--mtp','on','--mtp-max-drafts','3','--warmup-shapes','planned'])
    assert planned==run('auto-planned',['--warmup-shapes','planned'])
    one=run('override-k1',['--mtp-max-drafts','1'],width=1)
    assert one==run('explicit-k1',['--mtp','on','--mtp-max-drafts','1'],width=1)
    ordinary=run('disable',['--mtp','off'],mtp=False,subset=False)
    commands_auto=run('auto-commands',['--replay','commands'])
    assert commands_auto==run('enabled-commands',['--mtp','on','--replay','commands'])
    assert commands_auto==run('explicit-commands',['--mtp','on','--mtp-max-drafts','3','--replay','commands'])
    assert run('auto-explicit-k4',['--mtp-max-drafts','4'],width=4)==run(
        'explicit-k4',['--mtp','on','--mtp-max-drafts','4'],width=4)
    assert run('auto-k4-commands',['--mtp-max-drafts','4','--replay','commands'],width=4)==run(
        'explicit-k4-commands',['--mtp','on','--mtp-max-drafts','4','--replay','commands'],width=4)
    expanded=run('explicit-expanded-context',['--capacity','65536','--mtp','on','--mtp-max-drafts','3'])
    assert expanded==run('auto-expanded-context',['--capacity','65536'])
    assert ordinary==run('legacy-default',inputs=args.output/'legacy',mtp=False,subset=False)
    run('legacy-enabled',['--mtp','on'],inputs=args.output/'legacy',width=1,subset=False)
    small=args.output/'small-explicit'
    small_one=run('small-explicit-k1',['--mtp','on','--mtp-max-drafts','1'],inputs=small,width=1,rows=3)
    small_three=run('small-explicit-on',['--mtp','on'],inputs=small,rows=3)
    assert small_three==run('small-explicit-k3',['--mtp','on','--mtp-max-drafts','3'],inputs=small,rows=3)
    assert small_three==run('small-manual-k3',['--mtp','on','--mtp-max-drafts','3',
        '--mtp-draft-head',str(small/'mtp-draft-head.tsv'),
            '--mtp-draft-vocabulary',str(small/'mtp-draft-vocabulary.ids')],inputs=small,rows=3)
    assert small_three==run('small-auto-manual-k3',['--mtp-max-drafts','3',
        '--mtp-draft-head',str(small/'mtp-draft-head.tsv'),
        '--mtp-draft-vocabulary',str(small/'mtp-draft-vocabulary.ids')],inputs=small,rows=3)
    assert small_three==run('small-auto-vocabulary-file',
        ['--mtp-draft-vocabulary',str(small/'mtp-draft-vocabulary.ids')],inputs=small,rows=3)
    assert small_three==run('small-auto-head-file',
        ['--mtp-draft-head',str(small/'mtp-draft-head.tsv')],inputs=small,rows=3)
    assert small_three==run('small-auto-k3',inputs=small,rows=3)
    assert small_one==run('small-auto-explicit-k1',['--mtp-max-drafts','1'],inputs=small,width=1,rows=3)
    generic=args.output/'generic'
    generic_auto=run('generic-auto',inputs=generic,width=1,rows=3)
    assert generic_auto==run('generic-explicit',['--mtp','on','--mtp-max-drafts','1'],inputs=generic,width=1,rows=3)
    assert generic_auto==run('generic-head-original',inputs=args.output/'generic-head',width=1,rows=3)
    assert ordinary==run('generic-off',['--mtp','off'],inputs=generic,mtp=False,subset=False)
    run('reject-generic-stale',inputs=args.output/'generic-stale',error='source identity differs')
    run('reject-explicit-vocabulary-stale',
        ['--mtp-draft-vocabulary',str(args.output/'generic-stale'/'mtp-draft-vocabulary.ids')],
        error='source identity differs')
    if 'SUBSET' in head[0]:
        run('reject-wrong-rows',inputs=args.output/'wrong-rows',error='bound draft vocabulary')
    else:
        assert small_three==run('wrong-rows-auto-k3',inputs=args.output/'wrong-rows',rows=3)
    missing=args.output/'missing-head'
    original_auto=run('missing-head-auto',inputs=missing)
    assert original_auto==run('missing-head-explicit',
        ['--mtp','on','--mtp-max-drafts','3','--mtp-draft-head','original',
         '--mtp-draft-vocabulary',str(missing/'mtp-draft-vocabulary.ids')],inputs=missing)
    assert original_auto==run('missing-head-auto-original',
        ['--mtp-draft-head','original'],inputs=missing)
    run('reject-stale',inputs=args.output/'stale',error='source identity differs')
    assert ordinary==run('ignore-stale-disabled',['--mtp','off'],inputs=args.output/'stale',mtp=False,subset=False)
    assert ordinary==run('ignore-stale-zero',['--mtp-max-drafts','0'],inputs=args.output/'stale',mtp=False,subset=False)
    unused=['--mtp-draft-head',str(args.output/'missing-head.tsv'),
            '--mtp-draft-vocabulary',str(args.output/'missing-vocabulary.ids')]
    run('off-rejects-explicit',['--mtp','off',*unused],error='explicit MTP asset cannot be used: explicit_off')
    run('zero-rejects-explicit',['--mtp-max-drafts','0',*unused],error='explicit MTP asset cannot be used: drafting_disabled')
    run('random-rejects-explicit',['--sampling','random','--mtp-draft-vocabulary',str(small/'mtp-draft-vocabulary.ids')],
        error='explicit MTP asset cannot be used: sampling_or_draft_threshold')
    run('random-rejects-explicit-head',['--sampling','random','--mtp-draft-head',str(small/'mtp-draft-head.tsv')],
        error='explicit MTP asset cannot be used: sampling_or_draft_threshold')
    run('reject-auto-missing-vocabulary', ['--mtp-draft-vocabulary',unused[3]], error='draft vocabulary file does not exist')
    run('reject-auto-missing-head', ['--mtp-draft-head',unused[1]], error='draft head requires a bounded regular metadata file')
    full=run('full-explicit-original',['--mtp','on','--mtp-max-drafts','1',
        '--mtp-draft-head','original','--mtp-draft-vocabulary','full'],width=1,subset=False,head_rows=0)
    assert full==run('full-auto-original',['--mtp-draft-head','original',
        '--mtp-draft-vocabulary','full'],width=1,subset=False,head_rows=0)
    if 'SUBSET' in head[0]:
        assert full==run('full-auto-head',['--mtp-draft-vocabulary','full'],width=1,subset=False,head_rows=0)
        run('reject-explicit-compact-full', ['--mtp-draft-vocabulary','full',
            '--mtp-draft-head',str(args.inputs/'mtp-draft-head.tsv')], error='bound draft vocabulary')
    random=['--sampling','random','--temperature','0.8','--top-k','40','--top-p','0.95','--seed','7']
    assert run('random-auto',random,mtp=False,subset=False)==run('random-off',[*random,'--mtp','off'],mtp=False,subset=False)
    # Distinct explicitly planned prefill lengths exercise the aggregate cold
    # family search budget, not only the single-shape smoke case above.
    many=args.output/'planned-many';many.mkdir();lines=[]
    for index in range(12):
        prompt=many/f'prompt{index}.ids';prompt.write_text(' '.join(['1']*(index+2))+'\n')
        lines.extend((f'prompt {prompt.name}','generate 2'))
    many_schedule=many/'schedule.txt';many_schedule.write_text('\n'.join(lines)+'\n')
    observed=[]
    for name,extra in (('auto',[]),('explicit',['--mtp','on','--mtp-max-drafts','3'])):
        folder=many/name;folder.mkdir()
        command=[str(args.hyle),'--inputs',str(args.inputs),'--backend','cuda','--capacity','2048',
                 '--schedule',str(many_schedule),'--warmup-shapes','planned',*extra]
        (folder/'command.txt').write_text(subprocess.list2cmdline(command)+'\n',encoding='utf-8')
        with (folder/'stdout.log').open('wb') as stdout,(folder/'stderr.log').open('wb') as stderr:
            result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=300)
        error=(folder/'stderr.log').read_text(encoding='utf-8')
        assert result.returncode==0,(name,error)
        raw=json.loads((folder/'stdout.log').read_text(encoding='utf-8'))
        assert raw['mtp'] and raw['mtp_max_drafts']==3 and raw['future_shape_hints']>0,name
        assert raw['live_committed_versions']==0 and len(raw['sessions'][0]['result']['generated_ids'])==24,name
        assert max(map(int,re.findall(r'compiled text finite family variants=(\d+)',error)),default=0)>=11,name
        observed.append((raw['sessions'][0]['result']['generated_ids'],raw['sessions'][0]['ledger']))
    assert observed[0]==observed[1],'planned many-shape auto differs from explicit K3'
    print('PASS planned many-shape auto/explicit K3, outputs/ledger and retirement',flush=True)
    print('PASS automatic K3 with prepared or original head,source-bound subsets,unchanged legacy and random defaults,stale/row refusal,work/RNG/retirement',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for option in ('hyle','inputs','output','full-head'):p.add_argument('--'+option,type=Path,required=True)
    args=p.parse_args()
    for option in ('hyle','inputs','output','full_head'):setattr(args,option,getattr(args,option).resolve())
    check(args)
