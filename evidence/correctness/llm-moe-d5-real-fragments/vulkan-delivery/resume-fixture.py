from pathlib import Path
import hashlib

driver = Path('evidence/correctness/llm-moe-d5-real-fragments/vulkan-delivery/run-batch.py')
exec(compile(driver.read_text().split('# Compile the owner-only')[0], str(driver), 'exec'))
for stage in ('shared', 'source', 'static'):
    command = [sys.executable, str(driver.with_name('run.py')), stage,
               '--attempt', 'fixture-guard']
    if stage == 'shared':
        command += ['--reuse-shared-sdk']
    result = subprocess.run(command, env=environment, **flags)
    if result.returncode:
        raise SystemExit(result.returncode)

focused = root/'receipts/precision-current-driver'
focused.mkdir(exist_ok=False)
runtime = dict(environment)
runtime['PATH'] = os.pathsep.join([os.environ['SystemRoot']+'/System32', os.environ['SystemRoot']])
runtime['HYLE_D5_QWEN35_FIXTURE'] = str(Path('.scratch/d5-qwen35-six-profile-fixture').resolve())
test_filter = '*D5DeepSeekFullAttentionChain:*D5DeepSeekStatefulCompression'
receipt = {'candidate': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, **flags).strip(),
    'environment': json.loads((root/'environment.json').read_text()), 'filter': test_filter, 'steps': []}
for entry in ('direct', 'graph'):
    executable = root/'shared-consumer'/('d5_vulkan_'+entry+'.exe')
    command = [str(executable), '--gtest_filter='+test_filter,
               '--gtest_output=xml:'+str(focused/(entry+'.xml'))]
    started = time.monotonic()
    with (focused/(entry+'.log')).open('wb') as stream:
        result = subprocess.run(command, env=runtime, stdout=stream, stderr=subprocess.STDOUT, **flags)
    receipt['steps'].append({'entry': entry, 'argv': command, 'returncode': result.returncode,
        'seconds': round(time.monotonic()-started, 3),
        'executable_sha256': hashlib.sha256(executable.read_bytes()).hexdigest()})
    (focused/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('focused', entry, result.returncode, flush=True)
    if result.returncode not in (0, 1):
        raise SystemExit(result.returncode)
