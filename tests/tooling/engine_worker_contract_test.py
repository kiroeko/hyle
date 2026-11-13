"""Public native application/IPC contract; deliberately no model or device."""
from pathlib import Path
import argparse
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from engine.text.worker_client import WorkerClient


def check(engine: Path) -> None:
    scratch = ROOT / '.scratch'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='worker-host-', dir=scratch) as temporary:
        source = Path(temporary) / '输入 图'
        source.mkdir()
        (source / 'host-contract.txt').write_text('host fixture\n', encoding='utf-8')
        prompt = source / 'prompt λ.txt'
        prompt.write_text('7\n', encoding='utf-8')
        common = ['--inputs', str(source), '--capacity', '128', '--max-sessions', '1',
                  '--active-sequences', '1', '--draft-count', '0']
        result = subprocess.run([str(engine), 'generate', *common, '--tokens', str(prompt),
                                 '--max-output-tokens', '3'], capture_output=True, text=True,
                                encoding='utf-8', timeout=15, check=True)
        assert result.stdout == '7 8 9\n', result
        client = WorkerClient([engine, '--engine-worker', *common, '--profile', '图片',
                               '--draft-count', '0'], startup_timeout=15)
        try:
            assert tuple(profile.name for profile in client.execution_profiles) == ('default', '图片')
            assert all(profile.algorithm == 'host-fixture' and profile.max_draft_tokens == 0
                       and profile.input_modalities == ('text',) for profile in client.execution_profiles)
            for profile in (0, 1):
                request = client.submit([7], 3, execution_profile=profile)
                tokens = []
                while True:
                    event = request.read(timeout=10)
                    if event.kind == 'token':
                        tokens.append(event.token)
                        request.acknowledge()
                    else:
                        assert event.kind == 'completed', event
                        break
                assert tokens == [7 + profile * 10, 8 + profile * 10, 9 + profile * 10], tokens
        finally:
            client.close(timeout=10)
    print('PASS public native CLI/IPC: Unicode paths, profile identity, prepared facts, selection and retirement; host only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    check(parser.parse_args().engine.resolve())
