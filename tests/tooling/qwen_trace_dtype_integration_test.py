"""Check local Qwen diagnostic file suffixes against activation storage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--prompt-ids', type=Path, required=True)
    parser.add_argument('--activation', choices=('f32', 'f16', 'bf16'), required=True)
    parser.add_argument('--hidden-width', type=int, required=True)
    parser.add_argument('--probe-inputs', type=Path, help='prepared inputs with prompt.ids for the layer-zero probe')
    parser.add_argument('--output', type=Path, required=True, help='new local diagnostic directory')
    args = parser.parse_args()
    if args.hidden_width <= 0:
        raise ValueError('hidden width must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'prompt.ids').write_text(args.prompt_ids.read_text(encoding='ascii'), encoding='ascii')
    (output / 'schedule.txt').write_text('prompt prompt.ids\ngenerate 1\n', encoding='ascii')
    command = [str(args.engine.resolve()), '--inputs', str(args.inputs.resolve()), '--backend', 'cuda',
               '--capacity', '1024', '--schedule', str(output / 'schedule.txt'), '--mtp', 'off',
               '--replay', 'commands', '--trace', str(output / 'trace'), '--trace-level', 'full']
    with (output / 'stderr.txt').open('wb') as stderr:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=stderr, check=True, timeout=120)
    report = json.loads(completed.stdout)
    if len(report['sessions'][0]['result']['generated_ids']) != 1:
        raise AssertionError('diagnostic request did not generate exactly one token')
    prefill = output / 'trace' / 'session0_default' / 'prefill'
    hidden = prefill / f'hidden.{args.activation}'
    item_bytes = 4 if args.activation == 'f32' else 2
    if hidden.stat().st_size != args.hidden_width * item_bytes:
        raise AssertionError('hidden diagnostic suffix disagrees with its byte extent')
    histories = list(prefill.glob('layer*.conv.*'))
    if not histories or any(path.suffix != f'.{args.activation}' or path.stat().st_size % item_bytes for path in histories):
        raise AssertionError('convolution history suffix disagrees with its activation storage')
    if not (prefill / 'layer0.gdn.f32').is_file():
        raise AssertionError('expected F32 recurrent state diagnostic is missing')
    if args.probe_inputs is not None:
        probe = output / 'probe'
        command = [str(args.engine.resolve()), '--inputs', str(args.probe_inputs.resolve()), '--backend', 'cuda',
                   '--capacity', '1024', '--steps', '1', '--mtp', 'off', '--replay', 'commands',
                   '--layer0-probe', str(probe)]
        with (output / 'probe-stderr.txt').open('wb') as stderr:
            subprocess.run(command, stdout=subprocess.PIPE, stderr=stderr, check=True, timeout=120)
        files = list((probe / 'prefill').glob('[0-9][0-9]-*'))
        if len(files) != 9 or any(path.suffix != f'.{args.activation}' or path.stat().st_size % item_bytes for path in files):
            raise AssertionError('layer-zero probe suffix disagrees with activation storage')
    print(f'PASS: {args.activation} hidden and {len(histories)} convolution histories have matching suffixes')


if __name__ == '__main__':
    main()
