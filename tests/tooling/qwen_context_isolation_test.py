"""Real-model regression for late turn capacity errors and request isolation.

Run from the repository root via python -B -m tests.tooling.qwen_context_isolation_test.
Provide an existing Qwen3.5-2B prepared input directory and local prompt files.
"""
import argparse
from pathlib import Path

from engine.text.worker_client import WorkerClient


def consume(request):
    tokens, boundaries = [], []
    try:
        while True:
            event = request.read(timeout=60)
            if event.kind == 'token':
                tokens.append(event.token)
                request.acknowledge()
            elif event.kind == 'turn_end':
                boundaries.append((event.turn, len(tokens)))
                request.acknowledge()
            else:
                return tokens, boundaries, event
    finally:
        request.close()


def first_boundary(request):
    tokens = []
    while True:
        event = request.read(timeout=60)
        if event.kind == 'token':
            tokens.append(event.token)
            request.acknowledge()
        elif event.kind == 'turn_end':
            request.acknowledge()
            return tokens
        else:
            raise AssertionError((event.kind, event.message))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--next-prompt', required=True)
    parser.add_argument('--draft-count', type=int, default=0)
    parser.add_argument('--sampling', choices=('greedy', 'random'), default='greedy')
    parser.add_argument('--assert-early-eos', action='store_true',
                        help='also check the known local Q4 early-stop fixture')
    args = parser.parse_args()
    prompt = [int(value) for value in Path(args.prompt).read_text().split()]
    next_prompt = [int(value) for value in Path(args.next_prompt).read_text().split()]
    if len(prompt) != 105 or len(next_prompt) != 107:
        raise ValueError('expected the existing 105- and 107-token fixtures')

    def start(capacity):
        command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
                   '--capacity', str(capacity), '--max-sessions', '2', '--active-sequences', '2',
                   '--draft-count', str(args.draft_count)]
        if args.sampling == 'random':
            command += ['--sampling', 'random', '--temperature', '0.8', '--top-k', '40',
                        '--top-p', '0.95', '--seed', '123']
        return WorkerClient(command)

    worker = start(256)
    try:
        expected, boundaries, terminal = consume(worker.submit_turns([(prompt[:3], 1), (prompt[3:4], 1)]))
        assert len(expected) == 2 and boundaries == [(0, 1), (1, 2)] and terminal.kind == 'completed'
        anchor = worker.open_session(prompt[:3], 1)
        first = first_boundary(anchor)
        tokens, boundaries, event = consume(worker.submit_turns([(prompt, 32), (next_prompt, 32)]))
        assert len(tokens) == 32 and boundaries == [(0, 32)]
        assert event.kind == 'error' and event.message.startswith('invalid: ') and 'remaining context' in event.message
        peer_tokens, _, peer = consume(worker.submit(prompt[:3], 1))
        assert len(peer_tokens) == 1 and peer.kind == 'completed'
        anchor.append(prompt[3:4], 1)
        anchor.finish_input()
        tail, boundaries, terminal = consume(anchor)
        assert first + tail == expected and boundaries == [(1, 1)] and terminal.kind == 'completed'
        print('oversized second turn rejected; peer completed and live anchor matched both baseline tokens')
    finally:
        worker.close(timeout=60)

    if args.assert_early_eos:
        worker = start(64)
        try:
            tokens, boundaries, event = consume(worker.submit_turns([(prompt[:25], 12), (prompt[25:45], 12)]))
            assert event.kind == 'completed' and len(boundaries) == 2 and boundaries[0][1] < 12
            print('early EOS released context; the later turn remained admissible')
        finally:
            worker.close(timeout=60)


if __name__ == '__main__':
    main()
