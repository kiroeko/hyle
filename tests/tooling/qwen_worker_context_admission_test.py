"""Exercise real CUDA context boundaries across an open multi-turn session."""
import argparse
from pathlib import Path

from engine.text.contracts import WorkerError
from engine.text.worker_client import WorkerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--draft-count', type=int, choices=(0, 1, 3), default=0)
    parser.add_argument('--sampling', choices=('greedy', 'random'), default='greedy')
    args = parser.parse_args()
    if args.sampling == 'random' and args.draft_count:
        parser.error('random sampling test requires draft-count 0')
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
               '--capacity', '16', '--max-sessions', '1', '--active-sequences', '1',
               '--draft-count', str(args.draft_count), '--sampling', args.sampling]
    if args.sampling == 'random':
        command += ['--temperature', '0.8', '--top-k', '40', '--top-p', '0.95', '--seed', '7']
    worker = WorkerClient(command)

    def read_token(request):
        event = request.read(timeout=60)
        assert event.kind == 'token', (event.kind, event.error_kind, event.message)
        request.acknowledge()
        return event.token

    def boundary(request, turn):
        event = request.read(timeout=60)
        assert event.kind == 'turn_end' and event.turn == turn, (event.kind, event.turn, turn)
        request.acknowledge()

    try:
        assert worker.context == 16
        # The initial output leaves its last token pending: 3 + (14 - 1) fits.
        exact = worker.submit([1, 2, 3], 14)
        try:
            tokens = [read_token(exact) for _ in range(14)]
            assert exact.read(timeout=60).kind == 'completed'
        finally:
            exact.close()
        assert tokens[0] not in worker.stop_ids

        # One additional output cannot fit. Its rejection must stay local.
        too_large = worker.submit([1, 2, 3], 15)
        try:
            error = too_large.read(timeout=60)
            assert error.kind == 'error' and error.error_kind == 'invalid', (error.kind, error.error_kind, error.message)
        finally:
            too_large.close()

        # Stop after one output, then consume the released quota with a new
        # turn. The prior pending token takes one of the remaining positions.
        live = worker.open_session([1, 2, 3], 14, stop_ids=[tokens[0]])
        try:
            assert read_token(live) == tokens[0]
            boundary(live, 0)
            try:
                live.append([4] * 13, 1, timeout=60)
            except WorkerError as error:
                assert error.kind == 'capacity', (error.kind, str(error))
            else:
                raise AssertionError('over-capacity live turn was accepted')
            live.append([4] * 12, 1, timeout=60)
            read_token(live)
            boundary(live, 1)
            live.finish_input(timeout=60)
            assert live.read(timeout=60).kind == 'completed'
        finally:
            live.close()

        # A subsequent request still has a usable model and retired slot.
        followup = worker.submit([1, 2, 3], 1)
        try:
            assert read_token(followup) == tokens[0]
            assert followup.read(timeout=60).kind == 'completed'
        finally:
            followup.close()
        assert worker.healthy
        print(f'PASS {args.sampling} draft-count={args.draft_count}: exact capacity, local rejection, '
              'early-stop quota release, live-turn boundary, slot reuse', flush=True)
    finally:
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
