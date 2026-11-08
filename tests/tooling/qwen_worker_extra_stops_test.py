"""Exercise per-request stop IDs through the real local CUDA model worker."""
import argparse
from pathlib import Path

from engine.text.worker_client import WorkerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--sessions', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    command = [str(args.engine), '--engine-worker', '--inputs', str(args.inputs), '--backend', 'cuda',
               '--capacity', '128', '--max-sessions', str(args.sessions), '--active-sequences', str(args.sessions)]
    worker = WorkerClient(command)

    def collect(request, *, live=False):
        tokens, turns = [], []
        try:
            while True:
                event = request.read(timeout=120)
                if event.kind == 'token':
                    tokens.append(event.token)
                    request.acknowledge()
                elif event.kind == 'turn_end':
                    turns.append(event.turn)
                    if live:
                        request.finish_input()
                    request.acknowledge()
                elif event.kind == 'completed':
                    return tokens, turns
                else:
                    raise AssertionError((event.kind, event.message))
        finally:
            request.close()

    try:
        prompt = [1, 2, 3]
        reference, _ = collect(worker.submit(prompt, 4))
        assert len(reference) >= 2 and reference[0] not in worker.stop_ids, reference
        index = next((i for i in range(len(reference) - 1, 0, -1)
                      if reference[i] not in reference[:i] and reference[i] not in worker.stop_ids), None)
        assert index is not None, reference
        token = reference[index]
        requests = (
            ('flat', worker.submit(prompt, 4, stop_ids=[token])),
            ('turns', worker.submit_turns([(prompt, 4)], stop_ids=[token])),
            ('live', worker.open_session(prompt, 4, stop_ids=[token])),
        )
        for kind, request in requests:
            output, turns = collect(request, live=kind == 'live')
            assert output == reference[:index + 1] and turns == ([] if kind == 'flat' else [0]), (kind, reference, output, turns)
            print('PASS', kind, 'stop', token, 'sessions', args.sessions, flush=True)
        # Stop IDs are vocabulary entries, not prompt positions. In particular,
        # a valid stop list may contain more IDs than this session's context.
        many_stops = []
        for value in range(worker.vocabulary):
            if value not in worker.stop_ids:
                many_stops.append(value)
            if len(many_stops) == worker.context + 1:
                break
        assert len(many_stops) == worker.context + 1
        output, turns = collect(worker.submit(prompt, 1, stop_ids=many_stops))
        assert output == reference[:1] and not turns, (reference, output, turns)
        print('PASS', len(many_stops), 'extra stops with context', worker.context, flush=True)
    finally:
        worker.close()


if __name__ == '__main__':
    main()
