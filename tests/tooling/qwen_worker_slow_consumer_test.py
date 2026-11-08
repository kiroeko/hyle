"""Check that a stalled consumer does not hold another real model slot."""
import argparse
import queue

from engine.text.worker_client import WorkerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--draft-count', type=int, choices=(0, 1, 3), default=0)
    args = parser.parse_args()
    worker = WorkerClient([args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
                           '--capacity', '128', '--max-sessions', '2', '--active-sequences', '2',
                           '--waiting-limit', '2', '--output-capacity', '4', '--draft-count', str(args.draft_count)])
    requests = []

    def submit(prompt):
        request = worker.submit(prompt, 8)
        requests.append(request)
        return request

    def complete(request):
        tokens = []
        while True:
            event = request.read(timeout=60)
            if event.kind == 'token':
                tokens.append(event.token)
                request.acknowledge()
            elif event.kind == 'completed':
                return tokens
            else:
                raise AssertionError((event.kind, event.error_kind, event.message))

    try:
        slow = submit([1, 2, 3])
        peer = submit([3, 2, 1])
        first = slow.read(timeout=60)
        assert first.kind == 'token', (first.kind, first.message)
        # Leave slow's first token unacknowledged throughout both requests.
        peer_tokens = complete(peer)
        assert peer_tokens
        replacement_tokens = complete(submit([1, 3, 2]))
        assert replacement_tokens and worker.healthy
        # No later token may be delivered without returning the first credit.
        try:
            slow.read(timeout=.25)
        except RuntimeError as error:
            assert 'acknowledge' in str(error), str(error)
        except queue.Empty:
            raise AssertionError('unacknowledged output unexpectedly became readable')
        else:
            raise AssertionError('unacknowledged output unexpectedly became readable')
        assert slow.cancel()
        terminal = slow.read(timeout=60)
        assert terminal.kind == 'cancelled', (terminal.kind, terminal.message)
        final_tokens = complete(submit([2, 1, 3]))
        assert final_tokens and worker.healthy
        print(f'draft-count={args.draft_count}: stalled credit retained while peer={len(peer_tokens)} '
              f'and replacement={len(replacement_tokens)} tokens completed; '
              f'cancelled slow request and completed final={len(final_tokens)}', flush=True)
    finally:
        for request in requests:
            request.close()
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
