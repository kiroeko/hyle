"""Exercise bounded waiting, cancellation and reuse with a local CUDA worker."""
import argparse
import queue

from engine.text.worker_client import WorkerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--draft-count', type=int, choices=(0, 1, 3), default=0)
    parser.add_argument('--output-capacity', type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    worker = WorkerClient([args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
                           '--capacity', '128', '--max-sessions', '1', '--active-sequences', '1',
                           '--waiting-limit', '1', '--output-capacity', str(args.output_capacity),
                           '--draft-count', str(args.draft_count)])
    requests = []

    def submit():
        request = worker.submit([1, 2, 3], 8)
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
        active = submit()
        first = active.read(timeout=60)
        assert first.kind == 'token', (first.kind, first.message)
        # Retain the first output credit so this request cannot be drained.
        waiting = submit()
        rejected = submit()
        error = rejected.read(timeout=60)
        assert error.kind == 'error' and error.error_kind == 'admission', (error.kind, error.error_kind, error.message)
        assert 'queue is full' in error.message, error.message
        assert waiting.cancel()
        assert waiting.read(timeout=60).kind == 'cancelled'
        replacement = submit()
        try:
            premature = replacement.read(timeout=.25)
        except queue.Empty:
            pass
        else:
            raise AssertionError(('replacement produced before cancellation', premature.kind, premature.message))
        assert active.cancel()
        assert active.read(timeout=60).kind == 'cancelled'
        result = complete(replacement)
        assert result and worker.healthy
        followup = complete(submit())
        assert followup and worker.healthy
        print(f'draft-count={args.draft_count} output-capacity={args.output_capacity}: held credit, queue-full rejection, '
              f'waiting and active cancellation, replacement={len(result)} and followup={len(followup)} tokens', flush=True)
    finally:
        for request in requests:
            request.close()
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
