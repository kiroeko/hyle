"""Check that a stalled consumer does not hold another real model slot."""
import argparse
import queue

from engine.text.worker_client import WorkerClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--draft-count', type=int, choices=(0, 1, 3), default=0)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'apple', 'vulkan'), default='cuda')
    parser.add_argument('--cpu-ffn-layers', type=int, default=0)
    parser.add_argument('--prefill-chunk-tokens', type=int)
    parser.add_argument('--sampling', choices=('greedy', 'random', 'both'), default='greedy')
    args = parser.parse_args()
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', args.backend,
                           '--capacity', '128', '--max-sessions', '2', '--active-sequences', '2',
                           '--waiting-limit', '2', '--output-capacity', '4', '--draft-count', str(args.draft_count),
                           '--cpu-ffn-layers', str(args.cpu_ffn_layers)]
    if args.prefill_chunk_tokens is not None:
        command += ['--prefill-chunk-tokens', str(args.prefill_chunk_tokens)]
    if args.sampling == 'both': command += ['--additional-sampling', 'random,0.8,40,0.95']
    elif args.sampling == 'random':
        command += ['--sampling', 'random', '--temperature', '0.8', '--top-k', '40', '--top-p', '0.95']
    worker = WorkerClient(command, startup_timeout=120)
    requests = []
    sampling_index = 0

    def submit(prompt):
        request = worker.submit(prompt, 8, sampling_index=sampling_index, seed=42)
        requests.append(request)
        return request

    def complete(request):
        tokens = []
        while True:
            event = request.read(timeout=120)
            if event.kind == 'token':
                tokens.append(event.token)
                request.acknowledge()
            elif event.kind == 'completed':
                return tokens
            else:
                raise AssertionError((event.kind, event.error_kind, event.message))

    def exercise():
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
        # Compare the same R1 program before/after a real cancelled request;
        # a B2 trajectory can have a different numerical reduction order.
        reference = complete(submit([1, 2, 3]))
        cancelled = submit([1, 2, 3])
        assert cancelled.read(timeout=120).kind == 'token'
        assert cancelled.retire(timeout=120).kind == 'cancelled'
        assert complete(submit([1, 2, 3])) == reference, 'slot reuse changed the logical RNG stream or committed state'
        print(f'draft-count={args.draft_count}: stalled credit retained while peer={len(peer_tokens)} '
              f'and replacement={len(replacement_tokens)} tokens completed; '
              f'cancelled slow request and completed final={len(final_tokens)}; '
              f'sampling-index={sampling_index} R1 replay IDs identical after terminal retirement', flush=True)
    try:
        assert worker.placement.primary_backend == args.backend
        assert worker.placement.cpu_ffn_layers == args.cpu_ffn_layers
        profile = worker.execution_profiles[0]
        assert profile.algorithm == ('mtp' if args.draft_count else 'ordinary')
        assert profile.max_draft_tokens == args.draft_count
        for sampling_index in range(2 if args.sampling == 'both' else 1):
            exercise()
    finally:
        for request in requests:
            request.close()
        worker.close(timeout=60)


if __name__ == '__main__':
    main()
