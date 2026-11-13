"""Real-weight mixed execution; uses local prepared assets and never downloads."""
import argparse
import threading

from engine.text.worker_client import WorkerClient


def turn(request, ordinal):
    tokens = []
    while True:
        event = request.read(timeout=120)
        if event.kind == "token":
            tokens.append(event.token)
            request.acknowledge()
        elif event.kind == "turn_end":
            assert event.turn == ordinal and tokens, (event, tokens)
            request.acknowledge()
            return tokens
        else:
            raise AssertionError((event.kind, event.error_kind, event.message))


def finish(request):
    request.finish_input(timeout=120)
    assert request.read(timeout=120).kind == "completed"
    request.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--vocabulary", required=True)
    args = parser.parse_args()
    command = [args.engine, "--engine-worker", "--inputs", args.inputs, "--backend", "cuda",
               "--capacity", "128", "--prefill-chunk-tokens", "128",
               "--max-sessions", "3", "--active-sequences", "2", "--output-capacity", "2",
               "--draft-count", "1", "--additional-sampling", "random,0.8,40,0.95",
               "--profile", "subset", "--draft-count", "3",
               "--profile-asset", "draft.vocabulary=" + args.vocabulary,
               "--profile", "plain", "--draft-count", "0"]
    worker = WorkerClient(command, startup_timeout=120)
    sessions = []
    try:
        invalid = worker.submit([1, 2, 3], 2, execution_profile=3)
        error = invalid.read(timeout=120)
        assert error.kind == "error" and error.error_kind == "admission", error
        invalid.close()
        sessions = [worker.open_session([1, 2, 3], 8, execution_profile=1),
                    worker.open_session([1, 2, 3], 8, execution_profile=0),
                    worker.open_session([1, 2, 3], 8, execution_profile=2, sampling_index=1, seed=777)]
        # Hold subset's delivery credit while both incompatible peers progress.
        results, errors = {}, []

        def read(index):
            try:
                results[index] = turn(sessions[index], 0)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=read, args=(index,)) for index in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not errors and all(not thread.is_alive() for thread in threads), errors
        results[0] = turn(sessions[0], 0)
        assert set(results) == {0, 1, 2}
        for request in sessions:
            request.append([4, 5], 8, timeout=120)
        followups = [turn(request, 1) for request in sessions]
        # A cancelled profile retires before its slot admits new native work.
        assert sessions[0].cancel()
        assert sessions[0].read(timeout=120).kind == "cancelled"
        sessions[0].close()
        replacement = worker.open_session([1, 2, 3], 8, execution_profile=0)
        turn(replacement, 0)
        finish(replacement)
        finish(sessions[1]); finish(sessions[2])
        # Repeat with all physical slots free. Logical RNG coordinates must
        # not depend on the earlier co-resident requests or their slot order.
        # Both runs retain the same R1 mathematical shape.
        reference = worker.open_session([1, 2, 3], 8, execution_profile=2, sampling_index=1, seed=777)
        assert turn(reference, 0) == results[2], "mixed profiles changed independent random state"
        reference.append([4, 5], 8, timeout=120)
        assert turn(reference, 1) == followups[2], "mixed profiles changed continuation RNG/pending"
        finish(reference)
        # A multi-profile owner must also combine compatible requests instead
        # of serializing every request merely because profiles were declared.
        peers = [worker.open_session([1, 2, 3], 24, execution_profile=0) for _ in range(2)]
        errors.clear()

        def peer(index):
            try:
                turn(peers[index], 0)
            except BaseException as error:
                errors.append(error)

        threads = [threading.Thread(target=peer, args=(index,)) for index in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
        assert not errors and all(not thread.is_alive() for thread in threads), errors
        for request in peers:
            finish(request)
        assert worker.healthy
    finally:
        for request in sessions:
            request.close()
        worker.close(timeout=120)
    # R1 cold planning must include greedy routes even when sampler zero is
    # random. This also exercises the original Context/Queue capability query
    # with its metadata-only default host workspace on actual CUDA.
    ordered = []
    for random_first in (False, True):
        command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
            '--capacity', '128', '--max-sessions', '1', '--active-sequences', '1', '--draft-count', '3',
            '--profile-asset', 'draft.vocabulary=' + args.vocabulary]
        command += (['--sampling', 'random', '--temperature', '0.8', '--top-k', '40', '--top-p', '0.95',
                     '--additional-sampling', 'greedy'] if random_first else
                    ['--sampling', 'greedy', '--additional-sampling', 'random,0.8,40,0.95'])
        single = WorkerClient(command, startup_timeout=120)
        try:
            request = single.open_session([1, 2, 3], 16, sampling_index=1 if random_first else 0)
            ordered.append(turn(request, 0))
            finish(request)
        finally:
            single.close(timeout=120)
    assert ordered[0] == ordered[1], 'sampler order changed the prepared greedy K3 execution'
    # No profile acts as the asset loader for its peers. In particular the
    # default may need only target roles while later profiles also need MTP.
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', 'cuda',
        '--capacity', '128', '--max-sessions', '1', '--active-sequences', '1', '--draft-count', '0',
        '--profile', 'full', '--draft-count', '1', '--profile', 'subset', '--draft-count', '3',
        '--profile-asset', 'draft.vocabulary=' + args.vocabulary]
    reordered = WorkerClient(command, startup_timeout=120)
    try:
        for profile, expected in ((1, results[1]), (2, results[0])):
            request = reordered.open_session([1, 2, 3], 8, execution_profile=profile)
            assert turn(request, 0) == expected, 'profile load order changed prepared target mathematics'
            finish(request)
        request = reordered.open_session([1, 2, 3], 8, execution_profile=0)
        assert len(turn(request, 0)) == 8
        finish(request)
    finally:
        reordered.close(timeout=120)
    print("PASS native subset K3/full K1/plain random: co-residency, live turns, RNG, backpressure, cancellation and slot reuse")


if __name__ == "__main__":
    main()
