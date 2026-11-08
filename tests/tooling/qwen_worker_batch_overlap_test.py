"""Compare two resident requests with the same observed prefill batch shape.

This real CUDA diagnostic uses an existing local Qwen3.5-2B preparation. It
does not build or download a model. An uncontrolled prefill shape is an error,
not a passing comparison. Later decode batch shapes can still differ.
"""
import argparse
import queue
import subprocess
import threading
import time


def run(engine, inputs, sampling, overlap):
    command = [engine, '--engine-worker', '--inputs', inputs, '--backend', 'cuda',
               '--capacity', '128', '--max-sessions', '2', '--active-sequences', '2',
               '--draft-count', '0', '--sampling', sampling, '--overlap', overlap]
    if sampling == 'random':
        command += ['--temperature', '0.8', '--top-k', '40', '--top-p', '0.95', '--seed', '7']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding='utf-8', errors='replace', bufsize=1)
    frames = queue.Queue()
    log = []

    def read_frames():
        try:
            for line in process.stdout:
                frames.put(line.rstrip('\r\n').split('\t'))
        finally:
            frames.put(None)

    def read_log():
        log.extend(process.stderr)

    reader = threading.Thread(target=read_frames, daemon=True)
    logger = threading.Thread(target=read_log, daemon=True)
    reader.start()
    logger.start()
    values, finished = {'1': [], '2': []}, set()
    try:
        # Queue both requests before startup. Still verify the actual batch:
        # timing alone does not prove that they were admitted together.
        process.stdin.write('G\t1\t32\t1,2,3\nG\t2\t32\t1,2,3\n')
        process.stdin.flush()
        ready = False
        deadline = time.monotonic() + 180
        while len(finished) < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('worker did not finish both requests')
            try:
                frame = frames.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError('worker did not finish both requests') from None
            if frame is None:
                raise RuntimeError(f'worker output closed; exit={process.poll()}')
            if frame[0] == 'READY':
                ready = True
            elif frame[0] == 'A':
                continue
            elif frame[0] == 'T' and len(frame) == 3 and frame[1] in values:
                values[frame[1]].append(int(frame[2]))
                process.stdin.write(f'K\t{frame[1]}\t1\n')
                process.stdin.flush()
            elif frame[0] == 'F' and len(frame) == 3 and frame[1] in values and frame[2] == 'completed':
                finished.add(frame[1])
            else:
                raise RuntimeError(f'unexpected worker frame: {frame}')
        if not ready or any(len(ids) != 32 for ids in values.values()):
            raise AssertionError('worker did not complete the intended workload')
        process.stdin.write('Q\n')
        process.stdin.flush()
        if process.wait(timeout=60):
            raise RuntimeError('worker did not shut down successfully')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        reader.join(timeout=5)
        logger.join(timeout=5)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()
    prefill = [line.strip() for line in log if line.startswith('generation_batch') and ' T=3 ' in line]
    if prefill != ['generation_batch requests=2 T=3 role=0 full_hit=0']:
        raise AssertionError(f'{overlap}: actual prefill shape was not two requests: {prefill}')
    if overlap == 'on' and not any('planned_replay T=1 slot_count=3 ' in line for line in log):
        raise AssertionError('overlap did not prepare a relay program')
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--sampling', choices=('greedy', 'random'), required=True)
    args = parser.parse_args()
    serial = run(args.engine, args.inputs, args.sampling, 'off')
    overlap = run(args.engine, args.inputs, args.sampling, 'on')
    if serial != overlap:
        raise AssertionError('published token IDs differ with the same two-request prefill shape')
    print(f'PASS {args.sampling}: 2 x 32 matching tokens, actual two-request prefill in both arms; '
          'relay prepared', flush=True)
    print('The resident protocol does not expose the actual successor submission count.', flush=True)


if __name__ == '__main__':
    main()
