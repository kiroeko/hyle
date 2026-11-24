"""Exercise full-width MTP catch-up prefill with locally prepared Qwen3.5-2B."""

import argparse
from pathlib import Path

from engine.text.worker_client import WorkerClient


# Each turn crosses the 64-row chunk boundary. The first decode leaves a draft
# row to catch up when the second turn submits its first full chunk.
PROMPTS = (
    (248045, 846, 198, 40, 2688, 33572, 264, 7483, 12368, 39579, 364, 1004, 12174, 13, 1165,
     599, 4590, 22293, 11, 1330, 944, 7713, 8266, 11, 264, 2526, 2956, 11, 321, 264, 7786,
     314, 220, 16, 17, 15, 10867, 13, 66511, 1189, 4300, 47788, 11, 14656, 11, 466, 70386,
     26, 567, 4021, 3579, 65835, 19132, 29201, 13, 11150, 728, 3019, 279, 1484, 13, 79091,
     264, 14542, 64291, 11, 264, 5902, 22168, 1817, 11, 321, 1204, 310, 21078, 279, 22293, 6,
     944, 13, 28763, 279, 31626, 4661, 678, 11125, 748, 353, 628, 4123, 279, 3019, 440, 279,
     2015, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271),
    (248046, 198, 248045, 846, 198, 2381, 279, 12368, 39579, 11, 353, 1144, 310, 2404, 1777,
     5072, 4965, 1083, 264, 24347, 207894, 13, 9357, 12654, 1970, 364, 4563, 79175, 8532, 8,
     421, 25503, 328, 591, 11, 28493, 19201, 2391, 487, 479, 5564, 35169, 11, 57409, 449,
     4147, 803, 11, 321, 7225, 3979, 310, 381, 449, 7308, 494, 220, 16, 310, 220, 21, 13,
     3301, 264, 795, 1005, 13, 4844, 3165, 264, 709, 421, 42199, 3979, 539, 9653, 1973,
     93183, 279, 1879, 13, 28763, 4821, 6813, 17661, 76531, 6761, 321, 11173, 7129, 11, 321,
     10033, 799, 2790, 6355, 1783, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271),
)


def complete_turn(request, ordinal, output_tokens):
    for _ in range(output_tokens):
        event = request.read(timeout=120)
        assert event.kind == 'token', (event.kind, event.error_kind, event.message)
        request.acknowledge()
    boundary = request.read(timeout=120)
    assert boundary.kind == 'turn_end' and boundary.turn == ordinal, boundary
    request.acknowledge()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--backend', choices=('cpu', 'cuda', 'apple'), default='cpu')
    args = parser.parse_args()
    command = [args.engine, '--engine-worker', '--inputs', args.inputs, '--backend', args.backend,
               '--capacity', '512', '--max-sessions', '1', '--active-sequences', '1',
               '--prefill-chunk-tokens', '64', '--draft-count', '3',
               '--additional-sampling', 'random,0.8,40,0.95']
    worker = WorkerClient(command, startup_timeout=120)
    try:
        profile, = worker.execution_profiles
        assert (profile.algorithm, profile.max_draft_tokens) == ('mtp', 3), profile
        assert worker.placement.primary_backend == args.backend and worker.placement.cpu_ffn_layers == 0
        for sampler in (0, 1):
            request = worker.submit_turns(((PROMPTS[0], 16), (PROMPTS[1], 8)), sampling_index=sampler, seed=42)
            try:
                complete_turn(request, 0, 16)
                complete_turn(request, 1, 8)
                event = request.read(timeout=120)
                assert event.kind == 'completed', (event.kind, event.error_kind, event.message)
            finally:
                request.close()
    finally:
        worker.close(timeout=60)
    print(f'PASS {args.backend} K3 chunked prefill: 48 outputs, greedy/random continuation', flush=True)


if __name__ == '__main__':
    main()
