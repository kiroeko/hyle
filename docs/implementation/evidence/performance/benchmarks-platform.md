# Platform performance evidence

This page records only evidence that is valid for the current platform API and
the repository-wide performance instrument. Older mixed microbenchmark tables
used different APIs, peers, charging boundaries, and statistical rules; they
remain recoverable from Git history but are not current product evidence.

The immutable experiment definition is
[`platform-consumer-v1.json`](../../../../bench/contracts/platform-consumer-v1.json).
The closure decision and bounded investigation log live in
[`platform-v1-closure.md`](../../tasks/platform/platform-v1-closure.md).

## Current verdict

`PASS` for the frozen W1-WIN P1 engineering contract. This is not a Formal,
cross-machine, or all-platform SOTA certificate. The two admitted families
passed independently; neither result is pooled into a platform-wide score.

| family | consumer-shaped charged lifecycle | pinned peer | material / NI margin | current status |
| --- | --- | --- | --- | --- |
| worker idle handoff | one persistent caller and worker, pinned to two distinct same-class physical cores, exchange 64 request/response tokens; threads and mailbox owners remain outside measurement | oneTBB production `binary_semaphore` used by `concurrent_monitor` | 5% / 5% | `WIN`; 8/8 strata NI `PASS` |
| model loader mapping | open and map one warm 16 MiB file, probe and validate 64 fixed page-distributed bytes, then unmap and close | Boost.Interprocess `file_mapping` + `mapped_region` | 5% / 5% | `PARITY`; 8/8 strata NI `PASS` |

The two families are independent gates and are never pooled into a “platform
score.” The contract runs primary and function-placement variants, both
execution orders, both allocation orders, A/A, known 5% and 10% deltas, and a
timer-floor control. W1-WIN is the engineering verdict machine; Linux and
macOS correctness results do not get averaged into Windows performance.

The first diagnostic probe exposed an instrument error rather than a product
verdict: the default single-thread policy constrained both handoff threads to
logical CPU 2. HYLE `Parker` then measured about 3.5x slower than the Windows
SRWLOCK peer, while allowing two physical cores reversed the result by more
than an order of magnitude. A bounded affinity experiment reproduced the
direction switch whenever Windows co-located or separated the threads. The
contract therefore uses the general `cross-core-pair` adapter: W1 selects CPU
2 / CPU set 258 for the caller and CPU 4 / CPU set 260 for the worker, limits
the process to mask `0x14`, pins both roles, and rejects missing telemetry.
The same-core result remains a documented mechanism characteristic; it is not
silently relabelled as the scheduler's normal cross-core worker topology.

## Qualified evidence

- product candidate: `5b8dab7012e9f7c6cdaf97f863c0b83b4e8a5064`, tree
  `24a68b0ab3314902a765c6bf60f8ccb760d290f4`;
- evidence source/harness revision: `535e5820ec3b2beda5326dd3d41364d8d43116f4`, tree
  `e676563c0c4146afdb3e761bc9ec8177a907e49e`;
- contract SHA-256:
  `3a66896e30eeaa6eb645a88a63fd02be9cc90654129f6c7f3bc5e21687c360b3`;
- retained engineering bundle:
  `C:\hyle-scratch\platform-perf-qualify-535e5820-evidence`; manifest SHA-256
  `2c0fdc3d650a49d18e3af298b910a25f1152ef72b296b98b943ca0c3cd4ea1c6`, summary
  SHA-256 `3bf1fd3ec40dac45418000bde08a927ba4f8648f17717fb0585af91b752b21d2`;
- runner/verifier verdict: `QUALIFIED`, `paths_verified=true`, current engineering
  policy eligible. Artifact identity, MSVC 19.50 Release/static/LTO-off build,
  clean pinned peers, registered host, child affinity telemetry, A/A, 5%/10%
  known deltas, timer floor and placement controls all passed.

The n=15 run escalated by the frozen power rule to 31 independent process
clusters. Handoff's eight median advantage ratios span 19.0693--23.1283; the
worst 98.93% confidence lower bound is 18.1863. Mapping's eight medians span
0.991584--0.997888; the worst lower bound is 0.983392, inside the predeclared
5% non-inferiority limit. Later commits change sanitizer/test/delivery tooling
and documents only: platform product sources and the frozen performance
contract/harness are unchanged, so they do not renew or overwrite this exact
bundle identity.

## Structural performance gates

Most platform operations are OS control paths. Their useful SOTA property is
the absence of avoidable work, not a claim that a wrapper beats its syscall.

| capability | accepted current bound | disallowed hidden cost |
| --- | --- | --- |
| cached CPU/ISA/cache/page facts | one bounded probe, then O(1) reads | probing on every query or target-wide unsafe ISA/LTO |
| process-available CPU ids | O(OS CPU records + returned ids); live rather than cached | stale affinity/cpuset cache or an O(ND) nested scan |
| `Thread` | one native owner; default setup has no required handshake; callable convenience buys one cold allocation | per-operation refcount, task-pool policy, or silent detach |
| address wait / `Parker` | O(1), one atomic word, no heap/mutex; blocked transition uses the corresponding native wait/wake | polling loop or per-wait allocation |
| immediate VM mapping | one native mapping request; staged reserve/commit explicitly buys two phases | making the ordinary RW shape use staged calls for a benchmark-oriented API |
| NUMA facts and ordinary policy | cached membership O(1), input validation O(nodes), ordinary Linux policy O(1) | silent node-0 fallback, unbounded persistent stripe metadata |
| selected-node/chunk placement | O(stripes) for bounded lazy policy, or O(pages) bounded-batch placement with constant scratch | O(pages) persistent HYLE metadata or unbounded temporary arrays |
| `FileHandle` short operation | one native read/write/seek/sync; one-word owner | hidden lock, heap, or a second I/O operation |
| `read_exact` / `write_all` | O(bytes/native transfer limit), zero allocation | copying into an intermediate owning buffer |
| `MappedFile` lifecycle | one open plus one mapping and teardown; access O(1), no normal-path metadata | implicit full-file prefault, advice, lock, or second path lookup after owner transfer |
| range advice/flush/lock | O(1) validation plus at most the documented native operation; kernel page work may scale with range | pretending OS-dependent residency work is a nanosecond user-space primitive |
| directory visitor | O(entries), no owning entry allocation on the common path | unconditional per-entry `stat` or collected path list |
| `DynLib` | one loader operation; symbol lookup delegates to the loader | private symbol cache, TLS error channel, or non-owning `get` race |

These bounds are reviewed in source and exercised by correctness tests. A
syscall-dominated operation only becomes a numeric P1 gate when a named
consumer profile shows material contribution and a same-contract peer exists.

## Interpretation limits

- `Parker` is compared as a no-loss single-consumer worker mailbox. It is not a
  counting semaphore or a multi-waiter fairness benchmark.
- The P1 handoff claim is specifically the scheduler's cross-physical-core
  worker wake topology. On W1, a forced same-logical-CPU ping-pong favors
  oneTBB's Windows SRWLOCK path by roughly 3.5x; HYLE does not claim
  non-inferiority for that different topology.
- Mapping is compared for a model-loader lifecycle with a warm page cache and
  fixed random probes. It does not claim mmap wins for small sequential files.
- oneTBB is built in its supported shared-library form from the pinned source;
  HYLE is consumed directly from source into a static target. The DLL call
  boundary is recorded in the contract and cannot justify a near-margin win.
- Boost is a modular header-only peer. Every required submodule is pinned by
  the recorded clean superproject revision.
- Single-node hosts can verify NUMA API cost and fallback correctness, but not
  multi-socket physical placement or local/remote bandwidth.
- Advice, prefetch, lock, filesystem, VM, and loader timings are dominated by
  OS state. Diagnostic point values cannot authorize product changes.

## Reproduce

Run from a clean candidate checkout and use unique scratch/evidence roots:

```powershell
python scripts/hyle-perf.py validate `
  --contract bench/contracts/platform-consumer-v1.json

python scripts/hyle-perf.py probe `
  --contract bench/contracts/platform-consumer-v1.json `
  --machine W1-WIN `
  --peer onetbb=C:\public\project-hyle\oneTBB `
  --peer boost=C:\public\project-hyle\boost `
  --build C:\hyle-scratch\platform-perf-probe-build `
  --output C:\hyle-scratch\platform-perf-probe-evidence

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/platform-consumer-v1.json `
  --machine W1-WIN `
  --peer onetbb=C:\public\project-hyle\oneTBB `
  --peer boost=C:\public\project-hyle\boost `
  --build C:\hyle-scratch\platform-perf-qualify-build `
  --output C:\hyle-scratch\platform-perf-qualify-evidence

python scripts/hyle-perf.py verify `
  --manifest C:\hyle-scratch\platform-perf-qualify-evidence\manifest.json `
  --contract bench/contracts/platform-consumer-v1.json
```

`probe` only qualifies the instrument and estimates wall time. Only the
independent-process `qualify-platform` manifest and a successful deep replay
can carry the verdict; hand-written nanobench tables are not accepted evidence.
