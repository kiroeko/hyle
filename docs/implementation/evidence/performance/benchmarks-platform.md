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

`PENDING` while platform is `CANDIDATE`. A development smoke run is useful only
to prove that the benchmark executes and validates its results; it is not a
performance verdict. The page is updated to `PASS`, `BLOCKED`, or a named
limitation only from runner-produced independent-process manifests.

| family | consumer-shaped charged lifecycle | pinned peer | material / NI margin | current status |
| --- | --- | --- | --- | --- |
| worker idle handoff | one persistent caller and worker, pinned to two distinct same-class physical cores, exchange 64 request/response tokens; threads and mailbox owners remain outside measurement | oneTBB production `binary_semaphore` used by `concurrent_monitor` | 5% / 5% | `PENDING` |
| model loader mapping | open and map one warm 16 MiB file, probe and validate 64 fixed page-distributed bytes, then unmap and close | Boost.Interprocess `file_mapping` + `mapped_region` | 5% / 5% | `PENDING` |

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
```

`probe` only qualifies the instrument and estimates wall time. The closure task
records the exact `qualify-platform` command and manifest paths after the
candidate is frozen; hand-written nanobench tables are not accepted evidence.
