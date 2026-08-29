# Platform performance evidence

This page records only evidence that is valid for the current platform API and
the repository-wide performance instrument. Older mixed microbenchmark tables
used different APIs, peers, charging boundaries, and statistical rules; they
remain recoverable from Git history but are not current product evidence.

The immutable experiment definitions are
[`platform-consumer-v1.json`](../../../../bench/contracts/platform-consumer-v1.json)
for the unaffected v1 families and
[`platform-residency-p1-v2.json`](../../../../bench/contracts/platform-residency-p1-v2.json)
for progressive model-loader residency. The corresponding closure records are
[`platform-v1-closure.md`](../../tasks/platform/platform-v1-closure.md) and
[`platform-v2-inference-residency.md`](../../tasks/platform/platform-v2-inference-residency.md).

## Current verdict

`PASS` for the frozen W1-WIN P1 engineering contracts. This is not a Formal,
cross-machine, or all-platform SOTA certificate. The admitted families pass
independently; no result is pooled into a platform-wide score.

| family | consumer-shaped charged lifecycle | pinned peer | material / NI margin | current status |
| --- | --- | --- | --- | --- |
| worker idle handoff | one persistent caller and worker, pinned to two distinct same-class physical cores, exchange 64 request/response tokens; threads and mailbox owners remain outside measurement | oneTBB production `binary_semaphore` used by `concurrent_monitor` | 5% / 5% | `WIN`; 8/8 strata NI `PASS` |
| model loader mapping | open and map one warm 16 MiB file, probe and validate 64 fixed page-distributed bytes, then unmap and close | Boost.Interprocess `file_mapping` + `mapped_region` | 5% / 5% | `PARITY`; 8/8 strata NI `PASS` |
| progressive model residency | on one persistent warm 16 MiB mapping, grow the locked prefix through four disjoint 4 MiB deltas, expose the applied prefix, then unlock once | pinned llama.cpp Win32 `llama_mlock::grow_to` mechanism replayed as the same four `VirtualLock` calls and one `VirtualUnlock` | 5% / 5% | `PARITY`; 8/8 strata NI `PASS` |

The families are independent gates and are never pooled into a “platform
score.” Each applicable contract runs primary and function-placement variants, both
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

The current v2 progressive-residency receipt is:

- product candidate: `88085b671ac7dfb4d41e2819ceda0925d1ac0147`, tree
  `01f2183f4cb5c029293ba6c52b15dfbc40d27e13`;
- evidence source/harness revision: `a36ab184fd079a0e0652c6d4b2fa8b1fbc51a935`, tree
  `77f1dc7493656033fee991f582b94262927ab2e5`;
- pinned consumer peer: llama.cpp
  `7584430716ee229751771ed0d6bbcb780d105eeb`, tree
  `027524b2c81f6df300250aadf21ed6512e13e141`;
- contract SHA-256:
  `45896aa4f462bca5de1ee3e4ea8171c45111c4ecaae10153e037639f073d8e1c`;
- retained engineering bundle:
  `C:\hyle-scratch\platform-v2\perf-qualify-evidence`; manifest SHA-256
  `4c019d408a287336868d99b26e529c13d800c330ce4c77cc54c43864357095bd`, summary
  SHA-256 `0d148336277104322dc2fc8d1bd14ee332fb48290eee3ae9fea4db75ae46a65a`;
- current verifier: `qualification=QUALIFIED`, `paths_verified=true`, exact
  provenance, current environment policy, clean source/peer, stable host and
  artifacts, Release/static/LTO-off MSVC 19.50, and every declared control
  `PASS`.

The v2 run used 15 independent processes per stratum. All eight exact verdicts
are `PARITY`; median candidate/peer ratios span 0.986466--0.999315 and the worst
99.26% confidence lower bound is 0.964334, above the predeclared reciprocal 5%
non-inferiority limit (0.952381). Native page work dominates the roughly
0.56--0.59 ms lifecycle. The evidence supports “no material wrapper loss for
this normal progressive-residency shape,” not a claim that HYLE makes
`VirtualLock` itself faster.

The unaffected v1 receipt remains:

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
- qualification-time runner/verifier verdict: `QUALIFIED`, `paths_verified=true`, engineering
  policy eligible at that receipt. Artifact identity, MSVC 19.50 Release/static/LTO-off build,
  clean pinned peers, registered host, child affinity telemetry, A/A, 5%/10%
  known deltas, timer floor and placement controls all passed.

> **Verifier compatibility on 2026-08-29.** Commit `2995286c` added
> `TRACY_ONLY_LOCALHOST` to the global child sanitizer after this bundle was
> sampled and initially made the verifier stop on an in-place v1 policy change.
> The frozen contract fixes `HYLE_TRACE=OFF` and does not use the Tracy control.
> After environment-policy versioning, today's CLI deep-replays the unchanged
> manifest with `environment_policy_compatibility=LEGACY_COMPATIBLE_TRACE_DISABLED`,
> `paths_verified=true`, and `qualification=QUALIFIED`. See the
> [shared instrument rules](../../../developer/performance/instrument-v1.md).

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
| range advice/flush/lock | O(1) validation; progressive lock does no syscall inside the existing page coverage and one native call for a new delta; explicit quota adjustment is a cold query/set/retry path | relocking the old prefix, per-page metadata, or pretending OS-dependent residency work is a nanosecond user-space primitive |
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
- Progressive residency is compared only for the pinned Windows eager-lock
  lifecycle after one common process working-set setup. It does not rank Linux
  `MLOCK_ONFAULT`, page-fault/RSS outcomes, quota adjustment, partial unmap, or
  end-to-end model load/token latency.
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

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/platform-residency-p1-v2.json `
  --machine W1-WIN `
  --peer llama=C:\public\project-hyle\llama.cpp `
  --build C:\hyle-scratch\platform-v2\perf-qualify-build `
  --output C:\hyle-scratch\platform-v2\perf-qualify-evidence

python scripts/hyle-perf.py verify `
  --manifest C:\hyle-scratch\platform-v2\perf-qualify-evidence\manifest.json `
  --contract bench/contracts/platform-residency-p1-v2.json
```

`probe` only qualifies the instrument and estimates wall time. A new claim
requires an independent-process `qualify-platform` manifest and a successful
deep replay; hand-written nanobench tables are not accepted evidence. The
exact retained path under "Qualified evidence" documents the historical receipt
and exercises the explicit legacy-policy compatibility rule; verifying it does
not create a new sample or renew the receipt under today's environment-policy identity.
