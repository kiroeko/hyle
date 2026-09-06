# Platform current performance evidence

> **Platform v10 terminal adjudication（2026-09-06）：**candidate
> `3c1ef2392d1f05c79edfe1c5d4ea5558bd917e3c` / tree
> `63b58d1051d86c4aecb1bf736ae84265e359a5f5` 新增 NUMA node memory fact。该 query 是低频控制路径，
> 三平台 correctness、结构复杂度与 source/static/shared 交付均 `PASS`。预冻结的旧 host-planning P1
> holdout 数值上 4/4 non-inferiority `PASS`，median ratio `1.020066--1.022953` 均有利于 HYLE；但 n=15
> 下 5% known-delta 的 median 虽在接受带内，其 99.26% CI 未完整落入接受带，导致 qualification
> `FAIL`。因此本轮 `platform=BLOCKED / engineering performance=INCONCLUSIVE`，不能把 diagnostic
> bundle 写成 current P1 PASS，也不能称为产品回退。旧 v8 qualified receipt 只保留为未变 host 路线的
> supporting history，不自动重绑定到 v10 candidate。完整解释、raw hashes 与精确恢复条件见
> [`platform-v10-numa-node-memory-facts`](../../tasks/platform/platform-v10-numa-node-memory-facts.md)。

> **Platform v8 current adjudication（2026-09-06）：**第二候选
> `c39df6f94c636ab8fa94cb700dd263763a42d680`、tree
> `37889346564f3a5cdc895f26c9d610883076ba02` 上，host v2 的产品点值和 controls 都通过，但收尾复核发现
> peer 没有返回完整 memory record，也没有等价的 `Result` 失败通道，故其 same-contract PASS 已撤销。
> 修正后的 `platform-host-p1-v3` 唯一一次 exact 资格化已经通过：4/4 strata 的 5% NI 均 PASS，全部
> controls PASS；3 行 PARITY，另 1 行因 HYLE 可能快过等价带上界而只判 exact INCONCLUSIVE。
> Platform-thread 本轮只改变 integrated-source provenance/package，产品
> Parker bytes 未变，P0 structural 为 `PASS`。Platform-fs 随后的受限 resume 轮保持第二候选、合同、
> Folly peer 和 5% margin 不变，只执行一次 fresh qualification；31 个独立进程下全部 controls 与 8/8
> 产品 strata 通过，结论为 exact `PARITY / NI PASS`。旧 r3/r4 仍保留为当时失格的 diagnostic，不与新包
> 拼接。恢复记录见
> [`platform-fs-v2-control-resume`](../../tasks/platform_fs/platform-fs-v2-control-resume.md)，完整产品轮见
> [`platform-v8-os-resource-contract`](../../tasks/platform/platform-v8-os-resource-contract.md)。

Platform v7 新增异构 cache-class snapshot、opened-file duplicate/info 和同步 explicit-offset
`RandomAccessFile`。这些路线按 `P0 / STRUCTURAL` 验收：cache discovery 是显式冷路径，file owner
仍是一字，单次 duplicate/info/read_at/write_at 各直接落一个 native control；旧 cached facts、
FileHandle 顺序 I/O 与其它未调用路径没有新增 branch、lock、heap 或 metadata。不可变产品候选 identity 为
`e892977bd2012841b3ee8a217c33fdd7ebdad260`，tree
`e585215602a3430906201edd7a10726b74498158`；同合同规则和三平台证据记录在
[`platform-v7-cache-class-and-random-access.md`](../../tasks/platform/platform-v7-cache-class-and-random-access.md)。

Platform v6 新增两组显式 P0 控制路线：process-startup priority/QoS，以及已打开 `FileHandle` 上的
size/resize/preallocate/whole-file locking。它们没有给一字 `FileHandle` 增加字段，也没有改变
未调用时的 process 初始化、CPU facts、thread/Parker、FileHandle open/read/write/close 或 directory
路径。下列 v3 P1 receipts 因而继续描述各自未变的 normal lifecycle；它们不能被外推为新增 v6
控制路线的计时结果。v6 candidate 为 `246bd182042f02c2b37457c21cea4ceebd8739c7`，tree
`28c386bd62fd0c619202fbcf4993e267a0fe54c8`；结构门、三平台结果和比较前置条件记录在
[`platform-v6-file-control-and-process-policy.md`](../../tasks/platform/platform-v6-file-control-and-process-policy.md)。

This report is the current engineering-performance evidence for the factorized
Platform product. It covers only the host, native-thread/wait, and synchronous
filesystem factors. File mapping, shared mapping, and residency are now owned by
Mem VM and are reported in
[`benchmarks-mem-vm.md`](benchmarks-mem-vm.md). Earlier Platform v1/v2 reports
remain historical Git records; they are not current owner evidence.

The immutable contracts are
[`platform-host-p1-v1.json`](../../../../bench/contracts/platform-host-p1-v1.json),
[`platform-thread-p1-v1.json`](../../../../bench/contracts/platform-thread-p1-v1.json),
and
[`platform-fs-p1-v2.json`](../../../../bench/contracts/platform-fs-p1-v2.json).
The closure records are
[`platform-v3-capability-factorization.md`](../../tasks/platform/platform-v3-capability-factorization.md),
[`platform-thread-v1-factor.md`](../../tasks/platform_thread/platform-thread-v1-factor.md),
and
[`platform-fs-v2-control-resume.md`](../../tasks/platform_fs/platform-fs-v2-control-resume.md).

## Current verdict

The current factor-specific verdict is `platform=BLOCKED`,
`platform_thread=PASS`, and `platform_fs=PASS`. Platform v10's product and
all four host non-inferiority rows passed, but its evidence bundle was not
qualified by the frozen 5% known-delta control; it therefore has no renewed P1
PASS. The older v8/v7/v6
structural results and v3 P1 receipts below remain supporting history; they are
not silently rebound to the new candidate. This is not a Formal, cross-machine,
all-topology, or module-wide SOTA certificate. Every numeric row has its own
same-contract statement and 5% non-inferiority gate; rows are never pooled into
a Platform score.

| current factor | contract / evidence | result |
| --- | --- | --- |
| host / Platform v10 | `platform-host-p1-v3`; `C:\public\project-hyle\_validation\platform-v10-numa-node-memory\perf-evidence` | bundle `DIAGNOSTIC`; product 4/4 NI `PASS` and ratios 1.020066--1.022953, but `delta5 FAIL`; engineering `INCONCLUSIVE` |
| thread | `P0 / closure-correctness-v1`; unchanged Parker state machine plus exact source/license/package audit | structural `PASS`; no new timing claim |
| filesystem | `platform-fs-p1-v2`; `C:\public\project-hyle\_validation\platform-fs-v2-resume-evidence-r1` | exact `QUALIFIED`; 8/8 `PARITY` and NI `PASS`; all controls `PASS`; ratios 0.987503--1.012026 flip with order/allocation/placement and do not show fixed loss |

The previous v8 host receipt at
`C:\public\project-hyle\_validation\platform-v8-perf-host-evidence-v3-r1`
was exact `QUALIFIED`, with 4/4 NI `PASS` and all controls `PASS`; its medians
favored HYLE by 1.48%--3.79%. It describes the unchanged host route, but remains
historical because candidate identity changed.

| v7 capability | tier / same-contract evidence | result |
| --- | --- | --- |
| heterogeneous cache classes | `P0`; OS class/relation/sysfs facts, M4 exact `hw.perflevelN.*` replay, bounded caller-buffered snapshot | `PASS`; M4 preserved P-core 16 MiB and E-core 4 MiB L2 where coarse getter returned 4 MiB |
| opened owner duplicate/info | `P0`; one native duplicate/query, one-word owners, rename/unlink identity and shared-cursor tests | `PASS` |
| blocking explicit-offset owner | `P0`; one native `pread/pwrite` or Windows offset `ReadFile/WriteFile`, caller buffer, no public cursor/lock/heap | `PASS` |

| factor | same-contract normal lifecycle | pinned peer | strata | result |
| --- | --- | --- | --- | --- |
| host | process-available CPU topology plus host-memory planning snapshot | oneTBB host-information route | 4/4 | `PARITY`; NI `PASS`; median ratio 0.980281--0.985920 |
| thread | 64 request/response handoffs between one persistent caller and worker on distinct same-class physical cores | oneTBB production binary semaphore | 8/8 | `WIN`; NI `PASS`; median ratio 20.189201--25.202667 |
| filesystem | open one warm 4 MiB regular file, completely read and checksum it, then close | direct OS route shaped by Folly `File` ownership | 8/8 | `PARITY`; NI `PASS`; median ratio 0.996192--1.010228 |

All contracts use Release, static consumption, LTO off, MSVC 19.50, independent
process samples, both execution orders, the applicable allocation orders, a
placement perturbation, A/A, known 5% and 10% deltas, and a timer-floor control.
Fifteen independent processes were sufficient under the frozen power rule; no
post-result escalation or API/layout change was made.

## Qualified receipts

The current Platform-fs receipt binds product candidate
`c39df6f94c636ab8fa94cb700dd263763a42d680`, tree
`37889346564f3a5cdc895f26c9d610883076ba02`, and Folly
`65749da412d92fe27d87ae45a8ef670720712858`. Its immutable
contract-lock/manifest/summary hashes are
`4324C5FD0A97E9B63D2868290B9988A175DBFA344254A52F0F2A79C40F50BE06` /
`6BC5AE79776C0A4B561326297F0910E00BB3C44CD9D405368369A1C18E655C14` /
`6AE27668D8DC940535E7F1495525D21F7AC67346E14DC26BFCDFAE4E551E9A19`.
The frozen `[15,31]` rule selected 31 independent processes without a second
bundle. A/A, both injected deltas, timer floor and placement all passed. All
eight product confidence intervals include parity and all lower bounds exceed
the reciprocal 5% non-inferiority limit. Five point estimates are below one and
three above it; the direction changes across execution, allocation and placement
strata, so no product change is keyed by those sub-material values.

The v3 rows below are retained historical receipts for the earlier candidate;
they remain useful supporting evidence but are not the current Platform-fs
identity.

The v3 P1 evidence product candidate was
`b1a526d6fdb05b6bb42c4cb041788959e74511de`, tree
`f808df6fcbd0b5549777e0c6939240a3feab9512`. The clean evidence source is
`28987e3118c74cc04276300f8baa20764495793a`, tree
`9242fc3d2dd31e2034a9d36d8f1d689cbc5f7eac`; the commits between these
identities add closure records and frozen benchmark contracts, not Platform
product changes.

| contract | peer identity | retained bundle | contract / manifest / summary SHA-256 |
| --- | --- | --- | --- |
| `platform-host-p1-v1` | oneTBB `843162db8cf4b2531a4a6147b6ea3fd860d44668`, tree `d69c763c4dcb956c88f363e75cbd0f2009cf3171` | `C:\public\project-hyle\_validation\platform-v3\perf\host-evidence-v2-28987e31` | `489d628076eb9703d21ed8c59b67aec8baa55f2d7cc77acc87ebc42cd89ddfa4` / `a8f3103070faa548e8baea48d425b3f427deca46b6345e14eef77306c52ec2df` / `8d2fb36e249d6c9ce20d25e188e504dc373cdba12f2a8e5c2dd38acf445ffa76` |
| `platform-thread-p1-v1` | oneTBB `843162db8cf4b2531a4a6147b6ea3fd860d44668`, tree `d69c763c4dcb956c88f363e75cbd0f2009cf3171` | `C:\public\project-hyle\_validation\platform-v3\perf\thread-evidence-28987e31` | `99b99efa810c059c7344792560b58da3832114e99782d975baedccffdacce62c` / `6c1e8a01b38ee13e13479a8cb6c4a86de99a43a0d5a642631bb8ed3d3491028b` / `6db19125b4cf4f3876a558c3ad0678f74f995d02d13a53f0d7ce15326f530341` |
| `platform-fs-p1-v1` | Folly `eeab4246827eae9a11c0fa22529bff1a258aa83a`, tree `37160ffb1aa3cbc124f0f958ac10d900aa3995e1` | `C:\public\project-hyle\_validation\platform-v3\perf\fs-evidence-28987e31` | `f6816dab82a38e2a259cd1ba7484fa7cfd98b16220fedcefaf9ae7081ef39d56` / `df451a58c499bcf7a3fd1212c96aaaa19e139e2c6308ed11d268e78bef45e2a0` / `bfcb55857898c281d2ec2bc06d58782a51966f0db590a8b001f1ee81120fad92` |

Every retained manifest currently deep-verifies as `QUALIFIED`. The worst
99.26% confidence lower bounds are 0.965179 for host, 17.180837 for Parker,
and 0.966165 for filesystem, all above the reciprocal 5% non-inferiority limit
of 0.952381.

The host snapshot costs about 9.2 microseconds versus about 9.1 microseconds for
the peer route in this environment. Parker completes roughly 2.38--2.98 million
handoffs/s versus 0.115--0.120 million/s for the pinned oneTBB route. The full
4 MiB filesystem lifecycle costs roughly 1.14--1.17 milliseconds on both
routes. These values describe the frozen workloads, not individual syscall
latency.

## Structural gates

| capability | accepted bound | forbidden hidden cost |
| --- | --- | --- |
| cached CPU/ISA/cache facts | one bounded initialization, then O(1) reads | repeated probing or target-wide unsafe ISA |
| live topology snapshot | O(available CPUs + reported relations), buffer-fill core | stale affinity/cpuset cache or nested O(ND) reconstruction |
| heterogeneous cache-class snapshot | macOS O(classes); Windows/Linux O(P log P + relation/cache input), stack-first scratch | flattening unlike profiles, guessing unknown facts, stale cache or per-task discovery |
| host-memory snapshot | O(1) bounded native/file queries | payload-sized allocation or scanning memory ranges |
| secure entropy | O(bytes/native chunk), complete fill or error | TLS PRNG state or silent insecure fallback |
| native thread and affinity | one native owner; snapshot O(mask words) | per-operation refcount or scheduler policy |
| address wait / Parker | O(1), one atomic word, one corresponding native wait/wake | polling or per-wait allocation |
| `FileHandle` short I/O | one-word owner and at most one native operation | hidden lock, heap, or second I/O |
| handle `duplicate` / `info` | one native owner/query control; original and duplicate stay one word | reopen-by-path, copied cursor metadata, refcount or hidden synchronization |
| `RandomAccessFile` short offset I/O | one-word owner and one native positional/offset operation | implicit seek+restore, payload copy, mutex/event/heap or exposed false cursor promise |
| random-access exact/all I/O | O(bytes/native transfer limit), checked offset progression | intermediate owning buffer or unchecked wraparound |
| `FileHandle` exact/all I/O | O(bytes/native transfer limit), caller-owned buffer | intermediate owning copy |
| directory visitor | O(entries), zero owning entry allocation on the common path | unconditional `stat` or collected path list |
| C++ visitor exception bridge | normal path keeps one callback per entry and no owning allocation; exceptional path closes then rethrows | exception crossing the compiled noexcept iterator or per-entry exception allocation |
| recursive mkdir path scratch | 4096-byte stack first; at most one checked allocation for a longer input | a library-fixed 4095-byte ceiling or per-component heap allocation |
| explicit Windows thread priority | two bounded native controls at thread setup; unconfigured route unchanged | per-task QoS work or changing all threads by default |
| explicit process-startup priority | POSIX one scheduling control; Windows one QoS plus one priority-class control; no default call or library state | realtime, silent privilege fallback, worker/task hot-path work or hidden auto-configuration |
| handle `size` / `resize` | one native query or EOF control; existing one-word owner and cursor unchanged | path re-resolution, hidden allocation/lock or data copy |
| handle `preallocate` | one size query plus bounded native allocation/EOF controls; physical cost may scale with requested growth/filesystem | calling sparse resize and claiming backing, per-write bookkeeping or permanent HYLE metadata |
| whole-file lock / try / unlock | one corresponding native lock operation per attempt, except documented interrupt retry; no HYLE lock state | polling, lock-file allocation, in-process mutex or per-I/O branch |

## Interpretation limits

- The host row compares complete dynamic planning snapshots. It does not compare
  a multi-query snapshot with one cached scalar and does not claim every
  Platform fact is timed.
- Parker's claim is the scheduler's persistent cross-physical-core mailbox
  topology. It is not a counting semaphore, multi-waiter fairness primitive, or
  same-logical-CPU ping-pong claim.
- The filesystem peer adapter uses direct native calls because the locked Folly
  source defines the relevant one-handle ownership but does not provide a
  portable buildable Windows benchmark library in this checkout. The identical
  open/read/checksum/close lifecycle and adapter boundary are part of the
  contract.
- Thread creation, affinity configuration, entropy, directory mutation, and
  durability are control paths covered structurally and by correctness tests;
  they are not inferred from the three numeric rows.
- Handle-vs-path resize, sparse resize-vs-preallocation, blocking-vs-try lock,
  advisory-vs-mandatory/distributed locking, and process-vs-thread priority are
  different contracts. They are `NOT_COMPARABLE`, even when the final byte
  count or enum looks alike; no such pair is used to claim a win or loss.
- Preallocation time includes filesystem allocation policy and requested
  growth; blocking lock time includes contention. Nanosecond ranking would
  measure those external semantics, not avoidable HYLE overhead. Correctness,
  native-call count, object shape and absence from old hot paths are the P0
  evidence for v6/v7.
- `FileHandle::read` and `RandomAccessFile::read_at` are different cursor contracts and are not ranked against
  each other. POSIX `pread` and Windows synchronous offset I/O also differ in native cursor effects; the public
  random-access owner removes cursor observability but does not claim cross-OS concurrent completion semantics.
- Cache-class discovery is compared to the same OS facts, not to cached `cache_info()`. The M4 16 MiB/4 MiB
  distinction is a correctness result, not a nanosecond win; caching the dynamic process view to manufacture an
  O(1) benchmark would return stale affinity/class counts and violate the product contract.
- W1-WIN is the engineering verdict machine. Linux and macOS correctness and
  sanitizer results are not averaged into Windows timing.

## Reproduce

Use the clean evidence source checkout and unique build/output directories:

```powershell
python scripts/hyle-perf.py validate --contract bench/contracts/platform-host-p1-v1.json
python scripts/hyle-perf.py validate --contract bench/contracts/platform-thread-p1-v1.json
python scripts/hyle-perf.py validate --contract bench/contracts/platform-fs-p1-v1.json

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/platform-host-p1-v1.json `
  --machine W1-WIN --peer onetbb=C:\public\project-hyle\oneTBB `
  --build <unique-build> --output <unique-evidence>

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/platform-thread-p1-v1.json `
  --machine W1-WIN --peer onetbb=C:\public\project-hyle\oneTBB `
  --build <unique-build> --output <unique-evidence>

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/platform-fs-p1-v1.json `
  --machine W1-WIN --peer folly=C:\public\project-hyle\folly `
  --build <unique-build> --output <unique-evidence>

python scripts/hyle-perf.py verify `
  --manifest <evidence>\manifest.json --contract <matching-contract>
```

`probe` only qualifies the instrument and estimates duration. A new engineering
receipt requires `qualify-platform`; a hand-written timing table is diagnostic.
