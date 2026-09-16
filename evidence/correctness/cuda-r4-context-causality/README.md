# CUDA primary-context progress: native reproduction and interventions

[Summary](summary.json), [artifact identities](identity.json), and
[raw captures, logs, source and recipes](raw-and-recipes.zip) preserve this
[bounded investigation](../../../docs/implementation/tasks/compute/cuda-r4-context-causality.md).
These are diagnostics; R4 product qualification remains `BLOCKED`, and P1 remains
`INCONCLUSIVE` until a sampling repair passes its declared progress checks.

## What is now established

The standalone program links only CUDA, Windows and the C++ runtime. It calls
the same inventory/device-count/driver-init sequence, then
`cuDevicePrimaryCtxRetain`. It does not link HYLE, Graph, Sched, Marlin or cuBLAS.
Unlike the earlier `cudaMalloc` probe, it covers the directly observed API.

The alternating batch ended after **89 launches: 45 native and 44 full FFN**,
because it obtained the two predeclared captures. Both captures were native:

| Observation | Native process 40 | Native process 88 |
|---|---|---|
| All threads' initial affinity | CPU2 / `0x4` | CPU2 / `0x4` |
| Caller current priority | 24 | 24 |
| Helpers' current priorities | 10–11 | 10–11 |
| Caller CPU time at capture | 4.453125 s | 5.09375 s |
| SDK helper CPU time at capture | 0 | 0 |
| Single intervention | Helpers allowed CPU2+CPU4; caller stays CPU2 | Caller relative priority lowered to -2; process stays CPU2 |
| Result | Retain returns, release and process exit succeed | Retain returns, release and process exit succeed |
| Exit after intervention | 3.313 s | 0.062 s |

The first retain returned about 1.06 seconds after the affinity intervention;
its release then took about 2.27 seconds. The second retain and release finished
within about 62 milliseconds after the priority intervention. The actual phase
history and call durations are in the raw records; no timeout was counted as PASS.

Both minidump stacks reach the same NVIDIA driver wait loop as the earlier full
HYLE fixture. At driver RVA `0x2e350e`, the code passes zero timeout to
`WaitForSingleObject`; the captured handle queried as a **Thread**. The caller
polls while waiting for separate SDK work to complete. Disassembly, import-table
resolution, phase markers and native call logs agree.

This supports **scheduling starvation caused by the single-CPU restriction and
the actual caller/helper priority difference** in this Windows/driver combination.
Windows schedules higher-priority ready threads first; a zero-timeout wait does
not block the caller. See the primary
[Windows scheduling contract](https://learn.microsoft.com/en-us/windows/win32/procthread/scheduling-priorities)
and [wait contract](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-waitforsingleobject).
NVIDIA documents primary-context retain as the shared driver/runtime context
mechanism; the chosen API is appropriate for library interoperability.
[NVIDIA primary-context API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html).

## Limits and next action

The component that raised the caller's current priority to 24 is not identified.
The two earlier uncaptured 120-second delays cannot be assigned this exact phase
with certainty. This does not establish a defect in every CUDA driver or GPU.

The native reproduction shows HYLE is not necessary to trigger the captured
problem. No source evidence calls for changing HYLE locks, Sched, primary-context
ownership or GPU algorithms. A GPU sampling contract must give native helper
threads an actual scheduling opportunity while retaining a pinned caller and a
fixed CPU budget. A justified sampling repair and its bounded regression come
before any new P1; the failed old samples remain excluded.

The thread-query helper was checked against Win32 priority changes: current
priorities 8/6/8 match requested relative values 0/-2/0 in a normal process.
The `BasePriority` field returned by this native information class is relative
on this host; the separate Toolhelp `base_priority` is absolute. The first
capture's `wait_timeout_100ns` value was read after the target resumed and is
stale; it is retained as raw data but excluded from inference. Timeout zero is
established by the call instructions and the second observation instead.

No product, driver, global machine policy, common statistics, margin or P1
sample budget changed. The complete FFN processes passed their built-in output
checks; this diagnostic does not repeat or replace the prior independent oracle.
