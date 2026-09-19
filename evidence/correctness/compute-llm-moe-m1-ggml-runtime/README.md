# M1-G shared ggml resource evidence

Candidate: `04368a397859f2d03e71ab005a5d66227a5f9a0f`.
Baseline: `9401082b116334ca6dc9cfbd4e426bf11c497e0d`.
Verdict: **PASS, P0 resource integration only**.

- [Identity and archive member hashes](identity.json)
- [Native validation and limitations](validation.json)
- [Raw commands, logs, original selector receipts and task scripts](raw.zip)
- [Unified Compute + Graph implementation LOC](implementation-loc.json)

## Executed scope

| Scope | Evidence in raw.zip | Result |
|---|---|---|
| Windows selected native/header/component gates and 15 products' source/shared/static/relocated SDK consumers | `task/selected-result.json`, `task/continued-result.json`, `selector/`, `task/continued-*.log` | PASS after the isolated test-link correction |
| Windows CPU static resources, official complete-source archive native execution, CPU/Vulkan OFF isolation | `task/windows-extra-result.json`, `task/source-package.json` and stage logs | PASS |
| Windows CPU+Vulkan SDK consumers and private-header/license closure | `task/windows-vulkan-delivery-result.json`, `task/windows-vulkan-delivery.log` | PASS |
| WSL CPU resource failures/concurrency and ASan+LSan, UBSan, TSan | `task/hyle-m1g-*-fixed.log`, `task/hyle-m1g-ubsan.log`, `task/hyle-m1g-tsan.log` | PASS at the recorded test-fixture candidate |
| WSL, Air and Mini CPU static native and all three SDK deliveries | `task/wsl-cpu-delivery.log`, `task/air-cpu-delivery.log`, `task/mini-cpu-delivery.log` | PASS |
| Air CPU+Vulkan SDK consumers | `task/air-vulkan-delivery.log`, `task/air-vulkan-payload.log` | PASS for all three actual GPU consumers |
| Air Vulkan native Provider/Compute/Graph and shared resources | `task/air-vulkan-native.log` | PASS, eight tests with explicit Vulkan ON and actual GPU test names checked |

## Corrections retained

The first POSIX fault fixture incorrectly counted backing-buffer destruction as
context destruction. Commit `2c1011cb` corrected only that fixture. The initial
failure remains in the bundle.

The original broad Windows selection at `c76c041c` had 29 PASS gates, one failed
Vulkan bridge test link and 84 unrun gates. Commit `04368a39` added the test's
direct dependency on the migrated metadata owner. Its 88-gate continuation
reran the affected gate and maintenance/configuration checks, then executed the
remaining gates. Prior PASS receipts retain their original tree identity;
they were not rewritten as current-tree cache hits.

The initial Air script reused a CPU-only native build cache. Its five native
tests and misleading final success marker do **not** prove Vulkan native
coverage. Its three fresh SDK consumers did enable and execute Vulkan. A
separate explicit-ON run completed the missing eight-test native suite; no
product changes or SDK rebuilds were needed for this correction.

## Structural cost and limits

Prepare creates the metadata context, CPU pool, plan and workspace. Repeated
execution calls the original ggml graph executor without HYLE Provider search,
timing or pool/workspace creation. Fault fixtures check allocation accounting,
partial-start rollback and exact thread joins using real threads. Original
CPU math/fusion/worker loops and the twelve retained Vulkan patch entries are
unchanged. P1 is not applicable to this resource card; no throughput or winner
claim is made.

These results cover a private resource seam and affected delivery closure.
They do not establish a complete CPU Provider, ordinary model lowering, any
new cell in the 64-by-four model coverage matrix, or the former C1 milestone.
Graph metadata/payload/allocator/callback owners must outlive their slot;
concurrent slots share only immutable data and a single slot cannot run/close
concurrently. Arena construction is restricted to checked HYLE lowering, not
arbitrary external ggml graphs.
