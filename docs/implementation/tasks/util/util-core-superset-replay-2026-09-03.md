# util_core 超集/质因数回放（2026-09-03）

<!-- MODULE REPLAY RECORD: HISTORICAL -->

> Verdict: `CONFIRMED`. This is a product-non-mutating replay of an already
> `CLOSED` factor. Current product state remains owned exclusively by
> `docs/implementation/module-status.json`.

## Replay identity and budget

- reviewed round: `util-core-v2-capability-completion`;
- closure task: `docs/implementation/tasks/util/util-core-v2-capability-completion.md`;
- candidate: `d42d992cf99232669f60422f30fb3b50ec9ec4eb`;
- end: `12d123f2f567d05538d4d77cb99cfb4aef9ee2de`;
- product-identity anchor: `fff4f06aafe0808bcde71f15b7cfe86b59dc78a8`;
- replay source commit/tree: `47b800fed025e81b4998642360ef06ea064e1e3d` /
  `aa4c6758e4c2aa0964d33e04340764c0f50c6459`;
- reviewer/date: Codex / 2026-09-03;
- budget: 45 minutes total, at most one delivery correction, no new timing
  experiment unless product or relevant peer mechanism changed.

## Updated discovery inputs

The clean local references were fetched and fast-forwarded before the replay:

| Input | Current identity | Relevant change since v2 |
|---|---|---|
| llama.cpp | `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb` | no new Util-core task or semantic tier |
| llama.cpp vendored ggml tree | `b91bd1c0eadf24e7d56143f36942e1a838c1dd8b` | no new RNG/borrow/ownership/spin contract |
| standalone ggml | `d4716378882593333721eb33f153144b6885caf2` | unchanged |
| Folly | `eeab4246827eae9a11c0fa22529bff1a258aa83a` | selected Function/Random/ScopeGuard/spin paths unchanged |
| oneTBB | `843162db8cf4b2531a4a6147b6ea3fd860d44668` | selected spin/backoff paths unchanged |
| Boost | `cb94f33fcb13e90922905c2d8105508e71fa6db4` | selected smart/intrusive ownership paths unchanged |
| Abseil | `536d30bfea05a09c7561cbba749ca7f738d46821` | selected FunctionRef/nullability/Cleanup/random paths unchanged |

The v2 capability map already came from the accepted global factorization and
covered statistical RNG, lexical cleanup, intrusive strong ownership,
thread-confined ownership, short spin/hints, synchronous borrowed callables,
required-borrow vocabulary and diagnostic type identity. Every admitted `MUST`
remains `COMPLETE` or completely `ROUTED`; the updated inference and peer inputs
add no unowned lifecycle or semantic variant.

The seven owned public headers remain the complete surface. Relative to the v2
candidate, the only Util-core product-path diff is an `Allocator` to
`AllocatorRef` comment correction in `ref_counted.h`; API, layout, algorithm and
code generation are unchanged. The central edge remains
`util_core -> PUBLIC core`, and the Util façade still directly selects
`util_core` plus the compiled Util runtime.

## Dependency and consumer findings

- Mem and Sched continue to consume the short-lock/hint vocabulary through the
  declared direct factor edge.
- Platform callable adoption and the duplicate adaptive backoff inside the
  not-yet-rebuilt Container domain remain findings for those owner rounds; they
  do not require a broader Util-core API or a reverse dependency.
- Compute and Graph were inspected only as frozen demand sources and were not
  changed.

## Performance and delivery replay

- `python scripts/hyle-perf.py verify --contract
  bench/contracts/util-core-p1-v2.json --manifest
  C:/hyle-scratch/util-core-v2-perf-evidence/manifest.json` returned
  `QUALIFIED`, `paths_verified=true`, exact provenance and current environment
  policy compatibility.
- The frozen P1 product implementation and the relevant Boost/oneTBB mechanism
  paths are unchanged. The existing result therefore remains a valid bounded
  claim for the frozen Ref handoff and short-state-transition workloads; it is
  not a new all-operation, cross-platform or end-to-end SOTA certificate.
- Fresh MSVC 19.50 factor tests passed `2/2`, including all seven public-header
  self-contained checks and the structural codegen object.
- Fresh selected `hyle::util` source, relocated shared and relocated static
  consumers passed after correcting one stale package fixture in
  `build-util-facade-probe-fix-2026-09.md`.

The fixture had incorrectly required `container_intrusive` in a Util-only
package. It now requires `util_core` and asserts that the unrelated intrusive
factor remains absent. No product dependency or façade was changed.

## Six answers and verdict

1. **Third party:** no Util-core product dependency; standard library, compiler
   intrinsics and ISA instructions are the mechanisms. Peers remain evidence.
2. **Capability:** all admitted inference/general `MUST` journeys are complete
   or routed; updated references add no product trigger.
3. **Performance:** frozen P1 evidence remains qualified and the product and
   relevant peer mechanisms are unchanged. Broader SOTA claims remain forbidden.
4. **Native FFI:** clean; there is no public C schema, bridge, unmanaged export
   or FFI-only layout.
5. **Build/import:** factor isolation and source/static/shared Util journeys pass;
   the over-broad fixture assertion was fixed without changing the product graph.
6. **C++ API/docs:** task-oriented selection, ownership, lifetime, concurrency,
   failure and traps remain documented in `docs/user/util.md`; no benchmark seam
   appears in the public surface.

Verdict is `CONFIRMED`: no reopen trigger exists and `util_core` stays
`CLOSED`. The next product audit proceeds to the lowest unfinished Platform
factorization boundary.
