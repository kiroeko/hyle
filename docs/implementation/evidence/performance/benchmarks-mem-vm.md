# Mem VM performance evidence

This report owns current P1 evidence for the `mem_vm` factor introduced by the
Platform-to-Mem storage migration. It covers the affected model-file mapping
and progressive residency journeys only. It does not renew unrelated Mem v2
allocator, pool, buffer, or planner claims; those algorithms and their existing
evidence were outside this directed replay.

The immutable contracts are
[`mem-vm-mapping-p1-v1.json`](../../../../bench/contracts/mem-vm-mapping-p1-v1.json)
and
[`mem-vm-residency-p1-v1.json`](../../../../bench/contracts/mem-vm-residency-p1-v1.json).
The closure records are
[`mem-v3-vm-factorization.md`](../../tasks/mem/mem-v3-vm-factorization.md) and
[`mem-vm-v1-factor.md`](../../tasks/mem_vm/mem-vm-v1-factor.md).

## Current verdict

`PASS` for both frozen W1-WIN P1 engineering contracts.

| family | same-contract normal lifecycle | pinned peer | strata | result |
| --- | --- | --- | --- | --- |
| mapping | open/map a warm 16 MiB file, validate 64 page-distributed bytes, unmap/close | Boost.Interprocess `file_mapping` + `mapped_region` | 4/4 | `PARITY`; NI `PASS`; median ratio 0.977834--1.005194 |
| residency | grow one persistent mapping's locked prefix through 4, 8, 12, and 16 MiB, expose applied size, then unlock once | pinned llama.cpp Win32 `llama_mlock` mechanism | 8/8 | `PARITY`; NI `PASS`; median ratio 0.991597--1.001353 |

Both contracts charge the complete stated ownership lifecycle, use Release
static source consumption with LTO off, and pass execution/placement/allocation
order controls, A/A, known 5% and 10% deltas, and timer-floor qualification.
They establish no material wrapper loss for these normal inference-loader
shapes. They do not claim that HYLE makes native mapping or page-lock syscalls
faster.

## Qualified receipts

The product candidate is
`b1a526d6fdb05b6bb42c4cb041788959e74511de`, tree
`f808df6fcbd0b5549777e0c6939240a3feab9512`. The clean evidence source is
`28987e3118c74cc04276300f8baa20764495793a`, tree
`9242fc3d2dd31e2034a9d36d8f1d689cbc5f7eac`; intervening changes are closure
records and frozen benchmark contracts, not Mem product changes.

| contract | peer identity | retained bundle | contract / manifest / summary SHA-256 |
| --- | --- | --- | --- |
| `mem-vm-mapping-p1-v1` | Boost `cb94f33fcb13e90922905c2d8105508e71fa6db4`, tree `a937e0f99c16e55467464a22eeafc6f84acc974a` | `C:\public\project-hyle\_validation\platform-v3\perf\mapping-evidence-v2-28987e31` | `bc1cb4c6b6d29fbb54dac663e4955e92566ddc75dfead5850552eb3a9ad9e13e` / `3c0d00bf27ebd4de9e16d45e78206a858d2cdc6b0057d823e3db207d7472fb37` / `da197a28f4b5fd89377258db9d5845810ed440f9a253ad002b0891ade5a4875e` |
| `mem-vm-residency-p1-v1` | llama.cpp `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb`, tree `5d61cb84711c5ad70d6a3c213e45b949bcf7e3d7` | `C:\public\project-hyle\_validation\platform-v3\perf\residency-evidence-28987e31` | `77155a64c3be83d49004ec87b1b375493926650cfb6b66df6d266762556b429c` / `0bf159c851786802dbfe004f48f07dd465d727f3725a9f681bcb5dab5e80e3d9` / `9ae1135adfc5cab31b44ffec696e64be6dcc5f7ca203962d68a92219de7cdf51` |

Every retained manifest deep-verifies as `QUALIFIED`. The worst 99.26%
confidence lower bounds are 0.963487 for mapping and 0.965027 for residency,
above the reciprocal 5% non-inferiority limit of 0.952381. Residency costs
roughly 0.57--0.59 milliseconds per complete four-growth lifecycle on both
routes.

The mapping runner intentionally declares `measurement_batch=64` because each
callback validates 64 probes. Its reported op/s is normalized per probe, not a
claim that the complete map/probe/unmap lifecycle takes about two microseconds.

## Structural gates

| capability | accepted bound | forbidden hidden cost |
| --- | --- | --- |
| anonymous/file/shared mapping | one inline move-only native owner; one native mapping lifecycle | pimpl, refcount, virtual dispatch, or implicit copy |
| mapping access and checked ranges | O(1) validation and span construction | per-access allocation or path lookup |
| advice/prefault | O(range bytes/pages) only when explicitly requested | implicit full-file touch on ordinary mapping |
| progressive residency | O(newly covered pages/native range); existing prefix is not relocked | per-page persistent metadata or relocking old coverage |
| huge mapping / guarded stack | bounded native setup and one owner | scheduler policy or hidden global registry |

## Interpretation limits

- Mapping uses a warm page cache and deterministic distributed probes. It does
  not claim mmap is the best route for small sequential files.
- Residency covers the pinned Windows eager-lock lifecycle after one common
  process working-set preparation. It does not rank Linux on-fault behavior,
  RSS/page-fault outcomes, quota policy, sparse fragment release, or end-to-end
  token latency.
- The clean Boost peer checkout is
  `C:\public\project-hyle\_validation\peers\boost-cb94f33f`; the canonical
  Boost checkout had unrelated submodule changes and was deliberately not
  modified or used as qualified evidence.
- Single-node validation proves API cost and fallback behavior, not remote-NUMA
  bandwidth.
- Formal certification remains separate and pending.

## Reproduce

```powershell
python scripts/hyle-perf.py validate --contract bench/contracts/mem-vm-mapping-p1-v1.json
python scripts/hyle-perf.py validate --contract bench/contracts/mem-vm-residency-p1-v1.json

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/mem-vm-mapping-p1-v1.json `
  --machine W1-WIN `
  --peer boost=C:\public\project-hyle\_validation\peers\boost-cb94f33f `
  --build <unique-build> --output <unique-evidence>

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/mem-vm-residency-p1-v1.json `
  --machine W1-WIN --peer llama=C:\public\project-hyle\llama.cpp `
  --build <unique-build> --output <unique-evidence>

python scripts/hyle-perf.py verify `
  --manifest <evidence>\manifest.json --contract <matching-contract>
```

The repository's separate `benchmarks-mem.md` report covers broader historical
Mem families. This directed factor round does not rewrite or silently renew it.
