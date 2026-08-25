# Dependency upgrade impact ledger — 2026-08

This ledger binds the dependency refresh to real HYLE consumers and release
evidence.  The machine-readable authority is
[`../../cmake/HyleDependencyLock.json`](../../cmake/HyleDependencyLock.json);
this document records impact and required revalidation, not a second set of
CMake pins.

This work item is a dependency upgrade and project-health checkpoint, not a
module-closure pass. The previous closure decisions for
`core/error/platform/mem/time/obs/util/build-package` remain the design baseline
for unchanged scope. That does **not** mean this batch only changed downstream
dependencies: relative to the last aggregate closure checkpoint `6dcd038`, all
seven runtime modules have public-header and/or implementation changes, while
the build/package graph changed directly. Every changed seam is therefore
yellow until its API/ABI, correctness, affected performance and documentation
evidence is signed on one clean commit. Modules outside that strict baseline
receive only targeted validation for a directly changed upgrade seam; a
diagnostic performance gap there is backlog for the later module closure, not a
blocker for this project-health checkpoint. `compute/graph` remain half-finished:
their rows below require only the mechanical compatibility needed to keep the
existing component graph building and testing, not feature, API, architecture
or performance work.

## Actual HYLE touch matrix since `6dcd038`

This is a source-impact matrix, not a list of completed evidence. It prevents a
historical certificate from being reused over a changed public contract while
also preventing this upgrade task from silently becoming an all-module closure
pass.

| Scope | Actual public / implementation touch in this batch | Evidence that must be re-signed |
|---|---|---|
| core | C/C++ prefetch behavior, Clang unroll-hint conditions and intrusive-list callback/container-of contracts changed in public headers | C/C++ header consumers, supported compiler/ISA codegen, sanitizer behavior and any affected primitive point values |
| error | The public C status-code type became explicitly fixed-width and the C adapter/ABI checks changed | Pure-C and C++ behavior, symbol/surface and `-fshort-enums`/cross-compiler ABI; current benchmark values are separately stale because nanobench changed |
| platform | Public C/C++ thread, CPU-feature, filesystem and NUMA contracts changed together with their implementations, including versioned/FFI records and ownership/error paths | C/C++ API and ABI consumers, struct-prefix/unaligned FFI cases, cross-platform FS/thread/CPU/NUMA correctness, package consumers and performance of actually changed hot paths |
| mem | The C surface now shares platform NUMA types, versions region options and fixes flag widths; the mimalloc-v3 allocation/alignment/reallocation path and NUMA/huge-page adapters changed | C/C++ surface and ownership, v3 allocation/alignment/OOM/remote-free behavior, NUMA/huge-page contracts, shared/static/source consumers, sanitizers and allocator performance |
| time | Public `Instant` overflow behavior and C timer callback/reentrancy/cancel contracts changed with the clock/timer adapter implementation | C/C++ boundary and callback-lifetime tests, extreme-duration/overflow cases, cross-platform clocks, package consumers and affected timer/clock performance |
| obs | The C log record/level ABI, caller-storage sink, fiber tracing and Tracy text/plot adapters changed, together with the C implementation | 29-symbol C surface, short-record/unaligned FFI and sink lifetime, trace-OFF plus exact trace-ON static/shared shapes, package consumers and changed hot paths |
| util | Registry priority ABI, frozen-name overflow handling and demangle portability changed in public/implementation code | Fixed-width pure-C ABI, registry OOM/overflow and C++ consumers, package/sanitizer checks and affected registry performance |
| build/package | Dependency lock/provenance, component graph, FetchContent/offline/source-bundle and CI/package contracts changed directly | Fresh shared/static/complete-source delivery, modular installed consumers, offline inputs, all-component CPU graph and supported cross-platform/toolchain checks |
| container / simd / sched / io | Only upgrade-triggered seams are in scope: simdutf/hash/string, Highway/xsimd integrated paths, Boost.Context fcontext and nghttp2 HTTP/2 respectively | Targeted correctness, build/package and honest same-contract diagnostics only. Results do not certify a whole module; a loss is input to its later closure and does not by itself block this project-health checkpoint |
| compute / graph | Only mechanical dependency/API compatibility needed by the existing all-component build | Compile/test/package health of the existing adapters; no capability, public-shape or SOTA conclusion |

## Sibling-checkout audit snapshot

Observed under `C:/public/project-hyle` on 2026-08-25.  These local clones are
research inputs, not the definition of “latest”; several were behind the
official stable targets and therefore were not copied blindly.

| Checkout | Local HEAD / commit date / describe | Status and decision |
|---|---|---|
| ggml | `eced84c86f8b012c752c016f7fe789adea168e1e` / 2026-06-26 / `v0.15.3` | clean; old HYLE pin, superseded by official `v0.21.0` |
| nghttp2 | `a25ec3de429d038f08f334cd3dc3b637f05fe7bf` / 2026-07-05 / `v1.69.0-353-ga25ec3de` | clean; newer than old HYLE pin but not a stable release, so target official `v1.70.0` |
| glslang | `b53185b34e298b112153b630f1b49ed3a36fcfb2` / 2026-07-02 / `main-tot` | clean; arbitrary main snapshot, not mixed into the Vulkan SDK train |
| Vulkan-Headers / volk / googletest | no sibling Git checkout | use official tagged train/releases and verified codeload archives |
| mimalloc | `fef6b0dd70f9d7fa0750b0d0b9fbb471203b94cd` / 2026-04-29 / `v2.3.2-19` | clean; superseded by `v3.5.0` |
| simdutf | `282107c23196ddc6c7b8ef06567f81bbd056fc3c` / 2026-06-23 | dirty: untracked `singleheader/simdutf_c.h`; never a safe copy source, target official `v9.1.0` |
| xxHash | `e573d4d2aaeaba0f3e5a0a9a54144a1f2b4b56e7` / 2026-04-13 / `v0.7.4-1019-ge573d4d` | clean but no longer remote HEAD; superseded by `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53` |
| wyhash | `e4764a0b637d34d3421a7760affada9288b625a8` / 2026-03-23 / `wyhash_final4-26-ge4764a0` | clean; retained exactly |
| nanobench | `e4327893194f06928012eb81cabc606c4e4791ac` / 2024-10-06 / `v4.3.11-6` | clean; superseded by official `v4.6.0` |
| picohttpparser | `f4d94b48b31e0abae029ebeafcfd9ca0680ede58` / 2026-04-06 | clean; retained and newly made explicit |
| Boost superproject | `2813c94a3eefd1adbf623703f9c4d8e818f5398f` / 2026-06-28 / `boost-1.91.0-123` | clean, but not a standalone context provenance point; fcontext instead binds official Boost.Context `boost-1.92.0` commit |
| Highway | `83eff39371e5e6494b305b253f8a3c06f831faf4` / 2026-06-25 | clean historical benchmark pin; target newer official HEAD below |
| xsimd | `5d2490f515cf773f984da9ec0a1aec103d00d238` / 2026-06-11 | clean historical benchmark pin; target newer official HEAD below |
| Tracy | `0b120ba84f8e829619f7b3fbd95049c1ae08500d` / 2026-06-26 / `v0.13.1-1155` | clean but not latest stable; target official `v0.14.1` |

## Product and validation source inputs

| Dependency / class | Previous identity → current locked identity | Actual HYLE consumer and likely break surface | License / offline / install contract | Pre-release evidence gate |
|---|---|---|---|---|
| ggml / FetchContent | `eced84c86f8b012c752c016f7fe789adea168e1e` (`v0.15.3`) → `8599e0ea3756c4bac4ef813af2241cb1a8bbfb0b` (`v0.21.0`) | compute-05 `ggml_bridge`; private PIC static CPU + optional CUDA. Highest risk: backend/tensor APIs, target names, exported headers and upstream CMake options | MIT `LICENSE`; verified codeload archive online; offline/direct source accepts exact Git or the locked no-`.git` raw expanded tree; static SDK owns target/header/license closure | Existing host adapter and tests compile/run; static/shared source/package graph consumes the v0.21 target/options. CUDA/operator/ISA product completion is deferred with compute closure |
| nghttp2 / FetchContent | `68cb6900fde14c77f0cd7add0e094a862960eb99` (`v1.69.0`) → `85e300c79fb6dbcfa9c1013215c8710c1c2cd3d2` (`v1.70.0`) | io-11 C codec and HTTP/2 state machine; private `nghttp2_static`. Risk is C API/configured header and noisy optional-app probes, not transport ownership | MIT `COPYING`; archive/Git checks as above; static SDK exports core target and copied public headers | io-only configure/build; HTTP/2 codec/stream/flow-control tests; install + downstream static consume; prove app/HTTP3 dependencies stay absent |
| Vulkan-Headers / FetchContent | `b5c8f996196ba4aa6d8f97e52b5d3b6e70f7e4e2` → `e3b1eec08173d6b825cd3ac88c885a63b621504a` | compute-06 compile-time API and generated declarations | Apache-2.0 OR MIT license set; same archive/offline checks; installed SDK carries required headers | Existing adapter/layout assertions compile and package consumers resolve; no new compute API or capability work |
| volk / FetchContent | `d979819fd03de3a7606d6e23d6ff4968942599da` → `776893306c5d3b22b6185b5d4a258b81d94572bf` | compute-06 dynamic loader table; risk is generated symbol/table shape | MIT `LICENSE.md`; archive/offline verification; one private/exported closure as registered | Existing loader-present/absent tests and static/shared package smoke only; backend completion is deferred |
| glslang / FetchContent | `f0bd0257c308b9a26562c1a30c4748a0219cc951` → `168d452a4f460d24b588fed08477a81c44ee27a1` | compute test/benchmark shader compiler only; not production runtime | upstream multi-license `LICENSE.txt`; archive/offline verification; must not leak into production SDK closure | Shader compilation tests/bench configure; prove tests-off product does not fetch/export glslang |
| Vulkan train invariant | `vulkan-sdk-1.4.341.0` → `vulkan-sdk-1.4.357.0` | Headers + volk + glslang must move atomically | reader rejects a mixed version train | Mixed-train rejection, existing shader/loader tests and compile health; full real-driver compute certification is deferred |
| googletest / FetchContent | `b514bdc898e2951020cbdca1304b75f5950d1f59` (`v1.15.2`) → `063de7e9578f82b369302001269680b4b1553359` (`v1.18.0`) | test harness only; assertion/listener/discovery integration | BSD-3-Clause `LICENSE`; archive/offline verification; absent when tests are OFF | Configure/build/run aggregate tests; tests-off source/package shape |
| Tracy / external checkout | previously unconstrained `HYLE_TRACY_DIR` (local sibling was `0b120ba84f8e829619f7b3fbd95049c1ae08500d`) → `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c` (`v0.14.1`) | obs trace macros plus exactly one `TracyClient.cpp`; risk is public macro/API drift and multi-DLL singleton/export rules | BSD-3-Clause `LICENSE`; trace build requires exact clean Git HEAD/tree or exact official-archive expanded tree, plus license. Installed trace-enabled SDK copies Tracy public headers and license | Trace OFF zero-symbol/zero-dependency gate; trace ON static/shared compile; one-client symbol audit; install/downstream consumer; runtime viewer handshake |

The Vulkan three-way version is validated by the lock reader.  nghttp2 1.70
unconditionally probes application/HTTP3/doc dependencies before handling
`ENABLE_LIB_ONLY`; HYLE suppresses those probes inside the dependency block and
still keeps the protocol core and `Threads` contract.

## Checked-in source snapshots

| Dependency | Previous identity → current locked identity | HYLE use and upgrade risk | Required revalidation |
|---|---|---|---|
| mimalloc | `fef6b0dd70f9d7fa0750b0d0b9fbb471203b94cd` (`v2.3.2-19`) → `18b08671c9302247bfb682286e6bf3cc1773f801` (`v3.5.0`) | mem system allocator. v3 reorganizes arena/segment/page-map and thread/subprocess internals; public HYLE allocator ABI must stay opaque | Exact locked vendored-tree audit; mem allocation/alignment/OOM; cross-thread free/fiber migration; shared/static symbol surface; sanitizer and performance rebaseline |
| simdutf | `282107c23196ddc6c7b8ef06567f81bbd056fc3c` → `2d4e249dcfd978b39dd567b156f20a43db97377b` (`v9.1.0`) | text/Unicode amalgamation; ISA dispatch and error/result semantics may change | Normalized amalgamation tree/license audit; UTF validity/transcode corpus; all supported ISA compile/runtime dispatch; benchmark refresh |
| xxHash | `e573d4d2aaeaba0f3e5a0a9a54144a1f2b4b56e7` → `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53` (2026-07-27 dev HEAD) | container hash adapter uses the header-only XXH3 implementation; this is a dev snapshot, **not** the divergent `v0.8.3` tag | The four intervening commits change only upstream GitHub Actions. `xxhash.h` and `LICENSE` blob IDs remain exactly `dceb9216...`/`e4c5da72...`, so HYLE's vendored content/license hashes intentionally stay unchanged; known vectors/seed/streaming parity still gate the adapter and tag provenance must never be claimed |
| wyhash | `e4764a0b637d34d3421a7760affada9288b625a8` → unchanged | alternate hash implementation | Known vectors, seed behavior, scalar/ISA builds |
| nanobench | `e4327893194f06928012eb81cabc606c4e4791ac` (`v4.3.11-6`) → `b709728dba4df0356d467abfb0ba20781a8031c0` (`v4.6.0`) | benchmark harness only, but not point-value neutral: exact epoch/calibration, zero-duration handling, batch-relative accounting and performance-counter overhead changed | Record harness identity with every certificate; parser canaries cover relative and absolute tables; all benchmark targets compile; rebaseline every nanobench-derived current point claim before promotion; prove no product or install leakage |
| picohttpparser | previously unrecorded bytes (now proven to match `f4d94b48b31e0abae029ebeafcfd9ca0680ede58`) → same locked revision | io HTTP/1 parser snapshot | Parser corpus/fuzz/sanitizer; exact source and MIT license audit |
| Boost.Context fcontext | `d2142b69252150dafb5f6569b7eb9a42312a92a3` → `cc1b9e02cd530d6607becc984e1e5bfa4f7b3600` (`boost-1.92.0`) | sched assembly on six OS/arch ABI combinations | Byte/provenance audit; MASM/GAS compile; context switch, unwind/sanitizer and cross-thread scheduler tests on every supported ABI |

Every configure hashes each vendored directory as a sorted
`relative-path|SHA256(normalized-LF-content)` manifest and separately checks
locked license hashes.  A version-only edit, partial copy or local source drift
therefore fails before compilation.

## Benchmark rivals

| Rival | Previous pin → current pin | Contract |
|---|---|---|
| Highway | `83eff39371e5e6494b305b253f8a3c06f831faf4` → `239071390b18ca70f8fadde90ebefa6d8b12cb69` (2026-08-22 snapshot) | External benchmark/reference checkout only; not a product dependency |
| xsimd | `5d2490f515cf773f984da9ec0a1aec103d00d238` → `3b3151ff665514a00a437cc985f47b7de99b772b` (2026-08-22 snapshot) | External benchmark/reference checkout only; not a product dependency |

Published point values and ported-kernel provenance remain historical evidence
bound to their old rival pins.  They must not be relabeled; rerun all affected
same-ISA comparisons and establish new baselines before making current SOTA
claims.

The nanobench refresh independently invalidates point-value freshness even when
the product source and comparison peer did not move. Version 4.6.0 changes the
measurement/calibration implementation rather than only declarations. This
does not by itself reopen a module's functional/API closure, but every current
nanobench certificate must bind the harness revision, runner parser and runtime
filter/configuration, then be sampled again. SIMD, sched and IO historical rows
are therefore affected too. This invalidates old point values; it does not turn
the dependency-upgrade work item into a closure review of every benchmarked
module.

## Integrated product-source origins

The benchmark peers above are not drop-in product upgrades. HYLE's Highway
vqsort/bit-pack/math code and xsimd math code have been translated onto HYLE
D-tags, dispatch, masks, tails, error semantics and public contracts. The lock
therefore records two separate `integrated_source` origins:

| Integrated record | Origin | Current locked HYLE surface | Upgrade rule |
|---|---|---|---|
| `highway_port` | Highway `83eff39371e5e6494b305b253f8a3c06f831faf4` | vqsort port + dispatch, bit-pack port/adapters including the corrected u16 11/13 implementation, Highway math port/wrappers; Apache-2.0 plus the JPEG XL BSD-3-Clause `Cbrt` attribution | Do not replace from the 2026-08-22 peer wholesale. Diff exact upstream contrib paths, classify correctness/UB/portability changes, then port isolated changes through HYLE primitives and rerun sort/bit-pack/math oracle + ISA/perf gates. |
| `xsimd_port` | xsimd `5d2490f515cf773f984da9ec0a1aec103d00d238` | xsimd common/trigo math port and HYLE wrappers; BSD-3-Clause | Same rule: constants/polynomials/special-value branches may be candidates, but xsimd batch/type dispatch cannot replace HYLE's adapted interface wholesale. |

Each record hashes the declared adapted HYLE paths separately from the upstream
Git tree. This makes both origin and current fork bytes auditable without the
false claim that the port still equals upstream. A peer refresh triggers a
resynchronization review when upstream changes touch the corresponding contrib
path, fix correctness/UB/security/portability, materially improve the selected
algorithm, or invalidate a benchmark/oracle assumption.

The 2026-08-22 diff was reviewed commit-by-commit. Highway's full relevant diff
is broad (21 files, roughly +1700/-210): new f16/fp-arithmetic/gamma work,
base-layer changes and sort rewrites cross HYLE's shim and established baselines,
so wholesale resync is not low risk. Four scoped fixes are applicable and were
backported: `e7e66dd5` (`Atan2` negative-zero x), `1a4eb392` (`Expm1(-0)`),
`c9563e98` (`Sin(-0)`) and defensive heap `k=0` handling from `3ed31b26`.
`735d2064` fixes Highway Pack16<11>/<13>; HYLE's live product path already routes
those widths through its independently corrected `bitpack_hyle_u16-inl.h` with
exhaustive/layout tests, and that previously omitted file is now inside the
integrated content audit. xsimd's changed common-math area is mechanical C++17
traits cleanup; its large-argument temporary-buffer alignment fix does not apply
to the borrowed tanh D-tag path. No xsimd algorithm backport was justified.

The current Highway review also invalidated one old HYLE compiler assumption:
upstream explicitly marks AVX3 targets broken under the classic MSVC frontend,
and the hosted AVX-512 runner reproduced corruption in 64-bit, UInt128 and KV
vqsort cases. HYLE now keeps the process/product tier at AVX-512 while binding
only the sort module to its AVX2 TU under `cl.exe`; `runtime().sort.isa` and the
lower-tier feature bit expose that choice. clang-cl and GNU-like frontends retain
the AVX-512 sort TU. Focused MSVC fat public-sort tests and MSVC/clang-cl exact
artifact audits pass; execution on an AVX-512-capable hosted runner remains a
final CI gate for the new checkpoint.

JPEG XL is not an unresolved or invented third checkout here. The locked Highway
origin's `hwy/contrib/math/math-inl.h` carries the JPEG XL copyright immediately
above `Cbrt`; HYLE preserves that notice. `highway_port` therefore requires and
installs both Highway and JPEG XL license texts from one exact wrapper origin.

## Environment SDK boundary

OpenSSL, CUDA Toolkit and the Vulkan loader/ICD/driver are discovered host SDKs
or runtimes.  Their identity is a toolchain/build-environment fact, so the source
lock does not invent archive or Git pins for them.  Release configurations must
fix the feature state (`ON` or `OFF`), record the version actually found, use an
approved offline package/sysroot where applicable, and exercise both capability
and required-missing failure paths.  Tracy is deliberately different: its C++
source is compiled into HYLE, so it belongs in the source lock.

## P1 source-override and source-bundle impact

The lock schema is now version 3. Every `fetchcontent` record and the Tracy
`external_checkout` record adds an `expanded_tree_sha256` beside the archive
SHA-256.  The archive digest proves downloaded transport bytes; the expanded
digest proves the post-extraction directory/file/safe-link shape.  Its canonical
manifest hex-encodes paths and original portable link targets, hashes raw file
bytes, includes empty directories, rejects control/list-delimiter paths and
rejects absolute or root-escaping links.  Times, ownership and Unix mode stay
outside this cross-platform expanded identity and remain covered by the archive
digest.

This closes two release gaps without changing module C++ APIs: a direct
`FETCHCONTENT_SOURCE_DIR_*` can no longer bypass central provenance, and a
strict source bundle no longer needs to ship `.git`.  Exact Git remains the
stronger developer-checkout proof (commit + root tree + no tracked, untracked
or ignored additions). The
full tree scan is paid only for supplied expanded overrides/offline sources;
normal online FetchContent continues to use the archive `URL_HASH` gate.  A
nonmatching internal source requires explicit preload-only unsafe opt-in plus
independent provenance; it is not an ordinary release input and must be
reproved at pre-release readiness.

Schema 3 also adds `integrated_source`, class-specific license roots and an
adapted-content hash. Highway/xsimd port licenses are therefore consumed by
their own locked package registrations rather than an unlocked shared escape.
Duplicate adapted paths are rejected rather than silently de-duplicated, and a
plain cache-variable override cannot replace the central lock; malformed-lock
fixtures must attest one exact path from already executing trusted CMake code.

## Evidence captured in this change

Evidence below binds the named intermediate snapshots, not whatever happens to
be in a later dirty working tree. A subsequent implementation/API/package edit
invalidates the affected edge until rerun; only
[`pre-release-readiness.md`](./pre-release-readiness.md) may promote the final
committed tree to a current release certificate.

- JSON/schema projection and all seven normalized in-tree tree/license audits:
  pass via `cmake -P cmake/HyleDependencyVersions.cmake`.
- Both integrated-source path/content/license audits pass; changing an adapted
  file now disagrees with the lock even though it remains ordinary HYLE source.
- xxHash was rechecked against official dev HEAD `c0b5ea995d66`: the four
  commits after `e573d4d2` modify only upstream GitHub Actions, while the
  vendored `xxhash.h` and `LICENSE` blob IDs are byte-identical. The new commit,
  root tree and codeload SHA are locked; the unchanged vendored tree/license
  audit and focused Windows container/hash 5/5 tests pass.
- Ordinary configure audits only the in-tree snapshots consumed by the selected
  mem/container/io/sched/benchmark component graph, so core-only partial source does not
  inherit unused vendors. The selected-missing negative and unused-missing
  positive pass; complete-source packaging runs the full audit before archive.
- All six product/test FetchContent archives and Tracy match locked archive
  SHA-256. Fresh Windows extraction produced the seven locked raw expanded-tree
  hashes; ggml's safe relative symlink is preserved and included by original
  target spelling.
- Every product/test `FetchContent_MakeAvailable` call goes through the locked
  wrapper, including GoogleTest and test-only glslang. Direct exact Git and
  no-`.git` expanded overrides pass; modified/added/deleted files, added empty
  directories, link-spelling tamper and escaping symlinks fail.
- Strict offline CMake contract: exact Git and exact expanded positives plus
  unknown name, wrong HEAD/tree, tracked/untracked/ignored dirt, externally-
  attested link-free escape with bad-license rejection, unsafe mismatch with
  any symlink rejection, source-free/no-`deps` positive and selected-missing
  negative all pass. A direct override cannot self-enable the unsafe escape.
- Central lock contracts reject a plain `-DHYLE_DEPENDENCY_LOCK_FILE` swap,
  wrong class license roots, cross-vendor/integrated license paths, duplicate
  integrated paths and adapted-content mismatch; each failure has a negative
  fixture that checks the specific diagnostic.
- Package registry lock closure consumes one runtime asset per locked license
  path as a multiset. A single asset cannot satisfy two dependencies merely
  because their license bytes/hash are identical; the duplicate-hash negative
  contract passes. Extra assets in a locked registration, unsafe destination
  traversal and non-basename rename are also rejected.
- A sustainable Linux static+shared package gate is now defined for locked
  ggml + nghttp2 + Vulkan-Headers/volk, using a lock-keyed archive cache and
  `-j2`. Both shapes prove install/downstream features, licenses, a live HTTP/2
  SETTINGS exchange and host `GgmlBackend`; static requires its private closure
  targets/headers while shared rejects their leakage. Both reject glslang
  leakage. Its first hosted CI result remains release evidence to capture, not
  a locally claimed pass.
- An earlier Windows upgrade-stage snapshot configured all components with
  tests, ggml, HTTP/2 and Vulkan and passed 68/68 tests. Subsequent shared-tree
  product changes require the final Windows rerun; this historical result is not
  promoted to current-tree green evidence.
- Tracy `v0.14.1` exact checkout: trace-ON shared obs Release build passes; the
  prior sibling HEAD is rejected by the intended HEAD-mismatch gate.

These results are not a cross-platform release certificate. Final current-tree
Windows, WSL, macOS, sanitizer, package/source-delivery and upgrade-affected
benchmark evidence remain bound to the gates above. CUDA and full real-driver
Vulkan certification stay with the unfinished compute module rather than being
silently pulled into this dependency-health task.
