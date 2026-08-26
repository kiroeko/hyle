# scripts/

Project automation: environment setup and reproducible builds. The goal is that a
**fresh checkout on any machine** can build, test, and benchmark HYLE without
depending on a hand-crafted directory layout.

## `loc-report.py` — canonical module-closure LOC

Every module closure reports the same four `code` counts: all repository code,
HYLE-written native code with tests, HYLE-written native code without tests,
and the current module implementation. Run it from anywhere inside the checkout:

```bash
python scripts/loc-report.py --module core --self-check
python scripts/loc-report.py --module core --commit <candidate-commit> --self-check
python scripts/loc-report.py --module mem --json
python scripts/loc-report.py --module container_intrusive --self-check
```

Without `--commit`, the snapshot is the current content of Git-tracked files.
Modified/staged files are therefore counted and reported as dirty; untracked files
are not counted but also make the report dirty. `--commit` instead asks Git/`cloc`
to materialize and count that immutable tree, which is the required form when
checking a recorded terminal LOC snapshot. The repository total includes every language `cloc`
recognizes, including tracked docs, benchmarks, and in-tree third-party code.
The two self-written totals count only `C++`, `C/C++ Header`, `C`, and `CUDA`
under `include/hyle`, `src`, and (for the first total) `tests`; consequently
docs, benchmarks, samples, third-party code, CMake, and build/cache trees do not
enter those totals. A module count uses the same native language policy and
excludes tests, benchmarks, docs, and CMake.
In JSON schema 2, each metric's `files` is the number of files actually counted
by `cloc`; `selected_files` is the wider tracked-path selection before language
filtering. This keeps a selected `CMakeLists.txt` from being mislabeled as a
native implementation file.

`core`, `container-intrusive`, `util-core`, and `util-runtime` have explicit
logical boundaries in the script. Other native modules use
`include/hyle/<module>`, all matching `include/hyle/c/<module>*.h`, and
`src/<module>`. If a future closure unit does not fit either shape, repeat
`--module-path REPO_PATH`; the exact override is printed in the report so the
certificate cannot hide its counting boundary.
`--self-check` also verifies the fixed core boundary and the expected subset
relations. `cloc` and Python 3 must be available on `PATH`.

## `check-docs.py` — documentation contract

Runs without third-party packages. It enforces the root agent entry and four
human-documentation roots, rejects retired directories/C-language user guides
and checks relative Markdown links across the repository documentation surfaces:

```bash
python scripts/check-docs.py
```

The Linux baseline CI runs the same command. Content correctness still requires
module review; this script prevents structural and navigation drift.

## Module closure and performance evidence v1

`check-module-closure.py` validates the canonical current-state registry at
`docs/implementation/module-status.json` and its generated human projection.
Task, architecture, user, and performance pages may describe historical facts,
but they must link to that registry instead of declaring a competing current
`AUDITED`, `CLOSED`, or `PRODUCT LOCKED` state.

```bash
python scripts/check-module-closure.py
python scripts/check-docs.py
```

The first command is the fast editing loop. It validates the current registry's
schema, paths, component set, and generated status projection. It is an
engineering consistency check, not a Git anti-tamper system; use `--previous`
only for an explicit before/after state comparison. Candidate/tree/receipt checks
use ordinary read-only Git commands in the module review. CMake dependency and
delivery semantics stay with the real build/package contract tests rather than a
second parser here.

`scripts/hyle_perf/` is the shared schema/statistics/evidence/machine core for
performance instrument v1. It owns contract locking, canonical observations,
`advantage_ratio` (greater than one always favors HYLE), the independent
comparability/qualification/exact-performance axes, non-inferiority, manifest
binding, and registered-machine probes. Module integrations may only supply
workload/row mapping and genuinely module-specific artifact validators; they do
not fork the estimator, margin, controls, verdict, or stopping rule. The normative
protocol is
[`instrument-v1.md`](../docs/developer/performance/instrument-v1.md), and machine
roles are registered in `scripts/perf-machines.json` and explained in
[`machines.md`](../docs/developer/performance/machines.md).

`hyle-perf.py` is now the single v1 orchestrator. The first mappings are
`bench/contracts/container-intrusive-p1-v1.json` and
`bench/contracts/error-p1-v1.json`; their implementation history and remaining
qualification work are tracked in
[`closure-system-v1`](../docs/implementation/tasks/build/closure-system-v1.md).
The contracts and runner existing does not close either module: neither pilot
currently has a candidate-bound engineering `QUALIFIED` manifest.

Validate the immutable contract before building:

```bash
python scripts/hyle-perf.py validate \
  --contract bench/contracts/error-p1-v1.json
```

`probe` configures/builds the declared targets but permits a dirty candidate,
dirty peer and a reusable build tree. It runs exactly one fresh child process
per required product stratum and one per control order, always emits
`DIAGNOSTIC`, and returns no performance qualification regardless of attractive
ratios. The output directory must be absent or empty; the peer checkout must
still be at the exact revision pinned by the contract:

```bash
python scripts/hyle-perf.py probe \
  --contract bench/contracts/error-p1-v1.json \
  --machine W1-WIN --peer abseil=../abseil-cpp \
  --build .scratch/perf-probe-error-build \
  --output .scratch/perf-probe-error-evidence
```

Every emitted bundle can be replayed while all referenced files are retained.
`verify` strictly parses the manifest and contract lock; rehashes every file;
reparses the exact raw-log index, child telemetry and nanobench rows back into
observations and controls; recomputes an n=15 escalation decision; independently
replays the actual CMake cache, compiler/linker and product artifacts; and
rebinds the registered host gate. Every launched child reports a unique
`run_nonce`, so copying one process log into multiple sample slots is rejected.
A diagnostic manifest may verify successfully; integrity and product
performance are intentionally separate axes:

```bash
python scripts/hyle-perf.py verify \
  --contract bench/contracts/error-p1-v1.json \
  --manifest .scratch/perf-probe-error-evidence/manifest.json
```

`qualify-platform` requires clean candidate and peer checkouts, an absent/empty
fresh build tree, Release, all required controls/strata, stable inputs/host/
artifacts and passing validators. It always starts with n=15. A complete n=31
replacement run occurs only when the contract predeclares it, every required
control passed, at least one comparable non-inferiority gate is `UNRESOLVED`
solely for statistical power, and all other such gates are `PASS`:

```bash
python scripts/hyle-perf.py qualify-platform \
  --contract bench/contracts/error-p1-v1.json \
  --machine W1-WIN --peer abseil=../abseil-cpp \
  --build .scratch/perf-qualify-error-build \
  --output .scratch/perf-qualify-error-evidence
```

CPU v1 runs A/A, 5% and 10% known-delta controls in both orders. Its timer floor
is matched to each product executable's declared `measurement_batch`; the
faster side must be at least 20 times the most conservative same-batch floor.
The 20x check is a harness-qualification guard, not a 5% product margin and not
a switch to relax when a sub-nanosecond operation fails. The error plumbing
pilot exposed several batch=1 rows at only about 1.5x--9.8x floor. Those ratios
remain harness-dominated diagnostics until a consumer-valid replay or another
instrument passes the gate.

Before every child, the runner clears ambient `HYLE_BENCH_*`/`HYLE_PERF_*`,
sanitizer/profiler, loader, allocator and affinity-tuning variables; it fixes
common math runtimes to one thread and then applies only contract-reviewed
environment additions. This is a sanitizer, not a fully empty environment:
ordinary variables remain and the evidence stores a `PATH` hash rather than raw
environment contents.

The nanobench adapter v1 accepts static linkage only and verifies the actual
CMake cache has `HYLE_STATIC=ON`. This is an instrument limitation while shared
runtime dependency closure is unimplemented, not a restriction on HYLE's
static/shared/source delivery shapes. It also binds actual generator,
architecture, configuration, ISA/LTO, compiler/linker hashes, SDK/stdlib and
product artifact hashes.

The v1 bundle is unsigned mechanical evidence, not an authenticity proof
against an author who can coherently regenerate every raw file and manifest.
It also references live build/toolchain files: rebuilding them makes later
verification fail closed. Engineering use is valid while those references are
retained; Formal release remains unavailable until a trusted runner writes a
content-addressed immutable archive (and signing/provenance where required).

`qualify-platform --formal` is deliberately stricter: the machine registry
must already bind a matching stable host identity and independent hardware, and
the contract profile must already pin exact compiler/linker ID, version, binary
hash, SDK and stdlib values. Observed values cannot self-certify as expected.
W1-WIN has a registered host identity; W1-WSL is not independent, M1-MAC is not
yet identity-registered, and L1-LINUX is unprovisioned. The two pilot contracts
also do not yet pin `formal_toolchain`, so Formal currently fails closed by
design. `certify` only composes already-Formal manifests and rejects reused
physical hosts or mismatched candidate/contract/peer identities.

Old raw results therefore remain diagnostics or historical evidence; the
presence of a module-specific runner is not a module engineering `PASS`.

## `hyle-windows.ps1` — select an installed Windows C++ environment

CMake may know the VS 2022 generator but not a newer installed Visual Studio generation.
For Ninja, this wrapper asks `vswhere` for the latest installation with the x64 C++ workload,
imports its transient `VsDevCmd` environment, then invokes the `windows-ninja` presets:

```powershell
scripts/hyle-windows.ps1 -Action Configure -Preset windows-ninja
scripts/hyle-windows.ps1 -Action Build -Preset windows-ninja --parallel 2
scripts/hyle-windows.ps1 -Action Test -Preset windows-ninja --output-on-failure
scripts/hyle-windows.ps1 -Action ComponentContract -SmokeRoot C:\hyle-scratch\component
scripts/hyle-windows.ps1 -Action Workflow -Preset windows-ninja-check
scripts/hyle-windows.ps1 -Action PackageSmoke -SmokeRoot "$env:TEMP/hyle-package-smoke"
scripts/hyle-windows.ps1 -Action ProductShapeSmoke -SmokeRoot "$env:TEMP/hyle-shipping-smoke"
scripts/hyle-windows.ps1 -Action SourcePackageSmoke -SmokeRoot "$env:TEMP/hyle-source-smoke"
```

The same imported Windows SDK/MSVC ABI environment can drive clang-cl in a separate tree. Install
Clang 18+ first (for example the Visual Studio **C++ Clang tools for Windows** component) and make
sure `clang-cl` is discoverable on `PATH`:

```powershell
scripts/hyle-windows.ps1 -Action Configure -Preset windows-clangcl
scripts/hyle-windows.ps1 -Action Build -Preset windows-clangcl --parallel 2
scripts/hyle-windows.ps1 -Action Test -Preset windows-clangcl --output-on-failure
```

For an IDE solution, the same wrapper enumerates side-by-side VS 2022 MSVC toolsets and passes the
highest installed 14.43+ version explicitly. CMake's Visual Studio generator otherwise follows the
VS default v143 pointer, which may be older than another installed toolset:

```powershell
# Native VS 2026/v145 solution (CMake 4.2+).
scripts/hyle-windows.ps1 -Action Configure -Preset windows-vs2026 -DHYLE_COMPONENTS=core

# VS 2022 fallback; dynamically selects the highest supported v143.
scripts/hyle-windows.ps1 -Action Configure -Preset windows -DHYLE_COMPONENTS=core
scripts/hyle-windows.ps1 -Action Build -Preset windows --target hyle_core_tests --parallel 2
# VS 2026: build-windows-vs2026/hyle.slnx; VS 2022: build-windows/hyle.sln
```

It installs nothing and does not persistently change the caller's environment. A direct
`cmake --preset windows` is also valid when the VS default toolset already satisfies MSVC 19.43+.

## Build/package verification automation

`.github/workflows/build-system.yml` runs on every pull request, every push to `main`, and manual
dispatch. Every event gets the all-component deterministic CPU baseline on Windows/Linux/macOS,
Linux component/source-embed contracts, Linux production/package/complete-source checks, an
all-component Linux AArch64 cross build, a selected-module Windows clang-cl strict gate, and one
Linux locked-feature job for HTTP/2/ggml/Vulkan adapters, static/shared SDKs, and Tracy consumers.
Windows/macOS production-layout and installed-package checks run only on `main`/manual events; they
do not repeat complete-source/source-embed. This is a current-graph regression, not an acceptance
certificate for every module. TLS/CUDA and real Vulkan loader/driver/device matrices remain separate
periodic/release checks; the hosted workflow does not claim them.

The three smoke actions above import the selected MSVC environment and keep their work under the
explicit scratch root. `PackageSmoke` builds shared/static CPU packages and runs focused mem plus
package-driven per-component target consumers. `ProductShapeSmoke` runs accepted roots with production test layout.
`SourcePackageSmoke` requires a clean Git commit, creates the complete-source archive, verifies its
digest/manifest/legal payload, extracts it, and builds the all-component CPU graph without using the
checkout. The stable validation layers and module handoff rules are documented in
`docs/architecture/build-system.md`.

## `container-intrusive-codegen-audit.py` — intrusive-list structural gate

The `container_intrusive` CTest graph builds a maintenance-only noinline object
and passes it to this script. The script runs a native operation/layout probe
in every ordinary configuration, locates all stable probe symbols in MSVC COFF,
ELF or Mach-O, and applies the strict optimizer checks in NDEBUG builds. Those
checks lock the 64-bit hook/list/iterator footprint at 16/16/8 bytes, reject
allocation or out-of-line helper calls, require `node_of` to remain a constant
hook-offset adjustment, and compare the distinct-owner `HYLE_ASSUME` probe with
a same-signature constant control. It intentionally does not gate absolute
instruction counts: benchmark acceptance and constant-factor review remain
separate evidence.

Run it through CTest so the configured compiler, architecture, object and
runner stay bound together:

```bash
ctest --test-dir build --output-on-failure \
  -R hyle_container_intrusive_codegen
```

`cmake/toolchains/linux-aarch64-gcc.cmake` plus the `linux-aarch64-cross` preset provides the
whole-project cross-build entry. It never runs a target executable during configure; the SVE/RVV
QEMU scripts below remain ISA-specific product tests rather than substitutes for CMake migration.

## `fetch-bench-vendors.{sh,ps1}` — fetch pinned SIMD benchmark rivals

Every `HYLE_BUILD_BENCH=ON` configuration needs external, exact-revision
`highway/` and `xsimd/` checkouts. SIMD scaling/parity harnesses compile those
headers even when their optional comparison rows are off. Product code under
`include/` + `src/` has **zero** in-tree Highway/xsimd (simd-20 X4); only
benchmarks may consume these external checkouts.

```sh
# Windows Git Bash: default C:/public/project-hyle
scripts/fetch-bench-vendors.sh

# Windows PowerShell: same default
powershell -ExecutionPolicy Bypass -File scripts/fetch-bench-vendors.ps1

# macOS: default ~/vendor
scripts/fetch-bench-vendors.sh

# Linux/WSL: choose an ext4 directory explicitly
scripts/fetch-bench-vendors.sh --dir "$HOME/vendor"
```

The scripts read `cmake/vendor-versions.cmake`, a compatibility projection that loads the sole
dependency ledger `cmake/HyleDependencyLock.json`; pins must be changed in the JSON, not in the
projection. The fetchers refuse to
overwrite a non-Git or dirty checkout, and leave detached checkouts at the
recorded revisions. CMake independently rejects absent, mismatched, or dirty
checkouts. Pass the same root via `-DHYLE_BENCH_VENDOR_DIR=<root>`; Windows and
macOS have matching defaults, while Linux/WSL intentionally requires that
explicit path.

## `check-no-vendor.sh` — product zero-Highway/xsimd gate (simd-20 X4)

Fails if `include/` or `src/` `#include` real vendor headers (`hwy/...`,
`xsimd/...`), if `third_party/{highway,xsimd}` still exist, or if product CMake
still points at those in-tree paths. Run after any simd vendor-port change:

```sh
scripts/check-no-vendor.sh
```

## `setup-bench-libs.sh` — fetch opt-in peer references

HYLE performance work compares a frozen capability contract against strong,
real references (abseil, folly, mimalloc, marl, moodycamel, oneTBB,
crossbeam-deque, wyhash) — not just `std`. These heavy, opt-in rivals are not
vendored, so this script clones them. Highway/xsimd use the separate pinned
fetcher above; simdutf and xxHash remain in `third_party/`.

```sh
scripts/setup-bench-libs.sh              # clone all into the repo's parent dir
scripts/setup-bench-libs.sh --shallow    # --depth 1 (faster)
scripts/setup-bench-libs.sh marl oneTBB  # only the named ones
HYLE_BENCH_DEPS_DIR=/custom/path scripts/setup-bench-libs.sh
```

- **Idempotent**: a lib that already has a `.git` is skipped, never clobbered.
- **Bootstrap, not a pin**: this helper does not move an existing checkout and
  therefore cannot prove freshness. Before a formal comparison, deliberately
  select a current upstream revision, keep the checkout clean, and pass it to
  `bench-stat.py --peer NAME=PATH`; the provenance snapshot records the exact identity.
- **Default location** = the repo's parent dir (`$repo/../`), which is exactly
  where the gated-bench CMake defaults (`HYLE_<LIB>_DIR`) look. So after running
  it with no args, every `-DHYLE_BENCH_*=ON` works with **no extra `-D`**.
- **Custom location**: set `HYLE_BENCH_DEPS_DIR`, then pass the dir to the build:
  `-DHYLE_MARL_DIR=$HYLE_BENCH_DEPS_DIR/marl` etc. This name is deliberately distinct from CMake's
  `HYLE_DEPS_DIR`, which owns one build tree's FetchContent population.

## Running opt-in peer comparison benchmarks

Product builds have no external benchmark dependency. Any `HYLE_BUILD_BENCH=ON` build first needs the pinned Highway/xsimd checkouts above; individual comparison rows remain opt-in:

| flag | compares | exe |
|---|---|---|
| `-DHYLE_BENCH_ABSL=ON` | SwissTable vs `absl::flat_hash_map`; SmallVector vs `absl::InlinedVector` | `hyle_absl_bench`, `hyle_container_misc_bench` |
| `-DHYLE_BENCH_MOODYCAMEL=ON` | queues vs moodycamel | `hyle_queue_sota_bench` |
| `-DHYLE_BENCH_TBB=ON` | queue/vector vs oneTBB | `hyle_queue_sota_bench` |
| `-DHYLE_BENCH_RIGTORP=ON` | BoundedQueue vs `rigtorp::mpmc::Queue` (same-contract strict-bounded MPMC) | `hyle_queue_sota_bench` |
| `-DHYLE_BENCH_HIGHWAY=ON` | simd vs pinned external Highway; dtype vs Highway PromoteTo/DemoteTo; math vs Highway contrib/math | `hyle_simd_bench`, `hyle_dtype_bench`, `hyle_math_bench` |
| `-DHYLE_BENCH_XSIMD=ON` | math exp/log/tanh vs pinned external xsimd at the same ISA | `hyle_math_bench` |
| `-DHYLE_BENCH_SIMDUTF=ON` | utf8 validate+count vs vendored simdutf | `hyle_string_algo_bench` |
| `-DHYLE_BENCH_MARL=ON` | fiber sync vs marl | `hyle_sched_marl_bench` |
| `-DHYLE_BENCH_BOOST=ON` | IntrusiveList vs `boost::intrusive::list` (needs boost submodule checked out) | `hyle_container_intrusive_bench` |
| `-DHYLE_BENCH_WYHASH=ON` | hash_bytes vs wyhash | `hyle_hash_bench` |

**10 gates total.** Highway and xsimd are **pinned external** rivals rooted at
`HYLE_BENCH_VENDOR_DIR`; their gates only toggle comparison rows after the
mandatory bench checkout is available. SIMDUTF remains **vendored**. The other
7 are **clone** rivals (ABSL/MOODYCAMEL/TBB/RIGTORP/MARL/BOOST/WYHASH — each
needs its sibling clone + a `HYLE_<LIB>_DIR` path variable, defaulting to
`$repo/../<lib>`). All gates are B-category dev/bench controls and never affect
the shipped library.

### One command for ALL rivals — the `bench-all-sota` CMake preset

After fetching the SIMD vendors and cloning the sibling rivals, the
`bench-all-sota` preset flips every gate ON at once:

```sh
# Windows/macOS use their documented vendor defaults.
scripts/fetch-bench-vendors.sh
scripts/setup-bench-libs.sh
cmake --preset bench-all-sota          # configure with every -DHYLE_BENCH_*=ON
cmake --build --preset bench-all-sota  # build all bench exes

# Linux/WSL: choose the ext4 vendor directory and force the GCC14 toolchain.
scripts/fetch-bench-vendors.sh --dir "$HOME/vendor"
CC=gcc-14 CXX=g++-14 cmake --preset bench-all-sota -DHYLE_BENCH_VENDOR_DIR="$HOME/vendor"
cmake --build --preset bench-all-sota
```

> **These `HYLE_BENCH_*` gates are DEV/BENCH-ONLY, a separate category from the
> product build params** (`HYLE_STATIC` / `HYLE_SANITIZER` / `HYLE_WERROR` / the
> SIMD ISA tier). They ONLY add comparison rows to the benchmarks — they never
> affect the shipped library. The `bench-` prefix on the preset name says so.

- **Vectorized `exp/log/tanh` is always-on in the product**. External Highway/xsimd
  checkouts are benchmark-only reference rows, never product includes. The HYLE math row remains runtime-gated by
  `cpu_features().avx2`; older CPUs fall back to scalar libm.
- **boost::intrusive** needs the boost `libs/intrusive` submodule checked out
  (modular-boost clones leave submodules empty); without it `-DHYLE_BENCH_BOOST`
  fails the header check at configure time.

```sh
# manual equivalent (single bench, e.g. queue vs moodycamel + oneTBB + rigtorp):
# Keep build-bench separate from the product build cache.
cmake -S . -B build-bench -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DHYLE_BUILD_BENCH=ON -DHYLE_BENCH_MOODYCAMEL=ON -DHYLE_BENCH_TBB=ON -DHYLE_BENCH_RIGTORP=ON
cmake --build build-bench --target hyle_queue_sota_bench
./build-bench/bin/hyle_queue_sota_bench
```

## Legacy module runners

`run-container-intrusive-bench-evidence.py` preserves useful module-specific
artifact, placement, no-IPO, and lifetime-object checks from the pre-v1
instrument. It is now a **legacy diagnostic / possible Formal plugin**, not the
sole module certificate entry and not a template to copy. Its old 3% statistics,
host policy, resampling, and `RIVAL/BASELINE` ratio do not override the v1 common
core. The `container_intrusive` P1 engineering gate remains open until a frozen
5% `closure-perf-v1` contract produces a candidate-bound qualified manifest.

The same rule applies to other `run-*-bench-evidence.py` and module `*-stat.py`
drivers: retain narrow validators or parsers when useful, but migrate shared
machine, controls, statistics, status, and evidence composition into
`scripts/hyle_perf/`. New modules add a contract/plugin; they must not start by
copying an old module runner.

Calling `bench-stat.py` directly remains useful while developing one workload,
but it bypasses module-level contract/control/artifact gates and cannot advance
the canonical module state.

## `bench-stat.py` — paired process-level performance evidence

`bench-stat.py` is a legacy measurement adapter. It repeats one nanobench executable in fresh
processes and emits same-process `RIVAL/BASELINE` throughput ratios, where ratio below one favors
HYLE. The v1 adapter must invert that value into canonical `advantage_ratio` before aggregation;
never mix the two directions in one table. Its historical `EQUIVALENT` spelling maps to current
`PARITY`; its informational `NOT_COMPARABLE` marker belongs to the separate comparability axis.

The legacy script also produces a separate non-inferiority gate. Its upper
limit is `U=1+m`: a median-ratio CI whose upper bound is at most `U` is `PASS`, a CI wholly
beyond `U` is `FAIL`, and a CI that still crosses `U` is `UNRESOLVED`. This gate answers only
whether a material loss has been excluded. It never turns an exact `INCONCLUSIVE` result that
crosses the WIN/EQUIVALENT boundary into either exact verdict, and it cannot use a second or
post-hoc margin.

The equivalence band is reciprocal: `[1/(1+m), 1+m]`. At an evidence size of 15
independent processes and tail alpha `.01`, the interval is the third through thirteenth ordered
ratio (actual coverage about 99.26%). A historical Formal input ran the default and
peer-first orders and required compatible classifications; short/code-placement-sensitive kernels
also need the perturbation gate described in
[`benchmark-evidence-standard.md`](../docs/developer/performance/benchmark-evidence-standard.md). The script's
`[order_verdict]` and `[order_sota_gate]` are therefore evidence inputs, not a module certificate.
Default order, peer-first, and every applicable shape gate remain useful raw inputs. Only the v1
contract and common core may compose the current three axes and engineering gate. Absolute
per-variant spread and the sign-test p-value are diagnostics only.

A diagnostic run may omit provenance gates. A generic clean pair run uses `--require-clean`, declares
the workload-specific margin before sampling, and names every external Git peer that contributes
comparison code. The following direct `container_intrusive` commands are **provisional diagnostics
only**: they omit its frozen v1 contract, common controls, actual-object/no-IPO, placement and
artifact-freeze gates. Neither these commands nor the legacy dedicated runner alone is a current
engineering certificate.

```bash
python scripts/bench-stat.py build-bench/bin/hyle_container_intrusive_bench -n 31 \
  --peer boost=../boost --equivalence-margin-pct 3 \
  --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink"

HYLE_BENCH_RIVAL_FIRST=1 python scripts/bench-stat.py \
  build-bench/bin/hyle_container_intrusive_bench -n 31 \
  --peer boost=../boost --equivalence-margin-pct 3 \
  --pair "HYLE IntrusiveList" "boost::intrusive auto_unlink"
```

Use `--not-comparable-pair BASELINE RIVAL` when an absolute cost comparison is useful but the two
rows have different ownership, validation, allocation, batching, concurrency, ISA, or QoS
contracts. It prints the ratio as information while mechanically suppressing a performance claim.

The runner records the HYLE and peer HEAD/tree/dirty/remote identities, its own and the executable's
SHA-256, host, relevant CMake cache contract, HYLE benchmark environment, and the centrally locked
nanobench version/revision/header hash. For every requested pair it also emits one
`# HYLE_BENCH_PAIR_RAW ` schema-1 JSON record per process with baseline op/s, rival op/s, and the
paired ratio; a log containing only aggregate medians is not formal evidence. `--require-clean`
rejects a dirty or unidentified current source/peer and an executable outside a discoverable CMake
build tree. This is a runtime provenance snapshot, **not** a cryptographic build binding: a formal
result must also retain the fresh configure/build log that produced the recorded executable hash.
The runner does not guess which peers a benchmark uses; omitting a required `--peer` makes the run
incomplete evidence. Do not pass unrelated sibling checkouts merely to make a list look
comprehensive.

## Rust cross-language benches (`bench/rust/`)

`bench/rust/crossbeam_queue_bench` races HYLE's work-stealing queues against
crossbeam-deque (Rust can't link a C++ TU, so it runs side-by-side). It depends on
the **crates.io** release by default (portable; offline builds use the `~/.cargo`
cache):

```sh
cd bench/rust/crossbeam_queue_bench && cargo run --release
# to use a LOCAL clone instead of crates.io (no Cargo.toml edit):
cargo run --release --config \
  'patch.crates-io.crossbeam-deque.path="../../../../crossbeam-deque"'
```

## Reproducing the toolchain workflows

- **Windows (MSVC)**: `windows-vs2026` uses CMake 4.2+'s native VS 2026/v145
  generator; `windows` is the VS 2022 fallback with the highest supported v143
  selected by `hyle-windows.ps1`; `windows-ninja` uses the latest environment
  selected by that wrapper. Their build trees are separate.
- **Windows (clang-cl)**: `windows-clangcl` reuses the selected Windows SDK/MSVC ABI
  environment but fixes the frontend to a PATH-visible Clang 18+; its build tree
  is separate from every cl.exe tree.
- **Linux (GCC14, incl. ThreadSanitizer)**: separate build dir `build-wsl*/`,
  `CXX=g++-14 CC=gcc-14`. Lock-free code MUST pass TSan
  (`-DHYLE_SANITIZER=tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo`, run under
  `setarch -R`). Never reuse the Windows `build/` dir (CMakeCache conflicts).
- **macOS (AppleClang 18+)**: `macos` leaves generator selection to CMake and uses
  its own `build-macos`; it is the Apple Silicon/NEON and kqueue runtime gate.

## `mem-cert-stat.py` — whole-request mem latency

Runs section G of `hyle_mem_scenarios_bench` in independent processes and
aggregates p50/p95/p99 for complete 4096-operation requests. It preserves
paired reference/HYLE ratios, records the Git commit and dirty state, and can
write every raw row with `--raw`. On macOS it also records the requested QoS,
whether the QoS call succeeded, and the unsupported hard-affinity boundary;
an unsuccessful QoS request rejects the certificate. Use at least
`-n 15 --samples 4096` for a certificate; short runs are harness smoke tests only. See
`docs/implementation/evidence/performance/benchmarks-mem.md` §8 for the historical
commands and interpretation. A new certificate must also satisfy the current
instrument qualification standard.

## `mem-fiber-stack-stat.py` — scheduler-owned fiber stack latency

Runs `hyle_mem_fiber_stack_bench` in independent processes and aggregates the
p50/p95/p99 latency of complete 32-fiber create/run/drain/destroy batches. The
paired variants are scheduler-owned `FiberStackPool` reuse and direct guarded
VM reserve/release. It records commit, dirty state, compiler/platform, Linux CPU
binding and Apple QoS; dirty runs or a failed macOS QoS request are not release
evidence. Formal runs use `-n 15 --samples 2048` on Windows/Linux and `-n 31`
for each macOS QoS. See
`docs/implementation/evidence/performance/benchmarks-mem.md` §5.

## `mem-shared-link-stat.py` — mem shared standard-adapter certificate

Runs section `L` of `hyle_mem_bench` with one variant per fresh process, alternating
forward/reverse order, and writes the raw CSV used by the mem same-link table. It covers
ordinary alloc/free standard adapters plus direct/runtime realloc growth. The benchmark must
come from `HYLE_STATIC=OFF`, `HYLE_BUILD_TESTS=OFF` shipping configuration.

```bash
python3 scripts/mem-shared-link-stat.py build-bench/bin/hyle_mem_bench \
  --runs 15 --platform linux --compiler gcc-14.2 \
  --cpu 2 --raw evidence/performance/mem-shared-same-link-linux.csv
```

On macOS omit `--cpu` and pass `--apple-qos interactive` or `--apple-qos utility`; the benchmark
applies and reports that QoS itself, and the driver rejects a failed request. Use repeated `--prefix`
only when another external launcher is part of the recorded contract. The script rejects static
linkage and records commit/dirty/QoS state in every row.
