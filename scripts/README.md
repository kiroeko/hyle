# scripts/

Project automation: environment setup and reproducible builds. The goal is that a
**fresh checkout on any machine** can build, test, and benchmark HYLE without
depending on a hand-crafted directory layout.

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

The scripts read the sole pin source, `cmake/vendor-versions.cmake`, refuse to
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

## `setup-bench-libs.sh` — fetch other gated SOTA references

HYLE's headline claim is *SOTA performance*, so the benchmarks compare against
real references (abseil, folly, mimalloc, marl, moodycamel, oneTBB,
crossbeam-deque, wyhash) — not just `std`. These heavy, opt-in rivals are not
vendored, so this script clones them. Highway/xsimd use the separate pinned
fetcher above; simdutf and xxHash remain in `third_party/`.

```sh
scripts/setup-bench-libs.sh              # clone all into the repo's parent dir
scripts/setup-bench-libs.sh --shallow    # --depth 1 (faster)
scripts/setup-bench-libs.sh marl highway # only the named ones
HYLE_DEPS_DIR=/custom/path scripts/setup-bench-libs.sh
```

- **Idempotent**: a lib that already has a `.git` is skipped, never clobbered.
- **Default location** = the repo's parent dir (`$repo/../`), which is exactly
  where the gated-bench CMake defaults (`HYLE_<LIB>_DIR`) look. So after running
  it with no args, every `-DHYLE_BENCH_*=ON` works with **no extra `-D`**.
- **Custom location**: set `HYLE_DEPS_DIR`, then pass the dir to the build:
  `-DHYLE_MARL_DIR=$HYLE_DEPS_DIR/marl` etc.

## Running the gated SOTA benchmarks

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
| `-DHYLE_BENCH_BOOST=ON` | IntrusiveList vs `boost::intrusive::list` (needs boost submodule checked out) | `hyle_container_misc_bench` |
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

- **Vectorized `exp/log/tanh` is always-on in the product during the staged
  migration**: its current Highway/xsimd cores remain in `third_party/` until
  simd-20 ports them. The external checkouts above are benchmark-only reference
  rows, never product includes. The HYLE math row remains runtime-gated by
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

## Reproducing the dual-platform workflow

- **Windows (MSVC)**: build dir `build/` (Ninja). Needs a VS dev environment
  (`vcvars64.bat`) on PATH/sourced first.
- **Linux (GCC14, incl. ThreadSanitizer)**: separate build dir `build-wsl*/`,
  `CXX=g++-14 CC=gcc-14`. Lock-free code MUST pass TSan
  (`-DHYLE_SANITIZER=tsan -DCMAKE_BUILD_TYPE=RelWithDebInfo`, run under
  `setarch -R`). Never reuse the Windows `build/` dir (CMakeCache conflicts).

## `mem-cert-stat.py` — whole-request mem latency

Runs section G of `hyle_mem_scenarios_bench` in independent processes and
aggregates p50/p95/p99 for complete 4096-operation requests. It preserves
paired reference/HYLE ratios, records the Git commit and dirty state, and can
write every raw row with `--raw`. Use at least `-n 15 --samples 4096` for a
certificate; short runs are harness smoke tests only. See
`docs/performance/benchmarks-mem.md` §8 for commands and interpretation.
