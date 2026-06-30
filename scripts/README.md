# scripts/

Project automation: environment setup and reproducible builds. The goal is that a
**fresh checkout on any machine** can build, test, and benchmark HYLE without
depending on a hand-crafted directory layout.

## `setup-bench-libs.sh` — fetch SOTA benchmark references

HYLE's headline claim is *SOTA performance*, so the benchmarks compare against the
real reference libraries (abseil, folly, mimalloc, Highway, marl, moodycamel,
oneTBB, crossbeam-deque, xxHash, wyhash, sleef, xsimd) — not just `std`. Those
libs are **not vendored** (they're heavy and we only need them for opt-in benches),
so this script clones them.

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

The default build has NO external bench deps. Each rival is opt-in:

| flag | compares | exe |
|---|---|---|
| `-DHYLE_BENCH_ABSL=ON` | SwissTable vs `absl::flat_hash_map`; SmallVector vs `absl::InlinedVector` | `hyle_absl_bench`, `hyle_container_misc_bench` |
| `-DHYLE_BENCH_MOODYCAMEL=ON` | queues vs moodycamel | `hyle_queue_sota_bench` |
| `-DHYLE_BENCH_TBB=ON` | queue/vector vs oneTBB | `hyle_queue_sota_bench` |
| `-DHYLE_BENCH_RIGTORP=ON` | BoundedQueue vs `rigtorp::mpmc::Queue` (same-contract strict-bounded MPMC) | `hyle_queue_sota_bench` |
| `-DHYLE_BENCH_HIGHWAY=ON` | simd vs Highway; dtype vs Highway PromoteTo/DemoteTo; math vs Highway contrib/math | `hyle_simd_bench`, `hyle_dtype_bench`, `hyle_math_bench` |
| `-DHYLE_BENCH_XSIMD=ON` | math exp/log/tanh vs xsimd (same ISA) | `hyle_math_bench` |
| `-DHYLE_BENCH_MARL=ON` | fiber sync vs marl | `hyle_sched_marl_bench` |
| `-DHYLE_BENCH_BOOST=ON` | IntrusiveList vs `boost::intrusive::list` (needs boost submodule checked out) | `hyle_container_misc_bench` |

### One command for ALL rivals — the `all-sota` CMake preset

After `setup-bench-libs.sh` has cloned the siblings, the `all-sota` preset flips
every gate ON at once (no hand-assembling 8 `-D` flags):

```sh
scripts/setup-bench-libs.sh            # clone the rival siblings
cmake --preset all-sota                # configure with every -DHYLE_BENCH_*=ON
cmake --build --preset all-sota        # build all bench exes
```

- **SLEEF is NOT a clone** (it's a prebuilt lib): for vectorized `exp/log/tanh`
  add `-DHYLE_SIMD_SLEEF=ON -DHYLE_SLEEF_DIR=<prebuilt sleef>` on top of the preset.
  Without it the HYLE math row falls back to scalar libm (honest, no vectorization).
- **boost::intrusive** needs the boost `libs/intrusive` submodule checked out
  (modular-boost clones leave submodules empty); without it `-DHYLE_BENCH_BOOST`
  fails the header check at configure time.

```sh
# manual equivalent (single bench, e.g. queue vs moodycamel + oneTBB + rigtorp):
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DHYLE_BUILD_BENCH=ON -DHYLE_BENCH_MOODYCAMEL=ON -DHYLE_BENCH_TBB=ON -DHYLE_BENCH_RIGTORP=ON
cmake --build build --target hyle_queue_sota_bench
./build/bin/hyle_queue_sota_bench
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
