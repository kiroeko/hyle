# scripts/

Project automation: environment setup and reproducible builds. The goal is that a
**fresh checkout on any machine** can build, test, and benchmark HYLE without
depending on a hand-crafted directory layout.

## `setup-bench-libs.sh` — fetch SOTA benchmark references

HYLE's headline claim is *SOTA performance*, so the benchmarks compare against the
real reference libraries (abseil, folly, mimalloc, marl, moodycamel, oneTBB,
crossbeam-deque, wyhash) — not just `std`. Those libs are **not vendored**
(they're heavy and we only need them for opt-in benches), so this script clones
them. (Highway, xsimd, simdutf and xxHash are already **vendored in
`third_party/`** — no clone needed; their `-DHYLE_BENCH_*` gates just toggle the
comparison rows against the vendored copies.)

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
| `-DHYLE_BENCH_HIGHWAY=ON` | simd vs Highway; dtype vs Highway PromoteTo/DemoteTo; math vs Highway contrib/math (vendored — no clone) | `hyle_simd_bench`, `hyle_dtype_bench`, `hyle_math_bench` |
| `-DHYLE_BENCH_XSIMD=ON` | math exp/log/tanh vs xsimd, same ISA (vendored — no clone) | `hyle_math_bench` |
| `-DHYLE_BENCH_SIMDUTF=ON` | utf8 validate+count vs simdutf (vendored — no clone) | `hyle_string_algo_bench` |
| `-DHYLE_BENCH_MARL=ON` | fiber sync vs marl | `hyle_sched_marl_bench` |
| `-DHYLE_BENCH_BOOST=ON` | IntrusiveList vs `boost::intrusive::list` (needs boost submodule checked out) | `hyle_container_misc_bench` |
| `-DHYLE_BENCH_WYHASH=ON` | hash_bytes vs wyhash | `hyle_hash_bench` |

**10 gates total.** 3 are **vendored** (HIGHWAY/XSIMD/SIMDUTF — the lib is in
`third_party/`, the gate only toggles the comparison row); the other 7 are
**clone** rivals (ABSL/MOODYCAMEL/TBB/RIGTORP/MARL/BOOST/WYHASH — each needs its
sibling clone + a `HYLE_<LIB>_DIR` path var, defaulting to `$repo/../<lib>`). The
difference is only *where the lib lives* (in-repo vs clone-it-yourself) — both
are B-category dev/bench gates that never affect the shipped library.

### One command for ALL rivals — the `bench-all-sota` CMake preset

After `setup-bench-libs.sh` has cloned the siblings, the `bench-all-sota` preset
flips every gate ON at once (no hand-assembling 8 `-D` flags):

```sh
scripts/setup-bench-libs.sh            # clone the rival siblings
cmake --preset bench-all-sota          # configure with every -DHYLE_BENCH_*=ON
cmake --build --preset bench-all-sota  # build all bench exes
```

> **These `HYLE_BENCH_*` gates are DEV/BENCH-ONLY, a separate category from the
> product build params** (`HYLE_STATIC` / `HYLE_SANITIZER` / `HYLE_WERROR` / the
> SIMD ISA tier). They ONLY add comparison rows to the benchmarks — they never
> affect the shipped library. The `bench-` prefix on the preset name says so.

- **Vectorized `exp/log/tanh` is always-on**: the math cores (Highway + xsimd)
  are vendored in `third_party/` and compiled into production unconditionally —
  no clone, no `-D` switch. The HYLE math row is full-speed SOTA by default
  (runtime `cpu_features().avx2` gate; older CPUs fall back to scalar libm).
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
