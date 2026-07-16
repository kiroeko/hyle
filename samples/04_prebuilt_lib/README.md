# Sample 04 — consume HYLE as a prebuilt library (shared / static)

Samples 01–03 build HYLE from source (`add_subdirectory`). This one is the **other**
integration mode: HYLE is **already compiled** into a shared (`.so`/`.dll`) or static
(`.a`/`.lib`) artifact, and a **separately-compiled** consumer just links it. This is the
mode that **validates HYLE's export boundary** (`HYLE_<MOD>_API`):

- **shared** → the consumer's `dllimport` must match the library's `dllexport`; a clean
  link + run proves the cross-DLL symbol surface is correct.
- **static** → `HYLE_<MOD>_API` expands to nothing (`HYLE_STATIC`); proves that path too.

## Honest scope (what this does and does NOT validate)

- ✅ Validates: HYLE's **C++ API** reachable across a **shared-library (dllexport/import)**
  boundary, and across a **static archive**. The consumer calls `hyle::simd::dot` and reads
  `runtime().isa` from the prebuilt `hyle_simd`.
- ❌ Does NOT validate a pure **`extern "C"` C-ABI**: that flat-C-header shim is a planned
  milestone (P6), not implemented yet. This is "C++ ABI across a DLL", not "a C ABI". We
  deliberately call a C-friendly free function to keep the surface minimal.
- We use `hyle::simd` because it is **dependency-light**. Note: a prebuilt **static**
  `hyle_mem` additionally requires linking `mimalloc_static` (mem's private backing), and
  a multi-module static consumer must link every module + its transitive deps in order —
  the kind of thing a future `install()` + `find_package(hyle)` (P6 packaging) will
  automate. `hyle_simd` has no such transitive dep, so both modes link cleanly here.

## Build + run (both modes)

```bash
# 1) build HYLE as SHARED (default) into one dir, STATIC into another:
cmake -S <repo> -B /tmp/hyle-shared -DHYLE_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/hyle-shared --target hyle_simd
cmake -S <repo> -B /tmp/hyle-static -DHYLE_STATIC=ON -DHYLE_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/hyle-static --target hyle_simd

# 2) link the consumer against each prebuilt artifact:
cmake -S . -B build-shared -DHYLE_REPO_DIR=<repo> -DHYLE_PREBUILT_DIR=/tmp/hyle-shared -DHYLE_LINK_MODE=shared
cmake --build build-shared && ./build-shared/prebuilt_consumer

cmake -S . -B build-static -DHYLE_REPO_DIR=<repo> -DHYLE_PREBUILT_DIR=/tmp/hyle-static -DHYLE_LINK_MODE=static
cmake --build build-static && ./build-static/prebuilt_consumer
```

`build_and_run.sh` automates both modes on Linux. Expected output (both):
```
linked prebuilt hyle_simd OK
  active SIMD variant : avx2        (whatever the host supports)
  dot(1.5*2.0 x512)   : 1536.0 (expect 1536.0)
```
