# Sample 04 — installed package consumer

Samples 01–03 embed HYLE source with `add_subdirectory`; this sample validates the other
supported assembly path: install HYLE, then consume it through
`find_package(Hyle 0.1.0 EXACT COMPONENTS simd mem)` and namespaced imported
targets. Pre-1.0 package consumption deliberately requires the complete version.

The imported targets carry the public headers, C++23 floor, static/shared visibility
definition, module dependency closure and mem's private mimalloc backing. The consumer never
searches for individual `.lib`/`.a` files or reconstructs static link order. On Windows,
`hyle_stage_runtime()` copies the imported DLL closure beside the executable.

Run both shared and static paths on Linux/WSL:

```bash
CC=gcc-14 CXX=g++-14 bash build_and_run.sh
```

Equivalent manual flow:

```bash
cmake -S ../.. -B /tmp/hyle-build -G Ninja -DHYLE_COMPONENTS='simd;mem' \
  -DHYLE_BUILD_TESTS=OFF -DHYLE_INSTALL=ON \
  -DCMAKE_INSTALL_PREFIX=/tmp/hyle-prefix
cmake --build /tmp/hyle-build --target hyle_simd
cmake --install /tmp/hyle-build

cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/tmp/hyle-prefix
cmake --build build
./build/installed_package_consumer
```

Expected output includes the active ISA and `dot(...)=1536.0`.
