# Sample 05 — installed all-target and mem API consumer

This mixed C/C++ project requires the installed `mem` component, then checks every component
advertised by `Hyle_AVAILABLE_COMPONENTS`. Its C executable calls the mem and error C APIs
through `hyle::mem`; its C++ executable calls `SystemAllocator`; one minimal executable per
available component mechanically checks every imported target. The C allocation path also
makes shared-package tests load the real `mem -> platform -> error` module chain.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/hyle
cmake --build build
./build/installed_c_consumer
```

The same project works with shared and static HYLE packages. For the genuinely C-only,
error-only package contract, see `../package/error/c`; its CMake project starts with only the C
language, while a static HYLE package enables C++ solely as the linker driver. The matching
error C++ API probe lives in `../package/error/cpp`.
