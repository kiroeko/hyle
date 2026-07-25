# Sample 05 — pure-C installed-package consumer

This project enables only the C language, finds the installed `error` and `mem` components,
and links `hyle::error` plus `hyle::mem`. It proves the package does not force C++23 compile
features onto a C source while exposing HYLE's reviewed flat C ABI. The allocation also makes
shared-package tests load the real `mem -> platform -> error` module chain.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/hyle
cmake --build build
./build/installed_c_consumer
```

The same C-only project works with shared and static HYLE packages. For a static package,
the package enables CXX as the link-driver language and the imported target records its C++
link interface and private mimalloc archive, so CMake adds the toolchain's matching implementation
runtime without compiling the consumer's `.c` translation unit as C++.
