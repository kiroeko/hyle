# Sample 05 — pure-C installed-package consumer

This project enables only the C language, finds the installed `error` component and links
`hyle::error`. It proves the package does not force C++23 compile features onto a C source,
while still exposing HYLE's reviewed flat C ABI.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/hyle
cmake --build build
./build/installed_c_consumer
```

The same C-only project works with shared and static HYLE packages. For a static package,
the imported target records its C++ linker language so CMake adds the implementation runtime
without compiling the consumer's `.c` translation unit as C++.
