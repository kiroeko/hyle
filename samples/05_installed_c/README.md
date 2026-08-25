# Fixture 05 — installed package/schema conformance

This mixed C/C++ validation project is not a native-C customer example. It requires the installed
`mem` component, then checks every component advertised by `Hyle_AVAILABLE_COMPONENTS`. Its C
executable is a compiler/link fixture for the old mem/error FFI-schema candidates; those surfaces
remain unreviewed and may change or become FFI=N/A. Its C++ executable is the real user journey for
`SystemAllocator`; one minimal executable per available component checks every imported target.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/hyle
cmake --build build
./build/installed_ffi_schema_fixture
```

The same fixture runs against shared and static packages. `../package/error/c` deliberately starts
with only the C language to audit schema/compiler/link hygiene; it is not a genuinely C-only
product contract. The actual error C++ package consumer lives in `../package/error/cpp`.
