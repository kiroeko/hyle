# Fixture 05 — installed C++ SDK conformance

This source-independent C++ project requests the installed `mem` component,
executes the ordinary `SystemAllocator`/STL/PMR journey, and verifies every
component advertised by `Hyle_AVAILABLE_COMPONENTS` has a matching imported
target. It runs against both static and shared SDKs.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/hyle
cmake --build build
ctest --test-dir build --output-on-failure
```
