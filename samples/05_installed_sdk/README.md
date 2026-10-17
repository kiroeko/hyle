# Fixture 05 — installed C++ SDK conformance

This source-independent C++ project requests the installed `mem` component,
executes the ordinary `SystemAllocator`/STL/PMR journey, and verifies every
component loaded for that request has a matching imported target.
`Hyle_AVAILABLE_COMPONENTS` describes the SDK; `Hyle_LOADED_COMPONENTS` describes
this `find_package` request and its dependencies. Unrequested SDK components
need not be imported. The fixture runs against both static and shared SDKs.

```bash
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH=/opt/hyle
cmake --build build
ctest --test-dir build --output-on-failure
```
