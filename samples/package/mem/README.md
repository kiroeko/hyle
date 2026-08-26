# Installed `mem` package probes

These are source-independent consumers of an installed HYLE SDK:

- `cpp/` requests only `mem` and exercises the normal process-lifetime PMR
  resource plus the same bounded scratch composition with native RAII types.

The package smoke discovers the C++ project automatically and runs it against
shared and static SDKs. It intentionally does not include HYLE through
`add_subdirectory()` or read files from its source tree. The retired C-shaped
fixture is not part of the current product; Native FFI remains `N/A`.
