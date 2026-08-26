# Installed `obs-runtime` package probes

These are source-independent consumers of an installed HYLE SDK:

- `cpp/` requests only `obs` and exercises the normal formatted macro plus the
  move-only `SinkRegistration` lifetime path.

The package smoke discovers the C++ project automatically and runs it against
shared and static SDKs. It intentionally does not include HYLE through
`add_subdirectory()` or read files from its source tree. The retired C-shaped
fixture is not part of the current product; Native FFI remains `N/A`.
