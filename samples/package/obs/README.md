# Installed `obs-runtime` package probes

These are source-independent consumers of an installed HYLE SDK:

- `c/` starts as a C-only CMake project, requests only `obs`, and exercises
  checked level control, a structured borrowed record, opaque sink ownership,
  flush, and shared `hyle_status` cleanup.
- `cpp/` requests only `obs` and exercises the normal formatted macro plus the
  move-only `SinkRegistration` lifetime path.

The package smoke discovers both projects automatically and runs them against
shared and static SDKs. They intentionally do not include HYLE through
`add_subdirectory()` or read files from its source tree.
