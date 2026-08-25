# Installed `obs-runtime` package probes

These are source-independent consumers of an installed HYLE SDK:

- `c/` is a C-syntax schema/compiler/link fixture, not a native-C product
  journey. It exercises the old obs candidate; obs review may change it or
  declare FFI=N/A.
- `cpp/` requests only `obs` and exercises the normal formatted macro plus the
  move-only `SinkRegistration` lifetime path.

The package smoke discovers both projects automatically and runs them against
shared and static SDKs. They intentionally do not include HYLE through
`add_subdirectory()` or read files from its source tree.
