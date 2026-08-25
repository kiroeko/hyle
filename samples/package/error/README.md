# Installed `error` package probes

These are source-independent consumers of an installed HYLE SDK:

- `c/` is a C-syntax schema/compiler/link fixture, not a native-C product
  journey. It requests only `error` and exercises the old status candidate and
  shared/static link boundary; error review may change it or declare FFI=N/A.
- `cpp/` requests only `error` and exercises the normal `Status`, `Result<T>`,
  and rvalue `WithContext` path.

The package smoke discovers both projects automatically and runs them against
shared and static SDKs. They intentionally do not include HYLE through
`add_subdirectory()` or read files from its source tree.
