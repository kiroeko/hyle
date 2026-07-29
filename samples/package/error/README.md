# Installed `error` package probes

These are source-independent consumers of an installed HYLE SDK:

- `c/` starts as a C-only CMake project, requests only `error`, and exercises
  C status ownership, context, formatting, and the shared/static link boundary.
- `cpp/` requests only `error` and exercises the normal `Status`, `Result<T>`,
  and rvalue `WithContext` path.

The package smoke discovers both projects automatically and runs them against
shared and static SDKs. They intentionally do not include HYLE through
`add_subdirectory()` or read files from its source tree.
