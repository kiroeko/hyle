# Installed `error` package probes

These are source-independent consumers of an installed HYLE SDK:

- `cpp/` requests only `error` and exercises the normal `Status`, `Result<T>`,
  and rvalue `WithContext` path.

The package smoke discovers the C++ project automatically and runs it against
shared and static SDKs. It intentionally does not include HYLE through
`add_subdirectory()` or read files from its source tree. The retired C-shaped
fixture is not part of the current product; Native FFI remains `N/A`.
