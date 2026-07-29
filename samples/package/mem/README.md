# Installed `mem` package probes

These are source-independent consumers of an installed HYLE SDK:

- `c/` requests only `mem` and exercises the default allocator plus the
  advanced `QuotaAllocator -> Region` composition through opaque handles and
  shared `hyle_status` ownership.
- `cpp/` requests only `mem` and exercises the normal process-lifetime PMR
  resource plus the same bounded scratch composition with native RAII types.

The package smoke discovers both projects automatically and runs them against
shared and static SDKs. They intentionally do not include HYLE through
`add_subdirectory()` or read files from its source tree.
