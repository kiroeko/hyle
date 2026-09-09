# Installed `mem` package probes

These are source-independent consumers of an installed HYLE SDK:

- `cpp/` requests only `mem` and exercises the installed public façade through
  system PMR + bounded arena, isolated `Heap`, read-only model `FileMapping`,
  transient `BufferPool` reuse, payload-free 64-bit logical range allocation,
  tensor-free `LifetimePlan` instantiation, and
  guarded publication plus typed retirement of an immutable snapshot.
  Device/pinned/managed providers remain a Compute consumer journey so this
  probe cannot hide a reversed `mem -> compute` dependency.

The package smoke discovers the C++ project automatically and runs it against
shared and static SDKs. It intentionally does not include HYLE through
`add_subdirectory()` or read files from its source tree. The retired C-shaped
fixture is not part of the current product; Native FFI remains `N/A`.
