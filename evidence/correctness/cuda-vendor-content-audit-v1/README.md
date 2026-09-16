# CUDA vendor identity and consumer admission

Owned by [V1/V2](../../../docs/implementation/tasks/build/cuda-vendor-content-audit-v1.md).
This closes the identified hash/sorting evidence gap. The final applicable
[Windows selector](../apple-r5-windows-selector/README.md) now passes; the owning
task records the legal candidate and terminal transitions separately.

- Windows and Air independently reproduce both incorrect historical hashes with
  case-insensitive path sorting, and the correct hashes with the existing CMake
  case-sensitive manifest rule. Vendor file contents and revision pins are unchanged.
- Full in-tree vendor audit succeeds on both operating systems, including the actual
  complete-source archive consumed by the [Apple delivery](../apple-r5-product-delivery/README.md).
- For each vendor, mutate a disposable copy and enter its real enabled owning factor:
  configuration fails on content identity before `enable_language(CUDA)` can run.
  This proves the consumer check is wired; it is not a standalone hash-only probe.
- OFF consumers do not acquire or inspect disabled vendor implementations. Native CUDA
  computation and limited RTX 4070 / SM89 performance retain the registered R4 evidence;
  these configuration fixes do not establish new device, toolkit or algorithm claims.

`windows/` and `air/` hold original receipts and negative logs. The executed bounded
recipe and `artifact-manifest.json` make the inputs and raw evidence reviewable.
