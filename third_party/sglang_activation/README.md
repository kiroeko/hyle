# Pinned precise SGLang activation

Origin: SGLang `06992c92ad4494fe9f9636e75be3e35040512447`, Apache-2.0.
`provenance.json` records the exact upstream Git blobs and adaptations.
`cmake/HyleDependencyLock.json` binds the official archive and this adapted tree.

The owning adapter is `src/compute_provider_cuda`; its standalone device
compatibility code remains in HYLE implementation sources. Only packed
SiLU-times-value, FP16/FP32, 16-byte vectors, PDL/filter/reuse disabled,
and FP32 activation/multiply followed by one output conversion are admitted.
No fast-math is enabled. Uninstantiated upstream enum/template branches
do not expose additional HYLE operations. TVM/Python runtime is excluded.

HYLE maintains this narrow source adapter. Upgrades require review of
arithmetic, indexing, flags, vector eligibility, namespace isolation and
the host ABI together; renew blob/archive/tree hashes, oracle and boundary
tests, native lifetime/capture, three C++ deliveries and applicable complete
consumer performance. Replacing this directory alone is not an upgrade.
