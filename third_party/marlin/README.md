# Pinned Marlin kernel subset

Origin: vLLM commit `51918e252af4dfff381001f42b7ebe6a0454501b`, Apache-2.0.
`provenance.json` records the exact upstream paths, Git-blob SHA-256 values and
local adaptations. `cmake/HyleDependencyLock.json` locks the upstream archive
and the separate adapted directory identity.

HYLE maintains this narrow CUDA subset without the Torch host wrapper. The
adapter is in `src/compute_provider_marlin`: FP16 activation/output, symmetric
group128 Q4 weights, FP32 reduction, M=1..64, K divisible by 128, N by 64. It
binds the upstream launch configurations in cold preparation. Packing uses the
upstream GPTQ repack kernel and grouped scale permutation. No device arithmetic
is replaced. Only the adapter and its CUDA translation unit include these files.

Local changes isolate the compile-time scalar descriptor from Torch, use the
full-width ScalarTypeId on Windows LLP64, remove an unused iostream include,
and extract the repack device kernel into a private namespace. The descriptor
invariant check cannot receive runtime user input; the fallible HYLE adapter
validates public inputs before any launch.

Upgrades require reviewing the kernel, launch configuration, packing and scale
layout together, renewing both identities, and running the CUDA route's oracle,
lifetime, delivery and applicable performance checks. Replacing this directory
alone is not an ABI-compatible provider upgrade.
