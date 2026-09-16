# Apple R5 product delivery checkpoints

Owned by the finite [D1–D4 delivery card](../../../docs/implementation/tasks/build/apple-r5-product-delivery.md).
Air's three C++ journeys and WSL's OFF admission checks pass. Mini and the complete
Windows broad gate are still pending; this record does not close R5 or the delivery card.

## Air / actual MLX Metal

- Complete source archive `cbedb837`, SHA256
  `1aecfb41b9af31881152bdf78a652f7a6ab0f9ec67a4bfbad8993dc1f097d8ce`:
  offline `add_subdirectory` builds and runs ordinary public Compute/Graph consumers.
- Installed static and shared prefixes are relocated. Both ordinary consumers execute
  native RMSNorm and affine Q4 FFN with changing inputs, checked against an independent oracle.
- All producer `mlx.metallib` fallback paths are hidden during installed execution.
  Static additionally rejects an absent executable-side resource, then succeeds after
  restoring only that resource. Shared loads exactly one MLX dylib from the relocated SDK.
- Public header/dependency closure, four licenses, absence of producer/SDK absolute paths,
  and shared exports pass: 11 HYLE libraries, 841 exports inspected.

`air/static/` and `air/shared-source/` retain actual configure/build/install/run and loader
logs. The executed verification recipes sit beside them. Static producer inputs were
independently matched in the [lifecycle input identity](../apple-r5-lifecycle/product-identity.json).

## WSL / Apple OFF

On Ubuntu 24.04 native ext4 / GCC 14.2, core, no-backend Compute+Graph, and CPU-only
configurations pass without MLX/Metal source acquisition or CUDA/Objective-C++ probes.
Explicit Apple ON is rejected as unsupported before native SDK admission. Both Apple
public headers compile with the feature OFF and no Apple SDK. These are isolation and
header checks; they do not claim Apple native execution on Linux.

`wsl/receipt.json` and raw logs bind the complete source archive and actual compiler.
`artifact-manifest.json` hashes the evidence and records that all shipping source,
headers, CMake, samples and vendored inputs are unchanged between `cbedb837` and the
frozen R5 candidate `1636365b`. No product rebuild or P1 is caused by archiving these logs.
