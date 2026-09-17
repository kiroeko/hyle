# R5 final Mini correctness and delivery

Mac mini `Mac16,10`, macOS 26.6.2 / build 25G83: correctness and portability only.
This host does not sign the `M1-MAC` Formal performance profile.

- Candidate `1636365bbe47779de24e84d7c7d1ed8088fe1af2`: 11 Compute and four Graph
  native tests passed, plus the two standalone header checks.
- Delivery candidate `cbedb837791da15392019299f2652d9d3d70ec9d`: complete-source
  `add_subdirectory`, relocated static SDK and relocated shared SDK all performed
  actual RMSNorm and Q4 FFN decode/prefill computations with changing inputs.
- Original producer metallib fallbacks were hidden during installed execution.
  Removing the static consumer's staged resource produced the expected failure;
  restoring it restored successful computation. The shared consumer loaded exactly
  one MLX dylib from the relocated prefix.
- Source dependency acquisition was offline. Export paths, installed headers and
  locked licenses passed; CPU and CUDA remained OFF.

`delivery-identity.json` records the exact Git comparison: every shipping input
under `include`, `src`, `cmake`, `samples`, `third_party` and `CMakeLists.txt` is
identical between the delivery and final route candidates. The later test/benchmark
changes therefore do not invalidate these three delivery journeys.

`r5-final-mini-raw.zip` preserves the executed recipes, logs, native receipts,
artifact hashes and installed file inventory. SHA-256:
`ade9655463aa48dcd38165f7b357df7667630e9e9d247ee66a05cee3c9a64fe7`.
The adjacent identity manifest records every archived file hash.

Owners: [R5 native route](../../../docs/implementation/tasks/compute/apple-v1-four-route-foundation.md)
and [Apple product delivery](../../../docs/implementation/tasks/build/apple-r5-product-delivery.md).
These receipts do not claim a Mini performance result.
