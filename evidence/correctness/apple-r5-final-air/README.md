# R5 final Air host sanitizers

Immutable candidate `1636365bbe47779de24e84d7c7d1ed8088fe1af2` passed three separate
ASan, UBSan and TSan configurations on MacBook Air `Mac16,12`. Every configuration
built and ran the 15 native Compute/Graph tests. Compile-command receipts prove
instrumentation of the eight HYLE implementation/test translation units.

The 79 MLX vendor units and Metal GPU shaders were **not instrumented**. These
receipts establish HYLE host behavior with actual Metal execution; they do not
claim GPU memory instrumentation or MLX's unsupported macOS `USE_TSAN` mode.

`r5-final-air-raw.zip` contains the executed helper, build/configuration/test logs,
compile commands and executable hashes for each configuration. SHA-256:
`f78f1c61dc1ea100e2efb9f76532946e2118c0ead9b5b1f34f7e1cf6de48110b`.
The adjacent identity manifest records every archived file hash.

This satisfies the host-sanitizer obligation of the
[R5 card](../../../docs/implementation/tasks/compute/apple-v1-four-route-foundation.md).
Final P1 and route closure are separate verdicts.
