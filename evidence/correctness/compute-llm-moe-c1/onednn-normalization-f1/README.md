# C1-AA first correction: optimized normalization prefix

Implementation `ae270af9ceb948b09601b26f32454eb28d32dff3`. W1 static/shared and WSL GCC14 static each pass
the seven selected CTests: 58 gtest cases plus an independent executor consumer.
The new regression covers RMS/LayerNorm, scalar/vector broadcasts, single/multiple rows,
full-shape Add/Multiply, published intermediates and two independent invocation slots.
Graph source, relocated shared SDK and relocated static SDK consumers pass the original
dependency, feature, license, export and relocation checks, with adjacent DLLs and PATH unset.

The original static attempt stopped on a WERROR test-variable shadow warning. The name was
fixed and only the affected Graph group was resumed; the saved Git trees prove that the other
completed groups had identical product/build/header inputs. No product check was removed.

The adapter stops fusion before an unsupported full multirow binary operand and rejects
recognized generic fused implementations during exact preparation. The optimized prefix and
remaining Graph node execute through the existing resources and completion protocol.
The explicit package consumer requires a real two-node oneDNN region followed by Native;
an all-Native execution cannot satisfy that check.

Corrected performance is NOT RUN. Default preference rules, Native domain decisions and
the rest of C1 remain unfinished. The original qualified 4 wins/12 losses are preserved separately.
SIMD is unchanged; the archived learning note is a source observation, not an optimization claim.

Raw recipes, exact selector plans/skipped reasons, commands, failures, test output and CMake caches
are in raw-bundle.zip. result.json binds the source tree and each validation shape;
delivery-artifacts.json binds the actual SDK and consumer files. The verified 85,063,680-byte
WSL transfer archive was deleted. Active fixed build directories remain available for incremental
validation of the unfinished default rules; they are not permanent duplicate caches.

Unified LOC: 46,955 Compute+Graph implementation lines, 193 files, self-check PASS.
Tests, benchmarks and third-party source are excluded from that implementation count.
