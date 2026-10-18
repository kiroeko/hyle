# D5 Qwen3.8-2.4T R10 CPU router partial

Immutable candidate `66f4d80fb` adds a narrow mature ggml CPU Softmax
translation: plain, dense F32 rank-2 rows, final axis, no mask, scale 1,
standard Softmax, F32 accumulation. It leaves all other domains and provider
priority unchanged. The fixed ggml implementation performs the math; HYLE
only qualifies and binds it. P0 structure is one already prepared ggml CPU
Softmax node over 512 contiguous scores per token. Replays reuse the prepared
program or compiled Graph and perform no winner search or timing.

The model-derived Direct and Graph consumers use H=8192, a stored BF16
512 x 8192 router weight, BF16 projection output, explicit F32 cast,
Softmax over all 512 experts, stable top-10, and reduction/division to
renormalize selected probabilities. Each executes two changed inputs plus a
tie case and checks exact IDs and independently calculated weights. This
follows the selection order in pinned llama.cpp `qwen35moe.cpp` and
`llama-graph.cpp` (`4c9233c`): Softmax first, then top-k, selected-weight
normalization, expert work, and a separately gated shared expert.

The targeted CPU Direct and Graph tests passed 1/1 each. In the existing
`ggml-native-providers` Release/WERROR build with oneDNN enabled, complete
ggml CPU Provider, Direct, and Graph suites passed 40/40, 55/55, and 70/70,
with no skips. The selector selected these three factors, configure, docs,
and closure; 126 gates were skipped or reused. The owner test also verifies
that axis-0, scaled, and F16 Softmax remain unqualified.

An additional oneDNN-disabled cache passed Provider 40/40 and Direct 52/52.
Its Graph suite was 64/65 because an existing Qwen3.5 test explicitly
requires a oneDNN region. That configuration is not recorded as a Graph
PASS. It did pass the new R10 targeted Graph test.

This candidate stops before expert dispatch. The top-10 outputs have not yet
been wired to grouped expert IDs and weights, shared gate, or final combine.
The stored BF16 router MatMul result is explicitly cast to F32 here; its
storage rounding has not yet been compared with the pinned ggml model graph's
F32 score boundary. Therefore R10, complete profile cells, route-level
source/static/shared delivery, and P1 remain unsigned. The ledger stays
48 `RUN_PASS` / 128 `VALIDATION_NOT_RUN`. The canonical LOC report gives
57,949 Compute + Graph implementation lines excluding tests.
