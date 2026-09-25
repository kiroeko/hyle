# D5 Qwen3.8-2.4T expert selection and CPU Graph combine partial

Tests-only candidates `34a13aab` and `0c97b284` strengthen the official
BF16 synthetic-capacity consumer. The ten resident experts now have distinct
weights. The CPU Graph feeds one changing BF16 token to both the 512-expert
router and, through mature Cast/Gather, ten F32 expert slots. Softmax and
stable top-10 produce IDs that change order between replays; those IDs drive
three BF16-weight grouped expert projections. The selected probabilities are
renormalized, multiplied into the ten outputs, and reduced to one token.
The test checks selected IDs, every expert output, and the weighted result
against an independent oracle. The model's H=8192, E=512, top-10, expert
inner width 2048, dtype ports, and mature Provider-only execution are intact;
only resident expert capacity is reduced to ten.

The changed Graph test passed its targeted case 1/1 and complete Windows
`graph_route_cpu` suite 70/70 in the existing Release/WERROR oneDNN-enabled
build. Selector from `34a13aab` to the Graph candidate selected docs, closure,
configure and `graph_route_cpu`; 128 unrelated gates were skipped/reused.
The existing configured factor cache was reused because the candidate changed
only a test source. Docs and closure checks passed.

The distinct-expert fixture passed complete Windows CPU Direct/Graph 55/55
and 70/70, and MacBook Air Apple Direct/Graph 83/83 and 56/56. CUDA and
Vulkan Direct/Graph targeted grouped expert cases each passed 1/1 after the
fixture change. The fixture-only selector selected docs, closure and its
13/13 input tool tests; the route executions additionally verify the shared
test header on all four actual platforms. All tests executed without skips.

This is a partial R10 integration receipt. The CPU Direct path and the
CUDA/Apple/Vulkan Graph paths have not yet connected router output to expert
dispatch. The shared expert's separate sigmoid gate and final addition are
not in this Graph. HYLE's BF16 router MatMul stores BF16 before an F32 Cast,
whereas pinned llama.cpp `build_lora_mm`/`ggml_mul_mat` yields F32 logits;
that precision boundary needs an explicit decision and implementation before
model-equivalent R10 acceptance. R05, profile cells, three deliveries and P1
remain unsigned. Ledger stays 48/176 RUN_PASS; canonical Compute + Graph
implementation LOC excluding tests stays 57,949 at `0c97b284`.
