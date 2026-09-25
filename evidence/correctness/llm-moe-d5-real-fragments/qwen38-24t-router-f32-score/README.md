# Qwen3.8-2.4T BF16 router F32 score boundary

Candidate `774c3a32d` adds the narrow public `BFloatLinearF32` operation for
BF16 activation `[T,K]`, native output-major BF16 weight `[E,K]`, and F32
scores `[T,E]`. The fixed ggml CPU, CUDA, Metal, and Vulkan providers lower it
to `ggml_mul_mat` with F32 accumulation. Generic `MatMul` keeps its existing
contract. The operation is bound during preparation; execution does not search
providers or copy a persistent F32 weight matrix. Vulkan's fixed small-matvec
pipeline widens only the small BF16 activation on device before the multiply.

The prior BF16-score path failed the new CPU near-tie test: experts 9 and 10
became equal after BF16 storage and the tenth selected ID changed. The new F32
path passed the same 512-way, top-10, four-replay Direct and Graph tests, with
different inputs and an independently calculated membership and normalized
weight oracle. CPU Graph's separate grouped-expert test also connects changing
router IDs to ten distinct resident experts and a weighted combine. This is a
partial R10 result; the shared expert branch is not connected there.

Targeted Windows Release/WERROR results on RTX 4070:

| Route | Provider, Direct, Graph |
| --- | --- |
| CPU | `compute_ops` 1/1, ggml CPU 1/1, Direct 1/1, Graph 2/2 |
| CUDA | ggml CUDA 1/1, Direct 1/1, Graph 1/1 |
| Vulkan | ggml Vulkan 1/1, Direct 1/1, Graph 1/1 |

On the formal MacBook Air M4, the ggml Metal provider, Apple Direct, and Apple
Graph targeted cases each passed 1/1. The relevant complete Apple CTest suites
`compute_ops`, `compute_provider_ggml_metal`, `compute_route_apple`, and
`graph_route_apple` passed 4/4, serially, in 73.65 seconds. The Direct and
Graph tests executed the upstream `kernel_mul_mv_bf16_bf16_4` path; the
provider test executed `kernel_mul_mv_bf16_bf16_short`.

The immutable Windows change-impact plan from `d59ad12e2` selected 27 gates:
docs, closure, configure, affected Compute/Graph owners and consumers, the
public header check, and Compute/Graph deliveries; 105 gates were skipped or
reused. The first configure lacked the Vulkan SDK environment. A corrected run
reached the ggml CUDA consumer build but inherited ggml's all-architecture CUDA
default; it was stopped before spending further time on nonlocal GPU targets.
The interrupted selector lease prevents a verdict for that candidate. The
complete Windows selector and delivery result is therefore pending, and this
receipt does not mark those gates PASS.

This change does not sign a full Qwen3.8-2.4T profile cell. The CPU shared
expert and final addition, the other routes' dynamic dispatch/combine, complete
R05, relocated three-delivery consumers, and applicable P1 remain to be done.
The D5 ledger remains 48/176 `RUN_PASS`.
