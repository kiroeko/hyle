# M3-D4 self-operator retirement

Product candidate `91c2fb4df18f6937d64d195793a8c5b30b06a8ed` removes the
self-owned operator domains that the frozen retirement ledger found redundant,
unreachable, or unsupported by a qualified advantage. The change removes
Marlin completely, CPU native Scan/TopK, CUDA native Gather/Scan/TopK, five
dead Apple implementation files, and the Vulkan native F32 MatMul/SwiGLU
preference. Current routes now reach the mature ggml, oneDNN, CCCL, MLX and
ggml GPU implementations recorded in the ledger.

The commit changes 94 files with 277 insertions and 11,509 deletions. Within
the public and product `include/hyle` plus `src` boundary it changes 40 files
with 129 insertions and 3,685 deletions. The Marlin component ID remains only
as an append-only registry tombstone; it has no public header, implementation,
route dependency, installed export, or ordinary consumer.

## Correctness and route behavior

| Route | Exact candidate result | Covered seam |
|---|---:|---|
| CPU, Windows | 6/6 PASS | native residual, ggml standard indexing, Direct, Graph and public headers |
| CUDA, Windows RTX 4070 | 6/6 PASS | CCCL managed Gather/Scan/TopK, Direct, Graph and public headers |
| Vulkan, Windows RTX 4070 | 5/5 PASS | ggml region, Direct, Graph, headers and pinned-source contract |
| Apple, MacBook Air `Mac16,12` | 5/5 PASS | MLX/Metal Direct, Graph, Provider and headers |
| Apple, Mac mini `Mac16,10` | 5/5 PASS | independent correctness and portability |

The Apple machines consumed the exact Windows Git archive, SHA-256
`A11A9C5B3D10BC504DF9A5CDD0C86CBD4591ECB3ED2828A8F3282443EF48EBEF`.
The Air is the primary Apple correctness machine; the mini signs portability.

The Vulkan route retains a preparation-time adapter that materializes a
physically contiguous, logically transposed RHS before binding the managed
ggml region. This is an exact boundary conversion, not a restored HYLE MatMul.
CUDA's checked CCCL operations use managed execution and host-observed
completion; Graph prepares fixed commands without requiring native graph
capture. Static review confirmed that all mature choices are bound during
preparation and submit performs no Provider search, timing, or winner decision.

## Delivery

The component graph contract passed. The ordinary Compute and Graph external
C++ consumers each passed from source, a relocated installed shared SDK, and a
relocated installed static SDK. Shared consumers staged every required HYLE
DLL beside the executable; shared export hygiene inspected 15 Compute libraries
with 1,047 exports and 18 Graph libraries with 1,244 exports. No missing-DLL
dialog path was used.

D4 is a P0 retirement task. It did not run P1 or attempt to optimize an
operator that the mature route replaces. Full real-profile fragment cost and
coverage qualification remains assigned to D5.

## Accounting

The unified Compute + Graph report is **57,658 implementation lines in 228
files**, excluding tests, benchmarks, docs, vendor and generated
build/cache/scratch trees; see [implementation-loc.json](implementation-loc.json).
The machine-readable validation summary is [receipt.json](receipt.json).
