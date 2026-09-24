# M3-D3 ggml Vulkan native packed-weight binding

Product candidate `2757ecc96815fd12c3050faab176cc2ddad02ccd` binds the
ten public GGUF storage formats to the pinned ggml Vulkan backend without a
resident HYLE decode, float expansion, or submit-time repack. Q3_K, Q4_K,
Q5_K, Q6_K, Q8_0, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS and MXFP4 all ran native
`MUL_MAT` on the Windows RTX 4070; IQ4_XS grouped experts ran native
`MUL_MAT_ID`.

## Correctness and fixed execution

The candidate-bound Windows selector passed docs, closure, configure,
`compute_provider_ggml_vulkan`, the unchanged limited
`compute_provider_iree_vulkan` domain, `compute_route_vulkan`, and
`graph_route_vulkan`. Direct covered all ten formats with a real ggml
quantize/dequantize oracle and two submissions. Graph covered Q4_K and
IQ4_XS captured constants plus IQ4_XS grouped experts with two submissions.
Malformed storage, wrong ports, wrong accumulation, resource lifetime and
the existing terminal failure suites also passed.

The old route gate that reserved every rank-2 F32 MatMul for the plain native
shader was the product wiring defect: it prevented packed descriptors from
reaching the mature Provider. The candidate narrows that reservation to
plain/plain F32, so packed weights reach the fixed ggml region. IREE remains
limited to its checked Scan, BF16 conversion and named FP8 domains; it does
not claim GGUF coverage.

The compressed [selector result](windows-selector-result.zip) contains the
complete machine-readable run. M3-D3 is a P0 binding task, so P1 was correctly
skipped; D5 owns full-profile cost qualification.

## Source, static and shared delivery

The checked [external consumer](consumer/main.cc) runs Q4_K Direct and IQ4_XS
Graph through the public Vulkan API twice each. It passed from the exact Git
source archive, a relocated installed static SDK, and a relocated installed
shared SDK on Windows. Before either installed consumer configured, the
extracted source and all three producer build directories were renamed away.

The installed packages contain the public low-precision/Vulkan headers and
the locked ggml, IREE and Vulkan license payloads. Their CMake files contain
no source or producer build path. The static executable has no HYLE DLL
dependency. The shared package's runtime helper placed every required HYLE
DLL beside the executable, and the consumer sets the Windows non-interactive
error mode before loading the route, preventing missing-DLL dialog boxes.

Zero-packed delivery inputs are a format-valid link/resource/replay probe;
the nonzero numerical oracle is in the candidate-bound route and Graph tests.
Exact archive, executable and fixture hashes, device/toolchain versions,
artifact checks and timings are in [delivery-receipt.json](delivery-receipt.json).

## Accounting and remaining scope

The unified Compute + Graph report is **60,923 implementation lines in 252
files**, excluding tests, benchmarks, docs, vendor and generated
build/cache/scratch trees; see [implementation-loc.json](implementation-loc.json).

This is the Vulkan D3 candidate, not the full model-profile acceptance. D5
still owns reduced-capacity real model regions, complete transfer/synchronism
cost, applicable P1 and any declared MoltenVK portability receipt. Model
loading, tokenization, sampling and new HYLE math remain outside this task.
