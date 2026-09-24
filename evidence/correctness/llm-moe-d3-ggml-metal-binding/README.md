# M3-D3 ggml Metal native packed-weight binding

Candidate `5a1b61fbfc7a3f131fa3c6e801fc6458ca507804` implements the
Apple packed-weight seam fixed by the M3-D3 closure card. The product imports
the public GGUF carrier directly into the locked ggml Metal backend; it does
not decode, repack, quantize, or choose a provider during submission.

## Native correctness

| Machine | Exact candidate result | Role |
|---|---|---|
| MacBook Air `Mac16,12`, Apple M4, 16 GiB, macOS 27.0 | `compute_route_apple`, `graph_route_apple`, and `compute_provider_ggml_metal` all PASS | primary native correctness and delivery |
| Mac mini `Mac16,10`, Apple M4, 32 GiB, macOS 26.6.2 | the same three complete targets PASS | independent correctness and portability |

The provider target creates and runs real Metal work for Q3_K, Q4_K, Q5_K,
Q6_K, Q8_0, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, and MXFP4. It compares each
result with the locked ggml quantize/dequantize oracle and repeats submission.
The route targets additionally cover ordinary Direct and Graph Q4_K, IQ4_XS,
and MXFP4 MatMul, IQ4_XS grouped experts, MLX to ggml Metal to MLX Graph
composition, wrong carrier capacity/schema, and early rejection of a packed
value that cannot enter a Copy/MLX region. Zero-packed delivery inputs are a
format-valid link/runtime probe; the nonzero numerical oracle remains in the
provider test.

## Source, static, and shared delivery

The checked [external consumer](consumer/main.cc) runs Q4_K Direct and IQ4_XS
Graph through the public API, twice each. It passed from the exact Git source
archive, a relocated installed static SDK, and a relocated installed shared
SDK on the Air. Before both installed executions, the extracted source and all
three producer build directories were renamed away. The shared process loaded
`libmlx` from the relocated prefix; the static executable has no HYLE or MLX
dylib dependency.

The installed SDKs contain `mlx.metallib`, the locked ggml/MLX/Metal license
payloads, and the exported
`hyle::compute::apple::input_spec(TensorDesc const&)` symbol. Their CMake files
contain neither source nor producer build paths. The first consumer compile
correctly rejected a missing public `low_precision.h` include; only that
external fixture was corrected and rerun. No product input or candidate changed.

Exact hashes, machine versions, test timings, format list, artifact checks and
the P0 cost statement are in [delivery-receipt.json](delivery-receipt.json).
[Static links](air-static-links.txt), [shared links](air-shared-links.txt), and
the [dynamic loader trace](air-shared-run.txt) retain the artifact boundary.

## Windows selector and accounting

The candidate-bound Windows selector passed all seven selected gates: docs,
closure, Apple-OFF configure, the two Apple public-header gates, and the
Compute and Graph source/relocated-shared/relocated-static consumers. Apple
native execution is not claimed from Windows. The complete machine-readable
result is archived in [windows-selector-result.zip](windows-selector-result.zip).

The unified Compute + Graph report is **60,881 implementation lines in 252
files**, excluding tests, benchmarks, docs, vendor and generated build/cache
trees; see [implementation-loc.json](implementation-loc.json). M3-D3 is a P0
binding task, so no P1 was selected. Full model-profile fragment performance
remains assigned to D5.
