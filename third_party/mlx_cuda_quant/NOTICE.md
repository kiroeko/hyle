# Original MLX packed quantized equations on CUDA

Origin: ml-explore/mlx `2b2bb49790158ecd008d1685c94e3ef119792a00`, MIT.
The port expresses `backend/cpu/quantized.cpp::_qmm_t_simd`,
`fp_qmm_t_simd`, and `ops.cpp::affine_dequantize`, plus
`python/mlx/nn/layers/quantized.py::QuantizedEmbedding::__call__`, with the
original CUDA `qmm/qmv.cu` warp-to-row/gathered-vector structure. Original affine
equation and MX per-group dot-before-scale order are preserved, with F32
accumulation and the declared output/scale dtype boundary. Source FP4_LUT and
E8M0 exponent bias127 are unchanged. F32 arithmetic association follows the
native matrix kernel, rather than a claim of bitwise SIMD lane-order identity.

The adapter supplies checked shapes/indices and decodes codes into registers
for the current K group. It never materializes a global decoded group or float
expert library or substitutes bias+8*scale for original affine weights.
No model loader, new quantization format or SIMD extension is introduced.
Exact source URLs/hashes are retained in SOURCE.json.

CUDA carrier: original group arithmetic, source dtype/rounding and F32
accumulation above are unchanged. FP4 LUT/code extraction and F32 warp reduction
stay in the native kernel; the existing HYLE Queue status protocol checks IDs
before payload access and reports failures through Completion. No CPU fallback
or temporary float weight tensor is used. Original MLX narrow Element weight
rounding is not substituted for the widened CPU equation. Floating reduction
association remains subject to the existing numerical tests, not bitwise SIMD
lane-order identity. Embedding keeps its separate original dtype rounding.
