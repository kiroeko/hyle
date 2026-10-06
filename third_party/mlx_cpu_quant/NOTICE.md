# Original MLX CPU packed quantized tensor sequences

Origin: ml-explore/mlx `2b2bb49790158ecd008d1685c94e3ef119792a00`, MIT.
The port expresses `backend/cpu/quantized.cpp::_qmm_t_simd`,
`fp_qmm_t_simd`, and `ops.cpp::affine_dequantize`, plus
`python/mlx/nn/layers/quantized.py::QuantizedEmbedding::__call__`, in ATen C++.
ATen owns actual matrix, bitwise, lookup and cast execution. Original affine
equation and MX per-group dot-before-scale order are preserved, with F32
accumulation and the declared output/scale dtype boundary. Source FP4_LUT and
E8M0 exponent bias127 are unchanged. F32 arithmetic association follows the
native matrix kernel, rather than a claim of bitwise SIMD lane-order identity.

The adapter supplies checked shapes/indices and bounds temporary decoding to
one K group of selected experts. It never materializes the complete float
expert library or substitutes bias+8*scale for original affine weights.
No model loader, GPU implementation, new HYLE codec/GEMM or SIMD extension is
introduced. Exact source URLs/hashes are retained in SOURCE.json.
