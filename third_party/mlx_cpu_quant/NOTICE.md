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

Affine integer widths4/5/6/8 share the original continuous low-first U32
bitstream. The5/6bit ATen extraction joins the next word for crossing codes;
I32 storage is normalized to unsigned I64 before joining. Group slices use
exact group*bits/32 words. F32 affine decode/dot, typed embedding multiply
then add, and MX4/8 scale casting retain their prior numerical sequences.

Grouped activation sharing preserves the original codec and numeric sequence.
A token may supply one activation row for every selected expert, including a
real zero-slot-stride view with token padding. IDs and outputs retain separate
slots; no full T*S*K activation expansion or dense expert bank is introduced.

Explicit affine A*W keeps the original physical K bitstream and scale groups.
The adapter bounds decoding to an N tile of at most64 rows and one K group
of selected heads/experts, then uses F32 ATen matmul and accumulation. N may
have a tail. It does not physically transpose or expand a full float bank.
