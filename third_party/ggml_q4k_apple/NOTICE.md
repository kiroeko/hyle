# Original ggml Q4_K selected-row decoding on MLX GPU

Origin: ggml-org/ggml `456172ec733a135778adcd32d00e576a58232e45`, MIT.
`src/ggml-quants.c::dequantize_row_q4_K` and `get_scale_min_k4` are
expressed as MLX tensor operations over original144-byte/group256 blocks.
The two original half scales are widened before F32 scale/offset operations;
six-bit scale/min reconstruction and alternating low/high nibble groups are
unchanged. The output is the original F32 decode, not a replacement format.
The adapter first selects checked rows on the GPU; this header only decodes
those rows. No full weight expansion, CPU model execution, HYLE algorithm or
new shader is introduced. MLX owns temporary storage and native compilation.
Exact source sequences require runtime/encoding/resource qualification.
