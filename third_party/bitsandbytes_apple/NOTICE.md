# Original bitsandbytes NF4 preparation on MLX GPU

Origin: bitsandbytes-foundation/bitsandbytes
`833649043474794b8fe7a4136e0c40faf077b2e0`, MIT.
The original `bitsandbytes/backends/default/ops.py` tensor sequences
`_dequantize_4bit_compute` and `_dequantize_blockwise_compute` are expressed
in MLX C++ operations. High nibble first, supplied codebooks, block256 nested
absmax restoration/offset, F32 lookup/multiply and the final requested storage
cast remain original. No codebook, GEMM or HYLE mathematical kernel is added.
All payload operations use the owned GPU stream; MLX owns temporary arrays.
This source port requires separate output/resource and delivery qualification.
