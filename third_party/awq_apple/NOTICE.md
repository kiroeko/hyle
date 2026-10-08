# Original AutoAWQ GEMM packing preparation on MLX GPU

Origin: casper-hansen/AutoAWQ
`88e4c76b20755db275574e6a03c83c84ba3bece5`, MIT.
`awq/utils/packing_utils.py::unpack_awq`, `reverse_awq_order` and
`dequantize_gemm` are expressed with MLX operations. Column packing, the
eight-column reverse permutation, actual GEMM zero points, complete groups,
scale dtype and final [N,K] transpose remain original. U32 masked shifts
are equivalent to the original signed-word extraction for these four bits;
the subtraction remains exact in [-15,15]. No HYLE dot product is added.
MLX owns GPU arrays and layouts; output/resource/delivery qualification is
separate from source translation.
