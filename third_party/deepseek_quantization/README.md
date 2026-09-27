# Original DeepSeek V4 quantization roundtrip programs

The fixed official source and its identity are in `provenance.json`. Original
`act_quant_kernel` and `fp4_quant_kernel` execute group64/128 E4M3FN and group32 E2M1
quantization then dequantization to BF16. Both use upward power-of-two scales,
not a different MX scaling policy. Each independent group is one original row.

Only generated SM89 cubins enter the C++ product. No Python, PyTorch or TileLang
is needed to build or run HYLE. CUDA Driver API13040 is the currently qualified
runtime. Source, static and shared C++ all include the same artifacts/licenses.

Offline rebuild: install TileLang0.1.14 and apache-tvm-ffi0.1.12, then from a
CUDA13.3/MSVC developer environment run `python rebuild.py --output <scratch>`.
The script records raw hashes and compares the entire cubin against its pinned
embedded image. NVCC's eight-digit internal symbol hashes can vary with the source
path: only those eleven fields inside the indexed GEMM ELF string table are
ignored. Instructions, constants, relocations and every other byte must match.
Raw hash mismatches remain visible in the receipt. Python bytecode and generated
files stay outside this source directory. Compiler/CUTLASS runtime template
notices are preserved beside the official MIT license.

Indexed FP4 expert programs preserve the original `fp4_gemm_kernel` math.
`indexed_entry.py` changes only the generated entry qualifier and adds checked
expert/token pointer selection before calling that body with M=1. No FP4
arithmetic is added by HYLE. Four actual DeepSeek N/K pairs are instantiated;
the encoded group128 activation program supplies original E4M3 codes/scales.
The per-slot M32 tile utilization is part of this implementation's cost.
