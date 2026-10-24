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
The script records raw hashes and compares each quantizer cubin against its
pinned embedded image. Python bytecode and generated
files stay outside this source directory. Compiler/CUTLASS runtime template
notices are preserved beside the official MIT license.

The encoded group128 activation program supplies original E4M3 codes/scales
for `GroupedFloat4Linear`. Its GEMM is the HYLE-owned ordinary CUDA source
`src/compute_provider_cuda/grouped_fp4_f32_kernel.cu`. It widens each small
E4/E2M1 code tile exactly to BF16 and uses F32 MMA, scaling and accumulation;
packed weights remain compressed. The previous four FP8-MMA expert GEMM
images and their indexed adapter are no longer embedded or rebuilt. The
official upstream Python source remains unchanged for source attribution.
