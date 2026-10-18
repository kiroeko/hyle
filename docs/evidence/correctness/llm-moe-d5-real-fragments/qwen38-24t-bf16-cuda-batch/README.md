# D5 Qwen3.8-2.4T official BF16 CUDA fragment batch

Immutable product candidate `05b374e61` changes only the CCCL provider's CUDA
runtime qualification from exact 13030 to 13030..13040. SM 8.9, compiled
Toolkit 13030, and CCCL-300304 remain exact. No operator math, priority, or
other provider changes. The Windows build used CUDA Toolkit 13.3 headers
(`CUDART_VERSION=13030`, `CUB_VERSION=300304`); the RTX 4070 machine reports
runtime and driver API version 13040. The earlier exact rule rejected CCCL
before execution, causing both the new 2.4T and existing 27B Gather consumers
to fail preparation. CUDA 13 Windows can obtain its runtime from the display
driver, so these version numbers need not match: see the
[NVIDIA Runtime API version documentation](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART____VERSION.html)
and [minor-version compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

After the scoped rule change, the CCCL owner targeted suite passed 10/10 and
the complete owner suite passed 42/42. CUDA Direct passed 53/53 and CUDA Graph
passed 70/70, with the local Qwen3.5 fixture and no skips. The official
Qwen3.8-2.4T profile-derived subset passed 5/5 on each entry. It reuses the
CPU batch's exact H=8192, shared expert N=2048, GQA 64/4 x 256, GDN
K16/V128 x 128, and MTP 16384 shapes, exercising prepared/compiled replay
with changing inputs and independent oracles. The R11 BF16 grouped expert
already has separate four-route evidence.

This is partial fragment evidence. The R10 512-way router, stable top-10,
dispatch, and combine chain is not assembled here. The R05 model projection
and storage boundary around GDN is also not yet complete. Therefore no full
profile cell, route-level source/static/shared profile delivery, or P1 result
is signed. The D5 ledger remains 48 `RUN_PASS` / 128 `VALIDATION_NOT_RUN`.
The Compute + Graph implementation count excluding tests remains 57,938 lines.
Machine-readable details are in [receipt.json](receipt.json).
