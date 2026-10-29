# HYLE third-party notices

HYLE — Copyright 2026 The HYLE Authors.

HYLE's original code is licensed under the Apache License 2.0 (see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE)). The components below keep their own copyright and license terms. This index
is informational; the license text and notices shipped with each component, and the notices in
its source files, control.

Exact versions, revisions, content hashes and license file hashes are pinned in
[`cmake/HyleDependencyLock.json`](cmake/HyleDependencyLock.json). Installed packages copy the
license files of the components they actually contain into `share/licenses/Hyle/`.

## In-tree vendored implementations

| Component | HYLE use | License | Full text in source |
|---|---|---|---|
| mimalloc | `mem` system allocator backend | MIT | `third_party/mimalloc/LICENSE` |
| xxHash | `container` byte hashing | BSD-2-Clause | `third_party/xxhash/LICENSE` |
| wyhash | `container` long-input hashing | The Unlicense | `third_party/wyhash/LICENSE` |
| simdutf | `container` UTF-8 validation/counting and Base64 | MIT OR Apache-2.0 | `third_party/simdutf/LICENSE-{MIT,APACHE}` |
| Boost.Context fcontext assembly | `sched` context switching | BSL-1.0 | `third_party/fcontext/LICENSE_1_0.txt` |
| picohttpparser | `io` HTTP/1 parser | MIT option of MIT OR Perl | `third_party/picohttpparser/LICENSE` |
| nanobench | benchmarks only | MIT | header in `third_party/nanobench/nanobench.h` |

## Pinned fetched implementations included in packages

| Component | HYLE use | License | Full text |
|---|---|---|---|
| Google Highway | `simd` portable register, target and mathematics implementation | Apache-2.0 OR BSD-3-Clause | fetched checkout `LICENSE` |
| xsimd | `simd` target-bound register and mathematics routes | BSD-3-Clause | fetched checkout `LICENSE` |
| oneDNN | optional CPU dense MatMul provider | Apache-2.0 | fetched checkout `LICENSE` |
| MLX | Apple Metal native regions, arrays and kernels | MIT | fetched checkout `LICENSE` |
| Apple Metal-C++ | MLX Metal/Foundation/QuartzCore declarations | Apache-2.0 | fetched checkout `LICENSE.txt` |
| nlohmann/json | MLX implementation dependency | MIT | fetched checkout `LICENSE.MIT` |
| fmt | MLX implementation dependency | MIT | fetched checkout `LICENSE` |
| nghttp2 | `io` HTTP/2 protocol core | MIT | fetched checkout `COPYING` |
| ggml (CPU, CUDA, Vulkan runtimes) | Compute metadata, CPU execution resources and native GPU regions | MIT | fetched checkout `LICENSE` |
| FlashInfer | CUDA paged/ragged attention provider | Apache-2.0 AND BSD-3-Clause AND MIT, by file | fetched checkout `LICENSE`, `NOTICE`, and `licenses/LICENSE.{cutlass,flashattention3,fmt,spdlog}.txt` |
| IREE runtime | Vulkan native program runtime | Apache-2.0 WITH LLVM-exception | fetched checkout `LICENSE` |
| flatcc | IREE runtime dependency | Apache-2.0 AND MIT AND BSD-3-Clause AND Zlib, by file | fetched checkout `LICENSE`, `NOTICE`, `include/flatcc/portable/LICENSE` and original portable header notices |
| Vulkan-Headers | Vulkan API declarations | Apache-2.0 OR MIT, by file | fetched checkout `LICENSE.md`, `LICENSES/` |
| LibTorch / ATen (CPU) | CPU ATen provider for selected original tensor sequences; common runtime also used by CUDA ATen | BSD-3-Clause for PyTorch; bundled components keep separate terms | `third_party/torch_sdk/` and original SDK license files |
| PyTorch CUDA leaf runtime | CUDA ATen provider | BSD-3-Clause for PyTorch; bundled NVIDIA and other components keep separate terms | original wheel LICENSE/NOTICE and NVIDIA runtime license files, plus common CPU runtime notices |

The locked CPU LibTorch archives omit the main PyTorch terms and some bundled notices.
`third_party/torch_sdk/` supplements them with original texts and an `ORIGIN.json` recording
their sources and verification limits. Relevant bundled components include FBGEMM
(BSD-3-Clause), AsmJit (Zlib), libuv (MIT and notices in its LICENSE), Windows Intel OpenMP
(Intel developer-tools EULA and third-party-programs), Linux GNU OpenMP
(GPL-3.0-or-later WITH GCC-exception-3.1), and macOS LLVM OpenMP
(Apache-2.0 WITH LLVM-exception and legacy MIT notices).

The exact GNU and LLVM runtime source releases have not been established from these archives.
The retained fixed-source texts identify their terms; they are not proof of the exact binary
source or satisfaction of source-offer and other redistribution requirements. Keep this
limitation with any distribution that includes those binaries. Intel DLL hashes were matched
to the supplier's 2025.2.0 package RECORD; libuv's actual DLL declares version 1.40.0.

## In-tree patches and adapted kernels

These directories hold HYLE's patch manifests or narrow adaptations of upstream code. Each carries
the upstream license and a `NOTICE`/`NOTICE.md` that names the exact origin revision and the changes.

| Directory | Origin | License |
|---|---|---|
| `third_party/ggml_runtime`, `ggml_cpu_runtime`, `ggml_cuda_runtime`, `ggml_vulkan_runtime` | ggml resource-integration patches (IQ1 types from Unsloth llama.cpp) | MIT |
| `third_party/ggml_vulkan_shaders` | unmodified ggml Vulkan shader subset | MIT |
| `third_party/ggml_q4k_apple` | ggml Q4_K row decoding expressed with MLX | MIT |
| `third_party/ggml_metal_iq1` | narrow IQ1 Metal kernels from Anemll/anemll-flash-llama.cpp | MIT |
| `third_party/flashinfer_cache` | FlashInfer mixed FP8 cache tile patch | Apache-2.0 |
| `third_party/iree_vulkan_runtime` | IREE Vulkan resource-integration patches and generated programs | Apache-2.0 WITH LLVM-exception, MIT, Apache-2.0 |
| `third_party/mlx_cast` | MLX BF16 conversion and paged attention integration, plus Metal-C++ | MIT, Apache-2.0 |
| `third_party/mlx_cpu_quant`, `mlx_cuda_quant` | MLX packed quantized equations on CPU (ATen) and CUDA | MIT |
| `third_party/awq_cpu`, `awq_cuda`, `awq_apple` | AutoAWQ unpack/dequantization sequences | MIT |
| `third_party/bitsandbytes_cpu`, `bitsandbytes_cuda`, `bitsandbytes_apple` | bitsandbytes NF4 dequantization sequences | MIT |
| `third_party/compressed_tensors_cpu`, `compressed_tensors_cuda`, `compressed_tensors_apple` | vLLM compressed-tensors int4 unpacking | Apache-2.0 |
| `third_party/torchao_cpu` | TorchAO CPU tensor sequences | BSD-3-Clause |
| `third_party/sglang_activation` | SGLang activation kernel fragment (with Triton and FlashInfer notices) | Apache-2.0 AND MIT |
| `third_party/deepseek_fp8` | DeepSeek-V3 block128 FP8 decode kernels, prebuilt with Triton | MIT |
| `third_party/deepseek_quantization` | DeepSeek-V4 quantization kernels, prebuilt with TileLang (with CUTLASS notice) | MIT AND BSD-3-Clause |
| `third_party/ds4_quantization` | antirez/ds4 CPU quantization round trip | MIT |
| `third_party/random123_counter` | Random123 Philox4x32-10 counter core | BSD-3-Clause |
| `src/compute_provider_cuda/CCCL-LICENSE.txt` | NVIDIA CCCL (Thrust, CUB, libcu++) from the installed CUDA Toolkit, instantiated into HYLE's CUDA device code | Apache-2.0 WITH LLVM-exception, BSD-3-Clause and others, see file |

## Code ported into HYLE sources

| Origin | HYLE files | Selected license | Full text |
|---|---|---|---|
| Google Highway | `src/simd/sort/vqsort_hyle/` | Apache-2.0 option of Apache-2.0 OR BSD-3-Clause | `third_party/ported/LICENSE-HIGHWAY`; source files retain upstream copyright, origin and commit notices |
| Rust standard library futex Parker | `src/platform_thread/wait.cc` Parker state machine | MIT option of MIT OR Apache-2.0 | `third_party/ported/LICENSE-RUST`; adapted from Rust 1.94.1 commit `e408947bfd200af42db322daf0fadfe7e26d3bd1` |

HYLE's Apache-2.0 terms for its modifications and surrounding original code do not remove the
permissions or conditions that apply to these third-party portions.

## Optional, build-time and system dependencies

- Tracy (BSD-3-Clause): compiled and redistributed only in trace builds; its `LICENSE` accompanies
  that package.
- glslang (multiple licenses, see its `LICENSE.txt`): used only to regenerate Vulkan shaders in
  builds that need it; not linked into or installed with HYLE.
- GoogleTest (BSD-3-Clause) and nanobench: validation only, not part of an installed package.
- Windows CUDA packages can include Toolkit cudart/cuBLAS/cuBLASLt DLLs; their Toolkit EULA is
  installed alongside the package. PyTorch's bundled NVIDIA libraries have their own terms.
- OpenSSL, externally supplied SDKs, and GPU drivers remain governed by their suppliers' terms;
  any runtime libraries copied into a distribution also require the corresponding notices.
  Explicit Windows shared OpenSSL imports retain the selected supplier's LICENSE/copyright
  and optional NOTICE with staging and installation. Ambiguous UNKNOWN imports remain external;
  a DLL is not inferred from arbitrary strings in a library archive.
- volk (MIT) remains in the central dependency lock but is not currently consumed by a product
  target; a lock entry alone does not mean its implementation is included in a package.

Highway and xsimd are consumed by product builds from centrally locked sources. Parity benchmarks
use separate clean checkouts of the same identities so that candidate and reference code cannot
share include paths by accident.
