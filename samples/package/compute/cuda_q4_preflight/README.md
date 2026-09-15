# CUDA Q4 carrier preflight

This is a foundation research fixture, **not a shipping Q4 Provider or a performance claim**.
It uses only public HYLE C++ headers and CUDA/cuBLASLt SDK APIs. No private ggml kernels,
fake tensor buffers or Provider SPI are included.

The same Q4_0 weights are evaluated by a small native CUDA dot-product kernel and, after
explicit cold decoding/upload, by a fixed cuBLASLt FP32 algorithm. HYLE owns device buffers,
the Queue and final Completion. The expanded vendor weights and workspace are reported;
their extra residency is not treated as zero-copy or as native FP4 support. The two routes
are checked against a host double-accumulation oracle; this fixture does not time them.

In a Developer shell matching an installed CUDA-enabled HYLE SDK:

```sh
cmake -S samples/package/compute/cuda_q4_preflight -B <preflight-build> -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<hyle-sdk>
cmake --build <preflight-build> --target hyle_cuda_q4_preflight --parallel 2
ctest --test-dir <preflight-build> -R '^hyle_cuda_q4_preflight$' --output-on-failure
```

Source embedding substitutes `-DHYLE_SAMPLE_SOURCE=<hyle-source>` for the prefix.
CUDA Toolkit 13.3+ and nvcc are user-installed. C++23 consumer code and the CUDA C++17
native TU remain separate; the installed/static HYLE runtime is not duplicated in the kernel DLL.
The recorded task determines which delivery/platform commands have actually been run:
[carrier preflight record](../../../../docs/implementation/tasks/compute/cuda-q4-foundation-carrier-preflight.md).

The fixture covers M=1/7/33, N=65 (a partial output warp group), K=96, negative/subnormal
scale values, three changed inputs and invalid block dimensions. It does not yet prove Graph
route choice, ggml-owned residency, capture, managed-runtime costs, all quantization formats
or performance superiority. Cold conversion must stay outside replay in any future adapter.
