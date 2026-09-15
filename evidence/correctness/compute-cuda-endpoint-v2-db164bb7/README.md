# CUDA v2 endpoint correctness evidence

Candidate: `db164bb7dced649c6ae0f5c992be5e7005748c7e`, 2026-09-15.
The authoritative scope and verdict are in the
[CUDA v2 record](../../../docs/implementation/tasks/compute/compute-backend-cuda-v2-endpoint-identity.md).
This is correctness/P0 evidence, not a timing receipt or full Compute/Graph certificate.
Native log bytes, line endings and terminal blank lines are preserved by the evidence attributes.

## Inputs and commands

Windows: RTX 4070, driver 610.62, CUDA/nvcc 13.3.73, MSVC 19.50.35728, Release/Ninja.
The CUDA product, public header and tests are unchanged between the candidate and `aaaf96f7`.
Later common build changes fix incremental DLL staging; they were replayed in the delivery tests.
Builds used at most two jobs; CTest was serial with a 120-second timeout.

- `windows-runtime-graph.log`: final `LastTest.log` from
  `C:/hyle-scratch/compute-graph-v7-cuda`. Exact targets/tests were
  `hyle_graph_executable_test`, `hyle_compute_backend_cuda_test`,
  `hyle_compute_backend_cuda_late_failure_test` and CUDA header self-containment.
  Graph 284, runtime 13, late failure 1 and the header suite all pass.
- `windows-source.log`, `windows-shared.log`, `windows-static.log`: final `LastTest.log`
  from `custom-execution-v10-source-consumer`, `custom-execution-v10-shared-cuda-consumer`
  and `custom-execution-v10-static-consumer` under `C:/hyle-scratch`.
  Each builds `hyle_custom_cuda_sample`, then runs
  `ctest --test-dir <consumer> -R ^hyle_custom_cuda_sample$ --output-on-failure --timeout 120`.
  Shared/static SDKs are explicitly installed under the same scratch root, not Program Files.

WSL: native ext4 `/root/hyle-cuda-endpoint-v2-1644bdc2`, GCC 14.2, CMake 3.28.3,
CUDA/nvcc 13.3.73, the same physical RTX 4070 through WSL. It is not an independent performance machine.
The source is the `1644bdc2` archive plus canonical Windows `cuda_backend.cc`, `cuda_memory.cc`,
`cuda_backend_test.cc` and `HyleWarnings.cmake`, matching the candidate inputs.

```sh
scripts/hyle-unix.sh cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DHYLE_DEV_FACTORS=compute_backend_cuda \
  -DHYLE_COMPUTE_CUDA=ON -DCUDAToolkit_ROOT=/usr/local/cuda-13.3 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DHYLE_BUILD_TESTS=ON -DHYLE_BUILD_BENCH=OFF -DHYLE_WERROR=ON
cmake --build build-cuda --target hyle_compute_backend_cuda_tests --parallel 2
ctest --test-dir build-cuda -L '^compute_backend_cuda$' --output-on-failure --timeout 120
```

`wsl-release.log` contains runtime, late failure and header tests. `build-cuda-asan` and
`build-cuda-ubsan` use Debug and `HYLE_SANITIZER=asan/ubsan`; only the runtime test is executed
under sanitizers (`-R '^hyle_compute_backend_cuda_test$'`), as recorded in their logs.
Host sanitizers do not instrument CUDA device code. No new TSan qualification is claimed.

## ASan environment qualification

Default ASan prevents CUDA device initialization in this WSL environment. A control containing
only `cudaGetDeviceCount` (no HYLE) reproduces it; its unqualified failure, including failed-init
leaks, is retained in `wsl-asan-vendor-control.log`, not represented as a product failure or PASS.
NVIDIA's [DALI sanitizer instructions](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/compilation.html)
recommend `ASAN_OPTIONS=protect_shadow_gap=0` for the CUDA address-space conflict.

With that setting, the control initializes one device, but the WSL shim leaves two exit-time
allocations totalling 280 bytes. A suppression file contains exactly:

```text
leak:/usr/lib/wsl/lib/libcuda.so.1
```

Qualified command environment:

```sh
ASAN_OPTIONS=protect_shadow_gap=0 \
LSAN_OPTIONS=suppressions=/root/hyle-cuda-endpoint-v2-1644bdc2/cuda-wsl-lsan.supp \
./cuda-asan-control
```

`wsl-asan-qualified-vendor-control.log` and `wsl-asan.log` both show the same 2/280 suppression.
Leak detection remains enabled; HYLE, mimalloc and other CUDA driver paths are not suppressed.
This exception is local to this reproduced WSL driver condition, not a default for other hosts.
UBSan requires no suppression. The control source is:

```cpp
#include <cstdio>
#include <cuda_runtime_api.h>
int main() {
  int count = -1;
  const auto result = cudaGetDeviceCount(&count);
  std::printf("cudaGetDeviceCount=%d count=%d\n", static_cast<int>(result), count);
  std::fflush(stdout);
  return result == cudaSuccess ? 0 : 1;
}
```

It is compiled with GCC 14, `-fsanitize=address`, the CUDA include/lib64 directories,
`-lcudart` and an rpath to that lib64 directory; no HYLE headers or libraries are used.
