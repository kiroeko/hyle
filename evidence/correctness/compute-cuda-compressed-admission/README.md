# Compressed CUDA kernel admission

Research only; [scope and verdict](../../../docs/implementation/tasks/compute/cuda-compressed-provider-admission.md).
Not a shipping Provider, graph/capture qualification, throughput benchmark or SOTA claim.

The fixture compiles the unmodified original Marlin TU from reference commit
`1f25790bdd49fba53106164a24666dade68d7c90`. Its SHA-256 is
`dd7ee08c62cb0ef6d6f905bc6bbd8adfb7808a95f62127eb52707c56f72eabae`.
The host packing permutation is a C++ adaptation of that revision's `marlin/__init__.py`:
Copyright (C) 2024 Elias Frantar, [Apache-2.0 license](../../../third_party/simdutf/LICENSE-APACHE).
This is a research reference, not a new HYLE product dependency or a claim about the current vLLM implementation.

Windows RTX 4070, SM89, 46 SMs; CUDA/nvcc 13.3.73, MSVC 19.50.35728. The CUDA-enabled shared SDK is the
existing `C:/hyle-scratch/custom-execution-v10-shared-sdk` used in
[CUDA v3](../compute-cuda-resource-sync-v3/README.md), with candidate `be2d8b059ca858d2e9b3393b38d2530d4eb3dbe3`.
No HYLE product rebuild or product changes were required. The reference is clean after compilation.

```text
cmake -S evidence/correctness/compute-cuda-compressed-admission/probe -B C:/hyle-scratch/cuda-compressed-admission-20260915 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=C:/hyle-scratch/custom-execution-v10-shared-sdk -DMARLIN_REFERENCE=C:/public/project-hyle/marlin
cmake --build C:/hyle-scratch/cuda-compressed-admission-20260915 --target hyle_marlin_admission --parallel 2
ctest --test-dir C:/hyle-scratch/cuda-compressed-admission-20260915 -R ^hyle_marlin_admission$ --output-on-failure -V
```

Use the matching VS developer environment. The local source reference is supplied explicitly; the fixture does
not download dependencies. Only the kernel source is compiled, not the original PyTorch wrapper.

| Evidence | Result |
|---|---|
| [Configure](windows-configure.log) | PASS, finds the installed SDK and CUDA |
| [Initial build](windows-build.log) | FAIL, original TU needs relaxed constexpr for host constexpr helpers called from device code |
| [Corrected build](windows-build-corrected.log) | PASS after the single fixture correction: `--expt-relaxed-constexpr`, MSVC UTF-8; source unchanged |
| [Run](windows-run.log) | PASS, M=1/7/64, N=K=1024, group128, three changed inputs each |
| Retained public CUDA journey: [source](retained-source-consumer.log), [static](retained-static-consumer.log), [shared](retained-shared-consumer.log) | PASS, existing custom-kernel consumer binaries replayed without rebuilding unchanged products |

Weights (524,288 bytes) and scales (16,384 bytes) remain packed on device: 528 KiB total plus 512 bytes of
exclusive lock workspace. No expanded device weight buffer exists; host-only oracle data is not a runtime route.
After cold uploads and workspace initialization, all nine calls use the same packed storage and workspace;
HYLE Queue/Completion encloses each native stream launch and readback follows Completion.

The predeclared L1 error ratio threshold is 0.001, following upstream `test.py`, but checked against a host double
oracle over the actual rounded FP16 weights/inputs. Observed ratios: 0.0002763–0.0003879; maximum absolute
error: 0.02984. This is a limited carrier check, not bitwise FP32 accumulation, an elementwise relative-error
guarantee, arbitrary-model accuracy or qualification of exceptional floating-point inputs.

Known upstream warnings are retained: an unused variable and constant-zero divisors in guarded per-column
template specializations. This run selects group128; it does not qualify per-column behavior. The original launcher
also has unchecked CUDA attribute results and repeated launch selection/attribute setup. Those must be handled
in a future prepared adapter. Runtime-internal allocation, concurrency, capture, injected faults, Linux,
source/static delivery and performance were not tested here.

The last table row covers retirement of the old sample, not Marlin source/static delivery. A read-only
`git diff --name-only be2d8b059ca858d2e9b3393b38d2530d4eb3dbe3 6619da46 -- include src cmake CMakeLists.txt`
is empty, so existing product artifacts remain applicable. The change selector requests `delivery:compute`;
this bounded task replayed those existing public CUDA journeys instead of running its default fresh-SDK/oneDNN
build script. No automated selector receipt for that default command is claimed.
