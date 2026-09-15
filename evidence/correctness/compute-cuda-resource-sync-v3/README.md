# CUDA 同步资源修复 v3 证据

产品候选 `be2d8b059ca858d2e9b3393b38d2530d4eb3dbe3`；见
[closure card](../../../docs/implementation/tasks/compute/compute-backend-cuda-v3-sync-resource.md)。
原始 CTest 日志保留本机字节/行尾；日期展示编码不影响用例输出。没有性能计时资格声明。
下面的 Q4 样例路径和命令是历史运行事实；样例随后退役，可从 `6619da46` 恢复。CUDA 产品与专门的
同步回归仍保留，见 [退役与替代研究记录](../../../docs/implementation/tasks/compute/cuda-compressed-provider-admission.md)。

| 日志 | 内容 |
|---|---|
| [original-q4-preflight.log](original-q4-preflight.log) | `e8e0c95e` 样例 + v2 产品：cuBLASLt 第三次输出仍是 poison |
| [original-stream-gate.log](original-stream-gate.log) | `cd00f76d` 测试 + v2 产品：受控 gate 下 fill/copy 都提前返回 |
| [windows-runtime-graph.log](windows-runtime-graph.log) | v3：CUDA 15/15、late error、逐头、Graph 284/284 |
| [wsl-release.log](wsl-release.log) | v3：Linux runtime、late error、逐头 |
| [wsl-asan.log](wsl-asan.log) / [wsl-ubsan.log](wsl-ubsan.log) | 各 15/15；ASan 仅有已记录的 driver 豁免 |
| [windows-custom-source.log](windows-custom-source.log) / [shared](windows-custom-shared-cuda.log) / [static](windows-custom-static.log) | 两独立 Invocation、双 lane、commands/capture、变化输入、零 host wait |
| [windows-q4-source.log](windows-q4-source.log) / [shared](windows-q4-shared.log) / [static](windows-q4-static.log) | 未使用诊断 control 的 Q4 native/cuBLASLt 默认路线 |

Windows：MSVC 19.50.35728，CUDA Toolkit/nvcc 13.3.73，RTX 4070，driver 610.62。源码入口
`samples/package/compute/cuda_q4_preflight`；source prefix 为 canonical checkout，两个安装 prefix 为
`C:/hyle-scratch/custom-execution-v10-{shared,static}-sdk`，从对应 `-build` producer 显式安装。
Q4 缓存为 `C:/hyle-scratch/cuda-q4-carrier-{source,static}-20260915`，shared 使用
`C:/hyle-scratch/cuda-q4-carrier-preflight-20260915`。原 custom consumer 沿用
`C:/hyle-scratch/custom-execution-v10-{source,shared-cuda,static}-consumer`。

```text
cmake --build <build> --target hyle_cuda_q4_preflight --parallel 2
ctest --test-dir <build> -R ^hyle_cuda_q4_preflight$ --output-on-failure --timeout 120 -V
cmake --build <custom-build> --target hyle_custom_cuda_sample --parallel 2
ctest --test-dir <custom-build> -R ^hyle_custom_cuda_sample$ --output-on-failure --timeout 120
```

Linux：GCC 14，原 `/root/hyle-cuda-endpoint-v2-1644bdc2` native-ext4 archive，不是 Git checkout；不得
把该目录向上找到的旧 root Git HEAD 当 candidate。本轮从 Windows 同步 `cuda_memory.cc` 与 runtime test，
并 `cmp` 确认 CUDA 两个 cc、私有头、公开头、产品/test CMake、测试源及 HyleWarnings 与 canonical 一致。
相对原 archive 的其余新 common CMake 更改属于 Windows DLL staging，与 Linux 此 factor 不相干。
三个缓存分别为 `build-cuda`、`build-cuda-asan`、`build-cuda-ubsan`。

```text
cmake --build build-cuda --target hyle_compute_backend_cuda_tests --parallel 2
ctest --test-dir build-cuda -L compute_backend_cuda --output-on-failure --timeout 120
cmake --build build-cuda-asan --target hyle_compute_backend_cuda_test --parallel 2
ASAN_OPTIONS=protect_shadow_gap=0 LSAN_OPTIONS=suppressions=/root/hyle-cuda-endpoint-v2-1644bdc2/cuda-wsl-lsan.supp ctest --test-dir build-cuda-asan -R hyle_compute_backend_cuda_test --output-on-failure --timeout 120
cmake --build build-cuda-ubsan --target hyle_compute_backend_cuda_test --parallel 2
ctest --test-dir build-cuda-ubsan -R hyle_compute_backend_cuda_test --output-on-failure --timeout 120
```

shared producer、SDK、custom consumer 和 Q4 consumer 的 CUDA DLL SHA256 均为
`BD8890A92E567CD8E0FA9DAE6E25B35657748B9567A504E0DEDB3D43AE534B93`；独立 source Q4 producer 与
staged DLL 均为 `263DC810AEC487B017993BAE2C71A98BF9CE24E21E9DD8851D0C6937D6A60B4B`。
不同 producer 配置不要求二进制相同；各自 delivery 内已核对，不手工替换旧 DLL 掩盖 staging。
