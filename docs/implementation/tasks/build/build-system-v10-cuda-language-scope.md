# Build/package v10：CUDA 语言作用域修复

<!-- PROJECT UNIT REVIEW: CLOSED -->

> 本卡由 Mem v2 的真实 CUDA provider 验证触发。Build/package 是横向开发单元，
> 不进入 `module-status.json`，也不替 Mem 或 Compute 签产品能力。

## Closure card（AUDITED）

- unit / round / kind：`build-package` / `build-v10-cuda-language-scope` /
  `DEFECT`；
- baseline：`e27d11df3b5ddfa495342ab5a7a1802bb357348f`；
- 可复现缺陷：Windows、CMake 4.2.3 与 4.4.0、Ninja、CUDA 13.3，选择
  `HYLE_COMPONENTS=compute`、`HYLE_COMPUTE_CUDA=ON` 时 CUDA 编译器检测成功，
  但 generate 阶段因 `_CMAKE_CUDA_WHOLE_FLAG` 和
  `CMAKE_CUDA_COMPILE_OBJECT` 缺失而失败；
- 根因：`enable_language(CUDA)` 在
  `hyle_configure_compute_dependencies()` 的函数局部作用域执行，CMake 语言规则没有
  留在创建 CUDA target 的目录作用域；
- 有限 FIX：只把 CUDA Toolkit 探测、architecture/install 前置条件和
  `enable_language(CUDA)` 提升到 module-owned include 的目录作用域；依赖登记仍留在原函数；
- 非目标：修改 Compute API/算法、改变 CUDA AUTO/ON/OFF 语义、增加 option、重构其他依赖、
  更改 package ABI 或公共收口仪器；
- 验证：CUDA OFF 不探测；CUDA ON + Vulkan ON 可配置、编译并运行现有 Compute 测试；
  closure/docs 检查通过；
- 停止条件：上述两条配置路径通过即关闭；若仍由工具链阻塞则记录精确环境并停止，
  不扩张构建系统。

## FIX ledger

| ID | requirement | state |
|---|---|---|
| `BUILD-V10-CUDA-01` | CUDA 语言规则与 CUDA target 位于可见的共同目录作用域 | `FIX` |
| `BUILD-V10-OFF-02` | `HYLE_COMPUTE_CUDA=OFF` 保持零 Toolkit/CUDA language probe | `KEEP` |
| `BUILD-V10-SEM-03` | AUTO/ON/OFF、portable install architecture 与依赖登记语义不变 | `KEEP` |

## Verification 与裁决

修复把 Toolkit 探测和 `enable_language(CUDA)` 放在 selected Compute include 的目录作用域；
依赖登记及 Vulkan/ggml 发现仍在原 dependency function 内。验证结果：

- Windows MSVC 19.50、CMake 4.4、Ninja，CUDA OFF + Vulkan OFF：configure/build 通过，日志明确
  `dependency probe and CUDA language skipped`；
- 同机 CUDA 13.3 / RTX 4070，CUDA ON（sm_89）+ Vulkan ON：configure/generate、CUDA/Vulkan
  backend 编译及 `hyle_compute_test`、`hyle_compute_gpu_await_test`、
  `hyle_compute_vulkan_test` 全部通过；
- WSL GCC 14 Host Compute/Mem 与 public-header 测试通过；
- `check-module-closure.py` 与 `check-docs.py` 通过。

结论：`CLOSED`。这是一项语言作用域 correctness 修复，不改变 option、产品 target、runtime API、
CUDA architecture 合同或正常运行时成本。
