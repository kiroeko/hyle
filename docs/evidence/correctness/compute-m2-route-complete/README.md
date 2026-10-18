# M2 组合路由完整迁移证据

不可变产品候选是 `3d7e212972e27f8f532cb084f5eb5c4656b45016`，tree
`836df752e2a158f8f3b4d00bbeef9f916f1ac559`。路线产品改动冻结于
`f15204826ba001524b6a66b6fa2291400226a160`；两者之间的产品差异只有普通 package consumer 对零
workspace 的合同修正，路线源码、规则与优先级没有变化。

## 验证结论

- Windows RTX 4070 / SM89 / CUDA 13.3：CUDA backend、CUDA Provider、ggml CUDA、Marlin、Vulkan
  route、ggml Vulkan 六个定向套件通过；Compute CUDA source/shared/static 三交付均实际运行通过。
- WSL2 Ubuntu 24.04 / GCC 14：FlashInfer 独立头与四项真实 provider 测试通过，包括 FP16/BF16
  prefill/decode。
- MacBook Air `Mac16,12` / Apple M4：Apple route 独立头与 39 项 native 测试通过；显式
  `HYLE_COMPUTE_APPLE=ON` 的 Compute source/shared/static 三交付实际运行 MLX/Metal FFN 并通过。
- Windows Vulkan SDK 1.4.357.0：Compute+Graph source consumer，以及重定位 static/shared SDK 下的
  Compute 与 Graph 四个 consumer 全部运行通过；生产 build/source 在 SDK consumer 前已经隐藏。
- 精确候选上的统一组件合同通过。P0 源码检查确认 matcher 只在 Context 创建、proposal、
  prepare/compile 使用；submit/execute/bind/wait 不重新匹配，无新增分配、锁、原子、系统调用、设备
  查询或计时。

原始摘要见：

- `cuda-delivery-receipt.txt`
- `apple-delivery-receipt.txt`
- `cross-platform-route-receipt.txt`
- `vulkan-delivery-identity.json`
- `component-contract.log`
- `implementation-loc.json`

## Selector 边界

在路线产品候选 `f1520482` 上统一 selector 的 Compute/Graph gates 003--062 全部通过，之后
`factor:io_core` 因两个使用固定 UDP 端口的测试失败而停止，所以原始 selector verdict 保持
`BLOCKED`，未运行的后续 gate 也保持 `BLOCKED`，没有改写成 PASS。Windows 当时的 UDP excluded
range 是 `54319--54418`，覆盖测试端口 `54355` 与 `54394`；精确 baseline `c810475d` 在独立 build
中复现相同两项失败。候选与 baseline 的 IO 源码/测试没有差异，因此这是已证明的无关主机环境限制，
不是本候选回归。候选修正只改 package consumer，受影响的 CUDA/Apple/Vulkan 三交付均已在新候选
重跑；组件合同也在新候选独立通过。简要记录见 `selector-receipt.txt`。

## LOC

统一 `loc-report.py` 在不可变候选上以既定 Compute+Graph 边界统计 **58,666 行 / 251 个实现文件**，
排除 tests、bench、docs、third_party、samples 与构建脚本；self-check 为 `pass`。该存量统计不等于
本轮新增量。

## 限制

本轮迁移现存静态实现资格，不新增模型特例，不改变 Provider 顺序、数学、shape/resource/native
capability 检查或 fallback。没有未知 AMD/Intel/Apple 设备或其它 CUDA/Vulkan/MLX 版本的 SOTA
声明；规则只表达已有证据域，未知事实安全 decline。CPU SIMD ISA dispatch 继续由 SIMD owner 管理，
Apple Graph 的公开 Q4 storage 解码和 Vulkan reduction 资源计算继续属于语义/资源边界。
