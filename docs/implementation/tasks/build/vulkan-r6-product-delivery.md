# R6 Vulkan：公开产品与三种 C++ 交付

## Closure card / SCOPED

- baseline：`a1329dd2bbed2cf2b4e334345273b6c970052702`。R1–R5 与 Vulkan Windows 原生检查点保留。
- owner：`compute`、`graph`、`compute_backend_cpu`、`compute_backend_cuda`、
  `compute_provider_native`、`compute_provider_onednn`、`compute_route_cpu`、`compute_route_cuda`、
  `graph_provider_onednn`、`graph_route_cpu`、`graph_route_cuda`、`compute_route_apple`、`graph_route_apple`。
  这是现有两 facade root union；仅绑定公开 roots/交付变化，不重审它们的既有 C++ 实现。
  两 Vulkan owner 继续归 [R6 主卡](../compute/vulkan-v1-four-route-foundation.md) V1–V7。
- trigger：两个原生 Vulkan factor 已真实实现，但普通 `hyle::compute/graph` 产品尚未包含它们。
- 继承：[改动前 registry](../../../../evidence/correctness/vulkan-r6-product-delivery/inherited-baseline.json)
  保存十三 owner 的 candidate、terminal 与性能。CPU 例外、CUDA 限定资格、Apple M4 P1 均不重跑。
- 主旅程：Vulkan ON、CPU/CUDA/Apple OFF → 同一 C++ Direct/Graph 完整 F32 门控 FFN、变化输入、
  checked owning result；Windows RTX 4070 与 Air MoltenVK 的 source/static/shared。
- 正交旅程：native custom SPIR-V 的固定资源组合；OFF/无后端/core 裁剪、SDK relocation、Windows
  DLL 邻置、离线缓存与许可证。普通用户不使用内部 factor target 或手动 Provider 列表。
- 范围：下列 D1–D3，复用既有 package registry/staging；不重写构建系统、selector 或性能仪器。
- 审计预算：15 分钟核对 root/依赖/嵌入资源/样例；实现最多两次具名 package seam 纠正。
  重型验证串行，parallel 2；发现公共工具缺陷单独定界，不在本卡扩建工具。
- 验证：两系统三交付的真实 Direct/Graph + custom 计算、变化输入/独立 oracle；relocation、
  安装头/导出/许可证、offline、OFF 无 SDK/其它路线获取、逐头；按 selector 执行或复用 broad 门。
- 性能：P0，十三 owner 原 C++/算法/布局不变；Vulkan P1 归主卡稳定候选。
- stop：D1–D3 与适用门完成后 CANDIDATE→CLOSED；未运行项不签 PASS，不能把三交付留给 R7。

## 有限待审项

| ID | 缺口与动作 | 验收 |
|---|---|---|
| D1 | 两 facade 增加真实 Vulkan roots；OFF 沿既有 absent 规则 | 无 SDK/第三方泄漏，普通 target 可发现实际能力 |
| D2 | 嵌入 SPIR-V、Vulkan-Headers 与系统 loader 的源码/二进制责任 | shader 来源/许可证和 locked headers 随包；二进制用户不依赖 glslang/producer build |
| D3 | 普通 Direct/Graph 和 native custom 的同源样例与指南 | 两系统 source/static/shared、relocation/offline、checked 数值与资源生命周期 |

先固定 AUDITED FIX 再修改公开产品 roots。当前仅登记 scope，未签交付完成。
