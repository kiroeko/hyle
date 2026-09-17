# R6 Vulkan：公开产品与三种 C++ 交付

## Closure card / CANDIDATE

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

## 审计裁决与有限 FIX

- D1 FIX：只给现有 Compute/Graph facade 分别追加真实 Vulkan root。依赖方向仍从 Graph Vulkan
  指向 Compute Vulkan；两个因子的 OFF 已有 INTERNAL_ONLY_ABSENT，普通通用头不带 SDK。
- D2 KEEP/FIX：沿现有 locked FetchContent、package registry 的 EXPORT_TARGETS、
  DEVELOPMENT_DIRECTORY_SPECS 和许可证投影。SPIR-V 已实际编译后嵌入产品 C++，安装库不依赖
  glslang、外部 shader 文件或 producer build 路径。Vulkan-Headers 是公开 native 接口的一部分，
  Vulkan loader/driver 属用户系统 runtime；HYLE 用既有 Platform DynLib 冷加载，不隐式下载 driver。
  Windows HYLE shared DLL 继续由现有 helper 邻置；不新增无需要的资源 staging 系统。
- D3 FIX：现有 package Direct/Graph 样例按 HYLE_COMPUTE_VULKAN 条件增加完整 F32 FFN，
  使用相同权重/独立 oracle、M1/M16、变化输入与 replay；同一已编译 custom SPIR-V 在源码和两种
  SDK 交付中组合。Graph 用普通 typed DAG，Direct 使用原生 Recorder 与 typed prepared 片段，
  不引入私有通用 DAG、测试 Provider 或只 copy 的假消费。指南区分 generated owning result 与
  native fixed caller-output 合同，展示普通单算子 prepare/bind/submit。

七问：第三方 pin/许可明确；完整计算和资源消费驱动；复用原生路线/package/Platform；本卡仅冷交付
P0；纯 C++ 无 FFI；同源三交付；ordinary 与 native 专家责任可发现。D1–D3 固定为 AUDITED FIX，
适用门仍未验收。十三旧 owner 的 C++/规则/P1 沿 baseline，不因 roots 变动重跑性能。

## 两系统三交付检查点

`60de42121b1361cc9fadecc33806404ddc84a2df` 的
[Windows / Air 原始证据](../../../../evidence/correctness/vulkan-r6-three-delivery/README.md)
已完成 source/static/shared 的真实 Direct/Graph FFN + custom、变化输入与重复执行。
同一完整源码包在两系统以离线锁定依赖消费；两种 installed SDK 均迁移前缀，Windows shared DLL
邻置，Air 还隐藏全部原 producer 源码/build 路径后再编译运行独立消费者。头文件、嵌入 shader、
许可证和其它路线 binary 缺席检查通过；保留实际命令、日志、链接和 archive/entry hash。

本检查点只补交付证据，没有修改产品或启动 P1。中央 public roots 引起的 broad selected gates
仍须按输入身份执行或复用，且最终 review 尚未完成；十三 owner 保持 AUDITED，不提前签 CLOSED。

## 候选冻结

D1–D3 的公开 roots、依赖/嵌入资源和普通消费者已经固定，十三 owner 从 AUDITED 进入 CANDIDATE，
产品 commit 为 `8e45b10423686a0753aaa9d461c4ef3b117412c5`。它们原 C++/算法/规则与性能证据仍
沿本卡继承快照；本轮冻结的是 Vulkan 交付变化，不重新裁决 CPU/CUDA/Apple 性能。

最终完整源码交付快照为 `20a029aded4bc00c9fc80a182bd3051853e7c99e`；与产品候选的差异只有
benchmark 路径规范化、文档/状态/证据，包含校正后的路线导航。Windows 的原有 source-package
target 已生成并校验完整源码包，SHA256
`22d4cce9a4f9c9f912b62877c8bc2a13a6d17f08a96d40c5499a3dbb5e3c5fa6`。
生成包不等于消费验收：最终两系统 source/static/shared、中央 selector 及收口仍须完成。

## P0 继承与制品身份

[逐路径 Git 身份](../../../../evidence/correctness/vulkan-r6-product-delivery/p0-unchanged.json)
确认 baseline `a1329dd2` 到候选 `8e45b104`，十三既有 owner 的全部源码目录及 Compute/Graph
公开头目录共十五棵 tree 完全相同。本卡改变公开 roots/交付，不改变它们的 C++、算法或布局；
CPU/CUDA/Apple 原测量候选、限定域与 CPU 用户例外继续按 baseline 继承。

另对 `8e45b104`、Windows 测量 `3894b72e`、Apple 测量 `a41d43d5` 与最终源码快照 `20a029ad`
核对顶层 CMake、cmake/include/src/third_party/scripts/samples 七项 shipping 输入身份，全部相同。
不同的是 benchmark、文档、状态和证据；这不把尚未执行的最终三交付或 selected gates 写成 PASS。
