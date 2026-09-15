# Compute v15：受检的具体布局实例化

## Closure card / AUDITED

- owner：`compute`；baseline `6dcce2df`；承接 [v14](compute-v14-compilation-storage-facts.md)、C93 与共同
  foundation，原有预算/完成边界不变。这是仍活动的 shape/layout 合同后继，不重开其它 CLOSED factor。
- trigger：Graph G101 的 slice→permute→Copy 与 broadcast→Copy 已能绑定真实 strides，但现有
  `shape_domain.cc::bind_tensors` 只能重建 dense，`OperationPreparationAccess::instantiate` 不能消费
  Graph 冷推导的具体描述。把 strided envelope 当 dense 会改变寻址，不能只删除拒绝分支。
- 正交旅程：原有 dense 动态 RMSNorm 的输出、alias、workspace、owner 与重复运行保持。
- scope：C150/C151；不新增 kernel/算子族、公开任意 stride 表达式、Graph ID 或第二个 shape 求解器。
- 依赖复用：ComputeOps TensorDesc/schema、现有有限 shape program、Mem allocator、Error。
- 验证：布局族/错误标记、固定 shape 可变 strides、zero/stride/格式/上界/guard/arity；小域穷举；原 Native
  RMSNorm 的 supplied-description 正确执行、错误描述在 Provider 前拒绝、OOM/生命周期；Compute/Native/
  Graph/oneDNN 定向回放、逐头、Windows CUDA 与三交付。跨平台/sanitizer/独立红队/P1 仍随完整候选。
- P0：全部校验/实例化在冷路径，复杂度 O(D + guards + ports × rank)，无 payload 读取；原 exact prepared
  不增加字段/分配/分支。域 axis map 的额外布局族与上界是 opt-in 冷 metadata，不声称零字节。
- 预算：一轮表示/实现与一次产品纠正；若要改变热 replay、全域覆盖或 ownership，先回本卡审计。

## 固定改动集

| ID | 实现与边界 |
|---|---|
| C150 | 私有 ProviderTensorAxes 区分既有 kDerived 布局与显式 kBoundedStrided。后者仅 plain，保持 dtype/rank/维度映射，允许合法非负 strides（含零步幅）且 required_bytes 不超过显式上界；不自动承诺可并行写入/无重叠。固定 extent 的 strided 家族也不是 exact point。 |
| C151 | 私有 instantiate 增加 caller-supplied input/output descriptions 的受检重载：先求值 guards，逐维/布局/字节校验，再复用原 schema inference、Provider 冷 callback、workspace/private/alias 校验和 owner 保活。旧自动重建入口对未提供具体 strides 的家族拒绝，绝不猜布局。 |

kDerived 保持旧 exact strides/opaque 与动态 dense/blocked-dense 规则；不能通过新入口放宽旧域。上界
是可触及的 storage span，不是 logical element bytes；envelope 不是必然合法点。Provider 承诺 entire-domain
时必须覆盖所声明布局族中的每个合法描述，不能只支持其 envelope 的 strides。现有 Native RMSNorm 明确
拒绝新 strided 家族；oneDNN/typed exact-only 路线继续拒绝，不能因私有字段默认值误签全域。

本卡只解除受检描述传递的底层缺口，不代表 G101 已完成。Graph 仍必须证明 View 操作关系、alias offset
及 workspace 上界，并在同一共享 Plan 的冷实例化消费；动态 View attributes/offset 不在本卡被偷偷放宽。
三种 C++ 用户交付没有新增产品开关/FFI；same-build Provider 私有版本递增，不提供旧表兼容。
