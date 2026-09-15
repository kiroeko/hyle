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
- P0：新增布局校验在冷路径，复杂度 O(D + guards + ports × rank)，无 payload 读取；原有 schema、
  input×output alias 校验及 Provider 冷成本另计，未改变。原 exact prepared 不增加字段/分配/分支。
  域 axis map 的额外布局族与上界是 opt-in 冷 metadata，不声称零字节。
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

## 2026-09-15 实现检查点

产品 `b07f0fec039f875673e2d15f8c51a6684bef7f8a` 已实现 C150/C151，私有 Provider 合同为 v13。
Graph 冷实例化传入已推导的端口描述；检查在 Provider callback 前执行。原 schema、alias offset、
workspace/private 上界与制品保活继续校验，未扩大已接受域或改变 warm replay。

新增 11 个测试：6 个 Compute 布局族/小域穷举用例，5 个 Native 真实 RMSNorm、错误前置拒绝、
逐点 OOM、描述所有权与销毁用例。初次编译纠正了测试中 AllocatorRef 的参数个数，没有放宽产品合同。

| Windows Release 验证 | `C:/hyle-scratch/` 缓存 | 结果 |
|---|---|---|
| Compute/CPU/Native/Graph 与逐头 | `compute-route-v8-static-20260914` | 9/9 |
| shared 与真实 oneDNN direct/region/composition | `compute-route-onednn-v8-20260914` | 14/14 |
| CUDA runtime/late failure/逐头；Graph/逐头 | `compute-graph-v7-cuda` | 分别 3/3、3/3 |
| source CPU/CUDA sample | `custom-execution-v10-source-consumer` | 2/2 |
| installed shared CPU/CUDA sample | `custom-execution-v10-shared-cuda-consumer` | 2/2 |
| installed static CPU/CUDA sample | `custom-execution-v10-static-consumer` | 2/2 |

沿用 Windows wrapper 的精确 target/CTest、2 build jobs、串行测试、120s timeout 与显式 SDK prefix。
shared SDK 与实际 CPU consumer 的 Compute DLL SHA256 同为
`5275E81FA87F9E5DD065FD268BCDAFB26BDF4B7D6CF2DDDC47FA4953F4A1040E`。
selector `20c80597 → WORKTREE` 为 READY、跳过 86 项；closure 39/schema 4、docs 与 diff check 通过。
同提交 LOC self-check：全库 **683,536**，自实现不含测试 **125,183**；Compute **16,917**、Graph
**13,081**，合计 **29,998**。

本卡仍 AUDITED。完整动态 View 关系/偏移/内存上界、G103 联合优化、G106 packing/capture 与真实架构
探针仍未完成；Linux/两台 Apple、sanitizer、终局独立红队和 P1 随完整 foundation candidate 验证。
以上增量通过不是 foundation 完成声明，没有新性能点值或跨平台结论。
