# Foundation A：四路线 Graph 语义对称性

## Closure card

- component / owner：`graph`、`graph_route_apple`、`graph_route_vulkan`、
  `graph_route_cpu_apple`、`graph_route_cpu_vulkan`；共同静态语义由 Graph 拥有，
  原生 artifact、端口、完成与资源由各路线拥有。
- baseline commit：c0792501319ad1eab482c41b054e1e63a3ca7b8a。
- trigger：用户明确要求实施四路线对称性；
  [固定源码研究](../project/foundation-f45-reference-research-2026-10.md)与
  [设计 A-01..04](../../../architecture/compute-state-paged-design.md)证明
  CPU+CUDA 复用共同 compiler，而 Apple/Vulkan 及 hybrid 拒绝全部 symbolic/control 图。
- 正常 consumer：同一有界变长张量管线，冷准备有限精确形状，独立 invocation 选择固定形状，
  显式控制前驱虽没有数据消费者也必须执行；不支持的 effect/alias 明确拒绝。
- 主纵切：F4/F5 显式状态消费者的 pure initial/final 与有限 batch shape。
  正交纵切：渲染资产变长矩阵管线的独立分支排序与旧结果保活。
- public journey：GraphBuilder 的已有 shape/control/effect 校验 → 冷 native artifact 表 →
  原路线 invocation/bind/prepare/submit → 原 Completion/ResultLease；不引入共同 native runtime。
- scope：仅 A-01、A-02、A-03、A-04。实际可支持的 effect 需由端口和 permission 证明；
  拒绝不能包装成支持，pure control 不得只删除检查。
- non-goals：F4/F5 产品代码、全引擎、自动 placer、运行时 JIT、padding/general 伪覆盖、
  新数学/copy kernel、万能 Queue、公共验证/构建/性能工具重写、重开 F1–F3。
- 依赖：既有 Graph/Compute/Mem/Sched 与锁定 MLX/ggml/IREE；不升级依赖。
  直接消费者：原 native 与 CPU+Apple/Vulkan 的普通 Graph consumers，CPU+CUDA 仅作合同比较。
- 性能：P0，检查冷表与 warm 路径、复制/等待/私有成本；本卡不运行 P1 或全算子目录。
- 机器：Windows CPU/CUDA/Vulkan，WSL CPU/CUDA 与 sanitizer，Air Apple/Vulkan/CPU；
  Apple 运行前核对真实身份，仅 MacBook Air。WSL 软件 Vulkan 仅构建与包证据。
- 预算：静态审计两轮；有限四 FIX；每个 gate configure 300 秒、build 1800 秒、
  test 120 秒、consumer 180 秒。相同 gate 的失败/超时不随目录或提交重置。
  重型 gate 串行；库 build 按可用内存 j4≥12GiB / j2≥8GiB / j1≥4GiB，test1。
- 预声明验证：selector 选择 actual owners/直接消费者；共同 shape/control/effect 边界与 OOM；
  exact coverage 缺口/外来 ID/重复 binding/derived guard/overflow、独立 slots、旧结果与 close；
  control-only 活前驱与跨 fragment 排序，禁止 host wait 的拒绝；实际支持 effect 的权限/alias/失败。
  受影响 native/hybrid 普通 source 与 installed static/shared、公开头、缺路线、搬迁、runtime；
  共同静态代码适用 ASan/UBSan，新增并发共享元数据适用 TSan；无新增并发机制不重复全库 TSan。
  每轮 docs/closure/diff，registry 明确 previous-ref，生成 status，不手改生成表。
- stop：预算到达或 native 无法满足所需语义时如实 BLOCKED；新系统性 API/owner 事实退回
  SCOPED/AUDITED；原本不准入的 effect 保留明确拒绝与限制，不冒充 A 完成。

## 当前阶段

AUDITED。GraphBuilder::finish 已统一验证 shape program、控制边身份/拓扑及 effect 冲突；
共同 executable 的 bind_shape_values / specialize_bound_descriptions / make_specialized_graph
已拥有精确形状求值和原身份保留。BORROW 这些实现，禁止第二套表达式/guard 校验。
原生 lowerer 的数据剪枝不能保留 control-only 前驱，必须改成数据+控制可达性。
Apple evaluate_mlx 逐 region 等成功 outputs，hybrid 逐 fragment 等原 Completion；
控制端点冷分片并显式物化前驱输出可以复用这些顺序，不需要新 Compute runtime。
Vulkan 原生控制端点的 command barrier/managed region 仍需实现期对原录制代码定向核实。

### 能力与 API freeze

| 能力 | 当前裁决 | 有限动作 |
|---|---|---|
| GraphBuilder 形状/控制/effect 合法性 | COMPLETE | KEEP/BORROW 原规则 |
| native 精确形状 | MISSING | 共同冷 specialize 快照，保留原 ValueId/NodeId；不宣称 general |
| 有限原生 artifact 表 | MISSING | SELF 冷 `ExactNativeVariants<Artifact>`，声明表全量预检后编译；select 时校验 binding，未覆盖拒绝 |
| control-only 活性与顺序 | MISSING | SELF 冷可达性/端点分片，BORROW 原完成边界；等待/物化成本明确报告 |
| 原生 pure inputs/results/旧结果 | ROUTED | KEEP 原 native Program/Executable/Invocation/ResultLease |
| 外部 effects/resource ports | PARTIAL | 原 CPU+CUDA 规则 KEEP；native 无真实可写端口者明确 REJECT；不得假报 preserves_effects |

共同精确快照以 `specialize_exact(graph, shape, allocator)` 返回 Graph 与原
SpecializationSignature；只在冷调用分配，完整复用既有求值、输出推导和 ViewSpec 绑定。
native/hybrid CompileOptions 增加 `shape`，原无 dimension 调用保留原形状。
`compile_variants` 为各 route 的冷 typed factory，返回拥有原 artifact 的有限 family；
`select(shape)` 返回该 artifact 的共享 handle，再用原 create_invocation 管理独立 slot。
不在 submit 查表/JIT，也不增加万能执行接口；无法准备任一声明 shape 则整表拒绝，
缺省没有隐式增长，重复 signature 拒绝，完整 bounds 内的表外点仍 Unimplemented。
声明/选择都检查原 graph identity、derived guards、完整 independent bindings。

metadata 与原 native artifact 的 allocation/OOM 发生在冷准备，失败不消费调用者 tensor；
family 复制不复制 payload。共享 ResourceBudget 继续累计各 artifact 的原收费，未知
MLX/driver bytes 保留 UNKNOWN；无总预算配置不声称硬内存上限。常态每 slot 原生对象与
旧结果权限不变。control 默认允许原 managed waits，禁止等待按原 workload 拒绝。
source/static/shared 使用同一 typed entry；公开新头归 Graph；文档提供显式两形状选型。
所有新增入口回指有限 shape consumer，取消/close/producer 沿原 Invocation，family 不起任务。

七问：依赖锁与许可证沿固定研究 KEEP；功能仅按上表且外部 effects 不伪支持；复用原 Graph
求值/Compute runtime/F1 owner；P0 冷 O(V(D+N+E))、选择 O(DV)、warm 原 artifact；C++ typed
入口无新 FFI；新增公共头及四 factory 三交付需实际验证；guide 要明示 exact-only、等待与
未知 private 成本。当前均为合同冻结，正确性/交付/红队尚未裁决。

## 有限改动 ledger

| ID | 工作 | 当前结果 |
|---|---|---|
| A-01 | 共用既有静态合法性与身份，保留 native artifact/owner | 已接线；共同39项与 Win Vulkan/Air Apple source PASS；其他格待验 |
| A-02 | 冷有限 exact artifact 表与创建时 coverage 拒绝、独立 slots | 已接线；Win Vulkan/Air Apple native+hybrid PASS；CPU+CUDA新表入口/三交付待验 |
| A-03 | 真实 pure control liveness/顺序与禁止等待裁决 | 已接线；Win Vulkan/Air Apple native+hybrid PASS；完整风险/三交付待验 |
| A-04 | 真实支持 effect 的端口/权限/alias；其余明确拒绝与旧结果合同 | UNPROVEN |

## Gate

scope、architecture、correctness、delivery、usability_docs、red_team、engineering_performance
均未裁决；本卡建立不代表产品能力或 native 验证 PASS。

## 首批实施事实

[共同精确形状](../../../../evidence/correctness/foundation-a-shared-exact/README.md) 39/39 PASS，
包含原静态语义和新有限表、OOM、control reachability；
[Win Vulkan source](../../../../evidence/correctness/foundation-a-vulkan-source/README.md) native 与
CPU hybrid 的有限形状、两个独立 slot/旧输出、控制前驱错误与禁止等待旅程 PASS。
原 fixture 失败保留，累计预算在对应 evidence。代码仍 AUDITED，尚无 candidate/终局。

[Air Apple source](../../../../evidence/correctness/foundation-a-apple-source/README.md) native 与
CPU hybrid 的 exact/control/旧槽结果数学 PASS，真实可用内存7GiB 按规则 j1，build20秒。
这不代签 Apple 的全部失败/取消及 installed 交付。CPU+CUDA 保持原 compiler/公开路线，本轮新 finite-only table 的
自然入口仍需纳入实际 owner 审计。A-04 尤其需要 CPU 岛上的原 effect/footprint 接线资格，
不能把所有 native effects 的明确拒绝写成四路线整体对称完成。F4/F5 尚未启动产品施工。
