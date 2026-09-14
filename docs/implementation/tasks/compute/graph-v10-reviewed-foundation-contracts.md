# Graph v10：整体复核后的 foundation 合同

## Closure card

- component / owner：`graph` / Graph；round=`graph-v10-reviewed-foundation-contracts`；
- baseline：`610a219f57e613ac6f1aaf79b337742122e2569f`；承接未 terminal 的
  [v9](graph-v9-foundation-routing.md)，保留已实现的 node-set/三轴资源/Completion 骨架。
- trigger：[整体复核 A1--A6](compute-graph-route-unification-audit.md#整体架构复核与合同修正)证明独立
  bounds、全局 effect chain、一律预算失败和不区分 replay 成本不足以承载既定旅程。
- 主纵切：动态 token 的 normalization→matmul→gated activation、精确 capture 变体与 general 路线；
  反偏置：单节点 exact CPU、自定义异步资源任务；四引擎的来源与取舍见总审计。
- scope：G100--G107；这是复合 planning/replay 能力，不是模型/KV/batching policy。
- non-goals：算子目录扩充、第三个 public Plan、通用符号求解器、全局自动调优/缓存、Provider 内核重写。
- 依赖：Compute/ComputeOps/Container/Mem；复用 Error、Queue/Completion；没有新增反向依赖。
- 性能：迁移 P0；全部有限 FIX 与真实 CPU/CUDA/Apple 探针稳定后，统一 one-op/MLP P1，沿用基础卡
  5% 工程门与完整边界/反向顺序，不以 managed kernel-only 成本对比 prepared 的完整调用。
- 预算：shape、selector、资源/variant ownership 各一轮表示与一轮纠正；定向红队一次。不在这些任务里
  改公共仪器；正常开发中的编译/fixture 错误不冒充最终性能裁决。
- 验证：G100--G107 的反例、原 exact/node-set 全回放、OOM/并发/完成与销毁顺序、逐头、三交付；
  Windows/Linux/两台 Apple 与适用 sanitizer 在完整候选验证，不因 docs/状态提交重跑 P1。
- stop：热态 HYLE search/JIT/分配、接纳后换路、用模型策略填基础合同缺口，或 required 约束被静默忽略。

## 已审计的有限改动集

| ID | 能力现状与动作 | 固定验收边界 |
|---|---|---|
| G100 | FIX replay 成本传播与冷筛选；复用 Compute 的属性校验/比较，不再复制字段列表 | unknown 不等于 prepared；prepared-only 排除 managed/unknown；report 原样保留，热提交不筛选 |
| G101 | FIX independent/derived shape 与 guards，复用 Compute provider-neutral program | 独立维度只绑定一次；flatten/concat/head split/broadcast、越界/overflow/非整除；完整域证明或 exact-only |
| G102 | FIX finite exact variants + 完整 general fallback | compile 后不新增 variant；concrete binding 固定路由，capture 精确绑定 shape/storage/queue |
| G103 | FIX leaf/region/placement 同池有界选择与完整成本 | 硬约束优先；copy/repack/sync/workspace/persistent/关键路径与 qualified identity；无证据稳定回退，不承诺全局最优 |
| G104 | FIX 有限事务化重选与预算终止 | 搜索预算停止扩搜而非禁止完成已有候选；有合格可行解返回并报告 optimization-limited，strict caller 可拒绝；无解失败，不吞真实错误 |
| G105 | FIX resource/effect domains，替换全局 last-effect 假串行 | schema access 来自 ComputeOps，资源 ID 属于 Graph；同域冲突有序、独立域可并行、未知保守；bind alias 与 region 边界反例 |
| G106 | FIX constant borrow/materialize 与 variant 聚合 | 成功 materialize 后 source 可释放；兼容 immutable artifact 显式共享，不能共享的副本披露；unique resident/冷峰值/并发/capture 全计预算 |
| G107 | KEEP exact replay/lifetime，并完成 report 与交付 | shortlist/拒绝/成本来源有界且可追溯；安装包普通 C++ 与 custom execution 消费同一 Compute Provider，不依赖 src SPI |

这些 FIX 来自 architecture 已接受的具体矛盾，不因本卡延长而重新审全库。Compute 的公开 owning custom
factory、ComputeOps access schema 必须先在各自有限卡完成；本卡不越过 owner 写其产品代码。

## API / owner / 成本

保留 `GraphBuilder → Graph → Executable → Invocation`，Plan 仍私有。Builder 拥有 shape 表达式与资源
identity，compile 拥有有界候选和不可变制品，Invocation 只拥有本次 scratch/bind/capture/完成状态。
普通 exact 用户无需提供表达式、资源域或 profile；动态用户只绑定独立维度，派生维度由预构建有限程序
求值。冷编译默认允许已声明且满足硬约束的 prepared/managed 路线；显式 prepared-only 不放宽。

G100 的首步仅迁移已有 runtime facts：`ProviderReplayClass` 为 unknown/prepared/managed，字段
`replay_class` 默认 unknown，公共 value type 提供 `valid()` 与完整 equality；unknown 可被诊断/保留，
不能据此证明 prepared-only。实际政策筛选与 managed 成本预算不是这一个字段自动完成的能力。

派生 shape 不读 payload；资源访问不携带模型 KV/communicator 策略；常量共享不靠裸地址猜内容；无解/
真实 error 仍返回 Status。异步对象一直保活到真实 Completion，不能用 host 提交返回替代设备终态。
新私有存储布局和有界搜索实现可在上述合同内实验；若要改变 owner/默认政策/完成边界，先回到审计。

## 七问与当前状态

无新外部依赖/FFI；Provider 接入只复用已有因子，版本事实不按名称猜测。能力与易用性由四引擎的真实
旅程和 exact 反偏置约束；正确性/性能/三交付需按上述具体反例实证，不能以代码量、已编译或历史通过代替。
已有 AUDITED 轮按接受的架构基线绑定本后继有限卡；当前仍 `AUDITED`，尚未形成新的 Candidate/PASS。
