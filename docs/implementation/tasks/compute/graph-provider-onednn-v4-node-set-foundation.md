# Graph oneDNN Provider v4：node-set foundation region

Compute 后继 [v9](compute-v9-shape-replay-substrate.md) 的 C90/C94 也迁移本 factor 的 single-op 拒绝
适配器，属于 GP41/GP42 的私有 SPI 影响；它仍不提供 leaf，不改变已实现 node-set MLP 或 kernel。

C91 的 replay facts 属于 GP41：已准备 MLP 明确 prepared；proposal/artifact 的成本类别必须一致，
Graph 复用 Compute 的 value 合同校验，不增加 provider 热分支。

G100 的 prepared-only 冷筛选同属 GP41：region discovery 提前声明同一 prepared class，迁移 extension
v7，确保 oneDNN MLP 不被当成 unknown 排除。只迁移元数据并增加实际 strict 编译/执行回归，不改 kernel、
pattern、线程或 submit；不据此签发 GP44 性能结论。

G103/GP41 共用 Compute 的结构成本类型，删除未资格化的私有时间估值槽和重复 runtime 字段；region
extension 随布局升版。实际 MLP 的已知 copy/repack/sync/dispatch 如实保留，workspace/JIT persistent
未知仍为 unknown，不为选路捏造时间；kernel/pattern/submit 不变。

G104/GP40 迁移有限发现：writer 改为 `try_add() -> Result<bool>`，false 是 Graph 提供的候选容量不足，
Provider 正常停止发现；unexpected 或 pattern/use-table 错误仍失败。只修改冷发现循环，保留当前匹配、
artifact 和执行。预算受限是否可接受由 Graph 的显式政策和完整可行性决定，不由 oneDNN 隐藏裁决。

## Closure card

- component / owner：`graph_provider_onednn` / Graph oneDNN provider；round=`graph-provider-onednn-v4-node-set-foundation`；
- baseline commit：`49d32deacf80fbc046e53be64de47042144edc5a`；前轮活动记录为
  [v3 sequence SPI](graph-provider-onednn-v3-sequence-spi.md)；
- trigger：Graph v9 用可收缩 node set 取代 interval，旧 v3 卡没有该产品变化；foundation 需要一个真实
  non-insertion-order region 反证新合同，而不能只用 mock；
- 正常任务：oneDNN 识别 gated MLP 的逻辑 node set，声明精确 boundary/成本，以 shared-runtime artifact
  执行；未入选时 Native leaf 正常接管；
- scope：迁移 node-set proposal/compile、显式 boundary、exact-only shape proof、完整结构成本与 prepare
  outcome；消费 oneDNN v5 私有 BRGEMM seam；保留 v3 acceptance/EnqueueResult；
- non-goals：扩大 pattern catalog、oneDNN Graph API、动态 shape region、packing、kernel/tile 修改；
- 依赖 / 消费者：依赖 Graph/Compute/ComputeOps/Math/Mem/SIMD/compute_provider_onednn；Graph compile 消费；
- 性能：P0 证明一次 region episode、warm 零 metadata allocation；gated MLP 与 leaf/fallback 做 P1；
- 预算/验证：一轮 node-set migration、一轮 non-order probe、一轮修正；contractibility/overlap/re-entry、
  decline/partial acceptance、OOM、三平台、sanitizer、三交付、P0/P1、红队；
- stop：如果 pattern 必须依赖 insertion order、Graph 反向依赖 dnnl，或 provider 接纳后才尝试 fallback，
  退回 `SCOPED/BLOCKED`。

## AUDITED 结论与有限改动集

| ID | 允许改动 | 验收 |
|---|---|---|
| GP40 | proposal/compile 从 begin/count 改为 canonical node set + ports | 插入无关合法 node 后仍识别同一 MLP；非法重入/成环由 Graph 拒绝 |
| GP41 | 填 exact shape proof、resource/precision/runtime/完整结构成本 | unknown 与 zero 分离；Graph 不从 provider rank 猜隐藏成本 |
| GP42 | 迁移事务化 prepare/decline 与 v3 EnqueueResult | work acceptance 前才可 fallback，partial accepted error 只经 Completion |
| GP43 | 消费 OD52 BRGEMM seam并关闭三交付 | source/static/shared 只有一个 kernel owner，无公开 dnnl/header |
| GP44 | gated MLP normal/holdout + 反例 P1 | 完整 boundary 成本裁决，非苹果对橘子；固定 loss 必须查机制 |

公开用户仍只安装 provider 并从 CompileReport 观察选择；没有 provider-specific Graph API。当前为 `AUDITED`，
只授权 GP40--GP44。

## 2026-09-14 实现检查点

本 factor 的 single-op 拒绝适配器已迁至显式 unsupported outcome，并在 oneDNN-enabled Windows shared
构建中通过逐头与现有 region 测试。

GP40 已迁移：从 down MatMul 沿 producer identity 匹配 gated MLP，不再假定四个 node 连续，也不要求 gate
先于 up 创建。Graph 复制并 canonicalize node set，Provider 固定 semantic input 到 boundary port 的映射；
submit 只读该固定映射，BRGEMM kernel/tile 未改变。发现期通过 Graph 共用 helper 排除 control 重入，不把
合法的未融合 DAG 当成编译错误。

Windows oneDNN-enabled shared 通过真实数值测试：无关 node 插入、up-first 构造仍生成一个正确 region；
gate → 外部 node → gated 的控制依赖保留普通 leaf 路线且输出正确。原有 Graph/oneDNN region 回归和
Graph 逐头测试同时通过。非 Windows、sanitizer、三种完整安装交付和 P1 尚未在本 node-set 树上运行。

GP41--GP44 的完整 boundary cost、事务回退、BRGEMM seam 三交付与 gated MLP 终局 P1 仍待完成；
本检查点不构成 Candidate 或关闭声明。

G100/GP41 冷筛选迁移已通过 Windows oneDNN-enabled shared 定向 4/4：实际 gated MLP 在 prepared-only
政策下仍被选中并执行正确，discovery/resource/artifact 使用同一个 runtime 常量；Graph 与本 factor 逐头
同时通过。只验证当前 source-tree 交付，GP43/GP44 的最终安装/性能义务不被此检查点替代。

G104 的 writer 迁移已通过 oneDNN-enabled Windows shared 定向回归。新增真实 gated MLP：node 容量
小于整个 region 时不部分接收、不隐藏 OOM，Graph 用已安装 leaf providers 执行完整数值旅程，报告 node
发现受限。既有 exact/non-order/decline 等 region 测试继续通过；kernel/pattern/submit 未变化，无新 P1。

GP42 的事务化结果接入：region compile 改为公共 Compute prepare outcome，显式声明 prepared 或真正
error；现有 oneDNN compile 路径不把原始分配错误转换成安全 decline。只迁移冷回调合同和检查，不增加
pattern/算子、重新计时或修改 BRGEMM 内核；Graph 的原样错误传播与 rollback 验证覆盖这个 carrier。

C93/GP41 后继迁移固定：region proposal 给现有 gated-MLP 实现声明稳定 `ProviderRouteKey`，升版的
Graph extension 使用同一字段；它不是私有 prepare token 的别名或自动性能证书。Graph report 验证该
键在真实 MLP 路线上保留；pattern、执行、BRGEMM owner 与热成本不变。

G106 的 ConstantBinding/region extension 升版按 Graph v10 的 source-release 段直接迁移并复验：真实
MLP 用 materialize bindings，在 compile 后释放三个原权重，再执行与对照。Provider 继续借用 Graph
传入的有效 canonical view，不推断 aggregate packed count 能释放 source，不改 BRGEMM kernel。

### GP41 / G106 discovery allocator 审计

`propose_regions()` 的 use-table 与 contractibility helper 当前回用 Context allocator，绕过 Graph
compiler 配额。FIX：消费 Graph 在 propose/resource-query 参数显式传入的 allocator；所有发现 scratch
在返回前释放，同一次分配/释放 owner 不变。若 Context 中的 allocator 字段已无消费者，删除这个仅为
发现缓存的副本；不改变 Provider 的 context ownership 或 kernel/pattern/submit。
用实际 gated MLP 的 discovery allocation/OOM、原数值路径、Graph 逐头及三交付回放；额外 vendor
prepare 私有上界仍为 unknown，不因为 discovery 已正确记账就猜 oneDNN/JIT 内存为零。本项一轮有限
纠正，状态保持 AUDITED，属于既定 GP41 的资源/成本事实，不重开已 CLOSED 因子。

该纠正在 `bcd7ecbe` 落实：use-table 与 contractibility scratch 都走 Graph 传入的 allocator，Context
中重复的 allocator 字段已删除；extension v16 直接迁移，无兼容旧表。Windows shared 18/18 测试通过，
包括真实非连续 MLP 的逐点 OOM/零与精确配额/重试和原数值、并发、warm-no-allocation 路线。private
vendor/JIT 上界仍 unknown，有限 strict 编译拒绝，显式 allow 如实报告不完整。详情和未完成的 GP43/
foundation 验证见 [共同检查点](graph-v10-reviewed-foundation-contracts.md#2026-09-15-g106-编译-requested-storage-检查点)。
