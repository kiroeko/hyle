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
