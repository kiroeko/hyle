# Graph oneDNN Provider v4：node-set foundation region

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

