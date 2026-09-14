# oneDNN Provider v5：foundation proposal 与 region consumer

Compute 后继 [v9](compute-v9-shape-replay-substrate.md) 的 C90/C94 请求签名迁移属于 OD50；本 Provider
继续 exact-only。已有 kernel/公开工厂不变，非 exact domain 必须在准备重型 artifact 前拒绝。

C91 的 replay facts 属于 OD51：现有已准备 BRGEMM 路线明确 prepared，第三方 JIT 仍仅发生在 prepare；
不因默认 metadata 改写 warm allocation 或 executor 的 owner 合同。

## Closure card

- component / owner：`compute_provider_onednn` / oneDNN provider；round=`compute-provider-onednn-v5-foundation-route`；
- baseline commit：`49d32deacf80fbc046e53be64de47042144edc5a`；前轮活动记录为
  [v4 internal kernel seam](compute-provider-onednn-v4-region-kernel-seam.md)；
- trigger：Compute v8 修改本 factor 直接实现的私有 proposal SPI；同时 v4 尚欠真实 Graph region
  source/static/shared consumer，二者必须在一个新身份上共同验收；
- 正常任务：direct oneDNN MatMul 维持原精确 route；Graph oneDNN region 私有复用同一 BRGEMM mechanism；
- scope：迁移 proposal/outcome 与结构成本；先明确 exact-only shape domain；完成 v4 私有 seam 的真实 region
  consumer、rollback 和三交付；
- non-goals：新增 dtype/layout、权重 packing、oneDNN Graph API、AArch64 路线、修改 BRGEMM kernel/tile；
- 依赖 / 消费者：依赖 Compute/Error/Mem/oneDNN；graph_provider_onednn 是唯一新增私有 consumer；
- 性能：P0 证明 direct kernel/codegen/route 不变；完整 gated MLP 随 Graph 做一次 P1；
- 预算/验证：一轮 SPI migration、一轮 consumer composition、一轮红队；原 direct tests、pair/OOM/warm allocation、
  Graph region、三交付、依赖锁/许可证、P0/P1；
- stop：若迁移要求公开 dnnl 类型、复制 kernel 或热路径增加搜索/分配，退回 `SCOPED`。

## AUDITED 结论与有限改动集

| ID | 允许改动 | 验收 |
|---|---|---|
| OD50 | 迁移新 proposal/outcome，声明 exact-only | direct MatMul 路由、tile、scratch、mapping 与 Completion 不变 |
| OD51 | 给 route 填可审计结构成本，不提供跨 provider 私有 ns | Graph 可比较 boundary；unknown 不伪装 zero |
| OD52 | 由 graph_provider_onednn 消费 v4 opaque BRGEMM seam | 不安装私有 header、不泄漏 dnnl、无第二份实现 |
| OD53 | 回放 allocation/OOM/source-static-shared 与 P1 | v4 未决 gate 全关闭后才可 Candidate |

公开 `onednn::provider()` 不变；当前为 `AUDITED`，新卡取代未 terminal 的 v4 卡，只授权 OD50--OD53。

## 2026-09-14 实现检查点

OD50/OD51 的 exact 迁移已落地：direct BRGEMM 明确零 copy/repack/sync、一次 dispatch；opaque JIT storage
与尚未构造的 workspace 保持 unknown，未用假零补齐。prepare 的 unsupported、分配预算拒绝和真实错误已
显式分类。Windows oneDNN-enabled source-tree shared 构建以及 provider/composition/Graph/逐头测试 6/6
通过；OD52/OD53 的 node-set region 与完整 gated-MLP 证据尚未完成，状态保持 `AUDITED`。
