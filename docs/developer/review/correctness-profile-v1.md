# `closure-correctness-v1`：P0 候选验证 profile

`closure-correctness-v1` 用于没有真实 hotspot 证据的编译期、布局和冷路径基础能力。它验证“形状与
成本结构没有错”，不测亚纳秒点值，也不产生 SOTA、吞吐倍数或发布性能证书。P1/P2 使用
[性能仪器 v1](../performance/instrument-v1.md)。

## 冻结输入

执行前必须从 closure card 取得 component/round、candidate commit/tree、P0 理由、适用平台与
delivery shape、直接依赖、受影响消费者、预算和停止条件。若没有 candidate identity，输出只能是
`INCONCLUSIVE`；不得拿当前脏工作区补签旧候选。

每项证据记录命令、工具链/配置、结果和限制。候选 LOC 必须从不可变提交复算：

```text
python scripts/loc-report.py --module <component> --commit <full-candidate-commit> --self-check --json
```

source bundle 没有 `.git` 时只验证当前文件与 registry 结构，不冒充历史 Git identity 重放。

## 固定门

| Gate | 最低证据 | 失败含义 |
|---|---|---|
| capability / API | 自顶向下 capability-family map、全部现有公开入口反向追踪；每个 `MUST` 为 `COMPLETE` 或完整 `ROUTED`，不存在 `PARTIAL + KEEP`；ownership/failure/并发/复杂度/negative space 齐全 | `FAIL` 或回到 `AUDITED` |
| correctness | 风险矩阵逐项 `PASS/FAIL/NOT_APPLICABLE/BLOCKED`；公开头 self-contained；适用 sanitizer/target | correctness blocker |
| cost structure | 同复杂度档；正常路径无意外 allocation、lock、syscall、copy、registry 或常态 metadata | P0 `FAIL`；若实际是热点则重新 scope 为 P1/P2 |
| release structure | 对受 source hint/layout 影响的入口检查 codegen、symbol、object size 或调用边；不适用时写理由 | `FAIL/INCONCLUSIVE` |
| delivery | 真实适用的 source、relocated static、relocated shared journey；header-only 不制造虚构 binary | delivery blocker |
| consumers | 运行被本轮 diff 触碰、公开类型直接暴露或能反证合同的最小代表集 | `FAIL/INCONCLUSIVE` |
| docs / usability | architecture、user、task/evidence 分工及 `check-docs.py`、状态 registry 检查 | docs gate `FAIL` |
| red team | reviewer、日期、预算、逐 finding severity/disposition/evidence | 缺记录即 `INCONCLUSIVE` |

## 语义覆盖矩阵

行覆盖率、分支覆盖率和“测试数量”只用于发现完全未触达的代码，不能证明产品合同。每个 candidate 必须
在 task 中给出下面的逐能力矩阵；不能用一个 happy-path 单元测试同时替所有格签字：

| 能力族 / public journey | 正常与边界 | failure / OOM / overflow | ownership / move / teardown | cancel / deadline / concurrency | 主消费者纵切 | 正交纵切 | 平台 / delivery |
|---|---|---|---|---|---|---|---|
| `<task 填写>` | `PASS/NA/...` | `PASS/NA/...` | `PASS/NA/...` | `PASS/NA/...` | `PASS/NA/...` | `PASS/NA/...` | `PASS/NA/...` |

- 每个 `MUST` 能力族至少有一个只经公开 API 走通的主消费者旅程；存在领域特化风险时，还要有一个正交
  旅程。原子 primitive 可以由拥有它的最小下游 factor 测试组合，不要求在底层模块复制高层产品。
- 测试必须观察消费者最终依赖的结果与资源状态，而不只观察某个内部函数被调用。错误恢复、旧值保留、
  backpressure、晚到 completion 和 teardown 是旅程的一部分。
- 平台 backend、allocator、并发或交付形状不同会改变语义风险时必须分别取证；一个平台的 sanitizer
  不能冒充另一个平台的 native mechanism 测试。
- property/differential/fuzz 与 coverage 报告用于扩大输入空间；它们不能替代具名 lifecycle 和真实组合
  旅程。反过来，端到端样例也不能替代 checked arithmetic、OOM 和 race 等原子反证。
- 旧 `CLOSED` 记录若没有该矩阵，不自动变成失败；项目级 evidence audit 应把缺失项标为
  `UNPROVEN`，只有具体可复现缺陷或准入 `MUST` 缺口才触发重开。不得把“曾跑过全库测试”改写成逐格
  `PASS`。

“所有直接消费者”不是默认命令：底层组件可能扇出全库。closure card 必须按公开头暴露、语义调用、
本轮 diff 和失败半径选择代表集，并列出未选消费者及理由；否则一次 P0 回放会退化成全库验收。

## 证据复用与新鲜度

旧证据仅在以下条件同时成立时可复用：产品候选文件与公开合同未变；相关 compiler/dependency/build
变化不会改变被证明事实；原始证据 hash 匹配；本轮重新执行受影响的机械门。平台或 toolchain
改变且该门仍适用但无法重跑时写 `INCONCLUSIVE`，不能用 `NOT_APPLICABLE` 绕过，也不能把历史
PASS 改写为当前真机 PASS。只有合同上确实不适用才写 `NOT_APPLICABLE`。

P0 最终 verdict 为 `PASS/FAIL/INCONCLUSIVE`。只有全部适用门 `PASS`、其余明确
`NOT_APPLICABLE`、无 blocker 时才可支持产品 `CLOSED`。这个 verdict 只证明 P0 结构合同；具体
prefetch、padding、branch hint 或 inline 对消费者是否有收益，由拥有 workload 的模块重新证明。
