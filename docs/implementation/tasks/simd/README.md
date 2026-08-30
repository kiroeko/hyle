# SIMD 专题入口

本目录保存 SIMD 的实施事实。当前产品状态只看
[`module-status.json`](../../module-status.json)，长期产品边界看
[`simd-design`](../../../architecture/simd-design.md)，用户只需阅读
[`SIMD 使用指南`](../../../user/simd-usage.md)。

## 当前专题

- [SIMD v4 expert SDK](simd-v4-expert-sdk.md)：统一 source/static/shared 的锁定 provider header 与
  fixed/fat kernel 构建路线；
- [SIMD/Math epic](simd-epic-v1.md)：目标、顺序、预算、依赖和最终验收；
- [simd-v1 closure](simd-v1-closure.md)：上一轮的 capability audit 与终止记录；
- [tickets](tickets/README.md)：本专题唯一顺序施工队列。

ticket 的 `QUEUED/ACTIVE/DONE/BLOCKED` 只是 Jira 式实施状态，不是模块产品状态，也不授权跳过
`SCOPED/AUDITED/CANDIDATE/VERIFIED/CLOSED`。只有 `module-status.json` 指向的 closure card 能授权产品
改动；epic 和 ticket 负责把大模块拆成有限工作包。

## 历史材料

本目录中 `simd-01` 至 `simd-25`、`simd-module-closure.md`、旧 gap/ledger/research 文档均为历史事实。
它们可以提供复现、算法和失败经验，但其中的“终态”“下一步”或 benchmark 结论都不是当前产品状态、
授权或性能证书。新维护者不需要按编号重读全部历史；只有当前 ticket 明确引用时再打开相应材料。
