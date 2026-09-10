# HYLE 模块任务模板

复制到 `docs/implementation/tasks/<area>/<round>.md`，删掉不适用的提示和空表。小组件应有小记录；
完整事实、七问、适用验证和裁决不能省略。状态规则见[收口协议](module-closure-methodology.md)。

## Closure card

- component / owner：
- baseline commit：
- trigger 与证据：
- 正常 consumer 任务：
- 主产品纵切 / 具名来源：
- 正交反偏置纵切（若不适用，说明为什么不会领域特化）：
- 跨模块 public journey 与每段 owner：
- scope：
- non-goals：
- 直接依赖 / 直接消费者：
- 性能 tier 与原因：`P0 / P1 / P2`
- 适用机器 / 仪器 / material margin：
- 调查与纠正预算：
- 预声明验证：
- stop / blocker 条件：

## 能力与形状审计

| 能力族 | 层级 | 准入旅程/来源 | 语义与生命周期 | 原子分解/应用边界 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|---|---|
| | `原子机制/复合功能/应用策略` | | | | | `COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE` | `KEEP/FIX/BORROW/DEFER/REJECT` |

补充全部公开入口如何回指能力族；列出 benchmark-only、单向生命周期或不可自然组合的旧形状。对复合
功能还要写明普通用户完成旅程仍需自写哪些通用 glue；原子存在不能自动把这些 glue 判为 `ROUTED`。
对每个缺口先按能力完整性决策记录 `SELF/BORROW/ROUTED/APPLICATION` 承载裁决及其长期 owner；“某个
样例用到了”或“包装后更整齐”不能单独形成自实现理由。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 | 结果 |
|---|---|---|---|---|
| | | | | |

本表冻结后不顺手扩张。新事实改变范围时退回 `SCOPED/AUDITED`。

### API / usability freeze

- 普通 C++ 调用草图：
- 语义档位与选型规则：
- ownership / lifetime / failure / OOM / concurrency / teardown：
- 常态 allocation / copy / atomic / lock / syscall 成本：
- 性能成立所需的使用限制，以及任何并列 convenience 路线的显式额外成本：
- 默认值、误用诊断、source/static/shared 发现方式：
- guide recipe 与坑：

P1/P2 只能在本节和 correctness 实现固定以后开始。

## 七问

1. 第三方路径、版本、许可证、升级/fallback：
2. 真实消费者、主/反偏置纵切、标准和 peers 下的功能完整度，以及 public journey 是否可组合：
3. 已关闭下层组件与加速设施的复用：
4. 复杂度、同合同拟真性能和 material 固定输：
5. 旧 C/Native FFI 清洁度与跨语言边界：
6. source/static/shared 构建、安装、导出和依赖：
7. C++ 易用性、选型 guide 与陷阱：

每项给出位置或证据；不适用时写具体原因。

## Candidate

- commit（tree 由 Git/receipt 导出，不在 registry 重复保存）：
- public API、layout、算法和常态成本变化：
- 直接消费者迁移：
- candidate 后仅证据/文档变化：

## 验证

先按 [`closure-correctness-v1`](correctness-profile-v1.md) 填逐能力语义覆盖矩阵。测试数或 coverage
百分比不能代替主/正交 public journey、故障/生命周期以及适用平台与交付格；未取证写
`UNPROVEN/BLOCKED`，不得猜成 `PASS`。

| Gate | 适用风险与命令/证据 | Verdict |
|---|---|---|
| scope | | |
| architecture | | |
| correctness | | |
| delivery | | |
| usability_docs | | |
| red_team | | |
| engineering_performance | | |

性能记录还需写清：peer/version、同合同证明、normal/holdout shape、A/B 与 B/A、空白对照、有效样本、
机制解释和固定阈值。原始结果放 evidence，不粘贴成长篇 task。

P1/P2 计时前先填写：两侧从 admission 到 owner return 的逐阶段 `CHARGED/DEFERRED/N/A` 与源码锚点；
每 operation/owner 的 allocation/atomic/lock/wake/syscall/copy/metadata 静态账单；必要的对象布局/codegen
结论；以及同 workload route、同执行拓扑的短 A/A、双顺序和预计 MDE。通用算术 control 不替代实际
runtime/allocator/线程/设备 workload 的 A/A；preflight 不通过时不得启动完整样本。

## 红队与裁决

- 红队反证与处理：
- 未决限制：
- reopen triggers：
- LOC：全库 / 自实现不含测试 / 本模块实现：
- 最终状态与理由：`CLOSED / BLOCKED / 退回明确阶段`

若为 `BLOCKED`，记录 blocker、owner、`blocked_from` 和唯一可验证的恢复条件。

## 固定维护命令

```text
python scripts/check-module-closure.py --previous-ref <registry-change-before>
python scripts/check-module-closure.py --write-status
python scripts/check-module-closure.py
python scripts/check-docs.py
git diff --check
```
