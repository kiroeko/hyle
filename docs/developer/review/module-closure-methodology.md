# HYLE 模块审查与收口协议 v4

本文是新建、重构、复检和关闭一个 HYLE 组件的唯一施工流程。能力为什么准入由
[能力完整性决策](../../architecture/capability-completeness.md)定义；组件边界和顺序由
[架构路线](../../architecture/roadmap.md)定义；本文只规定一次工作如何有界推进、验证和停止。

当前机器状态只看 [`module-status.json`](../../implementation/module-status.json)，人读表是
[`status.md`](../../implementation/status.md)的生成区。历史 task/evidence 只证明已经发生的事实，
其中的命令式句子不是当前授权。

## 1. 唯一状态流

```text
QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> CLOSED
SCOPED | AUDITED | CANDIDATE -> BLOCKED -> SCOPED
CLOSED -> SCOPED
```

| 状态 | 含义 | 最小输出 |
|---|---|---|
| `QUEUED` | 身份已登记但本轮未启动 | component ID 与现状摘要 |
| `SCOPED` | 问题和预算已经冻结 | 一张 closure card |
| `AUDITED` | 能力、现状和有限改动集已经裁决 | coverage map 与有限 `FIX` ledger |
| `CANDIDATE` | API、算法和普通产品形状已经停止变化 | 完整 commit/tree 与候选验证 |
| `CLOSED` | 全部适用工程门通过，应停止日常施工 | terminal record、七门与工程性能结论 |
| `BLOCKED` | 一个外部或工具条件使本轮不能裁决 | blocker、owner、精确恢复条件 |
不能跳级。候选后若改变公开合同、算法、对象布局、常态成本或正常 workload，退回 `AUDITED`；
只补说明或证据不改变 candidate。`CLOSED` 表示当前范围已真实验收并进入变更控制，不表示永久 ABI、
所有机器上的绝对 SOTA，或未来永远不增加能力。

registry 只保存机器当前需要的字段：

- 所有状态：`module/state/summary`；
- 非 `QUEUED` 状态：再加 `record`；
- candidate/terminal：`candidate` 保存完整 commit；tree 可由 Git 唯一导出，不重复保存；
- terminal：再保存性能 tier、证据和适用的 material margin；`CLOSED` 本身已表示七门通过，registry
  不重复保存七个恒为 `PASS` 的字段；
- blocked：只加恢复所需字段。

scope、baseline、机器、仪器、限制、重开条件和 LOC 属于当前 record/evidence，不复制进 registry。
状态表只能运行 `python scripts/check-module-closure.py --write-status` 更新。

## 2. SCOPED：冻结一张小卡

改产品代码前，在 `docs/implementation/tasks/<area>/` 建立当前 task，并写清：

- component、round、owner、baseline；
- 一个真实 consumer、可复现缺陷、标准或成熟 peer 共同证明的需求；
- 正常用户任务、主消费者纵切、必要的正交反偏置纵切、范围、非目标、直接依赖和直接消费者；
- P0/P1/P2 以及适用测试、机器/仪器、material margin；
- 总调查预算、性能纠正预算、停止条件和可接受限制。

新组件先在中央组件图和 registry 以 `QUEUED` 追加登记。已有 `CLOSED` 组件必须由合法 reopen
trigger 启动，创建一张不同于旧 terminal record 的新卡，再直接进入 `SCOPED`；`REOPENED` 不是需要
持久化的产品状态。“再看一遍”、历史 TODO 或 benchmark 输赢本身都不是 trigger。需要先重新划分全库边界时，执行
[能力展开与组件质因数分解](../../architecture/capability-factorization.md)，不要边写代码边改 owner。

## 3. AUDITED：先定功能与形状

先从用户任务自顶向下建立能力族，再从全部现有公开入口自底向上回映。证据优先级是：

1. 当前真实消费者；
2. 本轮具名产品纵切中的具体操作序列；
3. 至少两个独立成熟实现共同提供的稳定任务；
4. 语言、平台标准或定义性算法；
5. 明确标为 diagnostic 的 synthetic shape。

每个能力族记录正常旅程、语义档位、ownership/lifetime、failure/OOM、cancel/close、并发、复杂度、
资源成本、当前路线与 `COMPLETE/PARTIAL/MISSING/ROUTED/OUT_OF_SCOPE`。每个公开入口必须回指需求；
只被 benchmark 使用的入口删除或私有化。属于本模块且有可靠准入证据的常用能力应尽量实现；纯策略、
模型格式或应用编排留给消费者。

模块不需要机械扫描所有产品域，但必须按能力完整性决策选一个相关主纵切；若主纵切可能把 API 做成
领域特例，再选一个正交纵切。项目级审计必须走通跨模块 public C++ journey；只有在具名 owner/依赖和
实际组合路线都存在时才可写 `ROUTED`，不能用“上层处理”掩盖用户必须重复手写的通用 glue。

API 设计顺序固定为：用户任务 → 真正不同的语义 → 生命周期/失败/并发 → 常态成本 → peer 取舍 →
自然 C++ 调用草图。不能先写 `fast/light/heavy` 入口再找场景，也不能为一个点值给所有对象增加常态
metadata。公开形状、默认值、错误形状、依赖暴露、source/static/shared 用法和 guide recipe 冻结后，
才允许开始 P1/P2。

审计结束只留下有限的 `KEEP/FIX/BORROW/DEFER/REJECT` 集。后续发现新能力必须重新 scope，不能无限
扩张本轮。

## 4. 每轮必须回答的七问

1. 第三方路径：直接/间接依赖、版本、许可证、升级和 fallback 是否清楚？
2. 功能完整：对当前真实消费者、选定产品纵切、反偏置纵切、标准和 strong peers 是否没有遗漏的
   `MUST`，跨模块 public journey 是否确实可组合？
3. 内部复用：是否合理使用已关闭的下层组件和加速设施，且没有为避开依赖而复制实现？
4. 性能：复杂度和成本结构是否合理；需要计时时是否同语义、公平、拟真且无 material 固定输？
5. 交付边界：历史 C/Native FFI 是否清理；仅在有跨语言 runtime 需求时才设计类型擦除 ABI？
6. 构建交付：源码、静态库、动态库的组件发现、依赖、导出和安装是否一致？
7. 易用与文档：普通 C++ 用户能否自然选型、正确管理生命周期并发现坑？

答案必须给出代码、测试、consumer、peer 或文档位置，不能只写“看起来没问题”。不适用项说明具体
原因即可，不制造虚构的 binary、peer 或消费者。

## 5. Correctness 与候选

实现阶段先验证语义、边界和失败路径。按风险选择：单元/属性/差分、并发进度、OOM、sanitizer、
逐头编译、直接消费者、三交付或平台测试。只有语义确实不适用才写 `NOT_APPLICABLE`；适用但未完成
就是 `BLOCKED/INCONCLUSIVE`，绝不能写 `PASS`。

API 与算法稳定、有限 `FIX` 全部处理、直接消费者迁移后，提交不可变 candidate commit/tree 并进入
`CANDIDATE`。文档、状态、证据提交不属于新的产品 candidate，也不得因此重跑产品测试。

产品实现只能在 registry owner 为 `AUDITED` 时提交。`validation-select.py` 对源码、公开头、factor
产品 seam、中央组件归属和已登记 vendored 产品输入执行该检查；`SCOPED` 仍只允许审计，`CANDIDATE`
已经不可变，`CLOSED` 已进入变更控制。测试和文档维护不会因此伪造 reopen。

## 6. 性能门

- `P0`：检查复杂度、分配/原子/锁/syscall/拷贝等成本结构和生成代码，不用计时证明纳秒差；
- `P1`：统一仪器上的同合同工程比较，主要回答是否存在 material 可修复劣化；
- `P2`：有可复现真实 consumer trace 时做端到端影响。

P1/P2 前必须逐项写明相同的语义、输入、输出、ownership、failure、setup/teardown、并发与完成条件。
若不同，这是能力差异，不是性能胜负。先做空白/同实现对照和 A/B、B/A 换序；地址/顺序反转、噪声带内
点值、亚纳秒值和未资格化样本只能诊断。只有正常任务中可复现的机制差异达到预先冻结的 material
margin，才允许继续改产品。比较证据遵守
[性能证据标准](../performance/benchmark-evidence-standard.md)。

只有 API/算法已固定的 candidate 才运行一次适用 P1；文档、格式、状态、测试专用改动不触发 P1。
公共仪器不足时登记独立工具 blocker，不得在模块任务里无限重写仪器。

## 7. 红队、停止与关闭

候选完成后由与实现过程分离的一遍红队检查：未映射公开入口、隐藏生命周期/OOM/并发问题、错误依赖、
苹果对橘子性能比较、benchmark 污染、交付差异和文档误导。红队发现产品问题退回 `AUDITED`；发现
证据问题只补证据；超出冻结范围的想法登记未来候选。

`CLOSED` 必须同时满足七门：

```text
scope, architecture, correctness, delivery, usability_docs,
red_team, engineering_performance
```

工程性能必须 `PASS`：P0 为结构成本通过，P1/P2 为预声明门通过。terminal record 保存实际命令、结果、
限制、reopen trigger 和 LOC；registry 只引用它。达到预算仍不能裁决时必须 `BLOCKED`，不能写
“继续研究”。

常见重开条件仅包括：可复现 correctness/security/UB/lifetime 缺陷；真实消费者证明的缺失常用能力；
公开合同、算法或交付变化；已采用依赖的相关升级；合格同合同证据显示 material 回归。新轮创建新 task，
引用旧 record，不回写历史 task。

## 8. 固定命令

每轮只运行 closure card 预声明且受改动影响的产品验证，并至少运行：

```text
python scripts/check-module-closure.py
python scripts/check-docs.py
git diff --check
```

registry 改动必须把改动前提交传给：

```text
python scripts/check-module-closure.py --previous-ref <previous-revision>
python scripts/check-module-closure.py --write-status
```

LOC 用统一工具；不手工相加历史快照：

```text
python scripts/loc-report.py --module <component> --commit <candidate> --self-check
python scripts/loc-report.py --registry-state CLOSED --commit HEAD --self-check
```

记录从[任务模板](module-review-template.md)开始，但模板是检查清单，不是篇幅要求。删掉不适用的空表，
保留足以复现裁决的事实即可。
