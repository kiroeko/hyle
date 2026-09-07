# 工程治理 v2：保留结论，删除防御性摩擦

<!-- PROJECT TASK: HISTORICAL -->

## 问题与边界

`util_runtime` v4 暴露了一个流程缺陷：产品 blocker 已经可复现且修复范围明确，但私有测试 fixture、
平台环境和产品修正共用一个“纠正次数”，最终把仍可直接完成的工程工作裁决为 `BLOCKED`。同时，普通
工程验证要求散列 compiler binary、backend、stdlib binary 与整棵标准库头目录，并在 Git common dir
留下永久 P1 attempt claim；这些机制主要防止有写盘权限的维护者伪造过程，却不能增加产品正确性，反而
让换机器、升级工具链和合法恢复失败采样变得困难。

本任务只简化维护合同和相应工具，不重开或修改任何 HYLE 产品模块。Formal 发布证据仍由独立、严格的
performance manifest 合同负责。

## 保留的硬门

- `module-status.json` 仍是唯一状态真相，`status.md` 只是受检投影；中央 component graph 仍只有一份；
- 产品实现前必须完成 scope、能力/依赖审计和有限 `FIX` 集；API/usability 先冻结，性能后验证；
- candidate 必须是不可变 commit/tree；公共 API、算法、layout 或 normal workload 漂移必须回到审计；
- correctness、适用 delivery、内部复用、文档/易用性、红队和分级性能必须给出真实 verdict；
- P1 仍只由显式 card 启动，必须绑定 candidate、factor scope、contract、机器、toolchain 描述和 peer；
  docs/state/test-only 变化不能触发 P1；并发 execution lease 仍防止两次采样同时运行；
- `CLOSED/FROZEN` 仍需限制、reopen trigger 与 terminal receipt；Git commit 自身承担历史身份。

## 删除或改正的摩擦

1. **纠正分类。** 产品机制、验证 fixture/tool、执行环境三类纠正分账。只有产品机制变化生成新
   candidate；确定性的 fixture/tool 修复只使受影响 gate 失效，环境失格只使该次采样无资格。后二者不
   消耗产品修正名额，也不能把产品写成 FAIL/PASS。达到 wall-clock 或性能调查预算且仍有 material
   未决问题时才 `BLOCKED`，不能只因一个任意的全局“纠正次数”耗尽而丢下可直接修复的问题。
2. **P1 尝试归任务所有。** 删除 Git common dir 的永久 one-shot attempt claim。task 记录有效、失格、
   中断和重跑理由；selector 只负责影响范围、候选身份、并发互斥和可复用 PASS receipt，不充当防篡改
   账本。有效 material FAIL 仍必须诊断后改产品/重定范围/阻塞，不能重复跑到绿。
3. **工程 toolchain 身份够用即可。** 普通工程门绑定 compiler path/id/version、stdlib id、显式 flags、
   machine profile 与影响 native 编译的环境变量；不再逐字节散列系统 compiler/backend/stdlib 和整棵
   标准库头目录。正式 performance runner 继续在自己的 manifest 中保存其声明所需的 artifact 身份。
4. **恢复只做 delta。** `BLOCKED -> SCOPED` 可以复用未漂移的能力、API、依赖和消费者审计，只重验
   blocker 及其影响闭包；不得为了形式重新扫描全模块、重跑全库或重采 P1。
5. **三项 LOC。** owner 只关心全库代码、自实现不含测试、当前模块实现。历史
   `self_with_tests` 只留在旧 Git revision；registry v2 删除并拒绝该字段，新的 `loc-report` 不再计算或输出它。
6. **去除自引用提交舞蹈。** 当前 terminal record 不再保存 `end_commit/registry_anchor_commit`；receipt
   与 registry 可以一次提交，Git 历史本身提供该提交身份，旧值只存在于旧 Git revision。
7. **Git revision 代替外部状态副本。** registry transition 用 `--previous-ref <revision>` 直接读取旧
   Git blob，不再要求维护者复制 JSON、计算外部 hash 再传临时文件。
8. **不维护格式证明账本。** 能机械证明的行级缩进/尾空白/CRLF 自动轻量处理；其余空白变化保守地按
   普通语义变更验证受影响 owner/consumer，不再因缺 blob-pair declaration 而阻塞开发。
9. **终态只有一个任务指针。** `closure_task` 已同时指向冻结卡和终态 receipt；当前 registry 删除并
   拒绝内容完全相同的 `closure_evidence`。同理删除自引用 `end_commit/registry_anchor_commit`；旧值只在
   Git 历史中保留。
10. **证据与风险成比例。** 模板是可裁剪检查清单，不是最低字数。七问、`MUST`、适用风险和 verdict
    保留；不适用整节用具体理由裁决后删除空表，不为小型 P0 因子制造 peer、binary journey 或样本。
11. **组件状态不复制到 task marker。** 联合 task 可以让不同组件得到不同 verdict；task 只标
    `ACTIVE/HISTORICAL`，逐组件 `CLOSED/BLOCKED` 只认 registry。旧 receipt marker 作为历史原文保留，
    不再是 v2 合同或兼容入口。

## 验收

- closure registry/schema、方法、模板、LOC 工具和说明使用同一 v2 合同；旧记录可由 Git revision 追溯，
  不把其兼容字段复制到当前 registry；
- validation selector 的 focused tests 证明：docs/state 不构建或采样，失败 P1 不产生可复用 PASS 也不
  永久锁死，candidate/card/machine/contract 仍受检，工具链上下文不再扫描系统源码树；
- `check-module-closure.py`、`check-docs.py`、LOC 与 validation-selector focused tests 全部通过；
- 本任务不运行产品测试、全库构建或 P1。

## 结果

`PASS`。实现删除 483 行、增加 231 行既有文件内容，净减少 252 行治理代码/合同；另保留本页作为
简短决策与验证记录。没有 HYLE public header、模块实现、组件图、benchmark 或 performance runner
产品合同变化。

- `validation-select.py` 不再创建或读取 `hyle-validation-attempts` 永久 claim；相同 candidate/context
  的短生命周期 execution lease、exact PASS receipt cache、P1 card、factor dependency scope、contract
  与 central machine profile 门仍保留；
- native engineering context 不再读取/散列 compiler backend、stdlib binary 或标准库 include tree，
  但仍把实际 compiler path/id/version、stdlib id、flags、host/environment 纳入 receipt identity；GitHub
  workflow 已同步使用轻量 context；
- closure registry 升级为 schema/methodology v2；terminal LOC 只保留三项，当前记录中的
  `self_with_tests/end_commit/registry_anchor_commit/closure_evidence` 已删除；`loc-report` schema 3
  不再计算测试行数指标；
- 方法、性能仪器说明、期望、任务模板和脚本入口都采用同一纠正分类与 delta-only 恢复规则。

验证只运行 change-impact 选出的维护门：docs、closure、closure-checker test、LOC test 和 validation
selector test；selected factor 为零，未启动 C++ build、delivery 或 P1。31 个 selector focused tests
通过，closure/LOC focused tests 通过，YAML parser 通过；`check-module-closure.py --previous-ref` 报
28 components / schema 2 PASS，`check-docs.py` 报 460 Markdown files / 1303 links PASS，三项 immutable
LOC self-check PASS，`git diff --check` PASS。

registry transition 直接以改动前 Git revision
`a333abd12455498b2d20b4fbefa6fc982d081364` 为输入；此前为了 v1 手工生成的外部 JSON 不再是合同或
证据依赖。

## 最终治理红队

没有剩余的 material 治理 blocker。七问、有限状态流、red team、三种 C++ 交付、immutable
candidate/tree、语义等价矩阵和 P0/P1/P2 分级看起来较重，但分别约束已经真实发生过的能力遗漏、伪
收口、苹果对橘子、ABI/安装断裂与微基准扭曲，继续保留。`CLOSED` 与 `FROZEN` 也暂不合并：`sched`
是当前合法 `CLOSED`、尚未通过新方法复核的反例。

两个非阻塞简化候选暂不扩建工具：`status.md` 仍是 tracked、受检的机械投影；两条一对一 component
identity migration 仍帮助从可能跨越更名的 Git baseline 做状态比较。两者都不产生旧 API/ABI alias、
额外产品验证或兼容实现。460 个 Markdown 中的旧 task/evidence 不批量改写；入口合同已经明确这些文件
只保存事实、其中命令没有当前授权，删除它们反而会破坏排障与决策追溯。
