# Validation selector v2：factor 额度与 native 身份

<!-- PROJECT UNIT REVIEW: ACTIVE -->

## SCOPED：有限工具任务

- unit / owner / kind：`build-package` / HYLE maintainers / `TOOL`；不进入产品 registry。
- demand：`SELECTOR-FACTOR-CONTEXT-2026-09-07`；用户只读复核发现三处缺陷和一个真实集成证据缺口，
  并明确要求先完成性能审计，再进行本工具修复轮。
- baseline commit：`be52a7abaa789554281189a4609d5cf79a237711`。开始时 canonical checkout
  `C:\public\project-hyle\hyle` 工作树干净，`main`、`origin/main` 与远端一致。
- 前置：性能审计已提交推送；19 个 FROZEN 模块继续 CLOSED。上一轮已完整阅读 AGENTS 指定的权威
  文档，本轮核对这些文档与 registry 在 `4f48e25f..be52a7ab` 均无变更；重读入口及适用验证合同。
- consumer：维护者与普通 CI 的 change-impact selector；风险是无谓构建、重复 P1 额度或错误证据复用。
- 范围：选择器、其有限 policy/合同测试、CI context 生产者、使用文档与本轮 receipt。
- 非目标：任何产品/API/layout/registry/模块 closure card 修改；Util/Sched/IO/Compute/Graph 实现；
  性能机会优化、P1 qualification、新性能样本、全库 native 测试、统计核心或中央 CMake 图重写。
- tier：P0 工具合同。复用 Git blob/tree、中央 CMake 图、原 factor targets/CTest labels 和已有 checker。
- 预算：一次有限实现，最多两轮由实际失败触发的修正；工具合同测试一轮最终验收；一个 Math factor
  native 集成候选和一次 configure/build/CTest 路线，失败仅针对具名问题修正，不扩到第二模块或全库。
  新 P1/性能样本预算为零。正常维护与 docs 收尾只跑 selector 选出的适用门。

## AUDITED：冻结 FIX 与验收

| FIX | 基线缺陷 / 有限修改 | 验收 |
|---|---|---|
| VS2-1 | factor 测试目录 CMake 被先判为 product seam；先按 tests owner 分类 | `tests/container_queue/CMakeLists.txt` 只选 queue correctness；无 consumer/header/delivery；公共 source CMake 仍按 seam 扩展 |
| VS2-2 | 单个全库 product hash 同时决定所有 P1 额度；改为每 factor 及中央图真实产品依赖输入的 scope | queue 失败后改无关 container 不重获额度；另一无关 factor 不被它阻塞；真实依赖输入改变产生新 scope；改 card/工具/机器/日志目录不补额度；multi-factor card 不因拆组/重组逃逸 |
| VS2-3 | 普通 native 接受只有 id 的 context；强制 compiler 文件身份、具体 stdlib 证据和明确 flags | 缺字段/文件 hash 漂移不能执行或复用 native；flags 进入实际 configure；计划与执行身份漂移拒绝；docs-only 不要求或准备编译器；同步 CI context |
| VS2-4 | 原 27 个合同测试只用 process spy | 独立验证副本从不可变工具候选，仅增加 Math test CMake fixture；真实 selector 只选 Math，构建 `hyle_math_tests`，CTest 精确 `^math$`，无 P1/全库/consumer gate |

P1 的产品依赖闭包由唯一中央图派生；split headers 继续按真实 owner，不能把 Container façade 作为
Queue 的依赖。共享 build/package 输入明确计入；外部 vendored 输入按有限、可审查的文件 owner
投影绑定，不复制 factor 依赖图。新 scope 不等于产品重开授权或允许改算法后直接采样。
旧 claim 保留；无法恢复作用域的旧格式不得靠升级 selector 自动获新额度。

native context 的文件摘要要与本机实际文件核对，flags 要用于执行命令；单个任意 `id` 或空泛
“Release”描述不足以签 native receipt。scope/环境不满足时 fail closed，selected/skipped/原因仍可审查。
仅作选择的无 context plan 继续可用于 CI 在准备编译器前决定是否需要 native；不能由它复用 native PASS。

## 预声明验证与停止

- 有限 selector 合同：test CMake 归属；factor/dependency/拆组 claim；失败/中断/竞争保护；弱 context
  拒绝、编译器/stdlib/flags 身份漂移、计划后漂移、正确 receipt 复用；既有 docs/format/未知路径回归。
- 真实 Math 集成只在明确的独立验证 checkout 执行。fixture 仅改该测试目录 CMake，产品文件与工具
  候选一致；不向 canonical 回传产品修改。记录候选/tree、context、configure/target/CTest 原始日志。
- `python scripts/check-module-closure.py`、`python scripts/check-docs.py`、`git diff --check`。
- 最终不可变工具候选 `loc-report.py --module build-package --commit <candidate> --self-check --json`。
- 四项 FIX 及适用验证完成即 CLOSED、提交推送；若必须改产品、统计核心或跑全库才能通过则具名
  BLOCKED，停止扩展。registry 不变，不运行状态迁移 `--previous`。

## 结果

待实现与有界验收；没有提前签 native PASS 或最终锁定 selector。
