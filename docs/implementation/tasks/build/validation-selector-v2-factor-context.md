# Validation selector v2：factor 额度与 native 身份

<!-- PROJECT UNIT REVIEW: CLOSED -->

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

三处工具修改已实现，最终 selector 合同测试 `34/34 PASS`，69.922 秒；真实 P1 次数为零，P1 路线
只在 disposable Git fixture 中用 process spy 检查状态与原子 claim。针对性回归期间修正了两处测试
夹具前置条件：保留 header suite 再检查 product seam，以及用新的 REUSED plan 检查执行前环境漂移；
没有据此修改产品或扩大构建范围。

Native 身份新增完整 stdlib include-root tree hash，既检查库文件，也防止标准库模板头变化后误复用；
显式 flags 同时用于直接 configure 与继承该 context 的嵌套 CMake。无 context 的 dry plan 保持可用，
但不能复用 native receipt。P1 schema v2 输出每个 factor 的依赖/input paths/hash/claim key；canonical
Git 账本目前没有旧 v1 claim，旧格式存在时仍保留并要求先做归属审计，不自动迁移成新额度。

不可变工具候选为 `66897618aeb2b0c043a4904d205f11abf8b5b892`，tree 为
`11b29222f9a33e28761d7c200e6d8883caf3af2f`。独立 Math fixture commit 为
`8caa65e5992412bcbfa0701c3868794c8daf035a`，tree 为
`e2ac47e6224d2ea4dca2428c20712a2266d1f037`：只为 `hyle_math_test` 增加 60 秒 CTest timeout，
差异唯一位于 `tests/math/CMakeLists.txt`；所有产品文件及 selector 与工具候选完全相同。fixture 不并入
canonical main；保留 patch 可在工具候选上重建相同 tree。

真实 selector native 路线 `PASS`：一次 configure，只构建 `hyle_math_tests`，执行 CTest `^math$`
并排除 headers label，`hyle_math_test` 为 `1/1 PASS`。所选门只有 `docs`、`closure`、`configure`、
`factor:math`，其他 63 门均有 skipped reason；没有独立 header/delivery/consumer/P1 门。
实际 CMake cache 确认 `HYLE_DEV_FACTORS=math`、`HYLE_BUILD_BENCH=OFF`，编译器为 MSVC
19.50.35728，flags 与 context 中的 `/DWIN32 /D_WINDOWS /GR /EHsc` 和 `/O2 /Ob2 /DNDEBUG`
逐项相同。context 绑定实际 cl/c1xx/c2/link 文件、MSVC STL/UCRT 库文件与 include-root tree hash。
随后仅运行只读 plan，四个已执行门全部 `REUSED`；未第二次构建或执行 CTest。

本轮 [receipt](../../../../evidence/validation-selector-v2/receipt.json) 与
[原始证据包](../../../../evidence/validation-selector-v2/native-integration.zip) 保留完整
selected/skipped 输出、context、逐 gate 原始 receipt/log、CMake cache、CTest 日志、fixture patch、
集成驱动和不可变候选 LOC。归档逐项核对所有 receipt/log SHA-256 并通过 ZIP 原字节 round-trip；
不会将规范化后的文本冒充原始日志。34 项工具测试结果来自命令返回码和终端输出，没有另存原始
测试 stdout，receipt 明确记录这一边界。真实 native 证据仅覆盖本轮 Windows Math 路线；不宣称
全模块、全平台或 P1 qualification。

`loc-report.py --module build-package --commit 66897618aeb2b0c043a4904d205f11abf8b5b892
--self-check --json` 为 `PASS`；以下数字绑定工具候选，不把后续文档/receipt 提交混入该快照：

| 口径 | code LOC |
|---|---:|
| 全库 Git-tracked，cloc 全语言 | 479440 |
| 自实现含测试，C/C++/CUDA | 121658 |
| 自实现不含测试 | 77122 |
| 当前工程单元实现，build-package canonical 边界 | 10419 |

canonical 收尾先由 selector 选择 `docs`、`closure`，其余 65 门跳过；实际执行
`python scripts/check-docs.py` 与 `python scripts/check-module-closure.py` 均返回 0，selector
结果为 `PASS`。`git diff --check` 为 `PASS`。这次 docs/evidence 变化未准备编译器、构建或补跑
工具合同/native/P1；工具实现与已验收候选完全相同。

终态 `CLOSED`：VS2-1..4 已完成并补齐一次真实 native 集成证据；达到有限停止条件。产品改动、
registry 改动、新性能样本和 P1 次数均为零。性能审计中三个双顺序负 median 项的 CI 均跨 1，
本工具修复不会把它们变成稳定产品缺陷，也不授权重开任何 FROZEN 模块。本卡不登记后续产品 FIX。
