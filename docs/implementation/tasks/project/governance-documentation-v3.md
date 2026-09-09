# 工程治理与文档体系 v3

<!-- PROJECT TASK: HISTORICAL -->

## 任务合同

- unit / kind / owner：`docs-governance` / 横向维护工程 / HYLE maintainers；
- baseline：`508d5b026638342d5a6bc8268111dd795d997fbc`；
- trigger：26 个真实验收模块都同时写成 `CLOSED + FROZEN`，证明 `FROZEN` 已不再区分状态；
  `VERIFIED` 只制造 candidate 验证与 terminal receipt 两次纯治理提交；当前 registry 又复制 task 中的
  scope、non-goals、机器、限制、重开条件和 LOC，达到 1761 行。仓库有 454 份人读 Markdown、约
  69,594 行，其中 implementation 历史占 366 份、53,217 行，当前文档与一次性历史容易混读；
- objective：让 `CLOSED` 恢复“当前方法下已验收并进入变更控制”的唯一含义；建立最小但可执行的
  registry/checker/status 合同；迁移所有现有状态而不丢失历史；让既有四类文档边界真正可维护，删除
  已被当前记录吸收且没有独立引用价值的过时重复页；
- non-goals：任何 `include/hyle/**`、`src/**`、组件图、公开 API、算法、依赖、benchmark、原始性能
  evidence 或产品状态结论变化；重开 26 个 CLOSED factor；为 compute/graph 设计产品；运行 C++
  configure/build/test、sanitizer、三交付或 P1；改写 Git 历史；
- normal consumer：下一位维护者能从一个短入口准确判断当前状态、合法启动/终止模块工作并找到证据；
  用户、维护者和架构读者不必维护或辨认重复状态叙述；
- cost / budget：P0 纯 Python/JSON/Markdown；一次 schema 迁移、至多两轮工具合同纠正；不以删除行数
  作为成功指标，也不为证明过程增加新的状态、marker、hash ledger 或生成文件；
- stop：中央组件集合、26 个真实 CLOSED 与 compute/graph QUEUED 未无损迁移，或当前 terminal record/
  performance evidence 失去可达性则 `BLOCKED`；否则全部维护合同通过即 `CLOSED`。

## 冻结设计

状态流简化为：

```text
QUEUED -> SCOPED -> AUDITED -> CANDIDATE -> CLOSED
SCOPED | AUDITED | CANDIDATE -> BLOCKED -> SCOPED
CLOSED -> REOPENED -> SCOPED
```

- `CLOSED` 本身就是唯一 terminal acceptance；删除 `FROZEN` confirmation、状态列和术语；
- 删除瞬时 `VERIFIED`；候选的适用工程门全部通过后可一次提交 terminal receipt 与
  `CANDIDATE -> CLOSED`。产品/API/算法在 `CANDIDATE` 后变化仍退回 `AUDITED`；
- 当前 registry 只保存机器真正消费的 module、state、round、record、candidate identity、工程 gate、
  工程性能摘要和一行人读 summary。scope、non-goals、baseline、机器/仪器细节、限制、reopen 条件与
  LOC 留在 record/evidence，不再复制到 JSON；
- Formal 发布认证不再为每个模块默认制造 `PENDING`。真正的 release certification 使用独立证据，
  不属于模块当前状态；
- `status.md` 保留为从 registry 机械生成的 GitHub 人读视图，提供显式 `--write-status`，不是第二真相；
- 组件历史更名由已有 build task 与 Git 保存；当前 registry 不永久携带 migration ledger。未来更名
  仍须一个显式横向任务并原子修改中央图和 registry；
- task marker 只区分 `ACTIVE/HISTORICAL`。历史文本永不授权当前工作，也不要求改写成今天的术语。

## 有限工作集

| ID | 动作 | 验收 |
|---|---|---|
| `GOV3-01` | checker/schema v3 与状态流 | focused tests 覆盖合法直关、非法跳级、blocked/reopen、retired 字段和中央图一致性 |
| `GOV3-02` | 无损迁移 28 个 registry record | 26 个 `CLOSED`、2 个 `QUEUED`；candidate、round、record、gates、工程性能与 summary 保持 |
| `GOV3-03` | 简化并可自动写入 status 投影 | 一条命令生成，checker 拒绝漂移；无 Freeze/Formal 列 |
| `GOV3-04` | 精简方法、模板、入口和相关维护工具说明 | 一个状态定义、一个操作流程；无手工 snapshot/SHA、自引用或双 terminal 提交 |
| `GOV3-05` | 文档分类和 reachability 审计 | architecture/implementation/developer/user 职责清晰；当前 record、当前性能页和入口全部可达 |
| `GOV3-06` | 删除/合并过时重复文档 | 只删已被现行权威页吸收且无当前入边的历史副本；Git 保留恢复；所有链接通过 |

## 预声明验证

- `python tests/tooling/check_module_closure_test.py`；
- `python tests/tooling/validation_selector_test.py`（registry schema/状态变更分类仍不得触发产品 build/P1）；
- `python scripts/check-module-closure.py --previous-ref 508d5b026638342d5a6bc8268111dd795d997fbc`；
- `python scripts/check-docs.py`；
- `git diff --check`；
- diff 反证不含产品源码、组件图、benchmark 或 raw evidence；C++/sanitizer/delivery/P1 全部
  `NOT_APPLICABLE`。

## 结果

### 状态与工具迁移

- `module-status.json` 从 schema 2 的 1,761 行迁为 schema 3 的 688 行；中央顺序、26 个真实
  `CLOSED`、`compute/graph` 两个 `QUEUED`、round、terminal record、candidate commit/tree、七门、
  工程性能 tier/verdict/evidence/margin 与摘要全部保留。
- `CLOSED` 成为唯一 acceptance；中间验证状态、第二 terminal confirmation、默认 Formal 轴、当前
  migration ledger 及 scope/机器/LOC/限制等 task 事实不再复制到 registry。
- checker 从 722 行降到 502 行，直接支持 `CANDIDATE -> CLOSED`、合法 blocked/reopen、严格 v3
  schema、中央组件顺序和 `--write-status`。一次性 v2 transition adapter 在迁移提交验证后删除；当前
  工具不兼容旧 registry。
- P1 card 的两个重复固定字段合为 `candidate_fixed=true`；文档/状态/格式仍不触发产品 build 或 P1。
- LOC 工具新增 `--registry-state`，按中央组件 owner 对当前实现文件做并集去重。当前 26 个 CLOSED
  组件为 84,603 行自实现、不含测试；QUEUED 的 compute/graph 为 5,286 行，两组恰好覆盖全库
  89,889 行自实现代码。

### 文档清洗

- 收口方法从 495 行降到 164 行，任务模板从 275 行降到 106 行；AGENTS、review/implementation
  入口和 scripts 说明只保留一套状态与命令。
- 删除 16 份当前性能页的第二 `CURRENT` marker 及其 checker；当前证据只由 registry 的
  `performance.evidence` 选择。
- `compute`/`graph` 架构改为诚实的 `QUEUED` 问题边界，删除重复的 `graph-02-design`；IO 架构只保留
  当前 factor、ownership、backend、协议和成本设计，移除 2026-07 施工日志与点值；macOS setup 不再
  把 compute 实验写成已验收状态。
- clay 逐文件资产表移到 implementation 历史 task，architecture 只保留“迁移来源不授予保留或兼容
  权利”的长期决策。历史任务和性能 evidence 保留原文，不为今天的术语批量改写。
- baseline `docs/` 为 454 份 Markdown、69,594 行；终局为 455 份、68,550 行。文件数增加来自本任务
  和 clay 历史迁移，`docs/` 总量净减少 1,044 行。总入口对 architecture/developer/user 的可达性为
  31/31、38/38、17/17；全部相对链接仍通过。

### 验证与边界

- schema 迁移提交上执行旧 registry `508d5b026638342d5a6bc8268111dd795d997fbc` 的 before/after
  比较为 PASS；移除一次性 adapter 后，以首个 v3 提交 `b90a9a83` 做 previous-ref 仍为 PASS。
- `check_module_closure_test.py`、`validation_selector_test.py`、`loc_report_test.py`、closure checker、
  docs checker 与 `git diff --check` 全部 PASS。
- diff 不含 `include/hyle/**`、`src/**`、组件图、benchmark 或原始性能数据；C++ build/test、sanitizer、
  三交付与 P1 均为 `NOT_APPLICABLE`，没有重开或重新裁决任何产品模块。

本页只保存本次迁移事实；当前方法只看 developer/review，当前状态只看 registry。
