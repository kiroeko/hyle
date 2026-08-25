# 模块审查任务模板

> 复制到 `docs/implementation/tasks/<module>/`。删除不适用项，但必须说明为什么不适用。
> 具体要求以 [module closure methodology](module-closure-methodology.md) 为准。

## 任务身份

- 模块 / owner：
- 状态：`OPEN`
- 起始 commit：
- candidate commit：
- 目标与成功条件：
- 范围：
- 非目标：
- 直接依赖 / 直接消费者：
- 真实 workload manifest：

## Capability ledger

| demand ID | 用户任务 / 来源 | 正常形状 | 处置 | public/internal route | strong peers | verdict |
|---|---|---|---|---|---|---|

完整性检查：

- [ ] 成熟库 capability 对账，不按现有 API 或 benchmark 名称盘点
- [ ] 每个公开 symbol 反向指向 demand ID
- [ ] 每个 `MUST/GENERAL` 正向指向实现、组合、依赖或裁决
- [ ] 语法消费者与语义重复均已扫描
- [ ] higher-layer、borrow、defer、reject 的理由明确

## 类型与依赖

| 形状 / 操作 | ownership / lifetime | concurrency / progress | failure / capacity | complexity | normal metadata/cost | 决定 |
|---|---|---|---|---|---|---|

- [ ] public/internal 分层自然，没有 benchmark-only shape
- [ ] 默认路径服务多数用户，可选优化显式 opt-in
- [ ] error/platform/mem/time 等底座使用正确，无同义私造
- [ ] 中央 component graph 与真实 target link 一致
- [ ] FFI=`N/A`，或链接到已通过准入门的独立任务

## 实现与正确性

- 算法/数据结构及 peer 对照：
- 平台 fast/fallback path：
- 最终 release codegen 检查：
- 正常、边界、OOM/系统失败、move/destroy：
- 并发、取消、关闭、晚到完成：
- property/differential/fuzz/stress：
- ASan / UBSan / TSan / 真 ISA/backend：
- public-header self-contained：
- source / installed static / installed shared C++ journeys：

## 性能资格与结果

- capability/contract/candidate：
- workload 与输入分布：
- peer/version 与计费边界：
- latency/throughput/resource 指标：
- 预声明 material margin：
- A/A negative control：
- known-delta positive control：
- timer floor / MDE：
- order / placement / affinity：
- consumer correlation 与 holdout：
- 原始证据：
- 五态 verdict 与适用范围：
- 产品改动的机制证据 + material 消费影响：

## 易用性

- [ ] 只看 user page 和 public headers 可完成 create/use/fail/compose/destroy
- [ ] 命名、默认值、overload、RAII、view/owner、Result 自然
- [ ] 线程安全、容量、取消、释放和性能坑能在调用点附近发现
- [ ] 没有要求用户理解内部 benchmark、任务号或源码目录
- 可用性走查者与发现：

## 文档与机器合同

- [ ] architecture 只保留长期需求、边界和决策
- [ ] implementation 保存本轮事实、commit、结果和限制
- [ ] developer 沉淀可复用方法/排障
- [ ] user 提供选型、示例、合同和坑
- [ ] 原始数据位于 `evidence/` 且被精确引用
- [ ] `python scripts/check-docs.py` 通过

## 独立红队

- 审查者 / 日期：
- 缺失能力：
- benchmark distortion / complexity：
- correctness / lifetime / concurrency：
- dependency / build / delivery：
- instrument attacks：

| 问题 | 严重度 | `fixed/disproved/accepted/out-of-scope` | 证据 |
|---|---|---|---|

## 签署

- 各门 verdict：
- 已知限制：
- reopen trigger：
- 结束 commit：
- 四个 LOC：全库 / 自实现含测试 / 自实现不含测试 / 模块实现
- 最终状态：`SIGNED` 或保持 `OPEN`
