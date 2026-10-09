# 未实现 Compute/Sched 预留项清理

<!-- PROJECT UNIT REVIEW: AUDITED -->

baseline `0fc45b5d86b4ba279fafa56c9fcac43e26e54e9f`。用户明确要求删除无需求的
`compute_sched`，并清理过期的当前说明。独立 build/tool 维护，不借 Vulkan 修改治理。

## 审计与有限 FIX

- Q1：该项只有无依赖的 QUEUED 声明与登记，没有源码、公开头、target 或消费者。
  现有 `ParallelExecutorRef`、Sched 和公开 CPU executor 样例已经提供组合入口；删除预留项。
- Q2：closure checker 目前禁止任何登记删除，包括从未施工的预留名称。仅允许撤回
  QUEUED；已进入施工的身份仍禁止删除、改名、重排，存续登记顺序保持。
- Q3：selector 对纯预留撤回不应扩大为全库构建。仅当删除的都是无产品文件的 QUEUED，
  所有存续 factor 元数据及 product roots 完全相同时，分类为状态维护。
- Q4：依赖真实 `compute_sched` 名称的治理测试改成自身临时预留 fixture；保留非空、
  非 QUEUED、公开边变化和重排的拒绝测试。不增加新 registry 状态、开关或永久白名单。
- Q5：更新当前 registry/生成状态页与长期撤回规则；历史 task/性能证据保留事实。
  当前 Vulkan 卡同步真实 25/25 状态结果，不改历史 FAIL 或其它路线终局。

预算20分钟，一次 focused tooling 批次；产品运行时、Sched、Compute API、依赖与数学不变。
验证 closure/selector 回归、实际两图投影、docs、明确 previous-ref 的状态检查与 diff。
不编译 C++、不跑 P1、不进行全库文档重审。失败仅修具名工具缺陷一次或报告具体阻塞。
