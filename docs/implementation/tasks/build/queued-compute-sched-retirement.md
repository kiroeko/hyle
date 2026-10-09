# 未实现 Compute/Sched 预留项清理

<!-- PROJECT UNIT REVIEW: CLOSED -->

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
  当前 CLOSED 摘要中的旧跨 owner 施工队列词也移除；state/record/candidate/性能证据不变。

预算20分钟，一次 focused tooling 批次；产品运行时、Sched、Compute API、依赖与数学不变。
验证 closure/selector 回归、实际两图投影、docs、明确 previous-ref 的状态检查与 diff。
不编译 C++、不跑 P1、不进行全库文档重审。失败仅修具名工具缺陷一次或报告具体阻塞。

## 结果

`compute_sched` 已从 manifest/registry/生成状态页撤回，没有运行时/公开接口可删除。
当前共57组件，52 CLOSED，剩余五项均为 Vulkan/Graph 当前收口范围；没有 QUEUED 尾项。
旧状态页日期移除，2026-09-17性能账本明确标为历史快照；当前 CLOSED 摘要里的过期
跨owner施工队列描述清理，所有 state/record/candidate/性能证据保持。历史任务不回写。

- `python tests/tooling/check_module_closure_test.py`：PASS，含撤回和已施工身份拒绝。
- `python tests/tooling/validation_selector_test.py`：60/60 PASS；非空/已施工/产品元数据
  变化不能借纯预留撤回跳过构建。
- `validation-select.py plan --base c6623c0c --candidate WORKTREE`：READY，仅选择
  docs/closure及上述两个工具套件；132门明确跳过，没有 configure/C++/P1/GPU采样。
- `check-module-closure.py --previous-ref c6623c0c`：57组件/schema4 PASS；docs/diff PASS。

本有限维护结束；不增加第二个调度器、不重开已关闭产品。Compute+Graph统一实现
仍62,410行/267文件，产品代码未改变。新消费需求或工具误分类才触发下一轮。
