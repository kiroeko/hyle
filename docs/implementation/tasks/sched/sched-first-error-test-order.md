# TaskScope 首错测试：区分结果就绪与 scope 完成

## 有限测试维护 / 已审计，待验证

- baseline：`2435e08d63d43dcc8ce4eb351f08b3fc838006cc`；产品候选及 Sched 的 CLOSED 状态不变。
- trigger：R6 Windows broad selector 在验证快照 `3894b72e` 实际运行 251 项 Sched 测试，
  250 项通过，`TaskScope.AggregatesFirstStatusFailureAndStillDrains` 返回 `later child`，
  违反该测试预期的 `first child`。原始失败不能删掉或重跑到 PASS。
- owner：Sched 测试维护；不修改 src/include/CMake、公共合同或性能仪器，不伪造产品 reopen。
- 有限 FIX：仅修正该用例的 happens-before；调查限这一个首错边界，最多一次修正。
- 预声明验证：新不可变测试快照的 selected Sched correctness 一次、该用例 32 次有界重复、
  docs/closure/diff；P0 源码身份证明。没有 API/算法/正常成本变化，不跑 P1 或三交付。
- R6 已通过的 84 门保留原候选/receipt；先保留原 BLOCKED result，再独立续跑原候选未执行的
  26 门。最终覆盖记录单列失败测试的修正快照，不伪造一个全新 111 门原候选 PASS receipt。

## 源码裁决

`include/hyle/sched/task_scope.h` 的 `ScopedTaskDriver<T>::entry` 先把值交给 JoinState；
`finish` 在 Scheduler 离开 Fiber 后释放 Task/Fiber/stack，再调用 `task_scope_publish`。
`src/sched/task_scope.cc` 的该函数先在控制锁内保留首错，随后 signal 每 child 的完成事件。
`TaskScope::remaining()` 读取的正是这些 scope 完成事件。

因此 `first.join()` 只确证 typed 结果可读，不能保证第一个 child 已进入 scope 错误汇总。
两个 worker 可以合法地让第二个 finish hook 先发布。这与当前文档的 scope 完整清理/完成合同
一致，未发现产品首错实现缺陷。原测试错误地用 typed 结果就绪建立 scope 完成顺序。

修正后先在 5 秒有界控制面等待中观察 `scope.remaining() == 0`，再接纳第二个 child。
保留原 NotFound/消息/full-drain 断言，不放宽为“任一失败都行”，不使用延时猜测顺序。
这不改变产品的 hot path、数据布局、失败排序政策或内存序。

## 验证与停止

实际失败、修正后的命令、测试与身份将在证据中分别保存。未执行项不写 PASS；若本次修正仍
失败，按该具体事实重新裁决，不扩大成 Sched 全库审计或增加性能采样。
