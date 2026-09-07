# Util-runtime 工程性能证据

<!-- PERFORMANCE EVIDENCE: HISTORICAL -->

> 当前状态和证据绑定只看 [`module-status.json`](../../module-status.json)。本页只承担 `util` runtime
> factor 的 P0 结构证据；`util_core` 的当前 P1 证据在
> [`benchmarks-util-core-v4.md`](benchmarks-util-core-v4.md)。

## v4 未通过的结构取证

[util-runtime v4 receipt](../../tasks/util/util-runtime-v4-inference-lifecycle.md) 记录产品
`0fff17ef3abe6830dba58a344c3878199dfb21ea`、测试 execution
`6145023f31590f0f175845470d6f4da32e8df1ee` 与全部原始证据。P0 为 **FAIL**：按值参数在 erased
invoker 内额外移动，允许准入的 const-reference noexcept factory 可能因此意外 terminate。
这是一项已复现的 C++ 转发/成本缺陷，不是微基准波动。

Linux/Mac 的分配断言和正确性用例支持 freeze O(1)/零分配、frozen lookup 无内部 allocation/lock 的
局部结论；Windows 测试、Linux 长符号判据、sanitizer 与其余资格门尚未全部通过，不能签发整体 P0 PASS。
三种 C++ 交付目前仅 Mac 本轮全部通过。当前无 P1/P2、peer ratio、固定输项或更快声明；计时样本为 0。
中央 registry 指向的新轮处于 BLOCKED 时，本页不标记 CURRENT；历史 v3 的 PASS 不能填补新轮证据。

## v3 历史命题

以下记录绑定 v3 候选。v4 已因具体依赖/交付变化重新登记，本页不为当前工作树签发通过证据；
当前轮只从 registry 指向的 closure card 读取。

Registry 的正常形状是启动期少量 registration/find/create/freeze，demangle 是日志、异常与 backtrace
冷路；没有 profile 证明它们进入 token/kernel 热路径。因此 v3 当时的等级是
`P0 / STRUCTURAL PASS`，timing peer comparison 为 `NOT_APPLICABLE`，不发布 lookup 纳秒或 SOTA 声明。

v3 candidate 与 [`util-runtime-v3 task`](../../tasks/util/util-runtime-v3-rebuild.md)曾证明：

- register 为 O(|key|+|help|)+expected-amortized O(1)；
- find/create 为 O(|key|)+expected O(1)，freeze 后读路径无锁、零 allocation；
- freeze 为 O(1)、零 allocation，callback/factory 不在 mutable mutex 内执行；
- demangle 为 O(input+output)，无常态 metadata；平台 serialization、临时 storage 和有界 fallback 都是
  显式冷路成本；
- 三平台 correctness/sanitizer 与 source/static/shared C++ consumer 已通过。

旧 util-runtime v1/v2、N=256 Registry 点值、早期 C adapter 和 util-core v1/v2 数字只属于各自历史
task/Git 记录。若真实 profile 把 Registry 或 demangle 提升为 P1/P2，应为该具名 consumer 新建合同，
不能复用这些旧点值。
