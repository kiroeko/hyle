# Util-runtime 工程性能证据

<!-- PERFORMANCE EVIDENCE: HISTORICAL -->

> 当前状态和证据绑定只看 [`module-status.json`](../../module-status.json)。本页只承担 `util` runtime
> factor 的 P0 结构证据；`util_core` 的当前 P1 证据在
> [`benchmarks-util-core-v4.md`](benchmarks-util-core-v4.md)。

## v3 历史命题

以下记录绑定 v3 候选。v4 已因具体依赖/交付变化重新登记，本页暂不为当前工作树签发证据；
当前轮只从 registry 指向的 closure card 读取。

Registry 的正常形状是启动期少量 registration/find/create/freeze，demangle 是日志、异常与 backtrace
冷路；没有 profile 证明它们进入 token/kernel 热路径。因此当前等级是
`P0 / STRUCTURAL PASS`，timing peer comparison 为 `NOT_APPLICABLE`，不发布 lookup 纳秒或 SOTA 声明。

当前 candidate 与 [`util-runtime-v3 task`](../../tasks/util/util-runtime-v3-rebuild.md)证明：

- register 为 O(|key|+|help|)+expected-amortized O(1)；
- find/create 为 O(|key|)+expected O(1)，freeze 后读路径无锁、零 allocation；
- freeze 为 O(1)、零 allocation，callback/factory 不在 mutable mutex 内执行；
- demangle 为 O(input+output)，无常态 metadata；平台 serialization、临时 storage 和有界 fallback 都是
  显式冷路成本；
- 三平台 correctness/sanitizer 与 source/static/shared C++ consumer 已通过。

旧 util-runtime v1/v2、N=256 Registry 点值、早期 C adapter 和 util-core v1/v2 数字只属于各自历史
task/Git 记录。若真实 profile 把 Registry 或 demangle 提升为 P1/P2，应为该具名 consumer 新建合同，
不能复用这些旧点值。
