# time-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> **当前阶段：`SCOPED`。** 本页是 `time-v1` 的唯一活动施工记录；既有 time task 与旧性能表只作
> 历史输入。权威状态见 [`module-status.json`](../../module-status.json)。本轮不继承旧的 C ABI、
> `SIGNED` 或 SOTA 结论，也不在有限 ledger 冻结前修改产品实现。

## Closure card（`SCOPED`）

- component / owner / round ID：`time` / time owner / `time-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`cb46d1183b7ed11757ab0661db893cdfd98b9f65` /
  `19f1c58c3d897240818642b2ca65467e3ad999d8`
- 进入原因与 demand/defect ID：`TIME-CAP-01`（重新核对基础 time 能力及成熟实现的正常形状）、
  `TIME-CONSUMER-01`（以 sched/io/platform 的真实语法与语义消费确定边界）、`TIME-CORRECT-01`
  （时钟、转换、饱和算术和侵入 timer 的边界/生命周期复检）、`TIME-PERF-01`（旧点值与旧 C ABI
  workload 不具当前资格）、`TIME-API-01`（C++23 易用性及无伪 FFI 表面）、`TIME-DOC-01`
  （architecture/user/performance 仍带 provisional、旧 C ABI 或历史结论）、`TIME-DELIVERY-01`
  （build/package v3 后重新验证 source/static/shared 组件闭包）、`TIME-ADOPT-01`（核对底层直接消费者
  是否重复或误用 time 语义）。
- 真实 consumer task / normal workload：每 worker 的 scheduler 在一个 event-loop tick 中批量 arm、
  cancel、advance 和查询下一 deadline；sched timed wait 与 io timer-vs-completion 仲裁；platform/io
  文件 metadata 的 signed Unix timestamp；裸线程控制面的 monotonic deadline 与 blocking sleep。
- 范围：`clock.h`、`unix_timestamp.h`、`TimerEntry/TimerList`、`TimerQueue`、
  `CommonTimeoutQueue`、`clock.cc`；公开 ownership/failure/concurrency/overflow 语义；直接依赖与代表性
  直接消费者；time-only C++ source/static/shared delivery；当前 architecture、user、performance 文档；
  现有 tests/bench 与统一仪器的窄 time adapter/contract。
- 非目标：日历、时区、locale、日期格式化/解析；scheduler/fiber/executor、callback 执行上下文、周期
  timer 的 skip/catch-up 策略、跨线程 timer service；下游模块自身收口；Native FFI/C API；稳定 C++ ABI；
  Formal release certification；为了参考库 API 数量或 benchmark 点值增加常态 metadata。
- 直接依赖 / 直接消费者：`PUBLIC core container_intrusive`；直接消费者为 `platform`、`sched`、`io`。
  本轮验证其 time seam，但不替尚未收口的消费者签署完整结论。
- tier 与理由：`P1`；默认钟、deadline 算术和 `TimerQueue` 位于 scheduler/event-loop 高频路径。
  Unix/OS 转换、sleep、Common queue 及 opt-in profiling clock 没有相同消费频率，分别按适用的结构、
  正确性或 diagnostic 门审查，不能拿来凑一个“time 总分”。
- material margin / resource budget：W1-WIN 同合同 timer consumer replay 使用 5% material 与
  non-inferiority margin；`TimerQueue`/`CommonTimeoutQueue` 正常 arm/cancel 零分配，前者 O(1) arm/
  cancel、后者在单一 duration 合同下 O(1) arm/cancel；不增加 benchmark-only node metadata；默认
  monotonic/realtime typed 与 raw 入口不得比对应 `std::chrono` 多一层动态调用。
- 总调查预算 / 性能预算：功能、API、依赖和文档调查最多 120 分钟；性能只允许一次 probe、一次
  fresh n=15 `qualify-platform`，仅在统一仪器停止规则允许时进行一轮全新 n=31；远程正确性与交付
  只执行冻结矩阵，不扩展研究姿势。
- 最多复现、机制诊断、产品修改次数：一次正常运行；合同内双 order 与一种适用 placement；一次
  codegen/profile 机制诊断；最多一个同时取得机制钥匙和消费钥匙的性能产品假设及重跑；全部产品修复
  合并为一个冻结 batch。
- machine profile：`W1-WIN` 是 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 只承担适用 correctness、
  sanitizer、ISA 与 portability 复核，结果不跨平台池化。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用
  `closure-correctness-v1` 风险矩阵和现有 component/package 合同。
- strong peers（按能力最多两个）：默认 typed clock/elapsed 参考 C++23 `std::chrono` 与 oneTBB
  `e349c15263bbb1d6b48f2b514c06201e1b90eea3`；profiling counter 参考 Folly
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`；任意 deadline queue 参考 Seastar
  `fd71a5b4c71b91abd4e93c9d20f88e1fd7465976` 的 `timer_set`；单一 duration queue 参考 libevent
  `e1f0335d2f3af1235421c939c62943bb99b3d652` 的 `common_timeout`。本轮只从
  `C:/public/project-hyle` 的已钉死本地源码取证。
- 无同合同 peer 时的预算及禁止声明：Unix/FILETIME/POSIX 转换按 checked/exact/round-trip、范围和
  零分配裁决；OS sleep 按不早醒、EINTR/平台降级及 syscall 结构裁决；profiling/coarse clock 若无具名
  产品消费者只保留绝对成本与 codegen diagnostic。不得声明跨平台“所有 time 操作 SOTA”，也不得把
  Common queue 的 peer-derived 一般能力伪装成已接入 sched/io。
- correctness gates：normal；zero/negative/past；全 `uint64_t`、signed epoch、2^53、ratio、NaN/Inf、
  overflow/underflow；EINTR/OS capability/fallback；timer reschedule/cancel/cancel_all/advance/FIFO 与
  wrong-owner；output-list/queue/entry lifetime；随机 oracle、million-churn、public-header self-containment；
  Debug misuse；ASan/UBSan/TSan（按 single-owner 合同适用）；MSVC/GCC/AppleClang 与 x86-64/ARM64 真 ISA。
- delivery gates：中央 component graph 与真实 target edge；time-only source build；relocated static/shared
  C++23 SDK consumer；代表性 platform/sched/io 编译/测试；产物与安装树没有 C schema、bridge、Native
  FFI symbol 或只为 FFI 存在的 option。
- performance gates：冻结 source-primary scheduler timer replay 的自然 tick 计费边界、node ownership、
  cancel/expiry 分布与同合同 faithful Seastar peer；W1-WIN 的 A/A、5%/10% delta、batch-matched floor、
  双 order 与适用 placement 全部通过后，required product strata 须通过 5% non-inferiority。clock/
  common/sleep 的 mechanism 行不进入该产品聚合。
- 预期输出：有限 capability/gap ledger 与唯一 `FIX` 集；immutable candidate commit/tree；correctness、
  delivery、usability/docs、red-team 与 P1 evidence；四项 LOC、known limitations 与 reopen triggers。
- 停止条件 / 可接受限制：全部适用工程门通过则进入 `CLOSED`；若真实产品缺口、同合同 P1 门或
  source/static/shared 交付在冻结预算内不能通过，则点名 blocker 和恢复条件进入 `BLOCKED`。无具名
  consumer 的 calendar/timezone、跨线程 service、periodic policy、Native FFI 和额外 timer family 是
  可接受 negative space，不因成熟库体积而自动扩面。

## 审计输入（尚非 `AUDITED` 结论）

- 当前实现能力已反向枚举为 typed/raw/default/opt-in clocks、signed Unix interchange、饱和 deadline、
  blocking sleep、`Instant`、arbitrary radix timers 与 one-duration timers；每个公开入口仍须落入 ledger。
- 当前真实消费确认：sched 使用 `TimerQueue`、deadline/remaining 与 callback/data；io 使用
  `TimerEntry` 和 `UnixTimestamp`；platform 公开 `UnixTimestamp`。`CommonTimeoutQueue` 暂无产品消费者。
- 已发现但尚未冻结的候选缺口包括：当前性能页仍含已删除 C ABI 的 workload；公开 header 还留有
  “C adapter/boundary” 私有 seam 叙事；Windows timestamp 转换只有读向而没有确认是否应提供 checked
  反向转换。它们必须经 capability/consumer/peer ledger 裁决后才能进入 `FIX`。

