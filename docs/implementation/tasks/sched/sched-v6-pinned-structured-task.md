# Sched v6：pinned、joinable、typed structured task

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

## Closure card（`CLOSED`）

- component / owner：`sched` / structured task admission and lifetime；round：
  `sched-v6-pinned-structured-task`；baseline：
  `cb87f5028b7f6bf8558054337de59538bb441f56`。前一轮当前范围由
  [Sched v5 terminal](sched-v5-io-completion-search.md) 与其
  [工程性能证据](../../evidence/performance/benchmarks-sched.md)固定；它们是本轮 baseline 事实，不授权
  重开 idle/IO-completion 工作。
- trigger：`SCHED-V6-PINNED-STRUCTURED-TASK`。跨产品公开 API 纵切已经证明
  `Scheduler::spawn_detached_on` 只能得到 pinned fire-and-forget，而 `spawn_task` 只能得到可 join 的
  typed task；两者没有交集。`samples/08_inference_service/main.cc` 因而手写 `launch_pinned`，重复处理
  coroutine frame failure、result/exception、错误上报和 detached lifetime。
- 正常 consumer 任务：把可能 suspend 的 `Task<T>` 提交到指定 scheduler worker，获得
  `JoinHandle<T>`；task 从首次 dispatch 到每次 timer/IO/foreign wake 都保持同一 owner，并能 join、
  detach 和 cooperative cancel。
- 主产品纵切：推理服务的 per-worker IO completion owner 与 arena/staging state。正交反偏置纵切：
  renderer 的 per-worker upload/resource queue。两者都要求稳定 worker ownership，但 result/error policy
  仍由调用者拥有。
- 锁定输入：HYLE inference-service sample 与现有 Sched/Fiber/JoinState；SGLang
  `dcebe8c4733a5cb802fb679e0c8fbb886e2801b0`、vLLM
  `51da0ca66c8065619c79e35dff97aa99aeaf5644`、ds4
  `f62ca29a308724cde5bc99134ede19104b2a3260`、llama.cpp
  `e71b80510c848c00175924ecf3c40333ccae8eb5` 用于真实 worker/backend 生命周期回放；Seastar、Folly、
  marl 与 oneTBB 只在 `AUDITED` 阶段锁定具体 API/机制证据。
- scope：审计并提供 `spawn_task_on(scheduler, worker, allocator, Task<T>, stack_size)` 或等价自然
  C++ 形状；复用现有 `TaskDriver<T>`、`JoinState<T>`、Fiber stack pool、admission accounting、pinned
  queue/reinject 和 `JoinHandle<T>`；覆盖 typed/void、首次运行与 suspend/wake owner、join/detach/cancel、
  invalid worker、OOM/stack/admission failure、shutdown 与 source/static/shared C++ 交付。
- non-goals：新 task lifecycle、priority/QoS/deadline lane、gang/SPMD、强杀、跨 scheduler migration、
  Compute/Graph、C/Native FFI、公共性能仪器、调度队列或 idle policy 重写，以及 F03 IO file watch。
- 直接依赖：现有 Sched 的 Core/Error/Container-intrusive/Util-core/Time/Platform/Mem/Mem-vm；不得新增
  dependency edge。直接消费者：IO per-worker operation owners、未来 Compute completion/staging owner、
  inference/rendering service；本轮不修改仍为 `QUEUED` 的 Compute/Graph。
- 性能 tier：暂定 `P1`。P0 必须先证明相对现有 `spawn_task` 只改变 admission route，不增加 task/cell/
  stack allocation、result copy、atomic、lock 或 syscall。P1 只允许同一 typed+suspending+pinned 生命周期
  的真实 consumer episode；若成熟 peer 无相同语义，则用现有手写组合路线作同合同迁移基线，不能拿
  unpinned、detached 或 non-suspending task 排名。material margin 5%。
- 仪器/机器：`closure-correctness-v1`；若审计后 P1 可比较，使用 `closure-perf-v1` 的已登记工程机器。
  correctness 覆盖 W1-WIN、W1-WSL 和一台 macOS；并发/唤醒风险需要 TSan，平台不支持时明确记录。
- 调查预算：能力/API/实现审计最多 90 分钟；至多两次产品纠正、一次性能机制归因和一次正式候选 P1。
  不为了点值调整公共 API、队列容量、pin policy 或常态 metadata。
- 预声明验证：public-header self-contained；typed/void/move-only result；invalid worker；frame/state/driver/
  stack OOM；closed/full admission；first-dispatch pin；timer/foreign wake 后仍 pin；join/detach/cancel；owner
  shutdown；Sched focused tests；受影响 IO integration；ASan/UBSan/TSan；source/static/shared public
  consumer；七问、分离红队、P0/P1 和固定治理命令。
- stop / blocker：`AUDITED` 后的有限 FIX 全部完成且七门通过则 `CLOSED`。若现有 Fiber owner/reinject
  不能安全承载 typed task、同合同性能或三平台 wake ownership 在预算内无法裁决，则 `BLOCKED`；不得
  通过 detached wrapper、弱化 result/lifetime 或无限 benchmark 姿势绕过。

## 能力与形状审计

锁定 peer：Seastar `78f64e2317b19c7e34ba61f5e8ea85f083eca1da` 的
`include/seastar/core/smp.hh:350-406` 以 `submit_to(shard, callable)` 同时表达目标 shard 与 typed
future；marl `b8406ab0a82524170ead6dd6c371b579bbc28b3e` 的公开 `schedule`、`Event` 与
`WaitGroup` 可组合相同执行旅程但没有 public worker-index admission；oneTBB
`e00c68dcac0e83947c9b54f6d5304c0616c9639f` 的 `task_arena` 约束 arena/concurrency 而不承诺一次
suspend/wake 生命周期的稳定 worker；Folly `65749da412d92fe27d87ae45a8ef670720712858` 的
Future/Executor 同样不把稳定 worker identity 纳入普通 future 合同。API 取 Seastar 的自然任务形状，
生命周期继续使用 HYLE 已有 JoinHandle，而不是复制 peer runtime。

| 能力族 | 准入旅程/来源 | 语义与生命周期 | 当前路线 | 覆盖状态 | 动作 |
|---|---|---|---|---|---|
| 指定 worker 的 suspending typed task | inference per-worker IO owner、renderer upload owner、Seastar `submit_to` | `Task<T>` 首次运行及 timer/IO/foreign wake 均回到同一 worker；返回 move-only `JoinHandle<T>` | pinned detached 与 unpinned typed 两条半路线 | `PARTIAL` | `FIX`：增加 `spawn_task_on` |
| result / error / cancel / detach / shutdown | 已有 `spawn_task` 与 `JoinHandle` | result 单次取得；task 与 handle 独立引用；取消协作式；scheduler drain 拥有运行中 task | `TaskDriver<T>` + `JoinState<T>` | `COMPLETE` | `KEEP`，禁止第二套 lifecycle |
| pinned admission 与 wake ownership | 已有 `spawn_on`、affinity test、IO owner | 合法 worker 首次入 private queue；所有 reinject 回 owner；无 migration | `Fiber::owner_worker` 与 per-worker pinned queue | `COMPLETE` | `KEEP`，新入口只路由到它 |
| 非挂起、typed、指定 worker callback | per-shard 小查询；Seastar 可表达 | worker stack 执行且返回 handle，不允许 suspend | 现有 inline queue 会由任意 worker 消费；要承诺 pin 需另一条 private-inline 机制 | `MISSING` | `DEFER`：语义和成本档不同，另立 consumer trigger 后审计 `schedule_task_on` |
| priority/deadline/gang/迁移 | 上层 execution policy | 不属于本交集，且会改变 queue/policy | 无 | `OUT_OF_SCOPE` | `REJECT` |

全部现有公开入口已回映：`schedule/schedule_task` 属 worker-stack non-suspending 族；
`spawn_detached/spawn_detached_on` 属无结果 fiber 族；`spawn_task` 属可窃取 structured task；
`spawn/spawn_on` 是 caller-owned low-level Fiber seam。它们没有 benchmark-only 入口，本轮只补最后一个
缺失交集。`spawn_task_on` 是 free function，与 `spawn_task` 对称；不把 placement 塞入 `Task`、
`JoinHandle` 或全局 Scheduler 默认值。

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 | 结果 |
|---|---|---|---|---|
| S6-01 | pinned fire-and-forget 与 typed joinable 不可组合 | 在 `join.h` 复用一个内部 admission helper，公开 `spawn_task_on`；合法性检查早于入口自有的 join-state/driver/stack allocation | typed/void/move-only、invalid worker、OOM/closed/full 全有确定 handle 结果 | `PASS`：`9f065791` 已实现并固定；终局 test-only 证据补齐全部失败路径 |
| S6-02 | owner 必须覆盖完整 suspend/wake 生命周期 | focused tests 覆盖首次 dispatch、yield、foreign Waker、cancel/detach/shutdown | 每次观测 worker 都等于所选 owner，无泄漏/挂死 | `PASS`：三平台 exact-candidate correctness 与 sanitizers 通过 |
| S6-03 | 当前 inference sample 手写生命周期 | 让真实 sample 与 package sample 使用新入口，更新用户 guide/architecture | 不再手写 Task resume/result/exception wrapper；三交付可发现 | `PASS`：inference sample 与 source/static/shared consumer 通过 |
| S6-04 | 交集成本没有独立证据 | 新增 opt-in v6 P1，同一 pinned+suspending+typed completion episode 对 baseline 已有公开 API 的手工组合；固定 5% 非劣门；既有 Marl 证据继续约束底层 wake/queue | 仪器 controls、双顺序/placement 和全部预声明 strata 通过 | `PASS`：31 个独立进程、4/4 strata 5% NI 与全部 controls 通过 |

本表之后不增加轻量 pinned queue、优先级或 IO 功能；出现相关需求另立 round。

### API / usability freeze

```cpp
auto handle = hyle::sched::spawn_task_on(
    scheduler, worker, allocator, request_loop(allocator, state));
auto result = handle.join();
```

- 参数顺序保持与 `spawn_task(scheduler, allocator, task, stack_size)` 一致，只在 scheduler 后插入必须的
  `worker`；不新增 options object 或 bool。
- `worker >= scheduler.num_workers()` 返回已经完成的 error handle，且不得取得 join state、driver 或
  stack；合法调用的 frame/state/driver/stack ownership 与 `spawn_task` 完全相同。
- task 可 suspend；所有 wake 回指定 worker。`JoinHandle` 的 join/detach/cooperative cancel 语义不变；
  handle 销毁不取消 task，Scheduler 仍负责 drain。
- 默认 stack tier 不变。相对 `spawn_task` 不新增 allocation/copy/atomic/lock/syscall、常态 metadata 或
  每次 resume/wake 成本；入口只增加一次可预测的 scheduler/worker 合法性检查，模板 shell 在 admission
  点用 `if constexpr` 静态选择 `spawn`/`spawn_on`，没有运行时路由分支。
- source/static/shared 都从 `<hyle/sched/join.h>` 取得同一 API；这是 C++ template API，不增加 C/FFI
  symbol。guide 必须明确只有真实 per-worker owner 才使用 pin，普通 work 保持可窃取。

## 七问（实现前裁决）

1. 第三方路径：运行时只使用 vendored Boost.Context assembly，锁/许可证由 `src/sched/CMakeLists.txt`
   的 `fcontext` package dependency 承载；本轮不变更第三方源码或依赖。
2. 功能完整：上述主/正交纵切的 suspending pinned typed 交集是唯一 `MUST`；新入口闭合后可直接组合
   IO owner、Task 和 JoinHandle。轻量 pinned callback 已显式分离而非伪装成同一语义。
3. 内部复用：必须复用 Error、Mem/Mem-vm stack pool、Time wake、Platform thread、
   Container-intrusive waiters与现有 Scheduler pinned queue；不增加同义 primitive。
4. 性能：算法仍为现有 pinned enqueue/reinject；P0 检查零新增常态成本，P1 对比 baseline 的
   `Task + spawn_detached_on + typed slot + per-task WaitGroup` 完整迁移 episode；既有 Marl P1 继续证明
   底层 completion spine，不把 unpinned/detached/non-suspending 路线当成 v6 比较对象。
5. 交付边界：没有 C/Native FFI；模板 API 只服务当前 C++ source/static/shared 产品。
6. 构建交付：component graph 与 `sched` target 不增边；头 owner 已存在。本轮只需证明三种 consumer 都
   能实例化并运行新入口。
7. 易用与文档：一个入口表达 worker + task + typed handle；invalid worker 可恢复；guide 同时说明
   pin 的适用场景、负成本与普通可窃取默认。

产品候选固定为 `9f0657919efd9daca6e5c22f4d01f9fa4fe39082`，tree 为
`e5396357310099aad3ce6d6bf4597963c8a37b8b`。候选后的提交没有改变 `include/hyle/sched` 或
`src/sched` 产品 seam；只补测试、证据、公共仪器修复和状态。

## 终局候选验证

精确候选以 SHA-256
`8184A504B0926C85604290231C7A780D75542773117B22533A8D5F413DCEF8DB` 的同一 archive 进入 WSL 与
macOS；Windows 使用 detached candidate worktree。远端副本只作验证，不是产品修改来源。

| 风险面 | 候选或 test-only 证据 | 结果 |
|---|---|---|
| W1-WIN correctness | MSVC 19.50 Release；完整 Sched + public-header；受影响 IO integration | `2/2 PASS`、Sched `181/181`；IO `15/15 PASS` |
| W1-WIN delivery | exact candidate 的 selected source、relocated shared、relocated static consumer；完整 inference-service sample | `3/3 PASS`；sample 编译通过 |
| W1-WIN sanitizer | exact candidate MSVC ASan 完整 Sched | `181/181 PASS` |
| W1-WSL portability/sanitizer | GCC 14.2 Release/header；ASan、UBSan、TSan（TSan 以 `setarch x86_64 -R` 满足运行环境） | focused/header `2/2 PASS`；三种 sanitizer 均完整通过 |
| W1-WSL delivery | exact archive 的 source、relocated shared、relocated static consumer | `3/3 PASS` |
| M1-MAC portability/delivery | MacBook Air arm64、AppleClang 21；Release/header 与三交付 | focused/header `2/2 PASS`；source/static/shared `3/3 PASS` |
| failure/admission 红队补证 | 产品 seam 不变后增加 `spawn_task_on` frame/state/driver/stack-failure 与 pinned queue-full cleanup 用例 | MSVC Release 完整 Sched `184/184 PASS`；新增四条 MSVC ASan `4/4 PASS` |

最后一行的 stack failure 使用 `SIZE_MAX` 触发 `VmRegion::reserve_guarded` 的确定性 overflow 拒绝，走与
OS reserve/commit OOM 相同的 `!stack_result` 清理分支；它不靠耗尽整机内存制造不稳定测试。task 参数在
调用表达式求值时已经拥有 coroutine frame，所以 invalid worker 只承诺不取得入口自有的 join state、
driver 或 stack，并会在返回时正常销毁 caller 已构造的 task；用户 guide 已明确这条边界。

## P0 与正式 P1

P0 对 candidate/baseline 产品 diff 与生成路径的检查结论：`TaskDriver<T>`、`JoinState<T>`、`Fiber` 及
其 layout 均未改变；合法调用仍为一个 coroutine frame、一个 join state、一个 driver 和一个可复用
guarded stack。没有新增 persistent metadata、result copy、atomic、lock 或 syscall。新增入口只有一次
`valid()`/worker 范围检查；`spawn_task_impl<true>` 的 admission 通过 `if constexpr` 静态落到既有
`Scheduler::spawn_on`，resume、yield、timer/IO/foreign wake 全部复用已有 owner/reinject 路线。

正式 P1 使用 [`sched-v6-pinned-task-p1-v1`](../../../../bench/contracts/sched-v6-pinned-task-p1-v1.json)
（SHA-256 `1424330e6f359cd34e8fdefbd29afbcb9a317397a095b70586685f7cd4b7c93b`），比较相同的
pinned+suspending+typed completion episode：HYLE 新入口对 baseline 公共 API 手工组合。W1-WIN、
MSVC 19.50、Release/static、portable ISA、31 个独立进程、双执行顺序与独立 placement 变体；A/A、
5%/10% known-delta、timer-floor、artifact/build/source/peer/host/input provenance 全部通过。

| stratum | HYLE / manual median（op/s） | paired advantage / 98.93% CI | exact | 5% NI |
|---|---:|---:|---|---|
| primary / HYLE first | 371,095 / 372,502 | 1.0005 / [0.9654, 1.0263] | `PARITY` | `PASS` |
| primary / peer first | 375,923 / 365,675 | 1.0325 / [0.9945, 1.0892] | `INCONCLUSIVE` | `PASS` |
| placement / HYLE first | 374,976 / 367,890 | 0.9980 / [0.9796, 1.0852] | `INCONCLUSIVE` | `PASS` |
| placement / peer first | 377,496 / 364,941 | 1.0339 / [0.9907, 1.0985] | `INCONCLUSIVE` | `PASS` |

四行 exact 聚合为 `INCONCLUSIVE`，不能宣称稳定胜出；四行 5% non-inferiority 与 qualification/
performance gate 全为 `PASS`，也没有固定方向的 material loss。manifest/summary SHA-256 分别为
`D4E7EDFBE41607B8484051A57CA2AF246F88C3B9CEF26F35AC7E5CBC8E78EDB9` /
`8CD00847AD0D8FC5CAE3E7F2446279652D2FEA5028527CA0A902FC795D6A7286`；独立 verifier 报
`QUALIFIED`、`EXACT` provenance。首次运行在采样前暴露 structured-pair placement validator 的不可满足
计数条件；独立工具任务 [`865008d4`](../project/performance-structured-placement-validator-2026-09.md)
修正路由身份后用全新目录重跑，失败运行没有形成样本。人读性能结论同步到
[`benchmarks-sched.md`](../../evidence/performance/benchmarks-sched.md)。

## 终局红队

- 全部公开 scheduling/spawn/join 入口已回映；新 API 填的是 pinned + suspending + typed 的真实交集，
  没有新增 `fast/light`、benchmark 参数、常态 metadata 或第二套 task lifecycle；
- invalid scheduler/worker、task frame、join state、driver、stack failure、closed/full admission 都返回可
  join 的 typed error handle；test-only 回归确认失败时 frame、state、driver 与 stack ownership 全部释放；
- first dispatch、cooperative yield、foreign wake、cancel、detach 与 drain 都保持 owner；底层并发由 exact
  candidate 的 TSan 覆盖，没有发现 UAF、double completion 或 shutdown 漏计；
- API 只把必需 worker 放在与 `spawn_task` 对称的位置；普通任务仍默认可窃取，guide 没把 pin 宣传成
  通用加速开关；caller 已构造 task frame 的边界也不再被文档掩盖；
- 组件边、compiled target、public header owner 与三交付不变；复用 Error、Mem/Mem-vm、Time、Platform、
  Container-intrusive 和现有 Sched owner queue，没有为避免依赖复制 primitive；没有 C/Native FFI；
- P1 双方语义、ownership、failure、setup/teardown、完成条件一致；placement 四个 executable 的路径和
  binary hash 全部不同，没有把 unpinned、detached 或 worker-stack callback 当成 peer。未发现未处理的
  medium/high 风险 finding；`schedule_task_on` 属不同的非挂起 private-inline 机制，只有新的真实 consumer
  trigger 才能另轮准入。

## 七门与裁决

| Gate | 证据 | Verdict |
|---|---|---|
| scope | 一个真实能力交集、两个产品纵切、有限 `S6-01..S6-04` | `PASS` |
| architecture | 只组合既有 structured lifetime 与 pinned admission；无新依赖或循环 | `PASS` |
| correctness | 三平台、失败/admission、owner wake、三 sanitizer 与受影响 IO | `PASS` |
| delivery | public header + source/static/shared relocated consumer + inference sample | `PASS` |
| usability_docs | 对称入口、明确 pin 适用条件、错误/lifetime/cancel 陷阱 | `PASS` |
| red_team | 上述分离红队，无未处理 medium/high finding | `PASS` |
| engineering_performance | P0 成本结构 + qualified W1-WIN 4/4 的 5% NI | `PASS` |

LOC 由 `python scripts/loc-report.py --module sched --commit
9f0657919efd9daca6e5c22f4d01f9fa4fe39082 --self-check` 计算并自检通过：全库 504,811 行，自实现不含
测试 90,560 行，Sched 实现 5,689 行。

最终裁决为 `CLOSED`：本轮 pinned、joinable、typed structured task 已完成且七门全过。当前工程声明只
覆盖上述真实 episode 的 W1-WIN 5% 非劣，不宣称全部 Scheduler 操作或跨平台 Formal SOTA。只有可复现
correctness/security/UB/lifetime 缺陷、真实消费者证明的新通用能力缺口、公开 API/layout/算法/依赖/
交付变化，或合格同合同证据显示超过 5% 的 material regression，才允许新 card 重开；旧点值、无新
事实的“再看一次”或单纯希望 pinned callback 不构成 trigger。

<!-- MODULE CLOSURE RECEIPT: CLOSED -->
