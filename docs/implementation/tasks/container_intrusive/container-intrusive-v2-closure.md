# container_intrusive v2 模块收口记录

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

> **记录性质：`HISTORICAL TASK EVIDENCE`。** 本页只记录本轮施工事实，不声明当前状态；
> [`module-status.json`](../../module-status.json) 的 `container_intrusive` 记录是唯一权威。本记录快照中，
> 产品形状已经冻结，但 P1 工程性能与最终独立红队 identity 尚未签署。稳定边界见
> [`container-intrusive-design`](../../../architecture/container-intrusive-design.md)，用户写法见
> [`user/container-intrusive`](../../../user/container-intrusive.md)，性能资格见
> [`benchmarks-container-intrusive`](../../evidence/performance/benchmarks-container-intrusive.md)。

## 身份、目标与范围

| 字段 | 值 |
|---|---|
| 起始基线 | `49512b77434c4f15fec355e4bd6d6ab5a8b9b650` |
| 历史提取提交 | `840a5a6`：从 Core 移到独立 `hyle::container_intrusive` target；不算 v2 签署 |
| owner | container 语义；独立 header-only 产品组件 |
| 实现边界 | `include/hyle/container/intrusive_list.h`、`src/container_intrusive/**` |
| 直接产品依赖 | `PUBLIC core`；无 runtime artifact |
| 本轮必要消费者回审 | time、sched；完整 container 聚合；Windows IOCP 特化反证 |
| 非目标 | 完整 container、time、sched、io 的模块签署；Native FFI；32-bit ABI |
| product candidate | commit `79c644c2b7cef51786862a5bf0dd6b565305a308`；tree `8588225e17ab4deecfbe7916a3f1d6ca7fe9a246` |
| 当前工程性能 | P1 `INCONCLUSIVE`；等待 `closure-perf-v1`、W1-WIN、预声明 5% margin 的 candidate-bound manifest |
| Formal | 独立 `PENDING`；不替代也不重复阻塞工程门 |

目标不是把它扩成 Boost.Intrusive 全集，而是确认一个真实用户能够自然使用的最小双向 member-hook
primitive：零分配、caller-owned 稳定节点、auto/safe 两种常用 retirement 合同、O(1) 任意摘链和
batch/range handoff。
旧 Core/container 任务与旧三平台点值只作为输入，不能继承绿色状态。

## Demand / capability ledger

| ID | Consumer task / normal shape | 处置与公开路线 | 复杂度 / failure / concurrency | 强 peer 与负空间 |
|---|---|---|---|---|
| INTR-MEMBER-01 | task/completion 等稳定 storage 允许 node 先销毁，需要自动撤销 membership | `MUST`：auto `IntrusiveListHook` + `IntrusiveList<T,&T::hook>` | 两指针 hook + 两指针 sentinel；两端 O(1)；auto-unlink；外部同步 | Folly/Boost auto-unlink；不增加 allocation、owner、lock |
| INTR-SAFE-01 | timer、waiter/owner state machine 规定只可在 owner lock/阶段内显式摘链，漏摘必须暴露 | `MUST`：`SafeIntrusiveListHook` + `SafeIntrusiveList<T,&T::hook>` | 与 auto 同 footprint/算法；Debug 析构 fail-fast，Release 是 zero-cost precondition；无 hook-level unlink | Folly/Boost safe-link；HYLE TimerEntry 真实消费；不让析构无锁修改共享链 |
| INTR-CANCEL-01 | 已知 timer/waiter node 在任意位置取消/retract | `MUST`：list `remove`、auto-only 幂等 `hook.unlink()`、O(1) `iterator_to` | O(1)；`remove/iterator_to` 要求正确 owner/membership；safe 不允许绕过 list；无 owner scan | Boost/LLVM node-to-iterator；不为 exact owner 给每个 hook 加第三指针 |
| INTR-BATCH-01 | ready/completion/waiter 子链或整链在 owner/queue 间交付 | `MUST`：whole/single/`[first,last)` splice、`append`、`consume_all` | relink O(1)，consume O(n)；callback 前摘除当前 node 且须 noexcept | Boost/LLVM range splice；不以逐项循环冒充同复杂度 |
| INTR-PREFIX-01 | common timeout 从有序队首交付满足 predicate 的最长前缀 | `OUT_OF_MODULE`：time 内 iterator scan + owner update，再调用 range splice | O(k) scan + O(1) relink；与专用组合函数相同，不增加公共 API | 只有 time 一个消费者；成熟 peer 无同形公共原语，避免 consumer/benchmark-shaped API |
| INTR-RANGE-01 | C++ 用户按惯用 iterator/range 做位置操作和稳定遍历 | `GENERAL`：bidirectional/const/reverse iterator、insert/erase、move/swap | iterator 一个指针；除被摘链 membership 外 node iterator 稳定；end 绑定 list sentinel；不是 `sized_range` | 标准 list / Boost / LLVM 形状；不提供隐藏 O(n) `.size()` |
| INTR-RECOVER-01 | null/already-linked 是调用方正常分支 | `GENERAL`：`try_push_front/back`、`try_insert` | O(1)，只检查 null/linked；position/owner/range 仍是前置条件 | 不把 `try_*` 冒充 untrusted-input validator |
| INTR-MULTI-01 | 一个 request 同时属于 ready 与 timeout 等独立 membership | `GENERAL`：每种 membership 嵌入一个 hook | 每个 membership 两指针；彼此独立 | member hook 成熟形状；不增加运行时 hook registry |
| INTR-FFI-01 | foreign runtime 操作 C++ pointer-to-member 与 caller-owned node | `REJECT / N/A` | 无自然窄 runtime task；source/static/shared 均为 C++ 交付 | 若未来有外语 timer/runtime，由上层 opaque node adapter 重新准入 |

`CountedIntrusiveList` 单独裁决为 `DEFER`：`TimerQueue` 的 count 是跨 bucket 业务状态，CV 的 waiter
count 是原子发布协议；`CommonTimeoutQueue` 的 count 虽等于单链 membership，却只是 time 专用的
`size()` 状态，尚不能确定 auto/safe 与 range splice 的通用 counted 合同。count 会增加 list 一个
word、所有 mutation 的计数写，且任意 range splice 需 O(k) 扫描或 caller 传 n。其余负空间包括
normal/base/function/offset hook、
公共 slist/hlist、exact owner、linear `.size()`、完整 list algorithms、allocator ownership、内部同步
和并发 reclamation；出现独立真实任务或新的成熟通用反证时重开。

## 公开入口反向追踪与 API 裁决

- `IntrusiveListHook`：INTR-MEMBER-01 / CANCEL-01 / MULTI-01；copy/move construction 创建未链接
  hook，assignment 保留 destination membership，析构 auto-unlink。
- `SafeIntrusiveListHook` / `SafeIntrusiveList`：INTR-SAFE-01；与 auto 共享 list API/算法和 footprint，
  但只能经 list 显式摘链，析构不充当恢复路径。
- `push_*`、`pop_*`、front/back、`empty`：INTR-MEMBER-01；空访问/弹出返回 null。
- `iterator_to`、insert/erase/remove、iterator/reverse iterator：INTR-CANCEL-01 / RANGE-01。
- 三种 splice、append、move/swap：INTR-BATCH-01 / RANGE-01。
- `consume_all`、`for_each`：INTR-BATCH-01；callback lifetime 合同分别文档化。
- prefix selection：INTR-PREFIX-01；策略留在 time，公共模块只提供 range splice 机制。
- `try_*`：INTR-RECOVER-01；没有泛化成 owner/range validation。

本轮补齐了 O(1) node-to-iterator 与 O(1) range splice；callback 实现由直接函数调用改为
`std::invoke`，使 member-function pointer 与公开约束一致。没有为了 benchmark 增加 count、owner、
cache、allocation 或分支元数据，也没有用线性 `.size()` 制造 C++23 `sized_range` 的复杂度谎言。

## 消费者、重复实现与依赖裁决

| 范围 | 扫描结果与处置 |
|---|---|
| time | `TimerEntry` 要求析构前 owner=null 且显式摘链，是 safe-link 真实消费者；`TimerList` 改用 `SafeIntrusiveList`，保留 `time -> PUBLIC container_intrusive` |
| sched ConditionVariable | 保留手写 waiter 链：摘链后复用 `next` 作锁外 wake chain；通用 list 需要额外字段/写，且 safe/auto 都不能自然表达该状态复用 |
| sched Semaphore | 保留手写公平 FIFO/batch wake 链；同一 `next` 在 wait queue 与 detached wake batch 间切换是 sched 私有状态机，不机械迁移 |
| Windows Proactor `IoOp` | 保留专用实现：`OVERLAPPED` 必须首成员且 link storage 与 owner free-list 状态机复用；换公共 hook 会增加状态/footprint，合同也不同 |
| cancellation hlist/free-list | 单链或 `prev_next` 状态机不是本模块双向 member-hook 语义，不机械迁移 |
| 完整 container | `PUBLIC` 聚合独立 target；不复制 implementation |

公开头直接使用该类型的模块必须有真实 direct `PUBLIC` target edge。此次扫描还发现 io 的公开 timeout
头直接暴露 time 类型；支持 fcontext transport 时补 `io -> PUBLIC time`。这是被本轮依赖审计触发的
相邻 build/package correctness 修复，不表示 io 已收口。

### 被拒绝的 sched 强制迁移诊断

本轮曾把 ConditionVariable/Semaphore 的手写链替换为 auto-unlink list。红队先发现合同错配，再以
同一 Release binary、固定 Windows P-core、双顺序 n=15、3% material margin 做限域 replay；A/A
双顺序为 `EQUIVALENT`，known-extra-walk 双顺序正确检出 `LOSE`，仪器具备本次裁决资格。结果只覆盖
锁内 batch relink seam，不冒充完整 scheduler episode：

| 事实 | rejected generic 形状 / 原特化形状 |
|---|---|
| Condition waiter layout | 56 / 48 bytes，+16.7% |
| Condition notify-all，8 waiters | 1.510x / 1.516x 时间比，两顺序 CI 下界均不低于 1.50 |
| Condition notify-all，64 waiters | 1.109x / 1.107x，两顺序均超过 3% 门 |
| Semaphore release-all，8 waiters | 1.731x / 2.342x；顺序敏感但两门都明确失败 |
| Semaphore release-all，64 waiters | 1.300x / 1.302x |
| Semaphore close | 近似平或 unresolved，不抵消其他失败 |

静态模型同向解释结果：Condition 从每 node 一个 byte membership store 变成约 `4N-1` 个 node pointer
stores，Semaphore release 从约 `6N` 增至 `10N+1` persistent writes；摘链后还需要额外 wake link。
原始诊断保存在本机历史目录
`C:\hyle-perf-evidence\container-intrusive-sched-consumer-diagnostic-n15-20260826`。该 benchmark target
在裁决后删除，避免把已拒绝的产品形状变成长期优化目标；最终证书只测实际 auto/safe 产品和 time
消费者。

## 算法、布局与 codegen

| 项 | 候选合同 |
|---|---|
| 常态 layout | 64-bit ABI 上 auto hook=16、safe hook=16、任一种 list=16、iterator=8 bytes；无 count/owner/allocator/lock |
| 两端 pop | 已知 sentinel predecessor/successor 的直接重链，避免通用 unlink 的冗余 load |
| insert | 缓存 `pos->prev_`，不重复读取 |
| single/range splice | 成熟 circular-list 六次 store / 边界重链；不 clear 后立即重写 hook |
| swap | 直接交换两组边界并修 sentinel，不再用 temporary + 三次 append |
| move assignment | 与 Boost.Intrusive 同档用 swap 做 O(1) membership exchange；moved-from source 接管旧 destination nodes，不承诺为空 |
| container_of | standard-layout + 已登记 MSVC/Itanium data-member-pointer ABI；未知 representation 编译期 fail closed |
| release precondition | `HYLE_ASSUME` 不产生 owner/check 热路径；Debug fail-fast 诊断可证明的常见误用 |

专用 noinline probe/runner 与 COFF/ELF/Mach-O parser 锁定结构属性，而不锁总指令数：禁止 allocation/
外部 helper，验证 constant hook offset、owner assume 无运行时 opcode，以及关键 O(1) primitive 的真实
布局和链结果。绝对指令数和亚纳秒点值不作为产品 API 门。

## Correctness / delivery / performance gates

| Gate | 覆盖与当前事实 |
|---|---|
| 单元/性质 | auto/safe conventional API、same/cross-list splice、multi-hook、auto-unlink、显式 safe retirement、move/swap、callback、异常传播、`std::list` 随机差分 |
| Debug misuse | double insert、erase(end)、remove(unlinked)、whole-list self-transfer、linked safe-node destruction fail-fast |
| consumer | time safe-list 正确性/性能与原 auto 算法对账；sched 保留特化且回退迁移后的测试绑定 candidate |
| header/component | self-contained；source standalone consumer；isolated/relocated static+shared SDK 均不得产生 intrusive binary artifact |
| component graph | direct `container_intrusive -> core`、`time/container -> container_intrusive` 与适用时 `io -> time` 由机器合同断言；sched 不伪造 direct edge |
| codegen | MSVC/clang-cl、WSL GCC 14 ELF、macOS AppleClang 21 Mach-O 预检通过；auto/safe hot op 同形，safe unlinked 析构零工作，auto linked-node 析构是无 helper/无循环的 O(1) 重链；最终 candidate identity 待取证 |
| performance | 历史三平台同合同 Boost 非劣只作输入；当前候选必须重新绑定 harness、peer、环境和 raw samples |
| docs/usability | 独立 architecture/user 页、选择指南、ownership/concurrency/failure/失效矩阵/负空间已建立；最终 stale/link check 待签署重跑 |
| Native FFI | `N/A`；没有 runtime artifact 或 raw hook export |

## 当前预检结果

- Windows MSVC Debug/Release + WERROR 的 intrusive/time 为 9/9；clang-cl Release intrusive 为 4/4。
- WSL GCC 14 Release + WERROR 为 intrusive 4/4、time 5/5；ASan 与 UBSan 各为 intrusive 3/3、
  time 5/5，均无诊断。macOS arm64 AppleClang 21 Release + WERROR 为 intrusive 4/4、time 4/4。
  两个平台使用同一个 1185-file dirty snapshot（tree SHA-256
  `186f471372f6c0a889257087ae268082afc8d36288bc305551e025deef3233bc`）；这是候选预检，不能替代
  clean commit evidence。
- Windows 与 macOS component contract 通过：source standalone consumer、relocated static/shared SDK、
  zero intrusive binary artifact、direct public edges 与 sample run 均受机器合同约束。
- Windows 旧 harness 的 dirty n=31 曾报告 A/A、known-delta、floor 与 9 个 peer workload 通过；随后
  红队加入 near-MDE control、双 fixed-floor budget、双 phase observer、allocation-order strata、实际
  artifact architecture/Release/tool identity 门，因此该轮已整体标为 `SUPERSEDED / UNQUALIFIED`，
  不能被写作当前 harness 资格。新的 `closure-perf-v1` P1 qualification 尚未执行；Formal 跨平台
  证书是完成工程门之后的独立任务。旧专用 runner 只保留为 diagnostic/可能的 Formal plugin。
- 当时的文档合同（253 个 Markdown、673 条相对链接）与 `git diff --check` 通过；这些计数是历史
  快照，不冒充当前工作树的复验。
- 当前 P1 manifest 与最终独立红队 identity 尚待完成，不能从预检提前写 `VERIFIED/CLOSED`。
  time consumer 的 N=1/8 peer sanity 仍有 time-owned 小批量差距，但 auto/safe layout 与
  hot-op codegen 同形，N=64/4096 未退化；不以 container 修改迎合该上层算法/仪器问题。

## Candidate、证据与 LOC

| 字段 | 值 |
|---|---|
| Candidate commit/tree | `79c644c2b7cef51786862a5bf0dd6b565305a308` / `8588225e17ab4deecfbe7916a3f1d6ca7fe9a246` |
| Peer identity | Boost.Intrusive `df8eeb90846648464e3c8e378e312f0cec772e83`；当前 v1 manifest 尚未绑定 compiler/binary identity |
| Current P1 evidence | `TBD`；性能页中的旧 raw 均为 `HISTORICAL / UNQUALIFIED` |
| 全库代码 | `303442`（candidate 快照） |
| 自实现包含测试 | `142952`（candidate 快照） |
| 自实现不包含测试 | `96244`（candidate 快照） |
| container_intrusive 实现代码 | `497`（candidate 快照；public header + `src/container_intrusive/**`） |
| 独立红队 | 初审 findings 已修；最终 candidate identity verdict 未签 |
| End commit / push | `1a465e23ea69a6e613840cf8d30d33054463ae1c`；这是首次把 candidate `79c644c2` 记为 `BLOCKED` 的 registry receipt，不是 `CLOSED` 签署 |

## 剩余限制与重开条件

当前候选是 64-bit、standard-layout、stable-address、单线程/外部同步的 raw-pointer member-hook list。
它不解决 node reclamation、exact owner、shared-memory offset pointer 或完整 STL-like intrusive
algorithms。受支持 compiler ABI 的 member recovery、真实 consumer lifetime、正确性/security、
直接依赖、交付形状，或同合同 peer 的可复现 material/复杂度优势变化时重开。LOC、类型数量、
命名偏好、单次跑分、N=32 placement 翻转或亚纳秒差异不构成重开理由。
