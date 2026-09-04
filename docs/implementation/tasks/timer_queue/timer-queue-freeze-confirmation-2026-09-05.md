# Timer-queue 冻结确认 — 2026-09-05

<!-- MODULE FREEZE CONFIRMATION: FROZEN -->

> 这是对既有 `CLOSED` factor 的只读确认，不是无触发重开，不建立新 product candidate，也不替换
> [`timer-queue-v1-factor`](timer-queue-v1-factor.md) terminal record。当前状态真相仍是
> [`module-status.json`](../../module-status.json)。

## 绑定身份与边界

- confirmation：`timer-queue-freeze-2026-09-05`；marker：`FROZEN`；
- reviewed repository commit / tree：
  `54ddb99fe43e4e45664f0150c4f1d84c95166b8e` /
  `19b813b2b999390f6677ce211d8ef88015b35e1b`；
- frozen candidate / tree：
  `381eb4b195973ed35ed23ca63c05f9605a132f9b` /
  `a5d48a259161c1a60e19450e42b4a6eac9a1a712`；
- candidate 到 reviewed commit 的 `timer_entry.h`、`timer_queue.h`、`common_timeout_queue.h` 与
  `src/timer_queue` product implementation 为 zero diff；后续变动只加固 host Time 测试/互操作与中央图；
- 本次 confirmation 没有产品修改。

`timer_queue` 是为解除 host-clock 与 intrusive storage 耦合而形成的 header-only 质因数：直接
`PUBLIC core,time,container_intrusive`，精确选择只解析这四个 factors。普通用户仍从 `hyle::time`
获得完整产品，不需要知道内部 factor；Sched/IO 可按真实边直接消费它。callback 执行、thread/fiber
wait、cross-thread command、periodic/cancellation policy 留在 runtime owner。

## 七问终答

### 1. 第三方产品路径

`PASS`。产品路径只有 Core、Time、Container-intrusive，没有第三方实现、类型或 runtime。Seastar
`143da84e51f8b5ac63a097056cef7d7e876223cf` 与 libevent
`d82464a277d0f42703702c4dfd9af6af38595a83` 仅是 peer；其相关 timer-set/common-timeout 路径相对冻结
输入没有实现变化。llama.cpp `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd` 与 ggml
`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` 没有新增低层 timer-node 数据结构形状。

### 2. 功能超集与推理消费

`PASS`。`TimerEntry/TimerList` 完成 stable-address caller ownership、payload/callback hand-off 与
delivery drain；`TimerQueue` 完成 mixed arbitrary deadline 的 arm/reschedule/cancel/advance/next/
cancel-all/teardown；`CommonTimeoutQueue` 完成单一 duration、非递减 arm、transactional reject、FIFO、
cancel/advance/teardown；system/manual/ref provider 都可组合且不存入对象。

需要 delivery 全序时排序返回批次；需要执行、wait、periodic、cross-thread arm/cancel 或 propagated
budget 时走 Sched/IO owning state machine。它们有不同 ownership/progress/常态成本，不应塞进零锁、
零分配 node queue。当前 Sched timer lifecycle 和 IO timed slot 是真实推理 runtime building-block
consumer；未处置 `MUST` 为 0。

### 3. 现有模块与加速设施复用

`PASS`。

| Needed mechanism | Existing candidate | Verdict | Reason |
|---|---|---|---|
| invariant/codegen | Core | `REUSE` | 统一 fail-fast 与 compiler contract |
| monotonic value、provider、deadline overflow | Time | `REUSE` | 不复制 clock、duration 或 saturation；同一 clock authority |
| stable intrusive membership/bulk hand-off | Container-intrusive | `REUSE` | safe hook/list 提供 O(1) unlink/splice 与 owner-lifetime diagnosis |
| allocation/owning callable/wait/cancel propagation | Mem/Util/Sched/IO | `NOT_APPLICABLE/ROUTED` | 本 factor caller-owned、callback-free、single-owner；加入这些机制会改变语义和常态成本 |
| recoverable failure | Error | `NOT_APPLICABLE` | common ordering 有 `try_schedule`；wrong owner/lifetime/clock-domain 是程序合同 |
| SIMD/topology/NUMA/hash/reclamation | 相邻 modules | `NOT_APPLICABLE` | radix/list pointer chasing 无统一 bulk layout 或 runtime dispatch 决策 |

没有私造 intrusive list、clock、allocator、reclamation 或第二 dispatch authority。取消最小项保留一次
conservative early wake 是 O(1) cancel 的明确取舍，不是遗漏 Time 或 Sched 能力。

### 4. 拟真、公平性能

`PASS / P1_ENGINEERING`，5% margin。冻结 workload 是当前 Scheduler N=64 完整生命周期：64 arm、45
cancel、64 advance tick、19 delivery/callback-state observation；与 faithful Seastar `timer_set` 保持相同
stable-node ownership、状态变换和计费。八个 execution/placement/allocation strata 均 `WIN`，median
advantage 约 12.9%--19.0%，最低 99.26% CI lower 为 1.0911；八项 5% non-inferiority 全部 `PASS`。

保留 bundle 的 verifier 返回 `QUALIFIED/EXACT/CURRENT`、paths verified 与当前策略 eligible。当前
Scheduler 文件 hash 仍为已审计 renewal 的
`32ec50841311ca5db0a67f735582bcf1a49c4abdc605ae460b4d636808700ef8`；timer ownership、计费阶段和
delivery 未漂移。该结论只属 W1-WIN、N=64 与冻结合同；CommonTimeoutQueue 没有当前 HYLE 产品性能
consumer，因此只签 O(1)/output-sensitive 复杂度和 correctness，不产生虚假排名或 universal SOTA。

### 5. 旧 C ABI / Native FFI

`CLEAN`。全部产品能力是 C++ header/type/template；没有 C schema/header/bridge/export/sample/option/
package metadata。header-only factor 没有 binary symbol，亦不制造伪 DLL 或跨工具链 ABI 承诺。

### 6. Source / static / shared 构建与导入

`PASS`。Windows、WSL、macOS 的精确 factor 都解析为
`core,container_intrusive,time,timer_queue` 并通过 header/unit tests；三端 `hyle::time` source、relocated
shared、relocated static consumer 均运行通过。中央 component contract 证明真实 direct edges 与 selected
headers；timer_queue 本身是 INTERFACE factor，static/shared journey 的 artifact `NOT_APPLICABLE`，但
完整 Time 产品在三种形态均交付其头和 host Time binary。

### 7. C++ API 易用性与文档

`PASS`。[`user/time`](../../../user/time.md) 按 mixed deadline、same-duration、total-order 和 executor
任务选型，并给出 explicit-now 批处理与 ManualClock recipe；stable address、single owner、provider not
stored、delivery drain、queue-first teardown、wrong-owner fail-fast 和 conservative wake 均就近说明。
公共面没有 benchmark-only batch、`fast/light` 分叉、额外索引 metadata 或 internal factor 名泄漏。

## 独立红队终审

`PASS`。本次只攻击 caller-owned timer factor 的既定合同：

- wrong-owner、entry-first destruction、linked-delivery rearm/destroy、倒退 advance、full-range radix、
  cancel/reschedule/cancel-all 与 queue-first teardown 已由三平台 focused suite、death cases 和两组
  million-operation oracle 覆盖；产品头/算法从候选到 reviewed commit 为 zero diff；
- callback 仍只是借用的 hand-off metadata，queue 零 invoke、零 waiter、零 owning callable、零 lock；
  periodic、cross-thread command 与 cancellation propagation 没有偷进低层 factor；
- minimum cancel 的一次 conservative early wake 是换取 O(1) cancel 的明示成本，不会漏发；
  CommonTimeoutQueue 没有真实性能 consumer，故没有把复杂度证据冒充 SOTA 排名；
- P1 的八个 execution/placement/allocation strata 全部在 5% margin 外胜出；公开 API、node layout 和
  metadata 没有为 benchmark 分叉或膨胀；
- TimerEntry 直接复用 Container-intrusive 的 safe list，deadline/provider 直接复用 Time，compiler 与
  fail-fast 直接复用 Core；精确 factor 与安装包没有泄漏内部产品名或制造 header-only DLL。

剩余 medium-or-higher finding、未处置 `MUST`、security/correctness/ownership/delivery blocker 和
material consumer regression 均为 0。

## 当前快照验证

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| exact `timer_queue` factor | 29/29 tests | 28/28 | 28/28 | `PASS` |
| Time product source/shared/static relocated | 1/1 each | 1/1 each | 1/1 each | `PASS` |
| current component graph | full component contract `PASS` | selected closure | selected closure | `PASS` |
| retained P1 bundle | verifier `QUALIFIED/EXACT/CURRENT` | portability only | portability only | `PASS` |

候选后的 WSL ASan/UBSan/TSan 与三平台 million-operation oracle 保持有效，因为三个 product headers、
算法、Core/Time/Container-intrusive 直接合同均无 material drift；本次没有把未重跑项目写成新的
cross-platform timing 或 Formal certificate。

## 冻结裁决与 LOC

没有发现新的准入能力缺口、错误复杂度、material 性能缺陷、benchmark-shaped API、内部复用缺口、
易用性/文档错误、旧 C ABI 或 build/package integration 缺陷。`timer_queue` 保持 `CLOSED` 并附加
`FROZEN` 标记；只有 registry 既有 reopen trigger 的新事实才能解除。

`python scripts/loc-report.py ... --commit
54ddb99fe43e4e45664f0150c4f1d84c95166b8e --self-check --json`：

| Metric | code |
|---|---:|
| 全库代码 | 439,646 |
| 自实现不含测试 | 70,651 |
| `timer_queue` factor 实现 | 438 |
