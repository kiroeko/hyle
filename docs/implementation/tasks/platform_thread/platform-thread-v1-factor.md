# Platform thread v1：native thread、current state 与 address wait 因子

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

## Closure card（`SCOPED`）

- component / owner / round / kind：`platform_thread` / Platform /
  `platform-thread-v1-factor` / `NEW`；baseline
  `f0662cdee82dd72d5813d17a4765f0fea0f16d87`；进入原因是已接受的 `B-F08 SPLIT_PRIME` 与
  `PLAT-A-02/A-17` 缺口；
- 真实消费者：Sched 长寿命 worker 的 create/configure/park/unpark，Compute 的 future backend poller；
  北极星是 llama.cpp/ggml 的 worker affinity save/apply/restore、SVE/RVV width 与 AMX/SME kernel thread
  enable；
- 范围：move-only native Thread、name/stack/affinity/priority、caller affinity 查询和显式恢复、
  current-thread scalable-vector/matrix state、32-bit address wait/wake 与 single-waiter Parker；
- 语义边界：Platform 提供事实与 mechanism，SIMD 唯一选择 kernel route，Sched 拥有 pool/task/fiber/
  cancellation/placement policy；current-thread state 不变成第二套 dispatcher；
- 预期依赖：`PUBLIC core,error,time`、`PRIVATE platform`；普通用户仍只选择 `hyle::platform`，factor 名仅供内部
  target、测试与维护；
- tier / workload：`P1`，沿用 persistent Scheduler worker 的 Parker request/response handoff 对
  oneTBB production binary semaphore，W1-WIN 5% non-inferiority；thread create/configure 和 ISA enable
  属控制面，只做 P0 cost/行为验证，不与 park 点值混榜；
- 常态成本：raw entry 不分配；owning callable 仅显式路线一次冷分配；best-effort 默认不握手，required
  才同步；affinity snapshot O(available mask words) 且只显式分配 owner convenience；wait 直达一个 OS
  primitive，Parker 只含一个 atomic word；
- peers：oneTBB `843162db8cf4b2531a4a6147b6ea3fd860d44668` 的 binary semaphore/spin wait；最新
  llama.cpp/ggml 的 threadpool/CPU traits 是北极星而非可直接冒充同合同 timing peer；
- correctness：normal、zero/past timeout、invalid/sparse affinity、unsupported/permission、required entry
  exclusion、move/join/destroy、wake-before-wait/spurious/race/stress、affinity restore、真 ISA 可用性、
  ASan/UBSan/TSan；OOM 只适用于 owning convenience；
- delivery：exact factor、Platform product 三交付、三平台、Sched direct consumer、header isolation、
  export/import 与旧 FFI 反证；
- 总预算与仪器共享 Platform v3 六小时/九十分钟上限；最多一次性能机制诊断和两次产品修正；
- 非目标与 registry 一致；API、复杂度和 `FIX` 必须在 `AUDITED` 冻结后才可实现。

当前状态仅为 `SCOPED`，不授权移动 `thread.cc/wait.cc` 或新增产品 API。

## Audit verdict（`AUDITED`）

正向证据是 llama.cpp 的 affinity save/pin/restore 与 ggml 的 SVE/RVV width、Linux AMX permission；
反向追踪覆盖 `thread.h`、`wait.h`、两份 source、Platform/Sched/Compute 消费与 tests。现有 native
Thread、required/best-effort config、address wait 与单 waiter Parker 均 `COMPLETE/KEEP`；缺口只有：

| FIX | frozen shape | cost / boundary | verification |
|---|---|---|---|
| `PLAT3-T01` | `this_thread` affinity count/fill + owning snapshot，可原样传给现有 set | O(mask words)，Mac 诚实 `Unimplemented`；不建 scheduler placement policy | save/set/restore、sparse/large mask、failure preserving original |
| `PLAT3-T02` | current-thread scalable/streaming vector byte facts与显式 AMX tile permission request | O(1) native query；SMSTART/kernel dispatch 仍属 SIMD | synthetic decode + available real ISA/QEMU + unsupported paths |
| `PLAT3-T03` | `platform_thread` 精确拥有 Thread/wait sources/headers，public-link time、private-link host facts | Parker 保持一 word/一次 OS wait，不增加常态 metadata | exact factor、Sched consumer、三交付、旧 P1 replay |
| `PLAT3-D01` | thread guide/architecture/error traps 与六问 | 无 product cost | docs/scans/red team |

API freeze 为 `FROZEN`（2026-09-03）：公开头自含、affinity save/set/restore、current ISA state、Thread
与 Parker normal journeys，以及 Windows/WSL 精确 factor 已通过。此后 P1 不得反向增加 light/fast 档、
改变 Parker layout 或削弱 required/best-effort 语义。不新增 detach、process priority、线程池或 cancel
queue。产品路径只用 OS API/C++23，无第三方实现和 Native FFI；公开 namespace 仍为
`hyle::platform`，普通消费者选择 Platform product，`platform_thread` 只服务内部精确开发/交付。

## 候选前实现检查点

- T01--T03 与 D01 的产品实现、测试、target/header ownership 和 user/architecture 文档已经落地；
- W1-WIN 为 `42 PASS + 1` 个需要第二 processor group 的硬件跳过；GCC 14 `-Werror` 为 `44/44`；公开头
  自含通过；
- 在该检查点 M1-MAC、sanitizer、Platform product 三交付、Sched direct consumer、P1 与红队尚未
  裁决；后续进度以紧接的 Candidate 节为准。

## Candidate（`CANDIDATE`）

product commit/tree 是 `e9b9a43d4782c60d84badc02b1a41fbe21207a00` /
`c6429316279aaac1e0f5d7a7a71872a60cf967b6`。T01--T03/D01、公开 API/layout 和 Parker normal workload
均已冻结；W1-WIN、GCC 14 `-Werror`、ASan/UBSan、公开头自含、Windows exact factor 及 Platform 三交付
已通过。M1-MAC、Sched direct consumer、正式 P1 与最终红队仍待签署。

首个共同 candidate 因 Platform host 的 Apple `getentropy` 声明缺失退回 `AUDITED`；本 factor 没有
产品缺陷或授权改动，修复 host 后只重放同一 tree 的验证。

## 第二候选（`CANDIDATE`）

共同 product commit/tree 是 `b1a526d6fdb05b6bb42c4cb041788959e74511de` /
`f808df6fcbd0b5549777e0c6939240a3feab9512`。M1-MAC 的公开头与实现测试已随五组件共同重放通过；
T01--T03/D01、API、layout 与 Parker workload 未改变。sanitizer、Sched direct consumer、正式 P1 与
最终红队仍待裁决。

## 最终验证与六问（`VERIFIED`）

本 factor 与 Platform v3 共用 immutable product candidate
`b1a526d6fdb05b6bb42c4cb041788959e74511de` / tree
`f808df6fcbd0b5549777e0c6939240a3feab9512`，完整矩阵、红队与前态 snapshot 见
[Platform v3 主记录](../platform/platform-v3-capability-factorization.md)。W1-WIN 实现/公开头、W1-WSL
Release/ASan/UBSan/TSan、M1-MAC Release/ASan、Platform 三交付、Sched 2/2 direct consumer、动态符号/
旧 FFI 扫描均 `PASS`。WSL TSan 按已记录的 `setarch x86_64 -R` 启动合同执行。

[Platform P1 报告](../../evidence/performance/benchmarks-platform.md)中的 Parker 行 8/8 strata 均为
`WIN` 且 5% NI `PASS`，median advantage ratio 20.189201--25.202667，最低 confidence lower
17.180837；该结论只属于 persistent cross-physical-core single-waiter mailbox，不外推到 counting
semaphore、multi-waiter fairness 或 same-logical-CPU ping-pong。Thread create/affinity/state enable 属
控制面，结构门确认没有 per-operation refcount、hidden scheduler policy、轮询或每次 wait 分配。

Owner 六问：产品只用 C++23/compiler/OS ABI，无第三方实现；native lifecycle、affinity save/restore、
current-thread scalable/matrix state 与 address wait/Parker 均完整，pool/task/cancel/placement 路由到 Sched；
公开 Native FFI/C ABI 为 `N/A/CLEAN`；内部 factor 可精确构建但普通用户只选择 `hyle::platform`；自然
C++ API 与单 waiter、外部同步、显式 restore 等陷阱已写入 guide。T01--T03/D01 全部 `COMPLETE`，七项
aggregate gate 全为 `PASS`，无 blocker；当前进入 `VERIFIED`，不在本节提前写 terminal verdict。

## Closure decision 与 terminal receipt

本 factor **`CLOSED`**：T01--T03/D01、七项 aggregate gate 与所有适用验证均通过。已知限制是
single-waiter Parker、调用者负责 affinity/state restore 和外部生命周期协调，且 P1 只覆盖 W1-WIN
persistent cross-physical-core mailbox；thread pool、task/cancel/placement 与多 waiter policy 仍由 Sched
拥有，没有 Native FFI 或稳定 C++ ABI 承诺。

只有真实 consumer 的 thread/wait/state correctness 或易用性缺陷、支持平台/编译器击穿机制、公开
API/layout/dependency/delivery 改变、material 正常负载回归或 formally admitted 的共同 thread mechanism
可以重开；API 数、单点 benchmark 或命名偏好不能。

immutable candidate 的四项 LOC 为 `434,612 / 110,971 / 70,261 / 833`，由
`loc-report.py --module platform_thread --commit b1a526d6fdb05b6bb42c4cb041788959e74511de --self-check`
重算通过。`833` 是 factor-owned `src/platform_thread`；共享公开头统一计入 Platform conventional
boundary，不能把两个数相加冒充去重产品总量。前态 snapshot/hash 与 Platform 主 receipt 相同；
registry 已绑定 terminal receipt `865413af2255cc58061943b556efe7d98d39a7ac` 与首个实际保存
`CLOSED` state/candidate 的 anchor `9c18057bc9e895558594f2b7f61cfeaafefc6548`；anchor 校准前态与
SHA-256 见 Platform 主 receipt，本次 evidence-only 校准不改变产品或 verdict。
