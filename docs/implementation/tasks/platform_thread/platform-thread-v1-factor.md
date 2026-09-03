# Platform thread v1：native thread、current state 与 address wait 因子

<!-- MODULE CLOSURE RECORD: ACTIVE -->

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
