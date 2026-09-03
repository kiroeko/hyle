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
- 预期依赖：`PUBLIC core,error,time,platform`；普通用户仍只选择 `hyle::platform`，factor 名仅供内部
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
