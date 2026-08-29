# platform-v2 推理映射驻留定向重开

<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 历史 terminal record：`platform-v2` 已 `CLOSED`。`platform-v1` 仍是未受影响能力族的冻结证据；
> 本轮只处理 HYLE 北极星本地推理引擎暴露的 file-mapping residency 缺口，并核验直接相邻的 CPU
> topology / NUMA placement。当前权威机械状态见 [`module-status.json`](../../module-status.json)。

## Reopen trigger

- trigger ID：`PLATFORM-INF-RESIDENCY-01`。
- 新事实：HYLE 已明确把本地量化推理引擎作为产品方向，而固定版本 llama.cpp
  `7584430716ee229751771ed0d6bbcb780d105eeb` 的正常模型加载路径会随着权重加载进度增量锁定新前缀，
  并在 POSIX 上释放未使用映射片段；Folly
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 还为大映射提供显式 eager / on-fault 锁定语义。
- 复现证据：[`Core--Obs llama.cpp 真实消费者扫描`](../project/core-obs-llama-consumer-scan-2026-08.md)
  已固定 `src/llama-mmap.cpp` 哈希
  `0673e5273594690a1a7b81010b18f158f5b153f502fed161a595c941e8e5808a`。当前
  `MappedFile` 只有 whole-map eager `lock()` 与一个 `bool`，无法表达“模型已加载到 N 字节时只新增锁定
  delta，失败仍保留既有驻留前缀”，因此先前 `PLATFORM-WATCH-04` 已被真实、已准入消费者击穿。
- 受影响范围：`MappedFile` 驻留控制的公共合同、实现、测试、交付与用户/架构/性能说明；topology 和
  NUMA 只核验当前公开能力能否自然承载 worker/arena placement，不因一个 peer 名字扩展 API。

## Closure card（进入 `SCOPED` 时冻结）

- component / owner / round ID：`platform` / platform owner / `platform-v2`
- task kind：`REOPEN`
- baseline commit / tree：`4053c9fd6c4961df98c96ad3c6bdc45e39f2f108` /
  `15595f8e48d14c3944cdab8a89c1262396daccd4`
- 进入原因与 demand/defect ID：`PLATFORM-INF-RESIDENCY-01`；不重新解释 v1 的其他能力族。
- 真实 consumer task / normal workload：本地量化模型 loader 建立一个长期只读 mapping，顺序解析并按
  已完成权重前缀逐步购买 residency；锁定失败可降级，但不得丢失此前成功前缀。CPU worker 根据当前
  process-available CPU/NUMA facts 选择 worker 与 host arena placement。
- 范围：
  - 审计并补齐 model-independent 的 progressive mapped-file residency 形状，包括 eager/on-fault 的
    明确平台语义、失败后的 applied state、move/reset/unlock 生命周期和零长度/边界行为；
  - 判断现有 `Advice::kDontNeed`、subrange mapping 与平台能力是否足以处理 unused pages；只有同一
    owner、跨平台可解释且消费者确需的缺口才进入本轮 `FIX`；
  - 对 `CpuInfo/CpuTopology/CpuAffinity` 和 NUMA topology/bind/selected-interleave 做定向能力 trace；
  - 更新受影响的 architecture/user/performance/task 文档，验证 platform source/static/shared C++ 交付。
- 非目标：重做整个 Platform；模型/tensor/token 专用 API；async/direct/scatter I/O；device/pinned memory；
  scheduler/stealing 策略；AMX/SME kernel enablement；JIT/shared-memory/process/plugin policy；稳定 C++ ABI；
  Native FFI/C API；为重叠任意 range lock 引入常态区间树/引用计数；Formal release certification。
- 直接依赖 / 直接消费者：现有 `PUBLIC core error time` 不变；直接消费者为未来 model loader、`mem`
  的 VM/NUMA backing，以及 `sched/compute` 的 placement seam。本轮不改半成品 compute/graph。
- tier 与理由：`P1`。progressive residency 是模型 load-to-ready / first-token 尾延迟和常驻资源控制路径；
  wrapper 本身必须保持 native-call 级开销，但不把 page-cache 噪声池化成“Platform 总分”。
- material margin / resource budget：W1-WIN source-primary 同合同 replay 使用 5% non-inferiority margin；
  已锁前缀查询 O(1)，单次增长做 O(1) 验证、零 heap/lock、至多一个 native residency call；失败保留旧前缀；
  normal access 无 HYLE dispatch/metadata；对象最多新增一个尺寸和一个小枚举，不按页存 metadata。
- 总调查预算 / 性能预算：总计 240 分钟；peer/consumer 审计 60 分钟，产品实现与正确性 90 分钟，
  交付/文档/红队 45 分钟，性能取证最多 45 分钟。
- 最多复现、机制诊断、产品修改次数：一轮 baseline probe、一次双顺序正式 replay；出现 material
  异常时最多一次同合同复现、一次 native-call/codegen 诊断和一次有机制依据的产品修改。
- machine profile：`W1-WIN` 为 P1 裁决机；`W1-WSL` 与 `M1-MAC` 验证真实 POSIX backend、sanitizer
  及 source/static/shared。没有多节点登记机器，不签物理 local/remote placement 或带宽。
- instrument profile：`closure-perf-v1`；correctness/delivery 使用 `closure-correctness-v1`。
- strong peers：
  - residency semantic：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`
    `MemoryMapping::{mlock,LockMode,LockFlags}`；
  - consumer shape：llama.cpp `7584430716ee229751771ed0d6bbcb780d105eeb`
    `llama_mlock::grow_to` 与 model-loader mapping lifecycle；
  - topology/NUMA 只以 llama.cpp 同 commit 的 thread/NUMA consumption 和 platform-v1 已钉死的 oneTBB
    `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 交叉核验，不新增第三个 peer。
- 无同合同 peer 时的预算及禁止声明：partial unmap、prefault/collapse、Windows working-set policy 若没有
  可移植同合同 route，只给 capability/negative-space verdict；不得声明全平台 model-load SOTA、物理驻留
  保证、所有 OS 相同锁定语义或多节点 NUMA 性能。
- correctness gates：normal；zero/end/unaligned mapping offset；prefix shrink/idempotence；over-size；quota/
  permission/unsupported；partial-growth failure；move/reset/destructor/unlock；read-only/RW/COW；真 Windows/
  Linux/macOS backend；public-header self-containment；ASan/UBSan（TSan 仅在共享可变状态存在时适用）。
- delivery gates：中央 component edge 不变；platform-only source consumer；relocated static/shared consumer；
  Windows import/export 与 Unix loader；安装头不出现 C schema/bridge/Native FFI。
- performance gates：冻结一个 `model-loader-progressive-residency` family。对同一映射和增长序列比较 HYLE
  source-primary 与等价 native consumer，计费包含参数检查、状态更新和 native call，双 order/A-A/
  known-delta 后按 5% NI；页触碰、quota 与 OS 回收作为分层资源结果，不拿不同语义点值裁决 API。
- 预期输出：受影响 capability map/public trace、有限 ledger 与冻结 `FIX`；immutable candidate；三平台
  correctness、三种交付、usability/docs、red-team、P1 verdict；四项 LOC 与 terminal receipt。
- 停止条件 / 可接受限制：全部已准入 `MUST` 为 `COMPLETE/ROUTED` 且适用门通过则 `CLOSED`；同合同 P1、
  OS 语义或交付无法在预算内裁决则点名 blocker 进入 `BLOCKED`。Windows/macOS 不支持 on-fault、没有
  多节点真机证书、partial unmap 不可移植均可作为明示限制，不用伪实现换绿色。

## Capability-family coverage map（`AUDITED`）

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|---|
| mapped-file residency | loader maps a multi-GB weight file, then advances the CPU-resident prefix as tensor ownership becomes known | whole/prefix; eager and Linux on-fault; preserve or explicitly permit process-limit adjustment | monotonic grow, exact applied prefix, failure preserves old prefix, one final unlock/reset, move-only and externally synchronized | whole eager `MappedFile::lock()` | `PARTIAL` | no prefix target/applied size/mode; default Windows working set rejects a normal model-sized request |
| mapped-file unused-range release | after offload, stop paying physical residency for known-unused ranges | best-effort page release; virtual-range destruction only when owner remains coherent | range boundaries, outstanding views, destructor, Windows/POSIX difference | range `Advice::kDontNeed`, exact subrange mapping, whole reset | `ROUTED` | POSIX partial `munmap` would punch holes into the public contiguous `bytes()` invariant and is a no-op in llama on Windows; no public partial-unmap `FIX` |
| mapped-file population / page collapse | explicit warm-up before first inference | will-need hint; synchronous touch remains consumer work; THP collapse is optional Linux policy | faults/RSS accounted separately from wrapper cost | `prefetch` / `Advice::kWillNeed`; caller access | `COMPLETE` for admitted warm-up | `MADV_POPULATE_READ/COLLAPSE` has no current same-contract material evidence and does not enter `FIX` |
| CPU placement facts | size workers from process-available CPUs, accept user masks, prefer known P cores without guessing | live available ids/info/order; single/set affinity; system count only for diagnostics | cpuset may change, ids sparse, macOS affinity soft/unavailable | `available_cpu_*`, `CpuInfo`, `CpuAffinity`, `ThreadOpts` | `COMPLETE` | llama consumes masks/NUMA groups, not physical-core/cache-sharing identities; exact domain topology remains scheduler-triggered negative space |
| NUMA placement | group available CPUs by node, bind worker sets, select local/preferred/interleaved host arenas | actual node ids, current/cpu node, preferred/bind/all/selected weighted interleave | fallible facts, single-node floor, whole-owner release, placement is not residency proof | `numa_nodes`, `numa_node_of_cpu`, `vm_bind_numa`, `vm_reserve_numa`, selected interleave | `COMPLETE` | current W1 cannot certify multi-node bandwidth, but the API route needed by ggml-shaped placement exists without a new type |

## Capability / gap ledger

| ID | requirement class | demand kind | exact consumer / peer / standard source | normal shape | semantic / artifact / delivery owner | ownership/failure/concurrency | complexity/resource target | coverage state | decision | severity | verification / negative space |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `PLAT-V2-01` | `MUST` | consumer | llama `llama_mlock::grow_to` + HYLE inference loader | grow locked prefix to loaded byte N | Platform `MappedFile` | monotonic; failure leaves prior prefix; move/reset transfer/release | O(1), zero heap/lock, one native call only when page coverage grows | `MISSING` | `FIX` | high | unit boundary/move/quota tests + native same-contract replay |
| `PLAT-V2-02` | `MUST` | mature peer + large-model resource shape | Folly `LockFlags::lockOnFault`, Linux `mlock2(MLOCK_ONFAULT)` | choose eager or lock-on-first-fault once, then grow in same mode | Platform; running kernel owns support | unsupported is `Unimplemented`; mode cannot silently change while a prefix is locked | same O(1)/one syscall; no eager page-in for on-fault | `MISSING` | `FIX` | medium | Linux true backend; Windows/macOS explicit unsupported |
| `PLAT-V2-03` | `MUST` | reproduced consumer defect | W1-WIN `VirtualLock`: 4 KiB pass, 1 MiB fails code 1453; llama adjusts working set | default preserves process policy; explicit option may raise soft/working-set limit after quota failure and retry once | Platform mechanism, application selects policy | global limit change is opt-in and not rolled back; hard limit/privilege failure remains Status | success fast path unchanged; failure path O(1), at most query+set+one retry | `MISSING` | `FIX` | high | Windows >1 MiB integration; POSIX soft-limit branch/unsupported boundary |
| `PLAT-V2-04` | `GENERAL` | consumer | llama POSIX `unmap_fragment` | release unused prefix/suffix or holes after offload | model loader owns tensor/view partition; Platform owns mapping/advice | public `bytes()` cannot remain a valid contiguous view after punched holes | arbitrary holes require O(fragments) metadata and per-fragment teardown | `ROUTED` | `REJECT` | medium | use exact mapping, `kDontNeed`, or separate owners; reopen only with coherent portable owner contract |
| `PLAT-V2-05` | `GENERAL` | peer | Folly `tryCollapseToTHP` | optional synchronous populate/collapse | Mem/Platform boundary depends on future backing journey | Linux version/THP policy; best effort | potentially O(mapped pages), not wrapper hot path | `COMPLETE` for current warm-up / extra variant absent | `DEFER` | low | `prefetch` + caller touch; require material load/RSS evidence before API |
| `PLAT-V2-06` | `MUST` | consumer trace | ggml NUMA init/thread affinity and model host allocation | available CPUs → node grouping → worker mask + arena policy | Platform facts/mechanism; Sched/Mem own policy | sparse ids, fallible queries, single-node floor | O(CPUs) initialization; no per-token Platform dispatch | `COMPLETE` | `KEEP` | high | existing cpu/thread/numa tests and source/static/shared route; no product edit |
| `PLAT-V2-07` | `GENERAL` | possible future optimizer | physical core/cache-sharing topology | avoid SMT/cache contention with measured engine policy | Platform facts only after a named consumer contract | OS portability and container restrictions | initialization O(OS records) | `MISSING` | `DEFER` | low | llama path reviewed here does not consume it; scheduler/engine trigger required |

### Public-surface trace

| public surface | reverse route | audit verdict |
|---|---|---|
| `MappedFile::{open,map,bytes,writable_bytes,flush,advise,prefetch}` | persistent contiguous mapping + warm-up/release hint | `KEEP`; already covers exact-range mapping and composable unused-page hint |
| `MappedFile::{lock,unlock,locked}` | whole eager residency | `FIX`; preserve source spelling while generalizing to prefix/mode/options and truthful applied size |
| `CpuFeatures/Isa/CacheInfo/CpuInfo/CpuTopology` + available CPU queries | kernel dispatch and worker facts | `KEEP`; cache capacity/system counts are not exact placement domains |
| `CpuAffinity/ThreadOpts/Thread` | application/scheduler-selected worker mask | `KEEP`; policy remains above Platform |
| `NumaCapability`, node facts, bind/reserve/interleave APIs | host backing placement mechanism | `KEEP`; HYLE already exceeds the fixed-size/dense Linux-only ggml representation |

- 现有公开入口反向追踪：受影响 `MappedFile` 入口全部映射到 residency lifecycle；CPU/Thread/NUMA 入口
  已映射到真实 inference placement journey，没有 benchmark-only 类型。
- 已有能力完整性检查：whole lock 缺正常 incremental update 与 applied-state query，是实际 lifecycle 缺边；
  arbitrary range lock 会带来重叠 ownership/reference-count 语义，不是 prefix consumer 的自然泛化。
- 语法消费者 / 语义重复扫描：HYLE 内当前没有 `MappedFile::lock` 产品消费者；这是 future inference
  loader 的基础机制，不修改 compute/graph。Mem 正确复用 Platform VM/NUMA；没有第二套 node binder。
- public/internal/higher-layer 裁决：mapping residency mechanism 属 Platform；是否 mlock、锁多大、tensor
  offload partition 属 model loader；working-set/rlimit 变化必须由调用方显式选择；direct I/O 属 IO。
- 本轮冻结 `FIX` 集：
  1. 新增 model-independent `LockMode`、`LockLimitPolicy`、`LockOptions`，以及 monotonic
     `lock_prefix(target, options)` / `locked_size()`；无参 `lock()` 源调用继续自然可用；
  2. 以 OS page 为粒度只锁新增 native delta；Linux 支持 `MLOCK_ONFAULT`，unsupported backend 明确失败；
  3. 默认不动全局限额；显式 adjustment policy 在 quota failure 后至多调整一次 soft/working-set limit 并
     重试一次；所有失败保留既有 applied prefix；
  4. 补齐正确性、source/static/shared、guide/architecture 与受影响 P1 replay。
- 相邻模块 backlog：partial/direct/async read 继续归 IO；pinned/device allocation 归 Compute；physical-core/
  cache-domain placement 在真实 Sched/engine policy 出现后重开；不修改这些模块。

## Owner 六问直答（审计阶段）

1. 第三方产品路径：产品实现只使用标准库与 Windows/POSIX OS API；Folly/llama/oneTBB 都是 pinned peer/
   consumer evidence，不进入 HYLE artifact。Platform 的现有 `PUBLIC core error time` 依赖不变。
2. 推理引擎能力完整度：本轮冻结的三项 residency `MUST` 修复前为 `NO`；CPU/NUMA placement route 为
   `COMPLETE`。partial unmap、THP collapse 和 exact core domain 均已给 owner/trigger，不伪装成缺失基础 API。
3. 拟真、公平性能：`P1`，只比较 source-primary progressive residency 与同义 native path；不拿 whole
   eager、on-fault、prefetch 或不同 page state 混成一个排名。
4. 旧 C ABI / Native FFI：本轮没有准入 slice；public C headers/bridge/options/exports 必须保持为零。
5. Source/static/shared：三条 C++ journey 都适用；`MappedFile` 是导出 owner，candidate 后复验符号与 relocation。
6. C++ API 易用性：默认无参 whole eager 行为保留；高级成本通过 options 显式；unsupported、quota、全局
   limit side effect 和外部同步要求进入 guide。

## Candidate

- product commit / tree：`88085b671ac7dfb4d41e2819ceda0925d1ac0147` /
  `01f2183f4cb5c029293ba6c52b15dfbc40d27e13`。
- 公开合同与 user journey 变化：`MappedFile` 新增 `LockMode`、`LockLimitPolicy`、`LockOptions`、
  `lock_prefix` 与 `locked_size`；无参 `lock()` 仍是自然 whole/eager/preserve-limit 入口。
- ownership / lifetime / concurrency / failure 变化：mapping 继续独占 residency lifecycle；prefix 单调增长，
  mode 在 unlock 前固定；control 操作外部同步；quota/unsupported/limit-adjust failure 返回 `Status`，旧前缀
  不丢失；显式 limit adjustment 的进程级副作用不回滚。
- 算法 / 复杂度 / 常态 metadata 与 layout 变化：`bool locked_` 替换为一个 native-prefix size 和一个小
  mode enum；普通 bytes access 无变化。未跨新 page 的 grow 无 syscall，增长/解锁一个 native call；
  opt-in quota failure 路最多 query + set + retry。没有 per-page/interval metadata 或 heap/lock。
- 直接消费者迁移：当前 HYLE 产品源没有 `MappedFile::lock` 调用方；future model loader 可直接消费新形状，
  compute/graph 不在本轮修改。
- candidate tests：W1-WIN MSVC 19.50、W1-WSL GCC 14.2、M1-MAC AppleClang 21 的 Release
  `hyle_platform_test` 与 public-header self-containment 均 PASS；最终测试还覆盖 prefix shrink、三种 mapping
  access mode、move/reset/unlock、unaligned offset、quota/unsupported/invalid option。
- harness / tool identity：产品 identity 只绑定上方 commit/tree；Windows build
  `C:\hyle-scratch\platform-v2\build-win`，WSL build `/root/hyle-platform-v2-build`，macOS build
  `/Users/Shared/HYLE/platform-v2-build` 是可重建验证 scratch，不进入产品身份。

## Verification manifest

candidate 到当前 `include/hyle/platform + src/platform` diff 为空。candidate 后只新增冻结性能合同/adapter、
package consumer 对新导出符号的真实调用、补充正确性测试和文档/registry 证据；没有改变 Platform 产品
API、layout、算法或正常运行时成本。

| Gate | Verdict | 证据与边界 |
| --- | --- | --- |
| scope / capability | `PASS` | 三项 residency `MUST` 全部 `COMPLETE`；CPU/NUMA 推理 placement trace 由现有动态 id/mask/node/bind/interleave 能力闭合；partial unmap 与 prefault/collapse 分别 `ROUTED/DEFER`，没有 `PARTIAL + KEEP` |
| architecture / API | `PASS` | 机制仍归 `MappedFile`，model/placement policy 留给 loader/sched；默认 whole eager/preserve-limit 自然，高级成本用 options 显式；一个 size + enum，无 range tree/per-page metadata；共享 `Status` 和既有 Platform OS abstraction 使用正确 |
| correctness | `PASS` | W1-WIN MSVC 19.50 Release 2/2、ASan 2/2；W1-WSL GCC 14.2 Release、ASan、UBSan 各 2/2；M1-MAC AppleClang 21 Release、ASan 各 2/2；三平台均含 public-header self-containment。TSan 对外部同步的 move-only control owner、无新增共享可变状态语义上 `NOT_APPLICABLE` |
| delivery | `PASS` | Windows、WSL、macOS 的 `HyleCiSelectedComponentSmoke` 均通过 source embedding、relocated shared SDK、relocated static SDK；commit `65ad806e` 后真实 package consumer 会调用 `lock_prefix` 并检查/释放 applied prefix；Windows DLL 确认导出 options `lock`、`lock_prefix`、`unlock` |
| usability / docs | `PASS` | architecture/user guide 覆盖 whole/prefix/on-fault 选型、页取整、失败保留、limit 副作用、外部同步与 partial-unmap route；watchlist 只保留尚未准入边界；无 C header/schema/bridge/Native FFI option 或 symbol |
| engineering performance | `PASS` | W1-WIN `platform-residency-p1-v2`、5% NI、n=15：八个 execution/allocation/placement strata 全为 `PARITY + NI PASS`；A/A、5/10% delta、timer floor、placement、source/peer/host/build/artifact identity 全 PASS，deep verifier `QUALIFIED` |
| red team | `PASS` | 下表 finding 均已修复或按 owner/trigger 限界，没有 correctness、security、delivery、usability 或 material-performance blocker |

性能 bundle 位于 `C:\hyle-scratch\platform-v2\perf-qualify-evidence`：product candidate 为
`88085b671ac7dfb4d41e2819ceda0925d1ac0147` / tree
`01f2183f4cb5c029293ba6c52b15dfbc40d27e13`，source/harness 为
`a36ab184fd079a0e0652c6d4b2fa8b1fbc51a935` / tree
`77f1dc7493656033fee991f582b94262927ab2e5`，llama peer 为
`7584430716ee229751771ed0d6bbcb780d105eeb` / tree
`027524b2c81f6df300250aadf21ed6512e13e141`。contract SHA-256
`45896aa4f462bca5de1ee3e4ea8171c45111c4ecaae10153e037639f073d8e1c`，manifest SHA-256
`4c019d408a287336868d99b26e529c13d800c330ce4c77cc54c43864357095bd`，summary SHA-256
`0d148336277104322dc2fc8d1bd14ee332fb48290eee3ae9fea4db75ae46a65a`；当前 verifier 为 exact
provenance、current policy、`paths_verified=true`、`QUALIFIED`。

八个 candidate/peer 中位数比为 0.986466--0.999315，最差 99.26% CI lower bound 为 0.964334，仍高于
5% NI 下界 0.952381。这个结论只说明同一 warm 16 MiB mapping 上“四次 4 MiB 增长 + 一次 unlock”的
HYLE source-primary wrapper 相对 pinned llama Win32 native mechanism 没有 material 损失；不宣称
`VirtualLock` 本身变快，也不外推 Linux on-fault、RSS/page-fault、quota adjustment 或端到端 tokens/s。

### Final red-team

| finding | severity | disposition / evidence |
| --- | --- | --- |
| 原 selected-component sample 只 map、不调用新驻留符号，可能给 shared/static 假绿 | material delivery false-pass | `FIXED` at `65ad806e`：真实 package consumer 调 `lock_prefix`、检查 `locked_size` 并在成功时 `unlock`；三平台三种交付重放全绿 |
| Windows ASan 被 `detect_leaks=1` 在 test main 前拒绝 | validation invocation | `DISPROVED` as product defect：MSVC ASan 不支持该 option；移除 unsupported leak flag 后同 binary 2/2 PASS，Linux ASan 仍启用 leak detection |
| 自动修改 memlock/working-set 会惊扰同进程其它组件 | high API trap | `FIXED BY CONTRACT`：默认 `kPreserve`；只有显式 `kAdjustIfNeeded` 才在 quota failure 后做一次 query/set/retry，成功副作用不回滚并进入 public comment/guide |
| 任意 range/partial unmap 看似更通用 | ownership/layout defect if admitted | `REJECTED`：重叠 range 需要 refcount/interval metadata；partial unmap 会破坏连续 `bytes()`，Windows peer 也无对等语义；精确 owner/subrange mapping/`kDontNeed` 已路由 |
| `kOnFault` 可能在不支持平台静默退成 eager | material latency-shape defect | `FIXED`：只在 Linux `mlock2(MLOCK_ONFAULT)` 成功时应用；Windows/macOS/旧 kernel 明确 `Unimplemented`，不偷换 latency/resource 语义 |
| peer adapter 没有编译整个 llama.cpp | claim-integrity risk | `ACCEPTED/BOUNDED`：固定 clean llama revision/source，人工核对 `grow_to` 的 page-rounded delta 与 Win32 raw call；adapter 对同一四个 disjoint range 计费，排除日志/model policy。结论只签该 native mechanism replay，不冒充 llama 端到端 |
| 当前机器均为单 NUMA node | physical-evidence gap | `ACCEPTED LIMITATION`：只签 API route、初始化复杂度、错误与 single-node floor；不签多 socket locality/bandwidth |

candidate 的 canonical LOC self-check（`python scripts/loc-report.py --module platform --commit
88085b671ac7dfb4d41e2819ceda0925d1ac0147 --self-check --json`）为：全库代码 **312,046**；
自实现包含测试 **125,130**；自实现不包含测试 **83,707**；platform 实现 **6,864** 行。

## Closure decision

- 本轮 `CLOSED`。三项 residency `MUST` 已 COMPLETE，CPU/NUMA inference route 已闭合，七项 aggregate
  gate 全部 `PASS`；没有 correctness、security、delivery、usability 或 P1 blocker。产品 candidate 保持
  `88085b671ac7dfb4d41e2819ceda0925d1ac0147` / tree
  `01f2183f4cb5c029293ba6c52b15dfbc40d27e13`，candidate 后没有 Platform 产品源码变化。
- terminal end / registry anchor：`1be5fb582b4fccb7ce5ddbd55de84b3cc01a682c`。该不可变快照保存
  `VERIFIED`、candidate identity、七项 gate、完整验证矩阵、性能 bundle identity、LOC 与 red-team 事实；
  terminal 提交只登记 `CLOSED` receipt、限制和 reopen triggers。
- raw local evidence：W1-WIN
  `C:\hyle-scratch\platform-v2\{build-win,asan-win,delivery-win-symbol,perf-qualify-evidence}`；W1-WSL
  `/root/{hyle-platform-v2-build,hyle-scratch/platform-v2-{asan-a36,ubsan-a36,delivery-symbol-65ad}}`；M1-MAC
  `/Users/Shared/HYLE/platform-v2-{build,asan-a36,delivery-symbol-65ad}`。这些是可重放本地 artifact，不冒充
  长期签名 Formal 证书；Formal 独立保持 `PENDING`。

- 可接受限制：P1 只覆盖冻结 W1-WIN progressive eager residency；Linux on-fault 只验语义/正确性；
  多节点 locality/bandwidth、partial unmap、同步 prefault/THP collapse、exact core/cache-domain placement、
  AMX/SME enable、direct/async I/O、JIT/shared-memory/process/plugin policy 仍按 owner/trigger deferred；
  Native FFI/C API 与稳定 C++ ABI 未准入。
- reopen triggers：
  1. 具名消费者复现 Platform 边界内的 residency/topology/NUMA correctness、security、ownership 或易用性缺陷；
  2. `MappedFile` 公开 API/layout/算法、受支持 OS primitive、直接依赖或 source/static/shared 交付变化；
  3. 同合同成熟 peer 给出更好复杂度档或正常 loader workload 上可复现超过 5% 的 material 劣势；
  4. multi-node、partial unmap/prefault、exact placement、AMX/SME、foreign runtime 等 deferred family 经
     独立能力决策正式准入；
  5. 性能 contract/peer/consumer provenance/machine/instrument 或 retained bundle 变化且当前声明需要续签。
