# mem-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`AUDITED`。本文是 mem 当前轮唯一 closure card 与施工记录；既有 mem task、设计页、
> guide 和性能页只作审计输入，不继承旧 `SIGNED`、Native FFI 或 SOTA 叙事。权威机械状态见
> [`module-status.json`](../../module-status.json)。能力族、全部公开入口、真实消费者与绑定版本的成熟
> 源码已经双向核对；本轮只实现下方冻结的九项 `FIX`，不得边施工扩面。

## Closure card（`SCOPED`）

- component / owner / round ID：`mem` / mem owner / `mem-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`c04e12c7dd65ea5858314d9ebfbea841119225ae` /
  `472e0cd3838c0852d85a8e0e3e8d69e37348283a`
- 进入原因与 demand/defect ID：`MEM-CAP-01`（按基础库完整性标准重新枚举 host-memory 能力族）、
  `MEM-NORTHSTAR-01`（模型加载、prefill/decode 临时区与通用 KV-lifecycle backing）、
  `MEM-API-01`（C++23 owner/view、失败、默认值和可发现性）、`MEM-ADOPT-01`（下游是否绕过 mem，
  或错误使用 error/platform）、`MEM-PERF-01`（旧性能页没有绑定当前统一仪器和当前 candidate）、
  `MEM-DOC-01`（设计页明确为 `NOT_REVIEWED`，历史性能叙事仍待裁决）、`MEM-DELIVERY-01`
  （mimalloc 外部依赖及 source/static/shared 交付复签）。
- 真实 consumer task / normal workload：
  - 本地量化推理与 graph evaluation 的 phase-lifetime 临时区：多种尺寸和对齐的小对象成批取得，
    epoch 结束整批释放；普通 block 复用，大对象不会永久抬高 retained footprint；
  - scheduler 的 guarded fiber stack：长寿命 worker 在多个 fiber tier 上 acquire/run/drain/release，
    跨 worker 归还后在 owner-level shutdown/trim 回收；
  - sched/io/graph/container 的 runtime allocator 注入：typed、PMR 与显式 alignment 路线保持相同
    backing identity、OOM 与 sized-free 合同；
  - 异步 state/node 的固定槽 handoff：固定容量、耗尽可观察、分配线程与释放线程可不同；
    owner-local cache 是显式 soft visibility policy，不冒充 hard capacity；
  - 工程计算留出：调用方固定 buffer 上的 variable-size allocate/free/reallocate/reset，以及
    NUMA/大页上的大块 host workspace；不引入模型、token、KV 或 backend 专用类型。
- 范围：`include/hyle/mem/**`、`src/mem/**` 的 alignment、system/runtime allocator、标准 allocator/
  PMR adapter、quota/stats、owned/external/borrowed scratch、fixed/growable/shared/typed pool、fiber stack、
  大页与 NUMA owner；全部 public surface 反向追踪；直接依赖和代表性直接消费者；mem-only C++
  source/static/shared；当前 architecture/user/performance 文档；统一仪器所需的窄 consumer replay。
- 非目标：device/VRAM、pinned/registered host buffer、backend coherence 与 staging ring；KV page/cell、
  graph liveness/reserve、io registered-buffer 和 scheduler blocking policy；generic `MemorySpace`、自动
  NUMA placement policy、稳定 C++ ABI、Native FFI/C API、下游模块整体收口、Formal 发布证书，以及
  为 benchmark 点值加入常态 metadata、TLS registry、padding 或公共入口。
- 直接依赖 / 直接消费者：`PUBLIC core error platform util_core`；代表性直接消费者为 `container`、
  `util`、`sched`、`io`、`compute`、`graph`。本轮只修 mem 及 correctness/build 阻断所需的直接边，
  不替仍为 `QUEUED` 的消费者签署整体结论。
- tier 与理由：`P1`；allocator adapter、region bump、fixed/shared slot 与 fiber-stack reuse 都可能位于
  高频或高资源贡献路径。NUMA/大页、统计快照、collect/trim 和构造/销毁控制面按 P0 复杂度、
  syscall/allocation 与空间结构门审查，不能池化成一个“mem 总分”。
- material margin / resource budget：W1-WIN 同合同 consumer/replay 使用 5% material/non-inferiority
  margin；默认 allocator/typed adapter 不增加隐藏 allocation、lock 或周期线程；Region 热分配 O(1)，
  一个 epoch 对 retained blocks 总遍历 O(blocks)，无 O(blocks²) 重扫；FixedPool fit/free/coalesce O(1)，
  move realloc 仅复制 `O(min(old,new))`；固定槽 acquire/release expected O(1)，hard core 不把 free 隐藏
  在线程 cache；fiber reuse 不增加 guard/VM 语义差异。峰值 reserved、retained footprint、碎片、
  tail latency 与 hard capacity 分别守门，吞吐不能抵偿资源退化。
- 总调查预算 / 性能预算：能力/API/依赖/实现审计最多 300 分钟；统一性能取证最多 120 分钟，只允许
  一次 probe、一次双 order/适用 placement 正式采样，以及协议允许的一次 material 异常机制诊断、
  一个产品修改与一次重跑。`FIX` 集在 `AUDITED` 冻结；预计实现超过一个工作日则拆成具名 defect，
  当前轮进入 `BLOCKED` 或重新 scope，不默默扩面。
- 最多复现、机制诊断、产品修改次数：每个异常一次反序复现、一次 codegen/profile/counter 机制
  检查、最多一个同时拥有机制钥匙和消费钥匙的产品修改及重跑；不得修改中央统计语义或发明第五种
  benchmark 姿势。
- machine profile：`W1-WIN` 为 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 承担适用 correctness、
  sanitizer、OS/backend 与交付复核，不跨平台池化性能。当前没有登记的多节点 NUMA 真机，因此本轮
  只签 selected-node/chunk 的语义、rollback、复杂度与单节点 fallback，不签跨 socket 带宽。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用
  `closure-correctness-v1` 与当前 component/package 合同。
- strong peers（每项 capability 最多两个，均绑定当前源码）：
  - system allocator、remote free、size/pressure：HYLE 锁定的 mimalloc v3.5.0
    `18b08671c9302247bfb682286e6bf3cc1773f801`，以及 oneTBB
    `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 的 scalable/cache-aligned allocator/resource；
  - owned region：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的 Arena，与 C++23
    `std::pmr::monotonic_buffer_resource` 标准路线；
  - external variable-size / fixed and typed pools：oneTBB 同 commit 的 fixed_pool/memory_pool，
    与 marl `b8406ab0a82524170ead6dd6c371b579bbc28b3e` 的 BoundedPool/PoolPolicy/Loan；
  - NUMA/大页与 guarded stack：mimalloc 同锁定版本及 oneTBB 同 commit 的 NUMA/allocator 能力，
    marl 同 commit 的 fiber-stack lifecycle 只作形状输入。
  llama.cpp `7584430716ee229751771ed0d6bbcb780d105eeb` 与 ggml
  `d99724f2b141fed9e1a9f402213506c081433465` 的 host buffer、arena 与 workload 生命周期只用于北极星
  取证，不冒充已有 HYLE 生产消费者。
- 无同合同 peer 时的预算及禁止声明：Quota/Stats、显式 selected-node chunk、preserving object 与
  scheduler-owned guarded-stack cache 若不能对齐完整语义，只按预冻结的 allocation/syscall/copy、
  peak/retention、tail 和 correctness 预算裁决并标 `NOT_COMPARABLE`。不得声明“mem 全部操作 SOTA”、
  物理多节点 placement/bandwidth、device/unified-memory 管理或未测 workload 的性能。
- correctness gates：normal；null/empty/zero；size/count/alignment/address/offset overflow；hard capacity/
  fragmentation/realloc；OOM/metadata OOM/native VM/NUMA failure；foreign/misaligned/double return 与
  outstanding object；move/release/reset/clear/destructor/Loan 早退；remote free、cache publish、并发
  acquire/release 与 shutdown quiescence；property/differential/randomized fragmentation/stress；
  ASan/UBSan/TSan（按合同适用）；Windows/Linux/macOS 真 backend 与 public-header self-containment。
- delivery gates：中央 `mem PUBLIC core error platform util_core` 与 include-owner 对账；mimalloc 锁、
  vendored/license/offline 投影；mem-only source build；relocated static/shared C++23 SDK consumer；
  代表性 container/util/sched/io/compute/graph seam；产物/安装树无旧 C schema、bridge 或 Native FFI 符号。
- performance gates：冻结四个独立 family，不互相池化：
  1. `runtime-allocator-adapter`：正常 size/alignment 分布、allocate/touch/free 与 PMR/typed consumer，
     对同一 mimalloc backing/direct seam 和同合同 oneTBB route 做 A/A、known delta、floor、双 order及 5% NI；
  2. `phase-region-epoch`：graph/inference scratch 分布的 allocate/use/clear、long-lived block reuse、
     peak retained 与 large-allocation release，对 Folly/标准路线做相同语义与资源门；
  3. `fixed-slot-handoff`：hard-capacity local/remote producer-consumer handoff，同时报告 throughput、
     p95/p99 与容量可见性；不以 soft cache 结果替 hard core；
  4. `scheduler-fiber-stack-lifecycle`：相同 guarded stack 语义下的 scheduler batch create/run/drain/
     destroy，比较 pooled reuse 与 direct VM lifecycle，并独立守 tail/retention 预算。
  FixedPool fragmentation/realloc 作为资源与复杂度 replay；不能对齐 peer 时保持 `NOT_COMPARABLE`，
  不伪造同合同 SOTA。
- 预期输出：capability-family map、全部 public-surface trace、有限 FIX 集、owner/复杂度/资源表、
  correctness 与直接消费者证据、统一 P1 bundle、三种交付、更新后的 architecture/user/performance、
  独立红队与四项 LOC。
- 停止条件 / 可接受限制：所有 `MUST` 为 `COMPLETE` 或完整 `ROUTED`、四个 P1 family 的适用工程门
  均为 `PASS`、其余能力通过 P0 结构/资源门、无 correctness/delivery blocker 即停止；多节点 NUMA
  物理带宽、Formal、foreign runtime 与 device memory 可保持明确限制。任何 material `MUST`、仪器资格
  或 P1 NI 无法在冻结预算内裁决，进入具名 `BLOCKED`，不无限调查。

## Capability-family coverage map

| Family | admitted user journeys / evidence | required semantic variants | lifecycle / boundary / ownership dimensions | current route | state | missing edge / negative space |
|---|---|---|---|---|---|---|
| aligned system allocation | HYLE containers、sched/io/graph runtime injection、mimalloc/oneTBB | compile-time default、runtime erased、typed raw/object ownership、zero/OOM/alignment/sized free、realloc preserving failure | system process lifetime；borrowed erased state；remote free；默认 route 不加统计或锁 | `SystemAllocator`、`Allocator`、typed helpers、good/usable size、collect | `COMPLETE` | stable foreign ABI、secure heap 与 vendor option registry 不进入此 family |
| C++ allocator interoperability | standard/third-party containers、PMR、`allocate_shared`；C++23、oneTBB | default stateless 与 runtime-injected 分型；allocator rebind/equality；PMR zero-byte、throw boundary、minimum base alignment | adapter/resource 借 backing；容器/control block 先销毁；source/static/shared 行为一致 | `SystemStdAllocator<T>`、`StdAllocator<T>`、`system_memory_resource()`、`PmrResource`、`AlignedPmrResource` | `COMPLETE` | `allocate_at_least` 是可选优化；当前用 `system_good_size` 完整服务显式 consumer，待具名标准库收益再准入 |
| admission and observability | strict service quota、opt-in leak/debug、长寿推理进程默认 heap 压力诊断；mimalloc + Folly/jemalloc | requested live quota；per-wrapper counters/inventory；process-wide allocator reserved/committed 与 process RSS/commit current+peak | quota/stats wrapper 借 backing；metadata 分域；全局 snapshot 是冷观测，默认分配热路零成本 | `QuotaAllocator`、`StatsAllocator`、`LeakTracker`；尚无 system heap snapshot | `PARTIAL` | 缺默认 `SystemAllocator` 的稳定非 vendor 压力 snapshot；精确 requested-live 必须显式套 `StatsAllocator`，不为全局统计污染热路；不开放任意 mimalloc mallctl/options |
| phase and stack scratch | inference/graph/request phase、caller buffer、pooled block、LIFO checkpoint；Folly Arena/PMR monotonic | owned growable、external fixed、borrowed block、LIFO；clear/release；alignment、limit、large-block release | single owner；bulk invalidation；borrowed/owned lifetime 可见；OOM 返回 null | `Region`、`FixedRegion`、`PooledRegion`、`StackAllocator` | `COMPLETE` | destructor registration、hidden TLS arena/merge 需要 runtime owner，保持 routed/deferred |
| external variable-size storage | embedded/realtime/caller-owned workspace；oneTBB `fixed_pool` | different-size alloc/free/realloc、fragmentation、alignment、bulk reset、no fallback | caller owns backing；single owner；reset quiescent 且批量失效旧指针 | `FixedPool` two-level indexed fit/coalesce | `PARTIAL` | 实现完整，当前 guide 错把 bulk reset 写成必须先逐块 free；并发版本需具名 consumer 或 caller synchronization |
| hard shared slots and typed objects | async state/node handoff、fixed quota、owner-local reuse；marl bounded pool/loan、mimalloc remote free | hard shared core、显式 soft cache、Reconstruct/Preserve、manual/RAII return、local/remote free | core thread-safe；cache single-owner；pool 比 cache/loan/slot 长寿；reset/publish 边界 | `SharedPool`、`SharedPoolCache`、`ObjectPool*`、`PreservingObjectPool`、`PoolLoan` | `PARTIAL` | erased allocator 首次 oversize/over-alignment 会 abort 而不是容量失败；reset 与 cache publication 合同未写全 |
| growable pools | single-owner stable-address typed churn；marl unbounded pool；owned growable variable-size standard route | typed fixed-slot geometric growth；arbitrary-size individual-free+bulk-release；constructor exception/OOM | single owner；typed reconstruct；trivial bulk teardown、nontrivial must destroy；standard resource lifetime | `GrowablePool<T>`；arbitrary-size route 为 `std::pmr::*synchronized_pool_resource` over HYLE PMR | `PARTIAL` | testing build currently rejects trivial bulk teardown while shipping/header allow it；通用 variable heap 已完整 `ROUTED`，不复制 mimalloc/PMR |
| guarded execution stacks | scheduler fiber lifecycle；marl runtime stack shape + platform VM contract | guarded owner、size tiers、bounded reuse、cross-worker return、trim/shutdown | move-only stack；pool thread-safe；VM release outside lock；wrong tier drops | `FiberStack`、`FiberStackPool` | `COMPLETE` | stack switching/context policy belongs sched；cache cap is explicit bounded metadata |
| large host span placement | large inference workspace、selected/weighted NUMA stripe、THP/1GiB reserve；oneTBB NUMA + mimalloc | any/preferred/local/hard bind/all-node/selected-node/chunk；requested vs accepted/applied；rollback | move-only owner；platform owns OS mechanism；soft policy may degrade observably，hard policy must fail honestly | `HugeRegion` + platform `VmRegion`/NUMA；`reserve_huge_1g_pages` | `PARTIAL` | hard-bind with `kNumaAny` currently succeeds without binding；`allocate_local` swallows current-node failure |
| device and model memory policy | ggml/llama.cpp backend buffers、future KV/liveness/staging | host/device visibility、pinned/registered/coherence、completion/backpressure、graph/KV policy | backend/graph/runtime owns device handle and completion lifetime | compute `Buffer/DevicePool`、graph allocator、io registered buffers | `ROUTED` | 不下沉 generic `MemorySpace`、pinned allocator、KV cell/page 或 staging ring 到 host mem |

自底向上的公开面追踪覆盖 17 个 public header：`alignment.h`；`allocator.h`/`system_inline.h`；
`std_allocator.h`/`pmr.h`；`quota_allocator.h`/`stats.h`；`region.h`/`stack_allocator.h`；
`fixed_pool.h`；`shared_pool.h`/`pool_cache.h`；`object_pool.h`/`pool_loan.h`/`growable_pool.h`；
`fiber_stack.h`；`hugepage.h`。每个公开 class/function/enum 均落入上表；`detail::*`、测试 hook 与
backend seam 未被误算成用户能力。现有公开面没有 C schema、Native FFI 或仅为 benchmark 存在的类型。

## Capability / gap ledger

| ID | requirement / source | normal shape / owner | ownership/failure/concurrency | complexity/resource | state | action / severity | verification / negative space |
|---|---|---|---|---|---|---|---|
| `MEM-SYSTEM-01` | MUST / current consumers + mimalloc/oneTBB | aligned alloc/free/realloc、typed helpers、size class、current-thread collect；mem | process backend；remote free；null on OOM | expected O(1)；default typed route direct call；零隐藏 metadata | `COMPLETE` | `KEEP` | tiny/default/over-aligned、overflow、remote free、shared/static direct seam |
| `MEM-SYSTEM-STATS-01` | MUST / inference pressure journey + mimalloc/Folly | stable POD snapshot of allocator reserved/committed and process RSS/commit current+peak；mem | process-global cold observation；concurrent activity may make fields temporal，不能伪称 per-owner quota/requested-live | O(number of allocator heaps) cold at worst；default hot path zero指令/状态增量 | `MISSING` | `FIX` / BLOCKER | wrap pinned mimalloc stats/process query privately；invariants/relative allocation test；不导出 vendor layout/options/reset |
| `MEM-CPP-ADAPTER-01` | MUST / C++23 + current container users | default/runtime allocator-template、PMR、alignment、standard exception boundary；mem | adapter borrows state；complete equality；zero PMR pair | allocation one backend dispatch；system form empty/direct | `COMPLETE` | `KEEP` | header/API recipes、vector/allocate_shared/zero/overaligned、三交付 |
| `MEM-REGION-01` | MUST / graph/inference + Folly/standard | owned/external/borrowed/LIFO scratch；mem | single owner；bulk invalidation；OOM null；hard reserved limit | hot bump O(1)；epoch O(blocks)；large clear O(large blocks) | `COMPLETE` | `KEEP` | overflow/OOM/move/clear/release/randomized reuse；no hidden scan/fallback |
| `MEM-FIXED-01` | MUST / engineering user + oneTBB | external variable-size alloc/free/realloc/reset；mem | caller backing；single owner；quiescent reset invalidates all old pointers | fit/free/coalesce O(1)；move realloc copy O(min)；约 4 KiB control | `PARTIAL` | `FIX` / CONTRACT | align code comment/guide/test to genuine bulk reset；fragmentation oracle and two-candidate structure gate |
| `MEM-SHARED-01` | MUST / async handoff + marl/mimalloc | hard fixed slots + optional explicit owner cache + erased allocator；mem | core thread-safe/cache single-owner；capacity failure null；publish before core reset | expected O(1) common path；no cached free in hard core；cache hit zero atomic | `PARTIAL` | `FIX` / BLOCKER | oversize/overalign initial erased request returns null；realloc preserves live block；reset/cache pitfall tests/docs |
| `MEM-OBJECT-01` | MUST / marl + foundation | reconstruct/preserve + move-only scoped loan；mem | pool outlives object/loan；constructor throw returns slot；nonthrow destructor | expected O(1) slot ops；preserve O(capacity) cold construct/destruct | `COMPLETE` | `KEEP` | exception/early return/double/foreign/outstanding；blocking route sched |
| `MEM-GROW-01` | MUST / existing public + marl | owner-local stable typed slots with geometric growth；mem | trivial live storage may bulk-die；nontrivial objects require destroy；OOM null | O(1) amortized acquire，O(log n) backing allocations，O(blocks) teardown | `PARTIAL` | `FIX` / CONTRACT | remove testing-only contradiction；nontrivial death + trivial normal teardown + move/OOM |
| `MEM-GENERAL-POOL-01` | MUST / standard + oneTBB | owned growable arbitrary-size individual free + release；standard PMR | user selects sync/unsync；upstream outlives pool/resource | mature size-class pool；no HYLE duplicate metadata | `ROUTED` | `BORROW` | guide compile recipe over system and runtime upstream；no new HYLE heap type |
| `MEM-FIBER-01` | MUST / sched consumer | guarded tier owner + bounded cache/trim；mem/platform/sched | stack move-only；pool shared；quiescent owner shutdown | cache hit O(1)+short lock；miss/release syscall；bounded address retention | `COMPLETE` | `KEEP` | guard fault、tier mismatch、cap/trim、sched lifecycle；no unsafe context API |
| `MEM-NUMA-01` | MUST / inference workspace + oneTBB/mimalloc | THP、local/preferred/hard/interleave/selected/weighted chunk；platform semantic + mem owner | hard policy error；soft result observable；move/reset rollback | setup O(nodes) or O(size/chunk)；access zero HYLE dispatch | `PARTIAL` | `FIX` / BLOCKER | reject bind-without-node；propagate local-node query；single-node/invalid/selected tests；multi-node physical bandwidth remains unsigned |
| `MEM-DIAG-01` | MUST / current debugging journey | strict quota、atomic counters、optional complete/incomplete leak inventory；mem | wrapper borrows backing；metadata separate；quiescent reset/destruction | quota/stats O(1) atomics；leak map expected O(1)+mutex only opt-in | `COMPLETE` | `KEEP` | composition order、OOM rollback、dropped record、concurrent peak/lifetime |
| `MEM-RESET-01` | MUST / existing public lifecycle | bulk clear/reset/release vocabulary and cache invalidation rules；mem/docs | reset requires operation quiescence, not prior individual free；attached cache must publish/empty | cold O(1) fixed pool or O(shards/blocks) owner reset | `PARTIAL` | `FIX` / MATERIAL | correct FixedPool/SharedPool docs/comments and add explicit cache-reset pitfall |
| `MEM-LEGACY-01` | MUST / accepted delivery policy | pure C++ source/static/shared terminology；mem/build/docs | no stable ABI/foreign schema | zero extra surface | `PARTIAL` | `FIX` / CLEANUP | remove “ABI-clean/current C schema” remnants from active code/docs；artifact symbol scan |
| `MEM-ADOPT-01` | MUST / direct consumer audit | allocator in container/util/sched/io/graph/compute；fiber stack in sched；Region in graph/io/util；mem + consumers | preserve dependency direction and owner-specific OOM policy | no extra wrapper/translation | `COMPLETE` | `KEEP` | central `mem PUBLIC core error platform util_core` equals target edge；candidate seam compile/tests |
| `MEM-PERF-01` | MUST / current evidence invalidation | four frozen consumer families under `closure-perf-v1`；bench/perf owner | same source-primary contract、backing、lifetime、touch、capacity semantics | W1-WIN 5% NI；resource/tail separate；no all-mem aggregate | `MISSING` | `FIX` / BLOCKER | build qualified bundle；legacy point tables remain history until replaced |
| `MEM-DELIVERY-01` | MUST / build-package contract | mem-only source + relocated static/shared SDK + mimalloc lock/license/offline；build/package | C++23 only；vendor symbol private except static link implementation | exact selected closure | `COMPLETE` | `KEEP` / blocker-at-verification | component/header/package/manifest checks on candidate |
| `MEM-DOC-01` | MUST / user discoverability | architecture decisions、scenario guide/pitfalls、current performance evidence；docs | N/A | no duplicated fragile point claims | `PARTIAL` | `FIX` / BLOCKER | remove provisional/stale reset/C text；document snapshot, request-fit and NUMA failures |
| `MEM-THREAD-ARENA-01` | GENERAL / Folly ThreadCachedArena | worker-owned Region registry/merge；sched/runtime owner | thread exit/zombie lifetime and global teardown require runtime owner | default Region keeps zero TLS registry | `ROUTED` | `DEFER` | admit only with named runtime owner; do not hide TLS in mem |
| `MEM-EXT-SYNC-01` | GENERAL / oneTBB preview fixed_pool | synchronized external variable heap；caller or future wrapper | mutex/owner policy explicit | caller lock adds O(1) serialization | `ROUTED` | `DEFER` | current FixedPool + caller synchronization is natural；second source/consumer can trigger named wrapper |
| `MEM-DEVICE-01` | GENERAL / ggml/llama.cpp | device/pinned/coherent buffers and allocation planner；compute/graph | backend handle、mapping、completion lifetime | backend-specific | `OUT_OF_SCOPE` | `REJECT` | compute/graph closure owns it；host mem must not infer device policy |

审计结论与边界：

- 当前语法消费者遍及 container、util、sched、io、compute、graph；语义上只有 allocator/Region/fiber
  stack 是现有生产消费。没有发现 mem 自己绕开 `error`/`platform`：冷失败统一用 `Status/Result`，VM、
  NUMA、page geometry 与 guard 全由 platform 承载，热 allocation 仍以 null 表资源失败。
- 下游扫描发现 sched/compute/io 仍有直接 `new/delete/malloc`，但它们分别是未收口 owner 的 pimpl、
  backend native object 或尚未审定 host backend；本轮不跨模块改写，登记为相邻模块 adoption backlog。
  error/platform/util_core 位于 mem 下方，内部自有 bootstrap allocation 不得反向依赖 mem。
- 成熟源码差异已逐项裁决：oneTBB growable pool、Folly TLS arena merge 与 marl copyable storage-owning
  Loan 不机械复制；标准 PMR、runtime owner 与显式 move-only Loan 已提供更清楚路线。oneTBB external
  concurrent pool 只有单一 preview 来源且无 HYLE consumer，保持 GENERAL；system heap snapshot 则同时
  有 mimalloc、Folly/jemalloc 与推理内存压力旅程，提升为 MUST。
- public/internal/higher-layer 边界：mem 拥有 host storage 与 allocation lifetime；platform 拥有 OS/NUMA
  mechanism；sched 拥有等待、TLS/worker lifetime 与 blocking policy；compute 拥有 device/coherence；graph
  拥有 tensor/KV liveness；io 拥有 registered-buffer completion。
- **本轮冻结九项 `FIX`：**`MEM-SYSTEM-STATS-01`、`MEM-FIXED-01`、`MEM-SHARED-01`、
  `MEM-GROW-01`、`MEM-NUMA-01`、`MEM-RESET-01`、`MEM-LEGACY-01`、`MEM-PERF-01`、
  `MEM-DOC-01`。其中 fixed/shared/grow/NUMA 是一个有限产品批次；system stats 是独立冷 API；
  reset/legacy/docs 是合同批次；performance 只建统一证据，不允许为点值修改 API/layout。
- AUDITED 前 registry snapshot 位于外部 scratch
  `C:/hyle-scratch/mem-v1/module-status-before-audited.json`，SHA-256
  `5A044DEE24341E08109305FC2FE4EA0B3794EECA20F63BB69295DF1B516CEDFD`。

## Candidate

尚未冻结。

## Verification manifest

尚未执行；未运行项不得写成 `PASS`。

## 独立红队

尚未执行。

## Closure decision

尚未裁决。
