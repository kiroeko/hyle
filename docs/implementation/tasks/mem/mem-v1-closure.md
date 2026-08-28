# mem-v1 重构与收口记录

<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 终态：`CLOSED`。本文是 mem 当前轮唯一 closure card、施工记录与 terminal receipt；既有 mem task、设计页、
> guide 和性能页只作审计输入，不继承旧 `SIGNED`、Native FFI 或 SOTA 叙事。权威机械状态见
> [`module-status.json`](../../module-status.json)。能力族、全部公开入口、真实消费者与绑定版本的成熟
> 源码已经双向核对；本轮只实现下方冻结的九项 `FIX`，不得边施工扩面。产品/API 修复已落在
> `0943c747`，三种交付 sample 对新冷快照的实调用落在 `2c71dfae`；产品/API candidate 已冻结，
> 当前统一性能资格化、exact-candidate correctness/delivery、最终红队均已通过；产品实现仍绑定
> `2c71dfae`。本轮已经停止，Formal 继续作为独立 `PENDING` 轴。

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
| admission and observability | strict service quota、opt-in leak/debug、长寿推理进程默认 heap 压力诊断；mimalloc + Folly/jemalloc | requested live quota；per-wrapper counters/inventory；process-wide allocator reserved/committed 与 process RSS/commit current+peak | quota/stats wrapper 借 backing；metadata 分域；全局 snapshot 是冷观测，默认分配热路零成本 | `QuotaAllocator`、`StatsAllocator`、`LeakTracker`、`system_memory_snapshot()` | `COMPLETE` | 精确 requested-live 必须显式套 `StatsAllocator`，不为全局统计污染热路；不开放任意 mimalloc mallctl/options/reset |
| phase and stack scratch | inference/graph/request phase、caller buffer、pooled block、LIFO checkpoint；Folly Arena/PMR monotonic | owned growable、external fixed、borrowed block、LIFO；clear/release；alignment、limit、large-block release | single owner；bulk invalidation；borrowed/owned lifetime 可见；OOM 返回 null | `Region`、`FixedRegion`、`PooledRegion`、`StackAllocator` | `COMPLETE` | destructor registration、hidden TLS arena/merge 需要 runtime owner，保持 routed/deferred |
| external variable-size storage | embedded/realtime/caller-owned workspace；oneTBB `fixed_pool` | different-size alloc/free/realloc、fragmentation、alignment、bulk reset、no fallback | caller owns backing；single owner；reset quiescent 且批量失效旧指针 | `FixedPool` two-level indexed fit/coalesce | `COMPLETE` | reset 不要求逐块 free；并发版本需具名 consumer 或 caller synchronization |
| hard shared slots and typed objects | async state/node handoff、fixed quota、owner-local reuse；marl bounded pool/loan、mimalloc remote free | hard shared core、显式 soft cache、Reconstruct/Preserve、manual/RAII return、local/remote free | core thread-safe；cache single-owner；pool 比 cache/loan/slot 长寿；reset 前 publish 所有 cache | `SharedPool`、`SharedPoolCache`、`ObjectPool*`、`PreservingObjectPool`、`PoolLoan` | `COMPLETE` | erased adapter 首次 request-fit 失败返回 null；旧 live block 的 sized/alignment precondition 仍 fail-fast |
| growable pools | single-owner stable-address typed churn；marl unbounded pool；owned growable variable-size standard route | typed fixed-slot geometric growth；arbitrary-size individual-free+bulk-release；constructor exception/OOM | single owner；typed reconstruct；trivial bulk teardown、nontrivial must destroy；standard resource lifetime | `GrowablePool<T>`；arbitrary-size route 为 `std::pmr::*synchronized_pool_resource` over HYLE PMR | `COMPLETE` | testing/shipping 对 trivial bulk teardown 一致；通用 variable heap 已完整 `ROUTED`，不复制 mimalloc/PMR |
| guarded execution stacks | scheduler fiber lifecycle；marl runtime stack shape + platform VM contract | guarded owner、size tiers、bounded reuse、cross-worker return、trim/shutdown | move-only stack；pool thread-safe；VM release outside lock；wrong tier drops | `FiberStack`、`FiberStackPool` | `COMPLETE` | stack switching/context policy belongs sched；cache cap is explicit bounded metadata |
| large host span placement | large inference workspace、selected/weighted NUMA stripe、THP/1GiB reserve；oneTBB NUMA + mimalloc | any/preferred/local/hard bind/all-node/selected-node/chunk；requested vs accepted/applied；rollback | move-only owner；platform owns OS mechanism；soft policy may degrade observably，hard policy must fail honestly | `HugeRegion` + platform `VmRegion`/NUMA；`reserve_huge_1g_pages` | `COMPLETE` | hard bind 必须显式 node；local discovery 失败原样传播；多节点物理 bandwidth 仍未签署 |
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
| `MEM-SYSTEM-STATS-01` | MUST / inference pressure journey + mimalloc/Folly | stable POD snapshot of allocator reserved/committed and process RSS/commit current+peak；mem | process-global cold observation；concurrent activity may make fields temporal，不能伪称 per-owner quota/requested-live | O(number of allocator heaps) cold at worst；default hot path zero指令/状态增量 | `COMPLETE` | `FIXED` | `system_memory_snapshot()` 私有适配 mimalloc 3.5；shipping/instrumented invariants + 三交付实调用；不导出 vendor layout/options/reset |
| `MEM-CPP-ADAPTER-01` | MUST / C++23 + current container users | default/runtime allocator-template、PMR、alignment、standard exception boundary；mem | adapter borrows state；complete equality；zero PMR pair | allocation one backend dispatch；system form empty/direct | `COMPLETE` | `KEEP` | header/API recipes、vector/allocate_shared/zero/overaligned、三交付 |
| `MEM-REGION-01` | MUST / graph/inference + Folly/standard | owned/external/borrowed/LIFO scratch；mem | single owner；bulk invalidation；OOM null；hard reserved limit | hot bump O(1)；epoch O(blocks)；large clear O(large blocks) | `COMPLETE` | `KEEP` | overflow/OOM/move/clear/release/randomized reuse；no hidden scan/fallback |
| `MEM-FIXED-01` | MUST / engineering user + oneTBB | external variable-size alloc/free/realloc/reset；mem | caller backing；single owner；quiescent reset invalidates all old pointers | fit/free/coalesce O(1)；move realloc copy O(min)；约 4 KiB control | `COMPLETE` | `FIXED` | code/guide/test 统一为 genuine bulk reset；fragmentation oracle and two-candidate structure gate |
| `MEM-SHARED-01` | MUST / async handoff + marl/mimalloc | hard fixed slots + optional explicit owner cache + erased allocator；mem | core thread-safe/cache single-owner；capacity failure null；publish before core reset | expected O(1) common path；no cached free in hard core；cache hit zero atomic | `COMPLETE` | `FIXED` | oversize/overalign initial erased request null；realloc preserves live block；reset/cache pitfall tests/docs |
| `MEM-OBJECT-01` | MUST / marl + foundation | reconstruct/preserve + move-only scoped loan；mem | pool outlives object/loan；constructor throw returns slot；nonthrow destructor | expected O(1) slot ops；preserve O(capacity) cold construct/destruct | `COMPLETE` | `KEEP` | exception/early return/double/foreign/outstanding；blocking route sched |
| `MEM-GROW-01` | MUST / existing public + marl | owner-local stable typed slots with geometric growth；mem | trivial live storage may bulk-die；nontrivial objects require destroy；OOM null | O(1) amortized acquire，O(log n) backing allocations，O(blocks) teardown | `COMPLETE` | `FIXED` | testing-only contradiction removed；nontrivial death + trivial normal teardown + move/OOM |
| `MEM-GENERAL-POOL-01` | MUST / standard + oneTBB | owned growable arbitrary-size individual free + release；standard PMR | user selects sync/unsync；upstream outlives pool/resource | mature size-class pool；no HYLE duplicate metadata | `ROUTED` | `BORROW` | guide compile recipe over system and runtime upstream；no new HYLE heap type |
| `MEM-FIBER-01` | MUST / sched consumer | guarded tier owner + bounded cache/trim；mem/platform/sched | stack move-only；pool shared；quiescent owner shutdown | cache hit O(1)+short lock；miss/release syscall；bounded address retention | `COMPLETE` | `KEEP` | guard fault、tier mismatch、cap/trim、sched lifecycle；no unsafe context API |
| `MEM-NUMA-01` | MUST / inference workspace + oneTBB/mimalloc | THP、local/preferred/hard/interleave/selected/weighted chunk；platform semantic + mem owner | hard policy error；soft result observable；move/reset rollback | setup O(nodes) or O(size/chunk)；access zero HYLE dispatch | `COMPLETE` | `FIXED` | bind-without-node rejected；local-node query propagated；single-node/invalid/selected tests；multi-node physical bandwidth remains unsigned |
| `MEM-DIAG-01` | MUST / current debugging journey | strict quota、atomic counters、optional complete/incomplete leak inventory；mem | wrapper borrows backing；metadata separate；quiescent reset/destruction | quota/stats O(1) atomics；leak map expected O(1)+mutex only opt-in | `COMPLETE` | `KEEP` | composition order、OOM rollback、dropped record、concurrent peak/lifetime |
| `MEM-RESET-01` | MUST / existing public lifecycle | bulk clear/reset/release vocabulary and cache invalidation rules；mem/docs | reset requires operation quiescence, not prior individual free；attached cache must publish/empty | cold O(1) fixed pool or O(shards/blocks) owner reset | `COMPLETE` | `FIXED` | FixedPool/SharedPool comments、guide 与 cache-reset pitfall test 对齐 |
| `MEM-LEGACY-01` | MUST / accepted delivery policy | pure C++ source/static/shared terminology；mem/build/docs | no stable ABI/foreign schema | zero extra surface | `COMPLETE` | `FIXED` | active allocator 注释与 guide 去除候选 C schema 叙事；Win/Linux 产物仅导出 HYLE C++，无 `mi_*` 泄漏 |
| `MEM-ADOPT-01` | MUST / direct consumer audit | allocator in container/util/sched/io/graph/compute；fiber stack in sched；Region in graph/io/util；mem + consumers | preserve dependency direction and owner-specific OOM policy | no extra wrapper/translation | `COMPLETE` | `KEEP` | central `mem PUBLIC core error platform util_core` equals target edge；candidate seam compile/tests |
| `MEM-PERF-01` | MUST / current evidence invalidation | four frozen consumer families under `closure-perf-v1`；bench/perf owner | same source-primary contract、backing、lifetime、touch、capacity semantics | W1-WIN 5% NI；resource/tail separate；no all-mem aggregate | `COMPLETE` | `FIXED` | 两份 clean n=15 bundle 均 QUALIFIED/PASS 并通过离线深重放；旧 821 行点值表已删除 |
| `MEM-DELIVERY-01` | MUST / build-package contract | mem-only source + relocated static/shared SDK + mimalloc lock/license/offline；build/package | C++23 only；vendor symbol private except static link implementation | exact selected closure | `COMPLETE` | `KEEP` / blocker-at-verification | component/header/package/manifest checks on candidate |
| `MEM-DOC-01` | MUST / user discoverability | architecture decisions、scenario guide/pitfalls、current performance evidence；docs | N/A | no duplicated fragile point claims | `COMPLETE` | `FIXED` | architecture/guide 已覆盖 snapshot、request-fit、reset/cache、NUMA failure；性能页只保留当前合同与诚实边界 |
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

### 冻结 FIX 的实现结果

- `0943c74753c85bf510eca6c628c8f21e0e7d8b89` 完成八项产品/合同修复：新增冷
  `SystemMemorySnapshot`；SharedPool/Cache erased adapter 的初始 request-fit 失败改为 null；
  GrowablePool trivial/nontrivial teardown 合同统一；HugeRegion 拒绝 bind-without-node 并传播
  local discovery 失败；FixedPool/SharedPool/cache bulk-reset 合同写入公开头与测试；删除把运行时
  type erasure 误称稳定 ABI 的注释。
- `2c71dfae9f4fb43ff0d68e5c1a4c55099fc0f101` 让 mem
  package sample 实际调用 `system_memory_snapshot()`，因此 installed shared/static consumer 会验证
  新编译符号，而不是仅证明头文件被复制。
- 默认 system allocation/free 没有新增 field、atomic、lock、TLS 或周期线程；mimalloc 3.5 shipping
  仍保持 `MI_STAT=0`。全局 snapshot 只读取 vendor 已存在的 essential/process stats；精确
  requested-live 继续由显式 `StatsAllocator` 承担。

### Candidate 前正确性与交付复核

以下结果绑定上述产品提交，只是 candidate 冻结前的施工证据；最终 manifest 仍须在 exact candidate
上重放适用门禁：

- Windows / MSVC 19.50：Release shared shipping `hyle_mem_test` PASS；`HYLE_TESTING=ON`
  instrumented `hyle_mem_test` PASS；17 个 mem public headers 逐头 self-contained PASS。
- W1-WSL / GCC 14.2：Release shipping、ASan、UBSan 全量 mem PASS；TSan + mimalloc `MI_TSAN=1`
  的 193 个非 death tests PASS。death-test subprocess 在 TSan 下与 test timeout 不相容，不伪记为
  product failure；同一 death contract 已由 Windows instrumented build 执行。
- macOS arm64 / AppleClang 21：Release shared 与 ASan static shipping mem PASS；17 个 public
  headers 逐头 self-contained PASS。
- Windows、W1-WSL、macOS 都通过 mem-only source embedding、relocated shared SDK 和 relocated
  static SDK C++ consumer；安装树含 HYLE/mimalloc licenses。Windows/Linux 动态符号检查确认
  `system_memory_snapshot()` 可见且无 `mi_*` vendor export。

### Candidate 前性能合同与 probe

- 冻结输入 `bench/mem/consumer_shapes_v1.json`（SHA-256
  `c72e00527ea82d5e32431fd29d996faa7fecb6d58c2d2514281c41a471a43b32`）同时保留当前
  container/io/sched 源码来源与显式 north-star 来源；没有把尚不存在的 ObjectPool 产品调用伪称为
  生产 trace。
- `mem-owner-consumer-v1`（合同 hash
  `911787ec1f4cc0b7c61d508cd2fea52a9baebbcb57ec29ebb19833e8df8e577e`）冻结 runtime
  allocator batch、HTTP-policy Region epoch 和 scheduler guarded-fiber lifecycle。oneTBB 通过独立 TU
  的 `memory_resource*` 走真实 runtime erasure；`Allocator` 对 `SystemAllocator` 的 compile-time
  direct 差异明确为 `mechanism + diagnostic + NOT_COMPARABLE`，不能因约 10–15% dispatch tax 改
  layout/API。默认用户本就有 direct `SystemAllocator`/`SystemStdAllocator`/
  `system_memory_resource()`；只有需要 runtime backing injection 时才支付 erased dispatch。
- `mem-handoff-consumer-v1`（合同 hash
  `415619a7875f77744444cb306c3214726291496370de605dc3e688768b202dd8`）冻结两颗物理角色、
  256 hard slots、64 个 unique scoped loan、双执行顺序、双 allocation order 与 placement。锁定 Marl
  的 `tryBorrow()` 含一行引用已删除 `Item::pool` 的陈旧编译错误；build-local overlay 只删除该行并
  改写三个相对 include，恢复其公开非阻塞容量语义，外部 checkout 保持 clean，source/overlay hash
  均进入合同与 validator。
- owner probe `C:/hyle-scratch/mem-v1-owner-probe-r2-output` 与 handoff probe
  `C:/hyle-scratch/mem-v1-handoff-probe-r1-output` 均完成。probe 固定只有每 stratum 一个进程样本且
  build 不声明 fresh，因此按协议只能是 `DIAGNOSTIC/INCONCLUSIVE`，A/A 与 known-delta 也必然因
  `n=1` 显示 FAIL；这不是产品失败。两份 probe 的 artifact、source/peer 输入、timer floor、placement、
  resource/tail validator 均通过，全部 product row 在每个 order/placement/allocation-order 方向一致：
  owner 三项约 `1.44–1.49x`、`1.58–1.63x`、`8.04–8.32x` 有利于 HYLE，handoff 约
  `1.91–2.25x` 有利于 HYLE。它们只证明正式采样值得执行，不作为签署点值。
- 第一份 owner probe 在 Windows 启动前以 `0xC0000135` fail closed：静态 HYLE benchmark 仍需要
  oneTBB 的动态 runtime，而全新 runner 目录未 stage DLL。benchmark peer 层现显式把精确
  `TBB::tbb`/`TBB::tbbmalloc` runtime 放到 consumer 旁；`r2` 使用全新目录复跑通过，没有保留或
  重解释失败样本。

## Candidate

- product commit：`2c71dfae9f4fb43ff0d68e5c1a4c55099fc0f101`
- product tree：`fc02363512a0807f4d46bb57427eae4137ea39fd`
- performance harness/docs commit：`dc9d4cb13d3893cb5bcc2adf5542c30f4df8ee8c`
- 产品冻结内容：`0943c747` 的八项实现/合同修复与 `2c71dfae` 的三交付实调用；其后没有修改
  `include/hyle/mem/**`、`src/mem/**` 的公开合同、layout、算法或正常 workload。性能合同、replay、
  validator 与用户/设计说明不改变 product identity。
- complexity/resource delta：新增冷 snapshot 为 O(number of allocator heaps/process query)；默认
  allocation/free 无新增 field、branch、atomic、lock、TLS 或 syscall。其余 FIX 是 failure/lifecycle
  合同收正，不改变已登记的正常复杂度档。
- 当前 runtime-erased `Allocator` 相对同一 backing 的 compile-time direct route 有可解释的一次
  indirect dispatch 税；两条用户形状已经分开提供且 direct route 无损。该行冻结为机制诊断，不能
  在 qualification 后改成产品 peer，也不能为点值扩大 handle/layout。
- `AUDITED -> CANDIDATE` 的前态快照保存于
  `C:/hyle-scratch/mem-v1/module-status-before-candidate.json`，SHA-256
  `711616A7F96A13E89E7D0865987E333D95653E8AF8121E1FF3C724B5691622BB`。

## Verification manifest

### W1-WIN P1 engineering 性能

2026-08-28 在 clean source `094b6cd1525df3363e174c3db27014c40d34d152` 上完成两份 fresh
`qualify-platform`，该 source 相对 product candidate 只增加性能 harness、文档与机械状态记录，
没有 mem 产品实现变化：

- owner bundle：`C:/hyle-scratch/mem-v1-owner-engineering-output/manifest.json`，SHA-256
  `be87b13ec7116e900b4415742e48a6aae27182fb1bf6f496d902904f6c282e8a`；contract SHA-256
  `911787ec1f4cc0b7c61d508cd2fea52a9baebbcb57ec29ebb19833e8df8e577e`；`QUALIFIED / PASS`；
- handoff bundle：`C:/hyle-scratch/mem-v1-handoff-engineering-output/manifest.json`，SHA-256
  `b082c033061973b1a50d6d24122670d21798df5849e6b4b99cee51ce4ef194fb`；contract SHA-256
  `415619a7875f77744444cb306c3214726291496370de605dc3e688768b202dd8`；`QUALIFIED / PASS`；
- 两份 manifest 都由独立 `hyle-perf.py verify` 深重放通过；artifact/build/source/peer identity、
  A/A、5%/10% known delta、batch-matched floor、placement 与模块资源 validator 全部 `PASS`；
- runtime allocator、Region、fiber lifecycle 与 cross-core ObjectPool handoff 的全部 strata 均为
  `WIN` 且 5% NI `PASS`；具体区间、资源/tail 数据与诚实外推边界只记录在
  [当前 mem 性能页](../../evidence/performance/benchmarks-mem.md)，不在 task 复制易漂移表格；
- runtime `Allocator` 对 compile-time direct baseline 仍是 `NOT_COMPARABLE` mechanism diagnostic；
  measured direct throughput 高 11.1%–13.4%，由一次间接分派解释，用户已有自然 direct route，
  不触发产品修改；
- 一次加 `--formal` 的 owner 运行按设计降为 `DIAGNOSTIC/FAIL`，唯一原因是 runner profile 未预先
  固定 exact `formal_toolchain`；其产品行和全部工程 controls 都通过。该包未被复用或签署。Formal
  本来就是 closure card 明确的非目标和独立 `PENDING` 轴，不能为本轮临时改合同或反向阻塞 P1。

### Exact-candidate correctness 与交付复签

最终复签在 clean `af991e1890eae742161415419ee2af6584255184` 上完成；该 HEAD 相对 product
candidate 只增加 benchmark、validator、文档与机械状态记录，`git diff 2c71dfae..af991e18 --
include/hyle/mem src/mem samples/package/mem` 为空：

- **W1-WIN / MSVC 19.50：**全新 shipping shared 与 `HYLE_TESTING=ON` instrumented build 的 mem
  public-header/unit 2/2 PASS；source、relocated shared、relocated static 三种 consumer PASS；shared
  DLL 有 HYLE snapshot 入口、零 `mi_*` vendor export，安装树含 HYLE/mimalloc notice；
- **W1-WSL / GCC 14：**Release shared、ASan static、UBSan static 的 mem public-header/unit 2/2
  PASS；mimalloc `MI_TSAN=1` 的 TSan static 运行 201 个非 death tests、33 suites PASS 且无 race；
  death subprocess 只因 TSan runner 合同排除，已由 Windows instrumented 与 WSL Release 执行；
  source、relocated shared、relocated static consumer PASS，ELF 零 `mi_*` vendor export；
- **M1-MAC / AppleClang 21 arm64：**Release shared 与 ASan static 的 mem public-header/unit 2/2
  PASS；Apple ASan 不支持 `detect_leaks=1`，同一二进制按平台能力用
  `detect_leaks=0:abort_on_error=1` 重跑 PASS，不把不支持的 leak option 伪记为产品失败；source、
  relocated shared、relocated static consumer PASS，deployment target 14.4、Mach-O 零 `_mi_*` export；
- 三平台交付 sample 都实调用 `system_memory_snapshot()`；中央 `mem PUBLIC core error platform
  util_core` 与 public include owner 对账，mimalloc 仍是 private implementation dependency；
- `check-module-closure.py`、`check-docs.py`、两份 contract validate 和两份 bundle 的独立 deep
  verify 全部 PASS。原始本地输出位于 closure card 已登记的 W1/M1 scratch；它们是可重放工程
  artifact，不冒充 Formal 发布证书。

## 验证矩阵（`VERIFIED`，2026-08-28）

| Gate | Verdict | 证据与边界 |
|---|---|---|
| scope / capability | `PASS` | 十个能力族全部 public surface 已双向追踪；本轮九项 `FIX` 全部关闭；通用 growable variable pool 有完整标准 PMR route，device/KV/staging/runtime policy 有明确 owner |
| architecture / API | `PASS` | host/device、semantic/artifact/delivery owner 分开；C++ owner/view、OOM、sized-free、reset、cache publish、NUMA soft/hard 与 direct/runtime 选型在头文件、architecture 和 guide 一致；candidate 后无产品 drift |
| correctness | `PASS` | W1-WIN shipping/instrumented，W1-WSL Release/ASan/UBSan/TSan，M1-MAC Release/ASan 的适用 mem suite 与 public headers PASS；边界/OOM/lifetime/fragmentation/remote-free/NUMA fallback 均有对应测试 |
| delivery / dependencies | `PASS` | 三平台 source、relocated static/shared consumer PASS；中央 direct edge 与真实 target/public include 对账；mimalloc lock/license/offline 投影与动态 vendor-symbol hiding PASS |
| direct consumers | `PASS` | package sample 实调用新 snapshot；container/util/sched/io/compute/graph seam 在 candidate 施工期编译/测试，最终产品 diff 为零；不替尚未收口的上层组件签整体结论 |
| usability / docs | `PASS` | 选择表覆盖全部能力族；guide 明示 invalid allocator、borrow lifetime、direct/runtime 成本、异常/null、bulk invalidation、cache、NUMA 与统计陷阱；当前性能页不保留旧总分或泛化 SOTA |
| engineering performance | `PASS` | W1-WIN `closure-perf-v1`、5% NI；owner 4 strata 与 handoff 8 strata 的全部 product row 为 `WIN + NI PASS`，A/A、5/10% delta、floor、placement、artifact/build/source/peer/resource validators 全 PASS |
| red team | `PASS` | 下表 finding 全部 fixed/disproved/accepted，无 correctness、security、delivery、usability 或 P1 blocker；没有为点值改 API/layout/metadata |

## Final red-team

- reviewer / date / budget：Codex（root）/ 2026-08-28 / 30 分钟只读终审；没有新增 workload、peer、
  第五种 benchmark 姿势或产品修改；
- 检查面：candidate→HEAD product drift、全部 public family、ownership/failure/concurrency、依赖与
  三种交付、性能同合同/controls/resource、用户陷阱、过时 C/Native FFI 与泛化 SOTA 叙事。

| Finding | Severity | Disposition | Evidence |
|---|---|---|---|
| `Allocator{}` 的 invalid sentinel 与 backing 借用寿命只在头文件隐含，guide 不够直接 | usability documentation | `FIXED`；guide 明示先 `valid()`/赋有效 backing、不得调用 invalid handle，并给出 direct/runtime 选型 | `docs/user/mem.md`；实现/API/layout 不变 |
| runtime-erased `Allocator` 相对同 backing direct route 慢 11.1%–13.4% | expected mechanism cost | `ACCEPTED/NOT_COMPARABLE`；一次 indirect dispatch，有 runtime injection 才使用；三条自然 direct API 已公开 | owner bundle mechanism row；未为 benchmark 放大 handle 或增加 metadata |
| Marl 锁定 header 的公开 `tryBorrow()` 含一行写入已删除 field | peer build integrity | `FIXED IN HARNESS`；build-local overlay 只删陈旧赋值并改 sibling include，算法/同步不变，source 与 overlay hash 均 fail-closed | handoff contract、validator 与 clean peer checkout |
| 全新 Windows static consumer 未自动找到 oneTBB runtime DLL | harness delivery | `FIXED IN HARNESS`；精确 `TBB::tbb`/`TBB::tbbmalloc` runtime stage 到 benchmark 旁后全新目录通过 | owner probe r1 fail-closed、r2 与正式 fresh bundle PASS |
| `--formal` 被误用于没有 exact formal toolchain 的工程 profile | operator/profile mismatch | `DISPROVED AS PRODUCT FAILURE`；该包按设计 `DIAGNOSTIC/FAIL` 且未签署，正确 engineering run 独立 PASS | Formal 是 closure card 非目标及独立 `PENDING` 轴 |
| 只有单节点开发机，无法证明跨 socket placement/bandwidth | claim integrity | `ACCEPTED LIMITATION`；只签 selected-node/chunk API、rollback、复杂度和 single-node fallback | guide/architecture/performance 均明确不外推物理多节点 SOTA |

没有未解决的产品或证据 blocker。owner 比较的 candidate/peer 使用相同尺寸、对齐、触碰和释放；
Region 共用同一 runtime upstream 并单列 retention；handoff 两边都是 256 hard slots、64 个 scoped loan
和远端析构；fiber 使用相同 scheduler/callback/guard/stack tier，仅改变 bounded reuse 与 direct VM。

## Closure decision

本轮 `CLOSED`：scope、architecture/API、correctness、delivery/dependencies、direct consumer、
usability/docs、engineering performance 与 red-team 八项 aggregate gate 全部 `PASS`，九项 ledger
`FIX` 全部关闭，没有 material `MUST`、正确性、安全、交付或 P1 blocker。产品 candidate 仍为
`2c71dfae9f4fb43ff0d68e5c1a4c55099fc0f101` / tree
`fc02363512a0807f4d46bb57427eae4137ea39fd`；Formal 保持独立 `PENDING`。

- VERIFIED precursor：`6f582f5289a9b72dfc6679a282dcf356f35530b9`；terminal end 与 registry
  anchor 由紧随本 receipt 的状态锚提交回填，避免在一个提交内容里伪造自身 hash；
- 可接受限制：P1 工程数字只覆盖两份冻结的 W1-WIN 合同，不代表所有 mem 类型、平台或 topology
  全面 SOTA；runtime-erased allocator 有一次间接分派，默认热点有无损 direct route；当前只在单节点
  机器签 NUMA API/rollback/fallback，不签跨 socket bandwidth；TLS arena merge、concurrent external
  variable pool、device/pinned/KV/staging 与 Native FFI 保持 routed/out-of-scope；Formal 为 `PENDING`；
- reopen trigger：具名消费者复现 mem 边界内 correctness/security/OOM/ownership/usability/常用能力
  缺陷；公开 API/layout/算法、直接依赖、OS placement primitive 或 source/static/shared 交付改变；
  成熟同合同 peer 证明更好复杂度档或正常 consumer 的 material 优势；现有 evidence/contract/peer/
  machine/instrument identity 改变且需要续签；或多节点 NUMA、foreign runtime、device/pinned 等能力经
  独立准入进入 mem 边界。

四项 LOC 由 immutable product candidate
`2c71dfae9f4fb43ff0d68e5c1a4c55099fc0f101` 运行
`python scripts/loc-report.py --module mem --commit <candidate> --self-check --json` 得到，self-check
`PASS`：

1. 全库代码：301,961；
2. 自实现包含测试：123,268；
3. 自实现不包含测试：82,768；
4. `mem` 实现：4,079（`include/hyle/mem` + `src/mem`，不含 tests/bench/docs/CMake）。

终态不再修改产品、正常 workload 或性能合同；只有上述 trigger 能重开受影响部分。
