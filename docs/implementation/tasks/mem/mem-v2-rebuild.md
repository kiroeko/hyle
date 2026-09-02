# Mem v2：统一 storage foundation 重构

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前轮由 [`mem-v1 capability replay`](mem-v1-capability-replay-2026-09.md) 的
> `PRODUCT_TRIGGER` 启动。旧 `mem-v1` task/evidence 只作 baseline 证据，不是本轮待办。
>
> 项目级阶段门：[`全库能力展开与组件因子化 v1`](../project/capability-factorization-v1.md) 已完成，随后
> [`build-system-v7 factor/façade projection`](../build/build-system-v7-factor-facade-projection.md) 已把内部
> factor DAG 与外部产品 façade 分开。中央 registry 已按本卡冻结的 `MEM2-01`～`MEM2-09` 有限 FIX
> 产出候选并进入 `CANDIDATE`；历史 factor 清单或 bundle 文字若与 v7 投影冲突，不再构成施工合同。

## Closure card（AUDITED）

- component / owner / round ID：`mem` / Mem / `mem-v2`；
- task kind：`REOPEN`；
- baseline commit：`e7b3a5f2f6bc90b194821c1777542c3c47a27c7b`；
- 进入原因与 demand/defect ID：`MEM-OWNER-2026-09-01`。未来 HYLE 推理引擎需要一个统一的
  storage foundation；旧 host-only 边界把同一条 memory lifetime 分裂到 Mem、Platform、Compute、IO；
- 真实 consumer task / normal workload：
  1. 映射并按需驻留/锁定模型权重，释放不再需要的页区间；
  2. 为 CPU、pinned host、discrete device 与 managed/unified memory 选择 resource，创建持久
     weights/KV/output 和 transient compute buffers，并安全取得 subview/imported view；
  3. 在同一 resource 或 provider 支持的跨 resource 路径做 read/write/fill/copy、map 与 coherence；
  4. 对 worst-case prompt/token graph 先 measure/reserve，再按 lifetime/alias 复用 storage；
  5. 普通 C++ 用户继续自然使用 host allocator、arena、pool、quota、stats、NUMA/huge span；
- 范围：统一 Mem 的 host allocator/heap/arena/pool 与跨域 resource/buffer/view；把用户级 virtual/file/
  shared mapping、residency 和 placement 收归 Mem；提供 resource pool/trim/stats 与不含 Tensor 的
  lifetime packing engine；迁移现有 Platform/Compute/IO/Graph adapters、直接消费者、build/package、
  architecture/user docs 和适用验证；
- 非目标：Tensor shape/dtype/operator、KV/sequence/eviction/model placement policy、Graph DAG 推导、
  backend kernel、Device/Stream/Event 调度、文件格式/异步 IO protocol、完整 CUDA/Metal/Vulkan 重做、
  稳定 C++ ABI、Native FFI/C API、所有 vendor knob、Formal 或“所有操作/平台 SOTA”声明；
- 直接依赖：`core`、`error`、`platform`、`util_core`；mimalloc 仍只是 system/heap mechanism；
- 直接消费者：`container`、`util`、`sched`、`io`、`compute`、`graph`；Platform 的 public owner 收缩另按
  direct-boundary trigger 留下可审计记录；
- tier 与理由：aggregate `P1`。allocator/ref、pool、BufferView 和 transient buffer reuse 可能高频；mapping
  与 plan construction 在没有端到端热点证据前按 P0 复杂度/资源门处理；
- material margin / resource budget：同合同 P1 默认 5%；source-primary normal route 不得多一次 per-item
  heap allocation、lock、syscall 或 copy；borrowed ref/view 常态 metadata 不超过四个 machine words，owned
  Buffer 不通过 pimpl 再分配；planner 为 O((N+E) log N) 或更好、scratch O(N)、不逐 byte/page 建表；
- 总调查预算 / 性能预算：九张有限 ticket；每个 P1 family 最多一次资格化重放、一次机制诊断和一次产品
  重跑。相同结果仍 `INCONCLUSIVE` 时按 blocker 停止，不扩建通用仪器；
- 最多复现、机制诊断、产品修改次数：每个独立性能 finding 为 `1 / 1 / 1`；correctness finding 只按可
  复现缺陷修改；
- machine profile：correctness/delivery 为 `W1-WIN`、`W1-WSL`、`M1-MAC`；P1 主机 `W1-WIN`；真实
  backend 只运行机器实际具备的路径，不能把 synthetic backend 写成硬件 PASS；
- instrument profile：`closure-perf-v1`；P0 family 使用 `closure-correctness-v1` 的结构/复杂度证据；
- strong peers：
  - allocator/heap/pool：locked mimalloc v3.5.0 + oneTBB
    `a8216b95202f3c55f9749870f4f8cfa4df0b32e3`；
  - resource/buffer：ggml `d4716378882593333721eb33f153144b6885caf2` + CCCL
    `b7aaea69a2b07e50f09e67f2962da0243e0b7c5d`；
  - mapping/residency：llama.cpp `e4b9af007beae34abba094cefd187658e89bbac8` + Folly
    `23ebefdf96c116700d3e134f04d95a2b60a509de`；
  - planner：ggml `d4716378882593333721eb33f153144b6885caf2` + oneDNN
    `5dc67b7f37b1215d8c1f054e99c1b7cfd40317a9`；
- 无同合同 peer 时的预算及禁止声明：provider-specific import/coherence 只签无额外 allocation/copy、
  正确 native ordering 与资源上界；不得宣称跨 backend SOTA；
- 使用原型 provenance：`REAL_CONSUMER`（llama.cpp/ggml）+
  `NORTH_STAR_JOURNEY`（HYLE inference）+ `STANDARD`/mature peer；benchmark-only 形状不准入；
- API 语义档位与选择前提：按 ownership/domain/lifetime 命名，不提供 `fast/slow/light` 档；host raw
  allocator、owned heap、arena、block/object pool、storage resource、Buffer owner/view、mapping、pool 和
  planner 是不同类型；
- 每档常态成本合同：见下表；
- 适用 gates：scope、architecture、correctness/sanitizer、source/static/shared delivery、C++ usability/docs、
  red-team、P1 engineering；Native FFI 为 `NOT_APPLICABLE`；
- 预期输出：一个有明确子系统的 `hyle::mem` C++ 产品面、无环 provider seam、迁移后的相邻模块、准确
  architecture/guide、可执行 consumer samples、限定性能证据与 terminal registry record；
- 停止条件 / 可接受限制：九票全部裁决且无未决 MUST 才可进入 candidate；真实 backend 不可用或同合同
  P1 仪器不合格时进入 `BLOCKED`，不以 benchmark 试错扩 API；backend-specific async optimization、KV
  policy 与 Formal 可以保留为明确相邻 backlog。

### 常态成本合同

| semantic tier | allocation / metadata | copy / syscall / synchronization | lifetime rule |
|---|---|---|---|
| direct host allocator | backing 原有 metadata；无 wrapper heap | direct path 不叠 runtime dispatch | sized return，remote free 由 backing 合同 |
| borrowed allocator/resource ref | inline small vtable/state identity | 每次 operation 至多一次 provider dispatch | backing/provider 长寿于 ref 与 allocation |
| BufferView | 不拥有、不分配；O(1) subview | 取 view 不 copy、不 syscall | owner/storage 长寿于 view 与 in-flight use |
| owned Buffer/mapping | owner 自身不 pimpl 分配；provider allocation 计费 | 只执行所请求 provider/OS operation | move-only；析构经原 provider 释放 |
| arena/pool | 按公开 capacity/growth/retention | 热路不隐藏 syscall；并发语义决定 atomic/lock | clear/reset/trim 的 invalidation 明示 |
| lifetime planner | O(N) records/scratch | 构建 O((N+E) log N) 或更好；不搬 payload | plan 与 instantiated buffers 分离 |

### 施工顺序：先功能与易用性，后性能

`MEM2-01`～`MEM2-07` 不按“实现一个类型、跑一轮 benchmark、再补下一个类型”的方式施工，而分四遍：

1. **能力与 API pass：**一次性画出全部 public taxonomy、ownership/resource/provider 关系、失败与并发
   合同，并写出 host heap、模型映射、persistent buffer、transient pool、planner 五条普通 C++
   compile-only recipe；同时完成场景选型表和误用路径；
2. **API / usability freeze：**确认每项 `MUST` 都有自然路线，锁定名称、类型关系、默认值、view
   invalidation、依赖暴露和 source/static/shared 可发现性。此门以前不运行 P1 计时；
3. **correctness / migration pass：**在固定接口后实现算法和 provider mechanism，完成边界/OOM/move/
   concurrency 测试并迁移直接消费者；接口若因 correctness 不成立而变化，显式更新本卡，不能悄悄漂移；
4. **candidate / performance pass：**冻结 product commit/tree 后，才运行同合同 normal consumer 与一个
   holdout。性能修复默认只换内部算法或零成本路由；只有 normal workload 的 material 机制证据证明
   public contract 本身有错，才退回 `AUDITED`。

全局因子图与 v7 façade 投影已经完成，因此这四遍顺序现在生效；相邻域只提供已经裁决的公开 factor
或私有机制，不会让 POSIX/Windows 某个 syscall benchmark 反向决定公共类型。

## API / usability freeze（FROZEN）

本节是 `MEM2-01` 的唯一 public-shape 合同。五条 compile-only journey、self-contained headers、
无环 target graph 与 source/shared/static relocated-package consumer 均已兑现，因此接口于
2026-09-03 进入 `FROZEN`。后续 correctness 不得悄悄改名或恢复旧 alias；若合同本身不成立，
必须按本卡显式退回 `AUDITED`。

### 单一用户心智与依赖方向

- 普通用户只看到 `hyle::mem`；`mem_core/mem_pool/mem_vm/mem_buffer` 是历史语义分解，
  不是要求用户选择的 namespace 或产品 target。当前 Mem façade 仍只投影 `mem`。
- Mem 拥有 storage 语义：host allocator/heap/arena/pool、mapping/residency、开放
  memory resource、Buffer owner/view/import/composite、sync operation、resource pool/stats 与
  tensor-free lifetime plan。
- Platform 只实现 VM/file/NUMA/page/OS-handle 机制；Compute 实现 pinned/device/managed
  provider 与有序 async operation；IO 实现 namespace/handle transfer/async protocol；Graph 只产生
  lifetime facts。固定方向为 `compute | graph | io -> mem -> platform`。
- Mem public header 不出现 `Device`/`Stream`/`Event`/Tensor、CUDA/Vulkan/ggml 类型。
  provider-specific import/async 作为上层自然 C++ adapter，接收 Mem `Buffer/View`；Mem 不建第二套
  queue/completion 抽象。

### 最终 public taxonomy

| header / family | public types | ownership 与选择规则 |
|---|---|---|
| `allocator.h` | `AllocatorRef`, `SystemAllocator` | `AllocatorRef` 是四字以内的 borrowed host-byte allocator；无 runtime injection 时用零状态 `SystemAllocator` |
| `heap.h` | `Heap` | move-only owned growable host heap；owner-thread allocate/reallocate，允许 remote free；全量 release 要求无 live allocation |
| `arena.h` | `Arena`, `FixedArena`, `PooledArena`, `StackArena` | 分别是 owned growable monotonic、caller-storage monotonic、borrowed-block monotonic 与 LIFO rewind；不伪装 individual free |
| `pool.h` | `FixedHeap`, `ConcurrentBlockPool`, `BlockPoolCache`, `ObjectPool<T>`, `PreservingObjectPool<T>`, `GrowableObjectPool<T>`, `PoolLoan` | 名字分开 variable/fixed-size、external/owned、hard/shared/owner-cache、reconstruct/preserve；不提供 `fast/light` 档 |
| `mapping.h` | `VirtualMapping`, `FileMapping`, `SharedMapping`, `MappingAdvice`, `ResidencyOptions` | 三种 owner 分开 constructor 和非法操作，共享 range/view/protect/advice/lock/flush vocabulary；async namespace/protocol 仍在 IO |
| `resource.h` | `MemoryResource`, `MemoryResourceRef`, `MemoryProperties`, `MemoryDomainId`, `ResourceAllocation`, `ResourceStats` | provider 是长寿命 polymorphic object；ref 是一字 borrow；通过属性表达 host/device accessibility、coherence、pinned/managed/exportable，不用闭合 kind enum |
| `buffer.h` | `Buffer`, `BufferView`, `ConstBufferView`, `HostMapping`, `CompositeBuffer` | `Buffer` 四字 inline owner、经原 provider 释放；view 四字且不拥有；host map 显式表达 map/unmap/coherence；composite 才支付 segment metadata |
| `buffer_pool.h` | `BufferPool`, `BufferLease`, `BufferPoolOptions`, `BufferPoolStats` | injected resource 上的 transient reuse owner；lease 自动归还；in-flight completion 前不得归还；不暗藏 Event |
| `lifetime_plan.h` | `LifetimeRequest`, `LifetimePlanner`, `LifetimePlan`, `MemoryPlacement`, `PlannedBuffers` | consumer 给 `[begin,end)` interval/resource/size/alignment/alias facts；plan 与 instantiate 分开，不出现 Tensor/Graph 类型 |

旧公开名称不是兼容负担：`Allocator` 改为 `AllocatorRef`；`Region/FixedRegion/PooledRegion/
StackAllocator` 改为上表 arena 家族；`FixedPool` 改为 `FixedHeap`；`SharedPool/SharedPoolCache`
改为 block-pool 家族；`GrowablePool<T>` 改为 `GrowableObjectPool<T>`。迁移完成后直接删除
旧名称，不留 deprecated alias。

### resource/provider 零成本边界

- `MemoryResourceRef` 只持有 `MemoryResource*`，default 为 invalid sentinel，从 lvalue provider
  显式构造，禁止绑定 temporary。provider 及其 backend/device state 必须比 ref、Buffer 和
  in-flight use 长寿。资源 identity 是 provider object identity；`MemoryDomainId{authority,value}` 允许
  同一 device/domain 的多个 resource 相关联，不声称跨进程稳定。
- `MemoryProperties` 是可扩展 bit flags + alignment/max-allocation/domain facts，不存储
  `Cpu/Gpu/IGpu` 类型。一个 provider 可同时 host/device accessible，pinned/managed 是属性而
  不是互斥类别。
- provider 同步 allocation 产生 `ResourceAllocation{native_handle,size,alignment}`。`Buffer`
  inline 保存 resource ref 和这三个字，不为 HYLE wrapper 额外分配 control block；backend
  原生 allocation 本身所需的 driver artifact 按 provider 合同计费。
- `BufferView`/`ConstBufferView` 保存 `{resource, native_handle, offset, size}`，所以
  `subview` 是 O(1)、zero allocation/copy，且不对 opaque handle 做 pointer arithmetic。owner reset/
  destruction 使 view 失效；owner move 转移 allocation 但不使基于 provider handle 的 view 失效。
- `read/write/fill/copy/flush/invalidate/map_host` 都以 view range 为边界并返回
  `Status/Result`。同 resource 先走 provider native route；跨 resource 只在 provider 明确支持或
  两端可 host-map 时执行，否则 `Unimplemented`，不偷做 hidden staging allocation/copy。
  overlap 只在同一 allocation 的 provider 合同明确支持时允许。
- 异步路线由 Compute/IO 提供类型化函数，例如 `compute::copy_async(Stream&, mem::BufferView,
  mem::ConstBufferView)`；它们检查 resource/domain identity 后用 native handle，Mem 不引入
  `void* stream` 或函数注册表。

### ownership / failure / concurrency / default

| family | failure shape | concurrency | default / invalidation |
|---|---|---|---|
| allocator / heap / arena / pool | 需要创建 backing/control 的 owner 使用 `Result<Owner>` 或可检查 `valid()`；纯 borrowed/external/lazy owner 由构造前置条件约束；热 byte/slot allocation 返 null/empty loan | 按具名类型区分 owner-only 与 concurrent；`Heap` 只承诺 remote free | 无隐式 fallback；reset/release 显式使旧 allocation/view 失效 |
| mapping | factory/range OS failure 返 `Result/Status` 并保留 native error | control operation 外部同步；普通字节访问由保护和宿主同步决定 | 不自动 prefault/lock/grow file；未支持 advice 明确 `Unimplemented` |
| resource / Buffer | provider/ref 非空、range/alignment 错误为 `InvalidArgument/OutOfRange`；OOM 为 `ResourceExhausted` | provider 必须支持独立 allocation 并发使用；同一 Buffer 的 mutation/map/control 外部同步 | `Buffer{}` 可作 invalid/moved-from；没有默认 resource，必须显式选 host/device/custom provider |
| BufferPool | create/acquire 返 `Result`；capacity/pressure 可观察 | pool 是 thread-safe；lease 中 Buffer 的使用仍由 caller 同步 | lease 归还前必须无 in-flight operation；trim 不等待 Event |
| lifetime plan | 溢出、非 2 次幂 alignment、非法 interval/alias/resource 返 typed Status | build/instantiate 外部同步；plan 不可变时可并发查询 | interval 为 `[begin,end)`；persistent/output 不复用；plan 不拥有 Tensor/Graph |

### 五条 compile-only normal journey

1. **host heap：**`Heap::create() -> Heap`，把 `heap.allocator()` 的 `AllocatorRef`
   注入 HYLE container，所有 allocation 归还后 `collect/release`。
2. **model mapping：**`FileMapping::open(path, ReadOnly)`，根据 loader 策略显式
   `advise(Sequential/WillNeed)` 与单调 `lock_prefix`，向 parser 传 `ConstBufferView`。
3. **persistent buffer：**从 Compute device 取 `MemoryResourceRef`，`Buffer::create`权重/KV，
   `write` 上传并把 O(1) `subview` 传给 kernel adapter；owner 活到 Event completion 之后。
4. **transient pool：**`BufferPool::create(resource, options)`，每次 `acquire` 得到
   `BufferLease`，completion 后析构/归还，pressure point 显式 `trim` 并查 `stats`。
5. **planner：**Graph/engine 构造无 Tensor 类型的 `LifetimeRequest` span，
   `LifetimePlanner::build` 先得 `LifetimePlan`/per-resource peak，再 `instantiate(resources)`
   一次建 backing，以 request ID 取 view。

### 用户选型与误用诊断

| 需求 | 应选 | 不应选 |
|---|---|---|
| 普通 host 小对象/容器 | `SystemAllocator` 或 `AllocatorRef` | `MemoryResourceRef`/`Buffer`；后者有跨域 owner 语义 |
| phase scratch | arena 家族 | variable-size heap/pool；individual free 在 arena 无效 |
| 固定尺寸并发 slot | `ConcurrentBlockPool` + 可选 cache | `FixedHeap`；它是单 owner 变长 external heap |
| 跨 host/device 持久存储 | `MemoryResourceRef` + `Buffer/View` | 把 native handle 当 host pointer，或把 `AllocatorRef` 扩成 GPU allocator |
| 映射模型 | `FileMapping` | 读入 heap 再伪称 zero-copy，或 open 时默认 prefault/lock 全文件 |
| async transfer | Compute/IO typed adapter + Mem views | Mem sync operation 内部暗藏 stream/event/staging |
| graph storage reuse | `LifetimePlan` | 让 Mem 理解 tensor/op/KV 业务策略 |

### freeze 未完成项

- [x] 上表类型关系全部落到 self-contained public headers，不泄漏 backend 类型；
- [x] 五条 journey 以正常 public API 编译，不借 test-only constructor/stub；
- [x] `sizeof(MemoryResourceRef) == 1 word`、`Buffer/View <= 4 words`、owner 无 wrapper pimpl allocation；
- [x] 头文件语法 include graph 与 CMake target graph 都满足上述无环方向；
- [x] source/static/shared selected-component 均可发现同一 API；
- [x] `docs/user/mem.md` 只写当前已落地产品，并已替换为 ownership/domain/lifetime/concurrency 选型表。

## Capability-family coverage map（AUDITED）

详细来源和 v1 反向追踪见
[`mem-v1 capability replay`](mem-v1-capability-replay-2026-09.md)。本轮只接纳下列有限终态：

| Family | admitted user journeys | required variants | baseline route | state | action |
|---|---|---|---|---|---|
| host allocator/adapters | 全库 allocator injection、STL/PMR | direct system + borrowed runtime | 完整但 borrowed 命名含糊 | `PARTIAL` | `FIX` public ownership shape；保留 direct fast path |
| isolated growable heap | individual free + bulk release/trim | single-owner owned heap + borrowed allocator ref | PMR-only route 不可回注 HYLE consumers | `PARTIAL` | `FIX` owned Heap over mature mechanism |
| arena/pool taxonomy | host scratch/block/object journeys | owned/external/borrowed；single/concurrent；hard/soft/growable | 机制完整，命名和失败形状不统一 | `PARTIAL` | `FIX` taxonomy/fallible construction；不重写已正确算法 |
| mapping/residency | model mmap/mlock/lazy range + VM/NUMA/shared map | anonymous/file/named、range policy、sparse release | 分散 Platform/Mem/IO | `PARTIAL` | `FIX` Mem owners + platform/io mechanism adapters |
| cross-domain resource | host/pinned/device/managed/custom provider | property query、identity、alignment/max/capacity | Mem 缺失、Compute 二值类型 | `MISSING` | `FIX` open resource/ref contract |
| Buffer owner/view/import/composite | weights/KV/output/transient scratch | owner、const/mutable view、subview、import、multi-allocation | Compute partial | `PARTIAL` | `FIX` Mem Buffer family，迁移 Compute provider |
| copy/coherence/async seam | upload/download/copy/fill/map/flush/invalidate | sync Mem operations + typed async provider extension | Compute-only partial | `PARTIAL` | `FIX` ownership seam；Stream/Event 仍在 Compute |
| resource pool/stats | transient reuse、pressure/trim | reuse/LIFO policy、threshold、capacity stats | host/Compute 各自实现 | `PARTIAL` | `FIX` generic semantics；保留 backend mechanism |
| lifetime packing | worst-case reserve、multi-resource reuse | intervals、alignment、alias/persistent/output、measure | missing | `MISSING` | `FIX` tensor-free planner + Graph adapter boundary |
| KV/tensor/model policy | engine business state | model-specific | absent | `OUT_OF_SCOPE` | `REJECT` from Mem |

## Capability / gap ledger 与冻结 FIX 集

| ID | class | source | normal shape | owner / boundary | complexity/resource target | baseline | action | verification |
|---|---|---|---|---|---|---|---|---|
| `MEM2-01` | MUST | replay + user decision | one coherent namespace and dependency direction | Mem semantics / provider mechanisms | headers/targets remain acyclic | host-only design | `FIX` architecture + public taxonomy | include graph + compile-only API journeys |
| `MEM2-02` | MUST | mimalloc + oneTBB + HYLE consumers | host AllocatorRef/SystemAllocator + owned Heap + standard adapters | Mem | direct route unchanged; heap alloc expected O(1), clear proportional to retained pages | PMR-only owned pool | `FIX` allocator/heap/arena/pool shape | correctness/OOM/remote-free + same-contract P1 |
| `MEM2-03` | MUST | llama + Folly + Boost | anonymous/file/named mapping with range residency | Mem owner / Platform+IO mechanism | O(1) owner ops; range syscall O(covered OS ranges), no payload copy | split owners | `FIX` mapping family | three OS paths, page rounding/failure/lifetime |
| `MEM2-04` | MUST | ggml + CCCL | property-bearing resource/ref and Buffer owner/view/import | Mem / Compute provider | ref/view O(1), zero heap/copy; owner no pimpl allocation | Compute partial | `FIX` resource/buffer core | synthetic provider differential + real available backends |
| `MEM2-05` | MUST | llama/ggml | host, pinned, device, managed/unified and external provider routes | Mem contract / Compute artifacts | provider call count unchanged; no hidden staging | device+pinned only | `FIX` providers and consumer migration | CPU/CUDA/Vulkan where available; exact skip otherwise |
| `MEM2-06` | MUST | CCCL + ggml pools + mimalloc | pooled resource/buffer with reuse, threshold, trim, capacity stats | Mem semantics / provider artifact | common reuse O(1) expected; bounded search policy explicit; retained bytes observable | split pools | `FIX` pool/stats | retention/pressure/fragmentation + P1 normal transient replay |
| `MEM2-07` | MUST | ggml gallocr + oneDNN planner | tensor-free lifetime requests -> placements -> reserve/instantiate | Mem planner / Graph facts | O((N+E) log N), O(N) scratch; no per-byte table | missing | `FIX` planner | property/differential, peak bytes, scaling structure |
| `MEM2-08` | MUST | direct consumers + build contract | migrate Platform/Compute/IO/Graph and source/static/shared | owning modules | no cycle/duplicate owner/compat alias | fragmented | `FIX` consumers/build/package | selected components + installed journeys |
| `MEM2-09` | MUST | closure method | guide/architecture/performance/red-team/LOC/registry | respective doc owners | no duplicated point claims or stale host-only text | stale after redesign | `FIX` closure evidence | check-docs/check-closure + independent red team |

冻结规则：上述九票是本轮全部 `FIX` 集。新增 API 只有在某票的 normal journey、ownership、failure、
concurrency 和 cost contract 内才允许；otherwise 退回 `AUDITED` 重新裁决。旧 API 不兼容不是限制：HYLE
尚未发布，不保留只为兼容而存在的 alias、bridge 或 deprecated header。

### 2026-09-02 增量实现记录（非 terminal verdict）

- `MEM2-01/MEM2-04` 的第一段可运行基础已落地：一字 borrowed `MemoryResourceRef`、开放
  property/domain provider、四字 move-only `Buffer` 与四字 mutable/const view、显式
  `HostMapping`、sync read/write/fill/copy/flush/invalidate，以及 process-lifetime
  `system_resource()`；不是 stub，也不泄漏 backend/Device/Stream/Event 类型。
- hot view operation 为 header-inline，source/static consumer 可继续内联；shared consumer 只有一次所请求的
  provider virtual dispatch，不额外穿越一层 `hyle_mem` DLL wrapper。generic 跨 resource copy 只在两端
  可直接 host-map 时执行，不创建 hidden staging。
- Windows MSVC shared/static 与 WSL GCC 14 shared 的 Mem header/test 通过；WSL GCC 14 ASan/UBSan
  通过。synthetic non-coherent provider 只证明 provider contract/correctness，不冒充 CUDA/Vulkan 硬件
  verdict；macOS 与真实 backend 尚未运行。
- 本记录不完成 `MEM2-01` 或 `MEM2-04`：`CompositeBuffer`、完整 taxonomy 的其余 header、五条
  compile-only journey、Compute provider/consumer migration 和 source/static/shared installed journey
  仍按后续有限票处理。registry 保持 `AUDITED`，API freeze 保持 `DRAFT`。
- `MEM2-02` 的 isolated owned-heap 机制已落地：一字 move-only `Heap` 直接复用 mimalloc
  per-heap，支持 owner-thread allocate/reallocate、remote free、collect，以及会显式失效全部 live
  allocation 的 bulk `clear/release`。allocator borrow 绑定 owner object，因此 successful `clear()` 后
  仍可用且不需要额外 control-block allocation；move/destruction 明确使 borrow 失效。Windows MSVC
  shared/static 和 WSL GCC 14 的 Mem header/test 通过。该票仍需完成 `AllocatorRef` 与 arena/pool
  taxonomy 迁移和固定接口后的同合同 P1，不能据此标成完成。
- `MEM2-03` 的 owner/mechanism 主干已落地：Mem `VirtualMapping/FileMapping/SharedMapping` 分别
  组合 Platform `VmRegion/MappedFile/SharedMemoryRegion`，匿名 mapping 新增通用 advice 与
  eager/on-fault lock mechanism；FileMapping 提供 RO/RW/COW、range advice/flush、渐进 lock 与直接
  `ConstBufferView` journey；SharedMapping 的 create/open/unlink、共享可见性和 flush 在 Windows/Linux
  通过。旧 `io::SharedMemory` owner 已删除，IO 指南/样例/CI 改走 Mem。该票仍需 macOS、完整
  Platform 文档迁移、sparse/residency edge 与最终 package journey 后才能完成。
- `MEM2-06` 的通用 transient pool 已落地：一字 move-only `BufferPool`、RAII `BufferLease`、显式
  take/reset、retention/slack/growth/bounded-scan options，以及 live/cached/peak/allocation/reuse/OOM-retry
  stats。实现使用 `sizeof(size_t) * 8 + 1` 个 size buckets（当前 64-bit 平台为 65 个）、每桶有界
  best-fit，常见复用为一次 mutex + O(1) bucket/有限
  inspection；首次 allocation 才创建一块 Entry metadata，复用不再分配 wrapper/provider。provider OOM
  只做一次释放 cache 后重试；pool 不包含 Event/Stream，completion 前归还是 caller contract。
  Windows MSVC shared/static、WSL GCC 14 shared 与 WSL GCC 14 ASan correctness/并发 accounting
  已通过；仍需 macOS、固定接口 P1 normal transient replay 和 Compute provider journey 后才能完成该票。
- `MEM2-07` 的 tensor-free planner 主干已落地：dense request/resource IDs、`[begin,end)`、任意方向
  acyclic alias chain/offset、persistent/output/no-reuse facts、不可变 `LifetimePlan`，以及每 resource
  一次 backing allocation 的独立 `PlannedBuffers`。build 用 active min-heap、按 capacity/alignment
  coordinate 压缩的 segment tree 和 reusable fixed slots，O((N+E) log N)、O(N) scratch，不按
  byte/page 建表；best-fit 是确定性通用 heuristic，不冒充全局最优。Windows MSVC shared/static、
  WSL GCC 14 shared 与 WSL GCC 14 ASan 的 correctness、随机 interval differential、
  alias/overflow/failure、10k reverse-order deterministic scaling 与 self-contained header 已通过；
  P0 cost review 确认 scratch 是与 N 成比例的固定数量 arrays、sort 原地、没有 per-request
  allocation 或 per-byte/page table，plan/instance 各只创建固定数量 metadata blocks 加每 resource
  一次 provider allocation。仍需 macOS 和 Graph adapter journey 后才能完成该票。
- `MEM2-04` 的 multi-allocation owner/view 已落地：一字 move-only `CompositeBuffer` 只在 opt-in
  shape 分配 segment/prefix metadata，普通四字 `Buffer/View` 不增加常态字段；factory 在移动输入前
  完成全量 validation 与 metadata allocation，失败时输入保持不变。logical subview 为 O(1)，
  piece discovery 为 O(log S)，跨段 read/write/fill/copy 为 O(log S + touched segments)，保留每段
  resource/native identity，不 flatten、不分配 staging。Windows MSVC shared/static、WSL GCC 14 shared
  correctness/header journey 与 WSL GCC 14 ASan correctness 已通过；macOS 与真实 Compute provider
  journey 仍按本轮后续验证，故该票不在此处提前完成。
- `MEM2-05/MEM2-08` 的 storage seam 已落地：删除 Compute 私有 `BufferType/Buffer/BufferKind`
  产品面，Compute 以 process-lifetime `MemoryResource` provider 提供 device、host/pinned 与可选
  managed storage，kernel typed shell 直接接收 Mem owner/view；raw backend binding 显式保留
  `{native_handle,offset,size}`。CUDA device/pinned/managed、Vulkan device/pinned（managed 明确
  `Unimplemented`）和 Host 三类路径均通过真实运行；CPU/CUDA/Vulkan 的非零 subview kernel
  验证 offset/range 没有被中间层丢失。Windows MSVC Host/CUDA/Vulkan 与 WSL GCC 14 Host
  correctness/header 通过。Compute/Graph 其余半成品能力未借本票重构；macOS、最终 selected
  package journeys 与相邻文档仍留在本轮后续票。
- `MEM2-01/MEM2-02` 的 public taxonomy 已完成源码迁移：旧 `Allocator/Region/FixedRegion/
  PooledRegion/StackAllocator/FixedPool/SharedPool/SharedPoolCache/GrowablePool/HugeRegion` 类型全部
  删除且不留 alias；最终类型经 `arena.h` / `pool.h` façade 暴露，全库直接消费者、样例、测试和
  benchmark source 一次迁移。五条 normal journey 分别由 owned Heap 注入 PodVector、只读模型
  FileMapping、Compute resource + Mem nonzero subview kernel、BufferPool lease 与 LifetimePlan
  instantiate 的 public-API 测试覆盖。Windows MSVC `all` 70/70、WSL GCC 14 selected
  mem/container/compute 48/48 通过；现行 architecture/user 文档只使用新名称。随后 Windows
  MSVC 与 WSL GCC 14 的 `mem` selected product smoke 均通过 source embedding、relocated shared
  SDK 和 relocated static SDK 三形状，安装包消费者实际运行 Heap、FileMapping、BufferPool 与
  LifetimePlan，故 API freeze 已进入 `FROZEN`。
- selected-package 扩展旅程在 WSL 暴露了 `mi_process_info` 小进程采样中 `peak_rss < current_rss`
  的真实缺陷（6.0 MiB 对 7.5 MiB）。`system_memory_snapshot()` 现于 API 边界把 process peak
  单调归一到至少 current，冷启动与 8 MiB 增长路径都检查不变量；不是放宽样例。Windows/WSL
  source/shared/static consumer 随后通过。
- WSL GCC 14 的 ASan 与 UBSan 均通过 Mem correctness 和 self-contained public headers。GCC TSan
  在普通 WSL ASLR 下于进入 `main` 前报已知的 `unexpected memory mapping`；按 Linux setup 的既定
  运行合同改用 `setarch x86_64 -R` 后，完整 `hyle_mem_test` **242/242 PASS**（含 concurrent pool、
  remote return 与 stats 路径），没有 race 报告。这里记录的是运行器前提，不把 runtime 启动失败
  写成产品 PASS，也没有为消毒器修改产品同步语义。

### 相邻模块 backlog，而非 Mem 产品范围

- Compute：Device/Stream/Event、async copy ordering、provider-specific queue allocation、kernel in-flight lifetime；
- Graph：从 DAG/tensor view/in-place rule 生成 `MEM2-07` 的 request；
- Engine：weights/KV/output placement、paging、offload、eviction、defrag；
- IO：async file/network completion 与 registered-buffer protocol；
- Native FFI：没有已准入 foreign-runtime consumer，保持 `N/A`。

## Owner 六问终态答案

### 1. 第三方产品路径

- Mem 的唯一直接第三方实现依赖是锁定的 mimalloc v3.5.0，用于 system/Heap backing；它由构建系统
  统一登记并静态编入 Mem，不出现在 HYLE public type 或调用合同中。
- oneTBB、标准 PMR、llama.cpp、ggml、Folly、Boost、CCCL 与 oneDNN 只提供 peer、consumer 或算法证据。
  CUDA/Vulkan provider 由 Compute target 实现并向 Mem contract 适配；Mem 不反向链接 Compute，也不
  泄漏 `mi_heap_t`、CUDA/Vulkan/ggml/CCCL 类型。

### 2. 推理引擎能力完整度

- `MEM2-01`～`MEM2-09` 九项 `MUST` 均已 `COMPLETE` 或由明确 owner `ROUTED`：host allocator/Heap、
  arena/pool、匿名/文件/共享 mapping 与 residency、开放 resource/ref、Buffer owner/view/import/composite、
  同步 copy/coherence、transient BufferPool/stats 和 tensor-free lifetime packing 都有自然公开路线。
- Graph 当前只有 task/data-parallel facts，没有第二套 storage owner 或 planner，因此没有可迁移代码；未来
  DAG 到 `LifetimeRequest` 的推导归 Graph。Tensor/KV/model placement、eviction、async queue/completion 和
  backend kernel 继续归 Engine/Graph/Compute/IO，不是以“交给别处”掩盖 Mem 缺口。

### 3. 拟真、公平性能

- P0 证明常态结构与复杂度：view/subview O(1) 且不分配；Composite 首次 piece discovery O(log S)、
  顺序遍历 touched segments；planner 为 O((N+E) log N)/O(N) scratch；BufferPool 为固定 bucket 选择加
  每桶有界 scan；mapping 只转发所请求 OS range operation，不搬 payload 或隐藏 fallback。
- P1 使用四个 normal consumer same-contract 形状。Heap 与 BufferView 八个 strata 全为 `PARITY`，
  FixedArena 与 BufferPool 八个 strata 全为 `WIN`；全部通过 5% 非劣门和资格 controls。结论不外推为
  “全部 Mem 操作/平台 SOTA”。

### 4. 旧 C ABI / Native FFI

- `NOT_APPLICABLE`。当前产品只承诺 C++23 source/static/shared；Mem 没有历史 C schema、bridge、option 或
  动态 C symbol。未来只有真实 foreign-runtime consumer 经运行时类型擦除准入后才另开 FFI slice。

### 5. Source / static / shared

- Windows、WSL 与 macOS 都从候选源完成 source embedding、relocated shared SDK 和 relocated static SDK
  consumer；sample 实际运行 Heap、FileMapping、BufferPool 和 LifetimePlan，不是只检查 version symbol。
- Mem header 自包含、中央 component closure、Windows import/export 与 Unix loader 均由这些旅程覆盖。

### 6. C++ API 易用性与文档

- public 类型按 ownership/domain/lifetime/concurrency 命名，没有 benchmark 名称、`fast/light` 档或 backend
  私有 handle 迫使用户选型；普通 host、model mapping、persistent/device buffer、transient pool 与 planner
  均有短的 compile/run recipe。
- architecture 与 guide 已区分 Platform 的低层 OS mechanism owner 和 Mem 的统一 storage owner，说明
  borrow/provider lifetime、view invalidation、pool completion 前提、mapping 页粒度与 unsupported 路径。

## Candidate

- 首个候选 `a7d7297e71d28214cf4b4c1d03665703c9f78822` 在独立 red-team P0 中被拒绝：
  `CompositeBuffer` 的跨段 read/write/fill/copy 每段重复二分，实际为
  O(touched segments * log S)，不满足公开的 O(log S + touched segments) 合同。
- 最终产品 candidate / tree：`9eda54f9b36bfb0792ecf1544dc39d7ac9682265` /
  `359c4832a91a498bb92e49b80766752a4ca9d9c6`。它只在首次 range lookup 二分，随后按 prefix 顺序遍历；
  没有增加 API、layout、常态 metadata 或 benchmark-only branch。
- candidate 后的 task、性能证据与 registry 维护不改 `include/hyle/mem`、`src/mem` 或直接 provider 产品
  实现。Mac 使用该 commit 的确定性 Git archive；Windows 与 Mac 两端 archive SHA-256 同为
  `230277064d2b516e5110a4f6b33b554483372880d2caa15ab260dcd7a09bbb68`。

## Verification manifest

| Gate | Verdict | 证据与边界 |
|---|---|---|
| scope / capability | `PASS` | 九项冻结 `MUST` 全部兑现；相邻 policy/async/backend-kernel 均有明确 owner，没有 `PARTIAL + KEEP` |
| architecture / API | `PASS` | `compute | graph | io -> mem -> platform` 无环；storage owner、OS mechanism 和 typed async seam 分工清楚；无旧 alias、重复 owner 或 benchmark-shaped public type |
| correctness / sanitizer | `PASS` | W1-WIN MSVC 19.50 Release 2/2（Mem 内部 242 tests）；W1-WSL GCC 14.2 Release、ASan、UBSan 各自 header+Mem PASS，`setarch x86_64 -R` TSan 242/242 且无 race；M1-MAC AppleClang 21 Release 2/2 |
| real provider | `PASS` | 当前候选在 Windows 真实启用 CUDA 13.3 与 Vulkan 1.4.357：Compute 78 PASS/2 有明确非本票 skip，Vulkan 16/16；Host/CUDA/Vulkan 的 resource、Buffer、nonzero subview 与 native offset/range route 均运行，不用 synthetic 冒充硬件 |
| delivery | `PASS` | Windows、WSL、macOS 的 selected Mem smoke 均通过 source、relocated shared、relocated static consumer；三种形状调用真实 owner/pool/planner 路线 |
| usability / docs | `PASS` | architecture/user guide/选择表/recipes/陷阱、当前 P1 专页与脚本入口一致；Native FFI 明确 N/A |
| red team | `PASS` | 下表 finding 均已修复、证伪或限界；没有 correctness、security、ownership、delivery、usability、复杂度或 material-performance blocker |
| engineering performance | `PASS` | `mem-v2-consumer-v1` 在 W1-WIN、5% NI、n=15、八 strata 下四项全部 PASS；A/A、5/10% delta、timer floor、placement 与 resource oracle 全 PASS，deep verifier `QUALIFIED` |

P1 retained bundle 位于
`C:\public\project-hyle\_validation\mem-v2-perf-evidence-9eda54f9`。contract SHA-256 为
`56083fe6817c74e010194aa96dae9f1f36d36698c4b5dd5206fd3d9ab31fd281`，语义矩阵 SHA-256 为
`85959bab834bf483f5b7b9506fcf6a7af79074819410be8ded78a1cd28edc4a8`，manifest SHA-256 为
`66919c54c9c61f696551c97c079e720eb6da8e89886bc6b8ecd3ad7a3588b983`，summary SHA-256 为
`aa43e2ec0be62563dbfea1fb98df1834b9c39783a4e8f5755b9cdf5dc535f7bc`；verifier 报
candidate eligible、environment current、paths verified、exact provenance 和 `QUALIFIED`。

| workload | eight-strata result | advantage ratio | candidate throughput |
|---|---|---:|---:|
| isolated Heap lifecycle / equal mimalloc PMR owner | 8/8 `PARITY`, NI PASS | 0.999--1.029 | 91.3--94.0 M op/s |
| FixedArena epoch / equal external PMR monotonic | 8/8 `WIN`, NI PASS | 1.196--1.223 | 546.9--552.3 M op/s |
| checked BufferView range fill / equal fused lower bound | 8/8 `PARITY`, NI PASS | 0.999--1.007 | 22.83--22.95 M op/s |
| BufferPool lifecycle / oneTBB memory_pool + charged semantic adapter | 8/8 `WIN`, NI PASS | 1.382--1.400 | 28.54--28.73 M op/s |

### Final red-team

| finding | severity | disposition / evidence |
|---|---|---|
| 首候选 Composite range operation 重复二分，复杂度合同不成立 | material complexity | `FIXED` at `9eda54f9`：一次二分后顺序 piece traversal；focused + 全 Mem 三平台复证 |
| SharedMapping 曾把 OS mapping capacity 当 payload size 暴露 | correctness/range | `FIXED` at `38ed316a`：owner 保存并传播 logical payload size，page-rounded backing 不泄漏为可访问 payload |
| SharedMapping read helper 曾吞掉源 range failure | error integrity | `FIXED` at `ee7277c8`：失败原样返回共享 Status，测试覆盖越界路径 |
| Platform 仍公开 low-level mapping/VM owner，看似与 Mem 重复 | architecture ambiguity | `BOUNDED`：Platform 是可直接使用的 OS mechanism；Mem 组合它们提供统一 storage/resource/view semantics，文档给出选择规则 |
| Graph 尚无 LifetimePlanner adapter | migration risk | `ROUTED`：当前 Graph 源没有 storage owner/planner 可迁移；未来 DAG/tensor facts 到 `LifetimeRequest` 的推导属于 Graph，不能反向塞入 Mem |
| BufferPool 常见复用是否偷偷成为线性搜索 | material hot-path | `DISPROVED`：固定 65 buckets、每桶默认最多 8 次 inspection；cold trim 才与 cached buffers 成比例，合同和 stats 可见 |
| CUDA/Vulkan seam 可能只由 synthetic provider 通过 | false hardware PASS | `DISPROVED`：最终候选在 RTX 路径真实构建并运行 CUDA/Vulkan resource、Buffer/subview tests；synthetic 只补 non-coherent differential |
| 为 Java/C# 预留旧 C ABI 会扩大未消费 surface | maintenance/API | `REJECTED`：本轮无 C header/schema/bridge/option/symbol，只有未来具名 foreign consumer 才可重开类型擦除 FFI |

candidate 的 canonical LOC self-check（`python scripts/loc-report.py --module mem --commit 9eda54f9
--self-check --json`）为：全库代码 **432,497**；自实现包含测试 **109,610**；自实现不包含测试
**69,085**；Mem 实现 **7,653** 行。

## Closure decision（待 registry 完成合法状态迁移）

验证事实已满足 `CLOSED` 条件；registry 仍须按 `AUDITED -> CANDIDATE -> VERIFIED -> CLOSED` 保存明确
快照。在该合法迁移完成并写入 terminal receipt 前，本节不提前宣告产品状态。

- 可接受限制：P1 只签四个冻结 W1-WIN same-contract workload；mapping/planner/Composite 依 P0
  complexity/resource 证据；不签多 socket NUMA、端到端 model load/tokens/s、cross-backend performance、
  KV/tensor/model policy、async queue/completion、Native FFI、稳定 C++ ABI 或 Formal。
- reopen triggers：
  1. 具名 consumer 复现 Mem ownership、lifetime、range、coherence、error、concurrency、OOM 或易用性缺陷；
  2. public API/layout/算法、mimalloc/direct dependency、provider seam 或 source/static/shared 交付变化；
  3. 成熟 same-contract peer 给出更好复杂度档，或正常 consumer workload 上可复现超过 5% 的 material 劣势；
  4. Graph adapter、multi-node locality、backend async、KV/model policy 或 foreign-runtime slice 经独立能力决策准入；
  5. performance contract/peer/provenance/machine/instrument 或 retained bundle 改变且当前声明需要续签。
