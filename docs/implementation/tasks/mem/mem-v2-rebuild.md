# Mem v2：统一 storage foundation 重构

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前轮由 [`mem-v1 capability replay`](mem-v1-capability-replay-2026-09.md) 的
> `PRODUCT_TRIGGER` 启动。旧 `mem-v1` task/evidence 只作 baseline 证据，不是本轮待办。
>
> 项目级阶段门：[`全库能力展开与组件因子化 v1`](../project/capability-factorization-v1.md) 已完成，随后
> [`build-system-v7 factor/façade projection`](../build/build-system-v7-factor-facade-projection.md) 已把内部
> factor DAG 与外部产品 façade 分开。中央 registry 当前把本轮登记为 `AUDITED`，因此只授权下述
> `MEM2-01`～`MEM2-09` 有限 FIX；历史 factor 清单或 bundle 文字若与 v7 投影冲突，不再构成施工合同。

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

## API / usability freeze（DRAFT）

本节是 `MEM2-01` 的唯一 public-shape 草案。`DRAFT` 期间允许用 compile-only journey
发现类型关系不成立；只有下列全部被真实头文件表达、五条 journey 可编译且
include graph 无环时，才可改为 `FROZEN`。后续 correctness 不得悄悄改名或恢复旧 alias。

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
| `heap.h` | `Heap`, `HeapOptions` | move-only owned growable host heap；owner-thread allocate/reallocate，允许 remote free；全量 release 要求无 live allocation |
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
| allocator / heap / arena / pool | 冷 owner 构造返回 `Result<Owner>`；热 byte/slot allocation 返 null/empty loan | 按具名类型区分 owner-only 与 concurrent；`Heap` 只承诺 remote free | 无隐式 fallback；reset/release 显式使旧 allocation/view 失效 |
| mapping | factory/range OS failure 返 `Result/Status` 并保留 native error | control operation 外部同步；普通字节访问由保护和宿主同步决定 | 不自动 prefault/lock/grow file；未支持 advice 明确 `Unimplemented` |
| resource / Buffer | provider/ref 非空、range/alignment 错误为 `InvalidArgument/OutOfRange`；OOM 为 `ResourceExhausted` | provider 必须支持独立 allocation 并发使用；同一 Buffer 的 mutation/map/control 外部同步 | `Buffer{}` 可作 invalid/moved-from；没有默认 resource，必须显式选 host/device/custom provider |
| BufferPool | create/acquire 返 `Result`；capacity/pressure 可观察 | pool 是 thread-safe；lease 中 Buffer 的使用仍由 caller 同步 | lease 归还前必须无 in-flight operation；trim 不等待 Event |
| lifetime plan | 溢出、非 2 次幂 alignment、非法 interval/alias/resource 返 typed Status | build/instantiate 外部同步；plan 不可变时可并发查询 | interval 为 `[begin,end)`；persistent/output 不复用；plan 不拥有 Tensor/Graph |

### 五条 compile-only normal journey

1. **host heap：**`Heap::create(options) -> Heap`，把 `heap.allocator()` 的 `AllocatorRef`
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

- [ ] 上表类型关系全部落到 self-contained public headers，不泄漏 backend 类型；
- [ ] 五条 journey 以正常 public API 编译，不借 test-only constructor/stub；
- [ ] `sizeof(MemoryResourceRef) == 1 word`、`Buffer/View <= 4 words`、owner 无 wrapper pimpl allocation；
- [ ] 头文件语法 include graph 与 CMake target graph 都满足上述无环方向；
- [ ] source/static/shared selected-component 均可发现同一 API；
- [ ] `docs/user/mem.md` 在实现可用前仍只写当前产品；待 migration 后一次替换为上述选型表，不提前宣传。

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

### 相邻模块 backlog，而非 Mem 产品范围

- Compute：Device/Stream/Event、async copy ordering、provider-specific queue allocation、kernel in-flight lifetime；
- Graph：从 DAG/tensor view/in-place rule 生成 `MEM2-07` 的 request；
- Engine：weights/KV/output placement、paging、offload、eviction、defrag；
- IO：async file/network completion 与 registered-buffer protocol；
- Native FFI：没有已准入 foreign-runtime consumer，保持 `N/A`。

## Owner 六问当前审计答案

### 1. 第三方产品路径

- system/Heap mechanism 允许复用锁定 mimalloc；其他 peer 只作 shape/algorithm evidence；
- public HYLE types 不泄漏 `mi_heap_t`、CUDA/Vulkan/ggml/CCCL types；
- provider implementation 可由 Compute target 编译，Mem target 不反向链接 Compute。

### 2. 推理引擎能力完整度

- 当前存在未处置 MUST：`YES`，恰为 `MEM2-01`～`MEM2-09`；
- 完成定义：llama/ggml 的 admitted storage journeys 均有自然 Mem route，不能只写“交给 Compute/Graph”；
- 不承诺模型/KV policy，也不按 ggml API 名字逐函数复制。

### 3. 拟真、公平性能

- 当前只有 v1 host 两份限定 P1 证据；它们不能证明 v2 resource/buffer/mapping/planner；
- candidate 前不发布新 SOTA 结论；每票按 closure card 的 comparison class 和预算验证。

### 4. 旧 C ABI / Native FFI

- 当前 policy 为 `N/A`；不得恢复旧 C schema/exports，也不为可能的 Java/C# consumer 预做 bridge。

### 5. Source / static / shared

- baseline mem-v1 三形状曾通过；v2 headers、exports、provider seams 与相邻 component graph 改变后必须全部
  重跑，旧 receipt 不能外推。

### 6. C++ API 易用性与文档

- baseline host guide 很详细，但“host only、device/mapper 留相邻模块”已经过时；
- v2 必须提供按 storage domain、ownership、lifetime、concurrency、failure 选型的一页表，以及模型加载、
  persistent buffer、transient pool、planner 四条可编译 recipe；
- 任何需要用户先理解 benchmark 名称、backend 私有 handle 或多 namespace 拼装 owner 的路线都判 `FAIL`。

## Candidate

尚未进入。产品代码只能按 `MEM2-01`～`MEM2-09` 顺序或显式无冲突的 ticket 边界施工；每票完成后记录
公开合同、复杂度、直接消费者和适用测试，不能把未完成票写成 PASS。

## Verification manifest

尚未进入。最终至少运行本卡预声明的三平台适用 correctness、sanitizer、selected component、source/static/
shared package journey，以及改变的 P1 family；真实 backend 缺失必须写 `NOT_APPLICABLE`/`BLOCKED` 的准确
原因，不能用 synthetic 路径冒充。
