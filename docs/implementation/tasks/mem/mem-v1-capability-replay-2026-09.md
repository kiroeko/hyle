# Mem v1 能力与语义所有权回放 — 2026-09

<!-- READ-ONLY CAPABILITY REPLAY: COMPLETE -->

> 本页是 product-non-mutating replay，只裁决冻结的 `mem-v1` 是否仍满足当前产品需求。它不把
> peer API 机械求并集，也不授权在旧 task 上继续施工。当前状态只看
> [`module-status.json`](../../module-status.json)。

## 回放合同

- reviewed round / candidate / end：`mem-v1` /
  `2c71dfae9f4fb43ff0d68e5c1a4c55099fc0f101` /
  `89841ab006433bceac39ba8083b01c634a641404`；
- replay baseline：HYLE `e7b3a5f2f6bc90b194821c1777542c3c47a27c7b`；
- demand / trigger：`MEM-OWNER-2026-09-01`。HYLE 的北极星消费者已明确为未来推理引擎；存储
  resource、placement、ownership、view、mapping、residency、pool、capacity 与 lifetime packing
  的语义所有者统一为 `mem`，不能再按“是否由 GPU/OS 实现”拆给 `compute`、`platform` 和 `io`；
- normal journeys：模型权重映射/锁页/按需释放，host-pinned-device-managed storage 选择，持久
  权重/KV/output 与 transient compute buffer，subview/import/copy/coherence，worst-case reserve 与
  lifetime reuse，以及普通 C++ host allocator/arena/pool；
- replay budget：只做公开面、真实 consumer 和成熟实现的静态差异审计；不新增 benchmark，不修改
  产品代码；
- reviewer / date：Codex / 2026-09-01。

锁定输入：

| source | clean identity | 本轮用途 |
|---|---|---|
| llama.cpp | `e4b9af007beae34abba094cefd187658e89bbac8`；vendored ggml tree `4660c26f69a7c1ab6c8309934214bde5622483ea` | 模型 mmap/mlock、lazy range、partial unmap、pinned upload、buffer placement、capacity 与 reserve |
| standalone ggml | `d4716378882593333721eb33f153144b6885caf2` | buffer type/owner、import、multi-buffer、sync/async transfer、tallocr/gallocr |
| CCCL | `b7aaea69a2b07e50f09e67f2962da0243e0b7c5d` | property-bearing resource/ref、device/pinned/managed pool、buffer 与 async resource |
| oneTBB | `a8216b95202f3c55f9749870f4f8cfa4df0b32e3` | scalable/cache-aligned allocator、owned/external variable-size pool |
| Folly | `23ebefdf96c116700d3e134f04d95a2b60a509de` | Arena/ThreadCachedArena、MemoryMapping、guard/huge-page ownership |
| oneDNN | `5dc67b7f37b1215d8c1f054e99c1b7cfd40317a9` | independent graph lifetime/in-place/persistent/transient memory planning |
| Boost | `78505eab9b203e3e26e83893d0685f160133fa0f` | Interprocess mapped storage、Compute buffer/map/copy、PMR/pool variants |
| HYLE product mechanism | mimalloc v3.5.0 / `18b08671c9302247bfb682286e6bf3cc1773f801` | system heap、isolated heap/arena、collect/stats/NUMA mechanism |

## 语义所有权裁决

是否 host-addressable 不是模块边界。新的边界是：

```text
mem      storage resource + allocation ownership + view + placement/property
         mapping/residency/protection + pooling/trim/stats + lifetime packing
platform OS syscall、page/NUMA/file-mapping mechanism
compute  device/backend provider、stream/event、async execution ordering、kernel
graph    从 tensor DAG 推导 lifetime/alias/in-place 约束
io       文件/网络传输与 completion；不拥有 named mapped storage 类型
engine   tensor/KV/model placement、eviction、offload 与 batching policy
```

因此：

- 当前 `compute::BufferType` / `compute::Buffer` / `compute::DevicePool` 的 storage resource、owner、
  view、coherence 与 pool 语义归 `mem`；CUDA/Vulkan allocate/free/VMM 实现和 stream/event operation
  仍由 compute provider 承担；
- 当前 `platform::MappedFile` / `platform::VmRegion` 的用户级 owner、range 与 residency policy 归
  `mem`；platform 只保留跨 OS mechanism；
- 当前 `io::SharedMemory` 的 named mapped-storage owner 归 `mem`；IPC protocol 与异步 IO 仍归 io；
- generic lifetime/offset packing 归 `mem`；Graph 只负责把 tensor DAG 翻译成 resource、size、alignment、
  live interval、alias/in-place 与 persistent/output 约束。

这个方向保持无环：`compute/graph/io -> mem -> platform`。Mem 不包含 Device、Stream、Event、Tensor、
KV 或文件格式类型；provider/adapter 从上层向 Mem 注入实现。

## Capability-family coverage map

| Family | admitted user journeys / evidence | required semantic variants | current route | coverage | missing edge / negative space |
|---|---|---|---|---|---|
| host byte allocation | 普通容器、HYLE 全库注入；mimalloc/oneTBB/PMR | direct system、borrowed runtime、aligned/realloc、STL/PMR | `SystemAllocator`、`Allocator`、std/PMR adapters | `COMPLETE` | borrowed ownership 名称和默认 usability 仍需形状复核，不因计时改 layout |
| isolated growable heap | oneTBB `memory_pool`、mimalloc heap、PMR pool；individual free + bulk release | owned、allocator injection、collect/trim、thread ownership | 只能绕到标准 PMR pool | `PARTIAL` | 标准 PMR resource 不能自然回注 HYLE `Allocator` consumer；缺 HYLE owner/ref 路线 |
| host arenas/scratch | ggml context/tallocr、Folly Arena、普通 request/frame | owned/external/borrowed、monotonic/LIFO、limit/clear/release | `Region`、`FixedRegion`、`PooledRegion`、`StackAllocator` | `COMPLETE` | device/non-host buffer 上的 linear suballocation 尚未覆盖 |
| fixed/typed pools | oneTBB/marl/Taskflow；固定块、typed churn、preserve/loan | hard/soft、single-owner/concurrent、fixed/growable | `FixedPool`、`SharedPool`/cache、Object/Growable/Preserving pool | `COMPLETE` | 名称没有直接表达 block-size/storage/concurrency；fallible constructor 形状待审 |
| quota/stats/pressure | 长寿进程 admission、leak diagnostics、mimalloc stats | owner-local exact、process/resource capacity、trim | quota/stats/system snapshot/collect | `PARTIAL` | 只有 system host domain；没有统一 resource free/total/reserved/used/trim |
| virtual/anonymous mapped memory | guarded stack、大 reservation、commit/decommit/protect、NUMA | reserve/commit/protect/advice/guard/placement | `platform::VmRegion` + `mem::HugeRegion` | `PARTIAL` | 两个 owner 重叠且 public owner 错位；policy 组合不统一 |
| file mapping/residency | llama model load、Folly MemoryMapping | subrange、RO/RW/COW、advice/prefetch、incremental lock、range release | `platform::MappedFile` | `PARTIAL` | Mem 无路由；lazy-range batch 与 POSIX sparse fragment release 缺失 |
| named shared mapping | multiprocess shared storage；Boost.Interprocess | create/open/unlink、RO/RW、flush、owner lifetime | `io::SharedMemory` | `PARTIAL` | public owner 错位；未与 mapping/view/property 模型统一 |
| cross-domain resource | ggml buffer type、CCCL `cuda::mr`、Boost.Compute | host/device accessible、pinned/managed/coherent、alignment/max、device identity | Compute 的二值 `BufferKind` | `MISSING` | Mem 没有 resource/ref/property；managed/unified/registered/custom provider 不可表达 |
| buffer owner/view/import | 权重/KV/output/scratch；ggml/CCCL/Boost.Compute | move owner、const/mutable subview、host mapping、adopt/import、composite | `compute::Buffer` | `PARTIAL` | Mem 无 owner；无 BufferView、external import、managed/composite/multi-buffer |
| data movement/coherence | ggml set/get/copy/clear，离散/统一内存 | sync read/write/fill/copy/map/flush/invalidate；async ordered extension | Compute Buffer 同步方法 + Stream 私有路径 | `PARTIAL` | 能力绑死 compute owner；跨 provider/composite/strided 路径和统一 async extension 缺失 |
| resource pooling | transient compute/scratch；CCCL pools、ggml CUDA/SYCL/Vulkan pools | arbitrary/LIFO、reuse、release threshold、trim、stats | `compute::DevicePool`；host 各自 pool | `PARTIAL` | 没有跨 resource 的 `PooledResource/BufferPool` 语义；driver pool 与 product API 混合 |
| lifetime packing/reserve | ggml gallocr、oneDNN memory planner、llama worst-case prompt/token reserve | multi-resource、alignment、persistent/output、alias/in-place、measure/reserve/reuse | 尚无通用路线；Graph 未完成 | `MISSING` | 缺不含 Tensor 的 interval/placement engine；Graph 仍应拥有 DAG 推导 |
| system/device topology | NUMA host placement、discrete/integrated/unified device | domain identity、capacity、access properties、preferred locality | Host NUMA + Compute Device 分裂 | `PARTIAL` | 没有统一 resource identity/query；不应把 topology 变成闭合 enum |
| model/KV policy | llama sequence KV、offload、evict/defrag、tensor dtype/layout | engine-specific | 无 | `OUT_OF_SCOPE` | Mem 只提供 backing/view/pool/planner，不拥有模型状态 |

## 现有公开面反向追踪

当前 Mem 的 17 个公开头可以归为六组：

1. allocator/adapters：`allocator.h`、`system_inline.h`、`std_allocator.h`、`pmr.h`；
2. diagnostics/control：`quota_allocator.h`、`stats.h`、`alignment.h`；
3. arena：`region.h`、`stack_allocator.h`；
4. variable/fixed/typed pool：`fixed_pool.h`、`shared_pool.h`、`pool_cache.h`、`object_pool.h`、
   `growable_pool.h`、`pool_loan.h`；
5. placement：`hugepage.h`；
6. runtime specialization：`fiber_stack.h`。

这些实现并非“没有赢”：host allocator/PMR、bounded external variable pool、hard shared slots、explicit
owner-local cache、typed reconstruct/preserve、quota/stats、arena、guarded stack 和 NUMA/huge span 都有清楚
合同，原则上应保留其机制。问题是产品面只覆盖 host allocation lifecycle，而推理消费者必须跨三套
namespace 拼装同一条 storage lifetime；这不是文档补一句即可解决。

### 自然形状 findings

| ID | finding | verdict |
|---|---|---|
| `MEM-SHAPE-01` | `Allocator` 实际是 borrowed runtime ref，默认 invalid；wrapper 的 `.get()` 与 backing lifetime 需要大量指南解释 | 重构时比较 `AllocatorRef`/resource-ref 形状；不得只为 microbenchmark 改四字布局 |
| `MEM-SHAPE-02` | `FixedPool` 是 external variable-size pool，`SharedPool` 是 concurrent fixed-block pool，`GrowablePool<T>` 是 single-owner object pool；类型名不能直接表达差异 | 公共 taxonomy 必须按 storage ownership、size model、capacity、concurrency 命名或分层 |
| `MEM-SHAPE-03` | 多个 fallible owner 通过构造后 `valid()` 报 OOM，另一些走 `Result::create` | 冷构造统一为自然的 `Result<Owner>`；热 allocation 保留 null/value failure |
| `MEM-SHAPE-04` | `HugeRegion` 与 Platform `VmRegion` 是重叠 owner，MappedFile/SharedMemory 又各自造一套 range/lifetime | 收敛到 Mem mapping/region family，OS/IPC mechanism 留原 owner |
| `MEM-SHAPE-05` | Compute Buffer 缺 view/import/composite/managed；BufferType 用闭合 kind enum 表达开放 provider 属性 | 改为 property-bearing resource/ref + owner/view，不把 backend 类型写死进 Mem |
| `MEM-SHAPE-06` | gallocr/oneDNN 的 graph reasoning 与 offset packing 容易被混成一个 Graph 类型 | Mem 提供无 tensor 语义的 packing engine；Graph adapter 提供 interval/alias facts |

## Replay verdict

结论：**`PRODUCT_TRIGGER`**。

理由不是“peer 函数更多”，而是三个独立事实同时成立：

1. 当前北极星推理 consumer 必须自然组合 mapping、host/pinned/device/managed resource、buffer view、
   pooling、capacity 与 worst-case reserve；
2. ggml/llama.cpp 与独立的 CCCL、Boost.Compute、oneDNN/Folly 分别给出相同能力族；
3. HYLE 已经实现其中相当一部分，却按 OS/backend 位置分散到 Platform/Compute/IO，导致 Mem 的公开
   边界和依赖图不能表达真实 storage ownership。

这命中 `mem-v1` 的显式 reopen trigger：“device/pinned memory 或另一 deferred family 通过单独能力决策
准入”。受影响范围是 Mem 的产品边界、公开 API taxonomy、相邻 provider/adapters 和文档；旧 host
实现的 correctness/performance 结论不会自动失效，但任何改名、layout、算法或 normal route 变化都须在
`mem-v2` 重新验证。

合法下一步是新建 `mem-v2` closure card，按 `CLOSED -> REOPENED -> SCOPED -> AUDITED` 冻结有限
`FIX` 集。不得直接在旧 `mem-v1` task 追加代码，也不得先为某条 benchmark 设计新类型。
