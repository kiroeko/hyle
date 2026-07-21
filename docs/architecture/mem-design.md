# HYLE mem/ 设计

> Tier 0 host-RAM 基础设施。上游约束见 [`hyle-scope.md`](./hyle-scope.md) 与 [`coding-conventions.md`](./coding-conventions.md)。

## 1. 边界

`mem/` 只管 host RAM：通用分配器、固定块池、bump/stack、对象池、fiber stack、NUMA/大页和诊断装饰器。device 显存属于 `compute/` 的 Buffer/DevicePool/VMM；host↔device 通过显式 copy/命令提交，不把显存伪装成 CPU allocator space。

pinned host RAM 也由 `compute/` 暴露，因为它的目的和生命周期与 DMA 绑定：

- `PinnedStaging` 当前直接拥有一个 pinned host block，并在该块内 bump；它不是 mem 池消费者。
- `pinned_allocator` 可以显式作为 `SharedPool` 的 backing 组合出 pinned 固定块池，但当前没有生产消费者，不把这种可组合性写成已接线能力。

## 2. 分配器分层

| 层 | 形状 | 用途 |
|---|---|---|
| `SystemAllocator` | 编译期无状态类型，header-inline 直调 mimalloc | 默认路径；静态/merged 构建零 vtable 接缝 |
| `Allocator` | `{alloc, free, realloc, state}` 运行时 C vtable | arena/mock/跨 DLL/自定义 backing 的 opt-in 注入 |
| 专用分配器 | SharedPool、Arena、StackAllocator、GrowablePool、FiberStackPool | 按生命周期和并发形状特化 |

容器默认使用 `SystemAllocator`；只有真正需要运行时多态时才显式传 `Allocator`。sized-free 必须带回原请求的 size/alignment。alignment 是真实契约，不能被 backing 静默丢弃。

通用 malloc 借 mimalloc，不自造任意大小通用堆。HYLE 自造的是生命周期明确、可以用结构换常数的专用层。

## 3. 固定块池：一个 sharded core，两种公开合同

`SharedPool` 和 `SharedPoolCached` 共用 `detail::ShardedSlab`：

- slab 按 block stride 切分，free-list link 存在独立 `atomic<uint32_t> next[]`，不占用用户字节；
- shard 数按硬件并发向上取 2 的幂，最大 128，且不超过 block 数；每个 head 独占 cache line；
- 正常 head 是 64-bit `(32-bit version, 32-bit index)`，用 64-bit CAS，不依赖 16-byte atomic；
- 只有精确使用进程 `system_allocator()` 的大 slab 才优先走 `HugeRegion::allocate_local(..., THP)`；自定义 backing 始终拥有 slab，不能把 pinned/accounted/domain-specific RAM 静默换成普通 host RAM；
- 两种池共享 OOM、对齐、reset 和 `as_allocator()` 语义。

### 3.1 SharedPool：硬容量默认

`SharedPool` 不缓存 block 到 TLS：

- home shard 成功 pop 和每次 push 正常各一次 CAS；home 为空时扫描其他 shard；
- completed `deallocate()` 立即把 block 发布到 shared core，任意线程随后可见；
- `allocate()` 第一次遍历全空后不立刻宣告耗尽，而是验证所有已观察 tagged head 未变。两次一致的全 shard 快照证明存在一个公共区间所有 shard 同时为空；若任一 shard 变化则重试。这样 block 在扫描背后跨 shard 移动也不会造成分片假耗尽。

硬容量是固定配额、跨线程 handoff 和默认 `ObjectPool<T>` 的安全合同。

### 3.2 SharedPoolCached：软容量高吞吐版本

`SharedPoolCached` 在同一 core 上增加每 `(thread,pool)` 的 owner-only TLS cache：

- 每线程 **128 slots**；
- TLS miss/overflow 以 **32-block magazine** 一次 CAS 批量 refill/flush；
- 同线程 free→alloc 命中时只是 TLS index push/pop，零原子；
- `publish_thread_cache()` 显式把当前线程 cache 发布到 shard；overflow、线程退出、pool 析构也会回流。

其合同是软容量：其他活线程 TLS 中已 free 的 block 暂时不可见，所以 `allocate()` 可能在 shared core 真空时返回 NULL。容量紧贴工作集或跨线程 free 后要求立即复用时不得使用 cached 版本，应使用 `SharedPool`。

TLS registry 的 attach/detach/reset/dtor 由 `hyle_mem` 内进程级单锁保护；节点由 `system_allocator()` 持有，与注入 backing 生命周期解耦。热路径不碰 registry lock。

### 3.3 tag 饱和：永久安全降级，绝不回绕

32-bit tag 是有限的。每个 shard 在 version 到 `0xFFFFFFFF` 时永久进入 saturated mode：

- head version 固定为 saturated sentinel；
- pop/push/batch/reset 在该 shard 上走 serialized lock；
- 独立 64-bit `saturated_changes` 每次变更递增，并参与稳定空快照比较；
- 不把 tag 重新置零，不让旧 `(index,version)` CAS 在长寿命池中重新变为有效。

正常池继续使用 64-bit tagged CAS；只有实际耗尽 32-bit tag 的 shard 承担 serialized 成本。

## 4. typed 对象池

| 类型 | backing | 容量合同 |
|---|---|---|
| `ObjectPool<T>` | `SharedPool` | **硬容量默认**；destroy 后槽立即可被任意线程复用 |
| `ObjectPoolCached<T>` | `SharedPoolCached` | **软容量**；同线程 churn 快，跨线程交接需 publish |

两者都用 placement-new / 显式析构，槽的 acquire/release 必须回到同一个 pool。C ABI 用 runtime size/alignment 擦除类型；默认 `HYLE_OBJECT_POOL_*` 宏绑定硬池，`HYLE_OBJECT_POOL_CACHED_*` 明确绑定软池。

## 5. bump、stack 与增长池

### Arena

`Arena` 只借 `SharedPool` 的一个 block，使用 `cur/end` pointer bump。它是单线程区域分配器：没有 per-object free，`reset()` 或析构整体回收。单块 borrow 不需要 TLS cache；硬池让 block return 的容量立即可见。

### StackAllocator

`StackAllocator` 自持一个 backing slab，同样使用 pointer bump，并提供 `mark()/rewind()` 的 LIFO 回退。它单线程、不可复制移动、无隐藏堆回退；空间不足返回 NULL。

### GrowablePool<T>

单线程固定尺寸对象反复 create/destroy 使用 `GrowablePool<T>`：intrusive free-list + 几何增长，无原子。它不提供固定硬容量；跨线程共享不是其合同。

## 6. 其他 host-RAM 能力

- `FiberStack` / `FiberStackPool`：guarded VM stack、尺寸分级与池化复用，供 sched 使用。
- `HugeRegion`：THP、大页、prefault、NUMA preferred/bind/interleave；策略失败不影响普通 RAM 正确性。
- `StatsAllocator` / `LeakTracker`：opt-in debug/test 装饰器，不给 system 热路径加税。
- `PmrResource`：把运行时 `Allocator` 适配到 `std::pmr::memory_resource`。

## 7. 选型

```text
通用大小 / 不确定                  → SystemAllocator
批量分配、整批释放                 → Arena / StackAllocator
单线程同尺寸对象反复复用           → GrowablePool<T>
多线程共享、容量必须准确            → SharedPool / ObjectPool<T>
同线程复用热点、池有充足余量         → SharedPoolCached / ObjectPoolCached<T>
fiber guarded stack                → FiberStackPool
GPU device 显存                    → compute::Buffer / DevicePool，不走 mem
pinned 单块 staging                → compute::PinnedStaging，不走 mem 池
```

## 8. C ABI

`include/hyle/c/mem.h` 导出：

- `hyle_allocator` 与 system usable/good-size；
- `hyle_shared_pool_*`（硬容量）；
- `hyle_shared_pool_cached_*`（软容量 + publish）；
- StackAllocator、Arena、GrowablePool、HugeRegion；
- hugepage/NUMA 进程策略。

C ABI 没有兼容别名。Arena 的 C 句柄只接受 `hyle_shared_pool_o*`。逐符号和使用合同见 [`../guide/c-mem.md`](../guide/c-mem.md)。

## 9. 验证边界

- 固定池正确性必须覆盖：耗尽/回收、跨线程 handoff、稳定双快照、tag 饱和 fallback、TLS publish/reset/退出、外来指针 fail-fast、OOM。
- lock-free/并发路径必须过 WSL TSan；生命周期必须辅以 ASan。
- 性能结论按硬池与 cached 池分别对标，不能把软容量 TLS 快路数字冒充硬池性能。
- 当前实测数统一记录在 [`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)，本文不复制易过期数字。

## 参考

- [mimalloc](https://github.com/microsoft/mimalloc)
- Treiber stack、tagged ABA protection
- tbb/jemalloc per-thread cache 与 magazine transfer 形状
