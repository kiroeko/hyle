# Mem：统一存储语义，多种明确的寿命

Mem 负责 raw allocation、资源、Buffer/View、pool/arena、mapping、quota 与 tensor-free lifetime
packing。它统一存储语义，不把所有内存都伪装成可解引用的 host pointer。示例与完整 API 选择见
[Mem 指南](../user/mem.md)，共同所有权原则见[基础机制](foundation-design.md)。

## 机制、owner 与调度分开

Platform 提供 raw anonymous VM/NUMA、page 和同步 FileHandle；Mem 提供 storage owner/view、
file/shared mapping 与通用复用。Compute 通过长寿命 provider 提供 host/pinned/device/managed
storage，同时拥有 Context、Queue、Completion 和异步执行顺序；Graph 推导 DAG 的 liveness/alias
事实，再交给 Mem 的非 tensor planner。Mem 不理解模型、KV、设备 stream 或算子。

用户选择 `hyle::mem`，内部包含 base `mem`、mapping/guarded-stack `mem_vm` 与 reader-lifetime
`mem_reclaim`。调用方无需手动挑内部 factor，维护者仍按直接消费声明依赖，见
[组件图](../../cmake/HyleComponents.cmake)。

## 为什么不是一个“更快 allocator”

| 用户寿命 | 形状 | 不可混淆的合同 |
|---|---|---|
| 通用 individual alloc/free | SystemAllocator、AllocatorRef、Heap | 静态默认、运行时借用、独立 heap owner 不同 |
| 外部固定 variable-size storage | FixedHeap | individual free，无外部 fallback |
| phase/region scratch | Arena、FixedArena、PooledArena | 批量失效，不是通用 free |
| LIFO scratch | StackArena | 严格栈式归还 |
| 固定块共享配额 | ConcurrentBlockPool | hard capacity、remote free 可见 |
| owner-local free-slot retention | BlockPoolCache | soft cache、publish 才恢复全局可见 |
| typed object | ObjectPool / PreservingObjectPool / Loan | 每次重构与归还保留状态不同 |
| opaque backing 分段 | RangeAllocator / RangeAllocation | 只分配 offset，不拥有 payload |
| 跨 provider storage | MemoryResourceRef、Buffer/View | handle、逻辑范围、mapping 与 host address 不同 |
| transient resource reuse | BufferPool / BufferLease | 归还前由执行 owner 确认最后使用 |
| 已知生存区间 packing | LifetimePlanner/Plan | 不包含 Graph、Tensor 或设备执行 |

block/object/variable size、owned/external、single-owner/concurrent、hard/soft/growable、individual/bulk
这几项共同决定类型。不能把缓存命中快描述成 hard quota，也不能因为 payload 可复用就跳过析构。

## 通用堆借核，适配保持显式

通用任意大小分配使用 mimalloc，不复制 malloc 算法。SystemAllocator 是静态默认路线；AllocatorRef
是借用 resource 的运行时值句柄。typed count 乘法、alignment 与范围都必须避免回绕；释放带回原始
owner 及匹配的 size/alignment。dynamic alignment 的成功分配不授权错误参数的 deallocate。

`SystemStdAllocator<T>` 保持默认 empty/always-equal 形状，`StdAllocator<T>` 显式接受 runtime
backing；标准 allocator/PMR 适配在对应 C++ 边界按标准抛 bad_alloc。PmrResource 适配借用 allocator，
AlignedPmrResource 才额外施加最小对齐，普通容器不承担 cache-line 对齐政策。

Heap 是 move-only 独立 host heap：分配/重分配由 owner thread 发起，remote free 允许跨线程；
collect 仅回收空闲页，clear/release 是活 allocation 必须已静止的 bulk invalidation。借用 allocator
不能活过 backing；不能因分配线程退出就销毁仍接受 remote free 的资源。

allocator 的 NUMA/large-page 启动设置可能在 native loader 初始化时已缓存，不能用一个返回成功却
无效的 runtime setter 伪装能力。需要 per-span placement 时选择 LargeMapping/Platform 的明确形状。

## Hard pool 与 soft cache 分型

ConcurrentBlockPool 固定 block size 和 runtime capacity，完成的 deallocate 进入 shared core。
快路径使用分片链；耗尽判定须避免块在扫描背后迁移产生假耗尽。tag 饱和后进入安全的串行模式，
不以回绕换取 ABA 风险。默认热路不为每个 slot 加无需求的 liveness 元数据。

BlockPoolCache 由调用者显式持有，可借 caller-owned index storage，也可显式选择 metadata allocator。
owner-local 命中零共享原子；被其他 cache 保留的 free block 暂不可见，publish/析构才归还共享池。
它不建立隐藏 TLS registry，也不改变底层 hard capacity。cache 本身 single-owner，pool 必须更长寿。

ObjectPool 的 create/destroy 每次构析；构造异常归还 slot。PreservingObjectPool 在池建立/销毁时
构析，borrow/return 保留对象内容。ObjectPoolCache 只缓存空闲 slot，不等于 preserve object。
PoolLoan 是 move-only RAII return，pool 仍须覆盖全部 Loan 寿命。基础池只提供 nonblocking try，
等待/取消/唤醒由拥有 scheduler 的上层组合。

## Arena 与逻辑 range

Arena 拥有可增长 backing，clear 批量使旧对象失效，并可保留合适 block 供下一阶段复用；固定外部
FixedArena 不分配 fallback；PooledArena 只借一个 hard-pool block；StackArena 表达 LIFO 失效。
对象析构与原始 storage 清空并非同一件事，调用者必须按对应类型的约定处理非平凡对象。

RangeAllocator 使用 64-bit aligned offset，metadata 与 payload 分离。RangeAllocation 归还逻辑范围，
owner reset 用 generation 使旧 lease 失效；metadata allocator 需活到依赖它的 handle 消失。
两级 size-class lookup/split/coalesce 常见为 O(1)，但精确对齐的 free-range fallback 最坏 O(F)，
不能为了宣传硬 O(1) 而永久吞掉 padding 或制造虚假 OOM。summary stats 与详细碎片扫描也分别计费。
它不执行设备搬迁、defragment callback 或 KV 淘汰。

## 开放 resource 与 Buffer/View

MemoryResourceRef 借用 provider，provider 及 backend state 必须覆盖 owner 与全部在途使用。
resource properties/domain 是权威事实，不用闭合的 BufferKind 枚举猜测所有未来存储类型。

Buffer move-only，拥有一份 provider allocation；BufferView/ConstBufferView 是资源身份、opaque
handle、byte offset/length 的借用逻辑范围。owner move 不改变 allocation token，reset/destruction
使 view 失效。native handle 不一定是 host pointer，不允许一般调用方对它做 pointer arithmetic。

HostMapping 才表达 CPU 访问；non-coherent storage 显式 flush/invalidate。同步 read/write/fill/copy
不隐藏 device staging 或异步 completion。外部 import/adopt 只有成功才接管回收责任，失败时原 owner
仍负责。CompositeBuffer 保留各段实际 backing，不能把虚拟拼接地址当成一块连续 native allocation。

BufferPool 只复用已归还 lease 的资源；“完成”由原 Compute/IO owner 裁定。取消、提交失败或等待返回
错误都不自动证明 native work 已退休，不能过早归还 pool。

## Mapping：lazy、warm-up 与孔洞都要诚实

`VirtualMapping` 是 Platform VmRegion 的别名。FileMapping/SharedMapping 拥有 native mapping，
不引入第二套匿名地址空间 owner。FileMapping 消费已验证 FileHandle；open 默认 lazy，不强迫预取。

prefetch 是 WillNeed hint，成功不证明 resident；populate 同步触页，成本与覆盖页数相关。连续 mapping
不常态维护 fragment tree，第一次有效 partial release 才建立所需 metadata。批量 loader ranges
先排序合并再与 live fragments 相交；逐项调用不冒充同等总复杂度。checked view 不能跨已释放 hole。
Windows 无安全的同合同 partial release 时返回 Unimplemented，调用方使用多个精确 owner。

advice 必须考虑 backing。Windows anonymous/private discard 不等价于 file-backed DONTNEED，不能
因系统有某个名字相似的 primitive 就套到 RO/RW/COW 文件视图。

SharedMapping 不在 payload 前插隐式 size header，保留 native page/cache alignment，也避免 opener
读取半初始化 header。精确 logical size、access 与 readiness 归 IPC 协议；部分系统只能报告 page-rounded
capacity，不能假装恢复精确长度。shared name 保留 native namespace identity，不静默截断或 hash。

## Lifetime packing 不知道模型

Graph 将 liveness、alias、resource 与 persistent/output facts 翻译为 `[begin,end)` LifetimeRequest。
Mem planner 检查同资源无环 alias，结合 resource alignment/单次 allocation 上限放置；结果是
resource-local backing index 加 local offset，不能用虚拟拼接 offset 代表跨 backing alias。

plan 与 instantiate 分开：plan 不拥有 provider/Tensor/Graph，实例才分配 Buffer；多个执行实例不能
因共享 placement 描述就共享可写 storage。retention 与并发 slot 数由实际执行 owner 决定。

## Quota、NUMA、大页和回收

QuotaAllocator 计费声明的资源域，不是总进程 RSS 硬上限。StatsAllocator/leak inventory 是显式诊断，
metadata OOM 必须让 incomplete/dropped 可见；system snapshot 的 reserved/committed/RSS 与
owner-local requested-live 不是同一数字。driver/外部库未知开销不能写成 0。

LargeMapping 提供显式 THP/prefault/NUMA 策略，requested、accepted、applied、实际 residency 分开。
selected-node 条带复用 Platform 的 node ID 和 weighted policy；大页 advice 可能改变更小条带的物理
粒度。process-global large-page arena 的 GiB 请求单位不能证明 OS 使用 1 GiB 页；partial reservation
无法统一 rollback，报告必须保留实际容量与 outcome。

FiberStackPool 是有 guard 的专用 owner；retention 0 禁用 cache，trim/clear 放在 worker/fiber 已 drain
的 owner 边界，不在每个 idle 热路扫描。

mem_reclaim 提供两代 reader lifetime、显式 guard、caller-owned retire record 和 grace-period 回收。
read path 不分配；普通参与 slot 满后仍有固定 fallback。它不拥有 payload 或 concurrent map 语义。
live ReclamationGuard 必须在同一 OS thread 进入与退出，不能跨可迁移 fiber 的挂起点持有。
同步 drain 不能由仍持有阻塞它的 read guard 的调用者执行。Container 的表退休和 Obs 的短快照保护
复用该机制，各自仍拥有 node/callback 的最终寿命。

当前无独立 C ABI。相关实现位于[公开头](../../include/hyle/mem/)；性能比较应同时对齐寿命、
容量、线程、失败、分配/回收与峰值，而非只比较最热一次 allocate。
