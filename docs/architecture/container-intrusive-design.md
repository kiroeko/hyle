# container_intrusive 架构：稳定节点的零分配 membership

> 本页记录长期产品边界，不签署当前状态或性能。施工状态只看
> [module-status](../implementation/module-status.json)，普通用法看
> [Container intrusive 指南](../user/container-intrusive.md)。

## 为什么是一个内部 factor

Timer、scheduler callback、arena object chain、LRU/wait-set 等对象已经由别的 owner 持有；若只为排队
再分配 owning list node，会增加 allocation、间接访问和失败面。`container_intrusive` 因而只管理
caller-owned stable node 的 membership，不拥有 storage。

它是只依赖 Core 的 header-only 质因数 target `hyle_container_intrusive`。Time 的 `timer_queue` 和
Sched 可以直接依赖它，不必反向引入完整 Container/Mem/Platform 闭包。普通用户不选择这个内部 target，
而是链接聚合它的 `hyle::container` 产品。源码、installed static 和 installed shared 三种交付使用同一
套 inline C++ API；模块没有 binary、注册表、Native FFI 或伪 C ABI。

公开能力分成三种拓扑、四种生命周期/metadata 选择：

| 需求 | 类型 | node / list footprint（64-bit） | 关键保证 |
|---|---|---:|---|
| 双向遍历、两端操作、range splice、已知 node O(1) 摘链 | `IntrusiveListHook` + `IntrusiveList` | 2 / 2 pointer | auto-unlink |
| 同上，但 membership 只能由 owner 显式退休 | `SafeIntrusiveListHook` + `SafeIntrusiveList` | 2 / 2 pointer | safe-link |
| 双向 safe-link 且 mutation 后必须 O(1) 查询数量 | `SafeIntrusiveListHook` + `CountedIntrusiveList` | 2 / 3 pointer | 只给选择者增加 count word/write |
| arena/FIFO/deferred chain，每 node 只肯支付一个 link | `IntrusiveForwardListHook` + `IntrusiveForwardList` | 1 / 2 pointer | explicit retirement；list 缓存 tail |
| bucket/callback membership，head 要小且已知 node 要 O(1) 摘链 | auto/safe `IntrusiveHListHook` + `IntrusiveHList` | 2 / 1 pointer | `next + previous_next`；只前向遍历 |

这不是按 benchmark 建立的 fast/light 层级。每种形状都对应不同的常态 metadata、可表达操作和生命周期；
不需要其语义时不会为它付费。

## 真实消费者与依赖

| Consumer | 采用的公共 primitive | 为什么 |
|---|---|---|
| `time::TimerQueue` / `CommonTimeoutQueue` | safe 双向 list | deadline bucket 要 O(1) cancel、双端/range handoff；timer 必须显式退休 |
| `sched::CancelCallback` | safe hlist | record 原来就是 `next + previous_link`；迁移后 record 与 head footprint、锁/reentrancy/fire 顺序均不变 |
| arena/object/deferred-work 用户 | forward list | append、遍历、批量 release；node 只付一个 pointer，list 的 tail cache 使 append/整链 handoff O(1) |
| wait-set/LRU/cache 用户 | counted safe list | mutation 后无需 O(n) traversal 就能读 membership count |

并非看到 `next` 就强制迁移。以下专用结构继续合法：

- Mem free slot 把 next 覆盖到已经释放的 storage；公共 hook 会让 live object 也付字段成本；
- Sched waiter 摘链后复用同一 link storage 形成锁外 wake chain；
- IOCP object 在 kernel inflight、completion 与 owner-thread free state 间复用 storage；
- atomic publish、ABA 与 reclamation 是 owning concurrent protocol，不是普通 intrusive hook。

中央依赖图允许任何更高 factor 直接声明 `PUBLIC/PRIVATE container_intrusive`；不得仅因“依赖不上”复制
同义 primitive。是否复用仍需同时满足语义、lifetime、metadata 与成本合同，而不是追求调用点数量。

## 生命周期与 member recovery

所有 hook 都必须是完整 `T` 的直接非静态数据成员；linked 期间 node 地址稳定、storage 仍存活，外部
同步覆盖 list mutation、遍历和 node/list 析构。一个 node 同时属于多种 membership 时，每种各嵌入一个
hook。Hook copy/move construction 创建未链接 membership；assignment 保留 destination membership，
不会暗中把 source 的链身份搬过来。

- auto 双向/hlist hook 可以在不知道 owner list 时 `unlink()`；hook 析构也会摘链；
- safe 双向/hlist 与 forward hook 必须在 node 析构前由 list 显式摘除；Debug fail-fast，Release 是调用
  前置条件，不增加 owner pointer；
- list 析构会把尚存节点恢复为 unlinked，但不销毁 node。

从 member hook 恢复 enclosing `T` 没有标准 C++ 逆运算。三类实现共用 factor-private
`IntrusiveMemberAccess`：只接受已验证的 64-bit MSVC direct-member representation 或 GCC/Clang
Itanium representation，遇到更丰富的 PMD ABI 编译期 fail closed。Node 不必是 standard-layout，hook
也不必是首成员；direct member、完整对象、稳定地址和支持 ABI 才是真边界。

## 双向与 counted 合同

双向 list 是 circular sentinel。默认 auto/safe list 不保存 count，因此 list 和 hook 均保持两个 pointer；
冷路径计数显式使用 `std::ranges::distance`。`CountedIntrusiveList` 复用 safe hook，只在 list 增加一个
`size_t`，是 `sized_range`，`.size()` O(1)。Auto-unlink 不能安全维护隐藏 list count，所以没有 counted
auto 组合。

双向公共操作包括 iterator/range、front/back、push/pop、insert/erase/remove、`iterator_to`、单项/
整链/range `splice`、append、swap/move、clear/consume 和
`reverse/remove/remove_if/unique/merge/sort`。Auto list 另有 typed
`List::swap_nodes(left,right)`；safe/count 不允许绕过 owner state machine 转移未知业务镜像。

| 操作 | 默认双向 | Counted |
|---|---:|---:|
| 已知 node/iterator mutation、single/whole transfer | O(1) | O(1) |
| range splice | O(1) | 普通重载 O(k) 计数；verified-count 重载 O(1) |
| size | O(n)，只通过 range distance | O(1) |
| filter/reverse/merge | O(n) | O(n) |
| stable sort | O(n log n) | O(n log n) |

verified-count 参数错误、错误 owner/range、重复 insertion 都是程序合同违反；不为 unchecked hot path 增加
owner scan。`try_push_*` / `try_insert` 只处理 null 或 already-linked 这两种正常可恢复输入。

## Forward-list 合同

Forward hook 的 self-next 表示 unlinked，linked tail 的 next 是 null。List 保存 before-head 与 cached tail，
因此 node 一个 pointer、list 两个 pointer；这是 arena/FIFO 的实际空间取舍，不是“比双向更快”的名字。

公开操作是 `before_begin/begin/end`、front/back、front/back push、`insert_after/erase_after/pop_front`、
whole/single/range `splice_after`、append/swap/move、clear/consume，以及与双向相同的 sequence algorithms。
已知 predecessor 的 single mutation 与 whole-list handoff O(1)；range 需要定位 range tail，为 O(k)。类型
不声称 arbitrary-known-node O(1) unlink，也没有 auto-unlink：需要该合同就选择 hlist 或双向 list。

## Hlist 合同

Hlist node 保存 `next + previous_next`，list/head 只保存一个 pointer。它支持 forward iteration、front、
push-front、known-position before/after insert、known-node erase/remove/replace、pop-front、clear/consume、
swap/move。Known-node unlink/replace 是 O(1)，不会为了 predecessor 扫描。

它刻意没有 back、reverse iterator、count、range splice、sort/merge；需要 sequence 语义时选择 forward 或
双向 list。Auto/safe 的含义与双向一致。`sched::CancelCallbackRecord` 使用 safe hlist，因为 record 的
退休必须在 cancellation tree 的 recursive mutex 和 detach-before-callback 协议内完成。

## Algorithms、异常与 iterator

Sequence `sort/merge` 稳定且不分配；`remove_if/unique` 只使被摘除节点 iterator 失效，其他 node 的引用/
iterator 继续有效。Predicate/comparator 抛出时原异常传播：已经完成的 filter/merge 步骤不回滚，但
每个 node 仍恰属一条有效 list；sort 的临时 bins 在异常路径全部汇回当前 list，不遗失 membership。
`consume_all` 则要求 `noexcept` callback，因为它先把 owner chain 整体 detach，再逐项交付。

双向 iterator 是 bidirectional；forward/hlist iterator 是 forward。Node iterator 随 splice/swap/move
中的 node 保持有效，但每个 list 的 sentinel/end 身份不会随 node 转移。任何被 erase/remove/pop/unlink
的 membership iterator 都失效；node storage 本身是否仍有效由 caller owner 决定。

## 并发、错误与负空间

所有类型 thread-compatible、不是 thread-safe；没有内部 lock/atomic/syscall/allocation/dynamic dispatch。
模块没有 OOM 或系统失败，因此 `Status/Result` 不适用。Auto-unlink 也不是 concurrent reclamation：遍历
与另一线程 node 析构无同步并发仍是 race/UAF。

明确不自实现的能力：

- atomic publication、lock-free list、hazard/epoch reclamation：由 Container/Sched/Mem-reclaim 的拥有
  protocol 提供；
- intrusive tree/hash/heap、base/value/function traits、fancy/offset pointer、shared-memory model：直接
  使用 Boost.Intrusive，除非至少两个真实 HYLE journey 证明现有依赖路线不足；
- allocator/disposer ownership：由 node/storage owner 决定；
- exact owner/`contains(node)`：会让所有 node 额外支付 owner metadata，当前不准入。

性能只对冻结的同合同 workload 签署：counted membership、cached-tail forward arena、hlist cancellation
consumer 和 stable sequence algorithms。亚纳秒点值、地址/顺序翻转或不同语义容器不能改变上述 API、
layout 与常态 metadata。
