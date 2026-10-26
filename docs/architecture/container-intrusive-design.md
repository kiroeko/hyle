# 侵入式容器：稳定节点的零分配 membership

timer、callback、arena chain 或 LRU 的对象已经有 owner；只为排队再分配一个 owning node 会增加
allocation、间接访问与失败面。侵入式容器只管理 caller-owned stable node 的 membership，不拥有
storage。使用示例见[侵入式容器指南](../user/container-intrusive.md)。

## 为何单独拆出 factor

header-only `container_intrusive` 只依赖 Core；Time/Sched 可直接复用，不必引入完整 Container/Mem
闭包。用户通过 `hyle::container` 获得能力，不把内部 factor 当作额外产品选择。source/static/shared
使用同一 inline API，没有独立 binary、registry 或 FFI 层。

| 需求 | 类型 | 64-bit hook / list footprint | 取舍 |
|---|---|---|---|
| 双向、两端、range splice、known-node O(1) unlink | IntrusiveList | 2 / 2 pointers | auto-unlink |
| 同上，membership 只能由 owner 显式退休 | SafeIntrusiveList | 2 / 2 pointers | safe-link |
| safe-link 且 mutation 后 O(1) size | CountedIntrusiveList | 2 / 3 pointers | 仅选择者多付 count word/write |
| arena/FIFO chain | IntrusiveForwardList | 1 / 2 pointers | 显式退休，list 缓存 tail |
| bucket/callback，head 小且 known-node O(1) unlink | auto/safe IntrusiveHList | 2 / 1 pointers | next + previous_next，仅前向 |

这不是 fast/light 层级：不同拓扑具有不同操作、寿命与 metadata。auto-unlink 无法在不知道 owner
list 时维护隐藏 count，所以不提供 counted-auto 组合。

## 已有消费者与保留的专用结构

Time TimerQueue/CommonTimeoutQueue 使用 safe 双向 list，支持 O(1) cancel 与批量 handoff；
Sched CancelCallback 使用 safe hlist，将摘链放在所属 cancellation owner 的锁与 detach-before-callback
协议内。forward list 服务单 link 的 arena/deferred chain；counted list 服务需要立即统计数量的成员集。

看到 next 字段不等于必须迁移：free slot 可复用已释放 storage；waiter 可能在摘链后复用 link 成为
wake chain；IOCP 可能在 native in-flight/completion/free 之间复用对象存储。atomic publication、ABA
与 concurrent reclamation 也不是普通 hook 可以代替的协议。复用同时满足 lifetime、metadata 和成本
才有意义，不能只追求调用点数量。

## Node lifetime 与 member recovery

hook 必须是完整 T 的直接非静态成员。linked 期间地址稳定、storage 存活；mutation、遍历与 node/list
析构受相同外部同步保护。同一个 node 参与多个 membership 时，各自嵌入不同 hook。

hook 的 copy/move construction 创建未链接 membership；assignment 保留 destination membership，
不会把 source 的链身份偷偷搬过来。这不表示可以把 linked node 搬到新地址后销毁旧对象而保留成员关系。

- auto 双向/hlist hook 可自行 unlink，析构也摘链；
- safe 双向/hlist 与 forward 必须先由 owner 摘除再析构，Debug 会诊断；
- list 析构恢复节点为 unlinked，不销毁 node，也不接管其 allocator。

从成员指针反求 enclosing T 没有一般标准 C++ 逆运算。三类实现共用 factor-private
IntrusiveMemberAccess，支持已明确的 64-bit MSVC direct-member 或 GCC/Clang Itanium representation；
更丰富/未知表示编译期 fail closed。node 不必 standard-layout，hook 不必首成员；直接成员、完整对象、
稳定地址与受支持 ABI 才是边界。不能拿通用 container_of 技巧掩盖这项平台合同。

## 操作复杂度

双向 list 使用 circular sentinel。默认 auto/safe 不存 size，显式 range distance 为 O(n)；counted
增加一个 size_t 并维护 O(1) size。splice/swap/move 维持 node iterator，但 sentinel/end 身份属于
原 list，不随节点转移。

| 操作 | 默认双向 | Counted |
|---|---|---|
| 已知 node/iterator mutation、单项/整链 transfer | O(1) | O(1) |
| range splice | O(1) relink | 默认 O(k) 计数；已验证 count 重载 O(1) |
| size | O(n) distance | O(1) |
| filter/reverse/merge | O(n) | O(n) |
| stable sort | O(n log n) | O(n log n) |

verified-count、owner/range 与重复 insertion 是调用方合同，unchecked mutation 不额外扫描 owner。
try insertion 只处理 null/already-linked 这类明确的可恢复输入，不是检查所有任意错误的安全容器。

forward list 保存 head/tail，append 与整链 handoff O(1)；已知 predecessor mutation O(1)，range
需找 tail 而为 O(k)。它不声称 arbitrary-known-node O(1) unlink，没有 auto-unlink。

hlist 保存 next/previous_next，head 仅一个 pointer；known-node unlink/replace O(1)。它不提供 back、
reverse、range splice 或 sort/merge；需要 sequence 语义时选择 forward/双向。

## 异常、迭代器与并发

sequence sort/merge 稳定且不分配。predicate/comparator 抛出时原异常传播，已完成的 mutation 不回滚，
但每个 node 仍属于一条有效 list；sort 临时 bins 在异常路径汇回，不遗失 membership。
consume_all 要求 noexcept callback，因为 owner chain 已整体 detach。

erase/remove/pop/unlink 使该 membership iterator 失效，node storage 是否仍可用归 caller。
其他 node 引用不因相邻 relink 失效。

这些类型 thread-compatible，不是 thread-safe；没有内置 lock、atomic、syscall、allocation 或动态
dispatch。auto-unlink 不等于 concurrent reclamation：与遍历并发析构仍可能 race/UAF。模块没有 OOM
或系统失败面，不为统一风格加 Status。intrusive tree/hash/heap、offset pointer 与 shared-memory
模型有新的语义成本，不因参考库已有就自动扩面。

实现合同见[双向 list](../../include/hyle/container/intrusive_list.h)、
[forward list](../../include/hyle/container/intrusive_forward_list.h)、
[hlist](../../include/hyle/container/intrusive_hlist.h) 和
[member recovery](../../include/hyle/container/detail/intrusive_member.h)。
