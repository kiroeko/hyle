# container_intrusive 架构：零分配侵入式双向链表

> **产品形状：`CANDIDATE / 复审中`。** 本页记录当前候选的长期边界，不是模块签署证书。
> 正确性、性能、交付与独立红队仍须绑定同一 candidate；唯一施工状态见
> [implementation/status](../implementation/status.md)。

## 定位

`container_intrusive` 为已有稳定 storage 的节点提供最小的、non-owning 双向链接机制。公开入口是
`<hyle/container/intrusive_list.h>`，公开成对入口是
`IntrusiveListHook` / `IntrusiveList<T, &T::hook>` 与
`SafeIntrusiveListHook` / `SafeIntrusiveList<T, &T::hook>`；产品 target 是
`hyle::container_intrusive`。它是只依赖 `hyle::core` 的 header-only `INTERFACE` 组件，不产生
runtime artifact，也不拉入完整 `container`、`mem`、`error` 或 `platform`。

每条同时存在的 membership 在 node 中嵌入一个 member hook。auto-unlink 与 safe-link hook 都只保存
前后两个 raw pointer，list 只保存一个双向 sentinel；二者共享同一重链算法，区别只在 node 析构
合同。这个选择服务 timer、waiter、completion batch 等地址本来就稳定、节点由其他对象拥有的场景。
若用户只需要 owning sequence，应先选择 `std::list`、
`std::vector` 或完整 container 中相应的 owning 类型，不应为了省一次 allocation 扩散生命周期责任。

该模块明确没有以下常态状态或行为：

- 不拥有、分配、释放或延长 node storage；
- 不保存 owner pointer 或元素 count；
- 不提供锁、原子同步或跨线程 reclamation；
- 不产生可恢复 OOM/系统错误，也没有 runtime registry、dispatcher 或动态符号。

## 真实需求与消费者

| 消费者任务 | 使用形状 | 归属裁决 |
|---|---|---|
| `time::TimerQueue` / `CommonTimeoutQueue` 保存 caller-owned timer，O(1) cancel，并批量交付 ready prefix/list | safe hook；`remove`、range `splice`、`append`、`consume_all`、`for_each` | timer 必须在析构前显式撤销；deadline prefix 扫描留在 time，结构转移复用通用 range splice；`time -> PUBLIC container_intrusive` |
| 用户节点允许先于 list 析构，希望析构自动撤销 membership | auto hook；FIFO、任意摘链与批量 handoff | `IntrusiveListHook` / `IntrusiveList` |
| 用户状态机要求在 owner lock 下显式摘链，漏摘必须被诊断 | safe hook；同一 list 操作面 | `SafeIntrusiveListHook` / `SafeIntrusiveList`；Release 不为析构检查付常态分支 |
| sched ConditionVariable / Semaphore 的 stack waiter | 摘链后复用原 `next` 作锁外 wake chain | 保留 sched 专用链；通用 list 会增加 waiter footprint 和锁内写，且不能表达 link storage 的状态复用 |
| 完整 container 用户需要同一轻量能力 | 传递公开 target | `container -> PUBLIC container_intrusive`，但模块仍独立可选 |

语义重复不按“看见 next/prev 就迁移”处理。Windows IOCP `IoOp` 保留内部特化链：`OVERLAPPED`
必须是对象首成员，完成时还要把同一个对象从 inflight 双链切换到 owner-thread free-list 并复用 link
storage；该状态机与 Win32 completion/lifetime 绑定，换成公共 list 不会消除专用 reset、recycle 和
kernel-lifetime 合同。Cancellation callback 的 `next + prev_next` hlist、pool free-list 等单链也不是
本模块双向 member-hook 合同的绕过。

## 类型、所有权与生命周期

两种公开 hook 的选择是生命周期合同，不是性能开关：

- `IntrusiveListHook` / `IntrusiveList` 是 auto-unlink：node 可以先析构，hook 析构会从当前 list 摘除；
  `unlink()` 可在不知道 owner list 时幂等撤销 membership；
- `SafeIntrusiveListHook` / `SafeIntrusiveList` 要求 node 析构前已由 owning state machine 显式
  `remove/erase/pop/clear`；违反时 Debug fail-fast，Release 是前置条件违反而非隐式无锁摘链；
- 两种 hook 的 default construction 均未链接，`linked()` 只表示“属于某条 list”，不能识别具体
  owner；copy/move construction 创建新的未链接 hook，assignment 保留 destination membership，
  不导入 source membership。

node 必须是 standard-layout，linked 期间地址稳定；hook 可以不是首成员。同一对象需要同时加入 ready、
timeout 等多条 list 时，必须为每条 membership 使用不同 hook。带 identity 的 node 通常应删除自身的
copy/move，避免“payload 已移动、原地址仍在链中”这种虽符合 hook 规则却不符合业务直觉的状态。

从 member hook 恢复 enclosing `T` 没有标准 C++ 逆运算。候选只支持已登记的 MSVC data-member
pointer ABI 与 GCC/Clang 使用的 Itanium 形状，并对未知或更复杂 representation 编译期 fail closed；
standard-layout 限制和三平台编译/sanitizer 证据必须在签署前保留。该限制不是允许调用方 relocation
linked node 的理由。

list 始终可以先销毁：析构会逐项清空 membership 而不销毁 node。只有 auto hook 的 node 可以在
同步得到保证时先于 list 析构并自动摘链；safe hook 的 node 必须先显式摘链。任何 iterator/reference
都不能越过实际 node storage lifetime。

## 操作与复杂度

| 操作 | 复杂度 | 合同 |
|---|---:|---|
| `empty`、front/back、push/pop、单项 insert/erase/remove | O(1) | 不分配；front/back/pop 空表返回 null |
| `iterator_to(node)` | O(1) | node 必须非 null、已链接且属于该 list；不做 owner scan |
| 单节点、整链及 `[first,last)` range splice | O(1) | range 不访问 node；transferred-node iterator 保持有效 |
| `append`、list move、`swap` | O(1) | 只重连 sentinel 与边界 hook |
| iteration、`clear`、list destruction | O(n) | 离开 list 的每个 auto/safe hook 都恢复未链接状态 |
| `consume_all(fn)` | O(n) | 当前 node 在 callback 前摘链；callback 必须 `noexcept`，可销毁或重链当前 node |
| `for_each(fn)` | O(n) | 结构不变；callback 不得改变被遍历 node 的 membership/lifetime |

Range splice 补的是不可由逐节点调用保持的复杂度档：在不保存 count 时，任意连续子链可以 O(1)
转移。`CommonTimeoutQueue` 的 deadline prefix 选择属于 time policy：它在 time 内做 O(k) iterator
扫描与 owner 更新，再用一次 range splice 完成 O(1) 重链。公共 container 因而不暴露只有单一
消费者的 predicate/callback 组合算法。

## 前置条件、`try_*` 与失败

普通 `push_*`、`insert`、`erase`、`remove`、`iterator_to` 和 splice 是已经证明前置条件的低层入口。
null node、重复插入、错误 owner/position、无效 range、把 position 放进同表转移 range 内部，或违反
linked-address lifetime 都是调用方 bug；无 owner metadata 的候选不会为这些情况在每次热操作扫描
整条 list。违反前置条件可能产生 UB 或链损坏，不能依靠 Release optimizer assumption 做输入验证。

Auto hook 的 `unlink()` 可用于 owner list 未知的幂等撤销；safe hook 刻意不提供该入口，必须经
list 明确摘链，避免绕过 owning state machine。`try_push_front`、`try_push_back` 和 `try_insert`
只处理两种正常可恢复输入：null node 与所选 hook
already-linked。它们返回 false 且不修改 membership。`try_insert` 仍要求 position 属于 destination；
`try_*` 不是通用 owner/range validator。若 untrusted input 需要精确诊断，应在拥有 owner、token 或
业务状态的上层先验证，不能给每个 hook 增加常态 owner 税。

模块自身没有 allocation 或系统调用，因此 OOM、partial progress 和 `Status/Result` 均不适用。
Callback 是否可抛由具体入口表达：`for_each` 可以传播异常并保持链结构，批量 detach/move callback
必须 `noexcept`，避免已经脱离 sentinel 的中间链失去可恢复 owner。

## Iterator 与失效矩阵

Iterator 是 bidirectional、一个 hook pointer 大小。下表中的“有效”包括可继续遍历；已经摘链的
iterator 即使 node storage 尚在且偶然仍可解引用，也必须视为失效，因为其 next/prev 已清空。

| 操作 / 事件 | 指向受影响 node 的 iterator | 其他 node iterator | `end()` iterator |
|---|---|---|---|
| push/insert | 新 iterator 有效；既有 iterator 不变 | 有效 | 仍绑定同一 sentinel |
| erase/remove/pop/`hook.unlink()` | 被摘链者失效 | 有效 | 有效 |
| 单节点、整链、range splice/append | transferred-node iterator 有效，之后遍历进入 destination | 有效 | source/destination 的 end 各自仍绑定原 list，不随 node 转移 |
| list move construction | transferred-node iterator 有效，之后遍历进入新 list | source 变空 | 旧 source end 与新 destination end 是不同 sentinel |
| list move assignment | source node iterator 转入 destination；原 destination node iterator 转入 moved-from source | 两组 membership O(1) 交换；source 有效但不保证为空 | 两个 list 的 end 均不转移 |
| `swap` | node iterator 有效，membership 随 node 链交换 | 有效 | 每个 list 的 end 仍属于该 list 对象 |
| `clear` 或 list destruction | 所有 membership iterator 失效；caller-owned node 本身可继续存活 | — | list destruction 后 end 也失效 |
| auto node destruction | 该 iterator/reference/pointer 随 storage lifetime 结束而失效；node destructor body 返回后，auto hook member 在自身析构时、storage lifetime 结束前摘链 | 其他 node 有效 | list 仍有效 |
| safe node destruction | 前置条件是已经显式摘链；仍 linked 时 Debug fail-fast，Release 为合同违反 | 其他 node 有效 | 合同满足时 list 仍有效 |

`consume_all` 在调用 callback 前使当前 membership iterator 失效，并允许 callback 销毁或重链当前
node；callback 不得销毁尚未访问的链或 list 本身。`for_each` callback 不得摘链、销毁或重定位当前
或后继 node。任何跨线程 iterator、hook 析构和 list mutation 都需要同一外部同步协议。

## 并发与进展保证

类型本身是 thread-compatible，不是 thread-safe。只读遍历也不能与另一线程的 insert/remove、
auto-unlink destructor、safe hook retirement、clear 或 list destruction 无同步并发。模块没有
lock-free 声明；类型的 thread compatibility 不会因 safe policy 而变成内部同步。

所有结构操作除线性 clear/drain 外都有确定的 O(1) pointer-relink 上界，不分配、不阻塞、不调用
系统服务。Callback 的进展与重入属于调用方；文档允许的当前-node destroy/relink 之外，不支持在
遍历 callback 中重入修改同一链。

## 明确负空间

| 未提供能力 | 裁决与理由 |
|---|---|
| `.size()`，包括 linear size | 不保存 count，也避免线性 member `size()` 让 C++23 range 错误建模为 amortized-O(1) `sized_range`；冷路径显式用 `std::ranges::distance` |
| counted list | `TimerQueue` 的 count 跨 bucket，sched waiter count 属原子发布协议；`CommonTimeoutQueue` 的 count 虽等于单链 membership，却是 time 专用 `size()` 状态，不能独自确定 auto/safe、range splice 的通用 counted 合同。count 会令所有 list 增加一个 word 与 mutation 写，range splice 还需 scan 或 caller 提供 n，故 `DEFER` |
| normal-link / 任意 hook policy | auto 与 safe 已覆盖常见生命周期；无检查/不初始化的 normal link 及完整策略矩阵借用 Boost.Intrusive，直到出现独立任务 |
| exact owner/`contains(node)` | 两指针 hook 无法 O(1) 回答具体 owner；不为罕见诊断向所有 node 加 owner pointer |
| base hook、function/value traits、fancy/offset pointer | 当前需求只要求 standard-layout member hook；polymorphic/shared-memory 等高级形状直接借用 Boost.Intrusive |
| intrusive `slist` / hlist | 单指针 footprint、tail/cache 与 unlink 合同是另一算法；当前 cancellation/free-list 特化不证明一个公共 slist 产品 |
| range erase convenience | safe/auto hook 都要逐项恢复未链接状态，循环 `erase` 已是相同 O(k)，没有复杂度缺口 |
| sort/merge/reverse/unique/remove-if 全套 | 本模块是链接 primitive，不是 `std::list` replacement；需要完整 STL-like intrusive algorithms 时借用 Boost，owning 数据优先标准容器 |
| disposer/allocator ownership | `consume_all(noexcept fn)` 已提供显式 handoff；allocator、delete policy 和异常恢复必须由 node owner 决定 |
| 内部同步、并发 reclamation | 属 sched/container concurrency 或消费方；不能让单线程 timer/list 为未命名协议支付原子与 owner 税 |

Folly 同时公开 auto `IntrusiveList` 与 safe `SafeIntrusiveList`，其 Semaphore/TimedMutex 等显式
retirement waiter 使用 safe 形状；Boost.Intrusive 为两种 link mode 提供同一算法底座。两者共同
构成生命周期合同与性能 peer。LLVM `simple_ilist` 的 range splice 与 node-to-iterator 用来检查
通用能力形状；oneTBB 的 intrusive list 位于内部 detail，不构成 HYLE 公共 parity 目标。

## Native FFI 与交付

Native FFI 为 `N/A`。模板参数、pointer-to-member、caller-owned C++ object lifetime 和 inline
iterator 都没有自然的窄 C ABI；当前也没有已命名外语消费者。若未来出现外语 timer/runtime
任务，应由拥有 opaque node 与 callback lifetime 的上层模块按
[Native FFI 准入策略](native-ffi-policy.md)建立 adapter，不导出 raw hook/list layout。

源码、installed static 与 installed shared SDK 中，`hyle::container_intrusive` 都是同一个
header-only imported/interface target；static/shared 选择不会产生 intrusive DLL、导出符号或运行时
stage。完整 container 与 time 通过真实 target edge 传播该依赖；sched 的专用 waiter 链不建立一条
虚假的 direct dependency。

只有真实消费者要求新 hook/link policy、受支持 compiler ABI 使 member recovery 不成立、发现
lifetime/correctness 缺陷，或同合同 peer 证明 material 的复杂度/消费者性能优势时才重开产品
形状。API 数量、LOC、另一个 list microbenchmark 或亚纳秒 pointer-store 差异不是重开理由。
