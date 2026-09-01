# HYLE container_intrusive：稳定节点的零分配链表

> 本页描述当前 C++23 API 和使用合同，不是稳定性或性能签署。唯一施工状态见
> [implementation/status](../implementation/status.md)。

`IntrusiveList` / `SafeIntrusiveList` 适合节点已经由 task、timer、连接、对象池或 stack frame 拥有，
并且需要零分配、O(1) 任意摘链或整段 handoff 的场景。List 不拥有 node；把普通 owning sequence 改成 intrusive
结构会把地址与析构顺序责任交给调用方，因此没有明确需求时优先使用 `std::vector`、`std::list`
或其他 owning container。

## 接入

侵入式链表属于 Container 产品，用户只选择一个稳定入口：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS container)
target_link_libraries(my_target PRIVATE hyle::container)
```

源码集成在 `add_subdirectory` 前设置 `HYLE_COMPONENTS=container`，再链接 `hyle::container`；不要
手工添加 include directory。`container_intrusive` 仍是 HYLE 内部独立、header-only factor，所以 Time
可以只依赖它而不会在内部图拉入完整 Container/Mem/Platform 闭包；该内部 target 不属于
`find_package COMPONENTS` 或用户兼容面。独立安装包 consumer 见
[`samples/package/container_intrusive/cpp`](../../samples/package/container_intrusive/cpp)。

## 先选择析构合同

| 任务 | 类型 | node 析构规则 |
|---|---|---|
| node 允许先于 list 销毁，希望 membership 自动撤销 | `IntrusiveListHook` + `IntrusiveList` | auto-unlink；析构必须与其他 list 访问同步 |
| owner/state machine 要求只在自己的锁或阶段内显式摘链，漏摘就是 bug | `SafeIntrusiveListHook` + `SafeIntrusiveList` | 先 `remove/erase/pop/clear`；仍 linked 时 Debug fail-fast，Release 为前置条件违反 |

两种形状的 hook/list 在 64-bit 上都分别只有两个 pointer，操作面和重链算法相同。Safe 不是“更慢的
检查模式”：它在 Release 析构不保留常态检查分支，而是禁止 auto-unlink 改写 owner 已失去同步保护
的共享链。需要一个 node 同时加入多条链时，每种 membership 仍各放一个匹配的 hook。

## 最小使用

```cpp
#include <hyle/container/intrusive_list.h>

struct Request {
  int id;
  hyle::container::IntrusiveListHook ready_hook{};
  hyle::container::IntrusiveListHook timeout_hook{};

  explicit Request(int value) : id(value) {}
  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;
};

using ReadyList =
    hyle::container::IntrusiveList<Request, &Request::ready_hook>;
using TimeoutList =
    hyle::container::IntrusiveList<Request, &Request::timeout_hook>;

Request first{1};
Request second{2};
Request third{3};
ReadyList ready;
ReadyList incoming;
TimeoutList timeouts;

if (!ready.try_push_back(&first)) return 1; // null/already-linked 可恢复
incoming.push_back(&second);                // 已证明该 hook 未链接
incoming.push_back(&third);
timeouts.push_back(&first);                 // 不同 hook，可同时 membership

auto first_moved = incoming.iterator_to(&second);
ready.splice(ready.end(), incoming,
             first_moved, incoming.end()); // O(1) range handoff

while (Request* request = ready.pop_front()) {
  request->id += 10;                        // 已摘链，storage 仍归调用方
}
first.timeout_hook.unlink();                // 不需要 TimeoutList reference
```

要求显式 retirement 时只替换 hook/list 类型，并经 list 摘链：

```cpp
struct Timer {
  hyle::container::SafeIntrusiveListHook hook{};
};
using TimerList =
    hyle::container::SafeIntrusiveList<Timer, &Timer::hook>;

Timer timer;
TimerList timers;
timers.push_back(&timer);
timers.remove(&timer); // timer 析构前必须完成；Safe hook 没有 unlink()
```

Hook 必须是 node 的直接非静态数据成员，但不要求 node 是 standard-layout，hook 也不必是首成员。
定义 list alias 时 `T` 必须完整，`&T::hook` 必须在该位置可访问；把 hook 设为 public 最简单。若业务
必须隐藏 hook，可像 `time::TimerEntry` 一样给一个只负责形成 alias 的命名 adapter 授予 friendship：

```cpp
struct PrivateNodeListAccess;

class PrivateNode {
public:
  PrivateNode() = default;

private:
  friend struct PrivateNodeListAccess;
  hyle::container::IntrusiveListHook hook_{};
};

struct PrivateNodeListAccess {
  using Type = hyle::container::IntrusiveList<PrivateNode, &PrivateNode::hook_>;
};
using PrivateNodeList = PrivateNodeListAccess::Type;
```

链中的 hook 必须始终属于一个仍存活的完整 node，且 node 地址稳定；不能把 linked node 放进可能
grow/relocate 的 `std::vector<Request>`。HYLE 只接受已验证的 data-member-pointer ABI 表示，未知或
更复杂的表示会编译期 fail closed，而不会猜测 enclosing object 地址。一个 hook 同一时刻只能属于一条
list；要同时处于 ready/timeout 两种 membership，就像示例一样嵌入两个 hook。

## 所有权与析构

- list 不分配、不销毁 node，也不延长 node lifetime；pointer/reference 永远是借用。
- auto hook 析构会从当前 list 摘链；safe hook 要求 node 析构前已显式摘链。list 析构会把两种
  hooks 都恢复为未链接，但不调用 node destructor。
- hook copy/move construction 得到新的未链接 hook，source membership 保持；hook assignment 保留
  destination membership，不会导入 source membership。
- auto list 的静态 `swap_nodes(left, right)` 是显式例外：它在 O(1) 内交换该 list 类型所选 hook 的
  链中位置。两者都 linked 时交换位置；只有一个 linked 时 membership 转移给另一 node 的同名 hook；
  两者都未链接时无变化。两个参数必须非 null。它只交换 membership，不移动 payload，也不会让 hook
  的普通 move 自动转移 membership。
- 因此复制或移动整个 node 不会“转移链表身份”。有稳定 identity 的 node 通常应像示例一样直接
  删除 copy/move，避免把 payload 从仍在链中的原 node 移走。
- `linked()` 只回答“该 hook 是否在某条 list”，不能识别是哪一条。模块不保存 owner pointer；只有
  auto hook 提供幂等 `unlink()`。只有 auto list 提供 typed `swap_nodes()`；safe list 必须由 owner
  明确改变 membership，因而没有该入口。

例如 `ReadyList::swap_nodes(&first, &second)` 只会交换两个 `ready_hook`。API 不接收裸 hook，因而不能把
`first.ready_hook` 与 `second.timeout_hook` 混在一次调用里；node 类型或 list 所选 member 不一致时也
没有可调用的重载。这是编译期 membership 边界，不是运行时 owner 检查。

`swap_nodes` 只修改 hook topology，不会同步业务侧的 owner pointer、count、state、deadline 或其他
membership 镜像。存在这些状态时，调用方必须在同一把外部锁/同一事务中一起更新。不要把它机械放进
通用 node move assignment：movable holder 必须先明确旧 destination membership 是摘除、保留还是交换，
再显式调用对应 `NodeList::swap_nodes`。

Auto-unlink 不是并发 reclamation。若一个线程遍历而另一线程析构 node，即使析构会摘链也仍是 data
race/UAF；insert、remove、iteration、hook/list destruction 必须使用同一个外部同步协议。

## 常用操作

| 任务 | API | 复杂度 / 结果 |
|---|---|---|
| 判断空表 | `empty()` | O(1) |
| 访问两端 | `front()` / `back()` | O(1)，空表返回 null |
| 插入两端 | `push_front/back(node)` | O(1)，前置条件由调用方证明 |
| 可恢复插入 | `try_push_front/back(node)` | null 或 already-linked 返回 false，不修改链 |
| 位置插入 | `insert(pos,node)` / `try_insert` | O(1)；两者都要求 pos 属于本表 |
| 两端摘链 | `pop_front/back()` | O(1)，空表返回 null；返回 node storage 借用指针 |
| 已知 node 摘链 | `remove(node)` / auto `node->hook.unlink()` | O(1)；前者要求属于该表；后者只属于 auto hook 的 owner 可选场景 |
| 已知 node 取得位置 | `iterator_to(node)` | O(1)；要求 node 已链接且属于该表 |
| iterator 摘链 | `erase(pos)` | O(1)，返回 successor |
| 单项/整链/range 转移 | `splice(...)` | O(1)，不访问 transferred nodes |
| 尾接整链 | `append(source)` | O(1)，source 变空 |
| 交换两个 auto membership 的位置 | `NodeList::swap_nodes(left,right)` | O(1)；两指针非 null；绑定同一 `T` 和 member hook；safe list 不提供 |
| 交换/移动 list | `swap`、move construction/assignment | O(1) pointer relink |
| 全部摘链 | `clear()` | O(n)，不销毁 node |

模块有意没有 `.size()`。它不保存 count，也不能让线性 member `size()` 使 C++23 range 错误建模为
amortized-O(1) `sized_range`；确需冷路径计数时显式使用 `std::ranges::distance(list)`。需要始终
O(1) size/owner check 时，应选择拥有这些 metadata 的上层类型，而不是让每次 mutation 付计数税。
若需求确实是通用 counted intrusive list，可在定义 node/hook 时选择 Boost.Intrusive 的
`constant_time_size<true>`；不要在 HYLE list 旁机械维护一份看似通用、却未与 range splice/move/swap
共同定义的 shadow count。出现具名 HYLE consumer 后，才会评估不影响默认形状的独立 counted type。

## `try_*` 能检查什么

`try_push_front`、`try_push_back` 和 `try_insert` 只检查 node 是否为 null、所选 hook 是否已经
linked。它们不检查 iterator owner，也不能判断 linked hook 属于哪条 list。因此：

- null/already-linked 是正常分支时使用 `try_*`；
- 调用方已经由业务状态证明前置条件时可用普通 O(1) 入口；
- `try_insert(pos,node)` 仍要求 `pos` 属于 destination；
- `iterator_to`、`erase/remove` 与 splice 的 node/position/range 都是 unchecked contract；
- 同表 range splice 时，`pos` 不能位于 `[first,last)` 内部；空 range 或已经紧邻 position 的 range
  是 no-op。

违反 unchecked 前置条件可能导致 UB 或破坏链，不会返回 `Status`。这不是运营失败：模块没有
allocation、系统调用或容量失败。若输入不可信，应在拥有 owner/token/状态机的上层先验证。

## Iterator 与失效

Iterator 是 bidirectional。把摘链后仍存活的 hook pointer 当作 iterator 使用并不安全：虽然直接
解引用可能仍能看到 node，`++/--` 会访问已清空的 link。请按下表处理：

| 事件 | 规则 |
|---|---|
| push/insert | 既有 node iterator 保持有效 |
| erase/remove/pop/auto `hook.unlink()` | 被摘链 node 的 iterator 失效；其他 node iterator 有效 |
| auto `NodeList::swap_nodes(left,right)` | 该 list 所选 hook 对应的两个 node iterator 失效；其他 node iterator 有效 |
| 单项、整链或 range splice/append | transferred-node iterator 保持有效，并开始在 destination 中遍历 |
| move construction | source node iterator 跟随 nodes，source 变空；source/destination 的 `end()` 不随 nodes 转移 |
| move assignment | 两条 membership O(1) 交换；source nodes 进入 destination，旧 destination nodes 进入 moved-from source；source 有效但不保证为空；两个 `end()` 均不转移 |
| `swap` | node iterator 保持有效，membership 交换；每个 list 对象自己的 `end()` 不交换身份 |
| `clear` / list destruction | 所有 membership iterator 对遍历失效；caller-owned node 可继续存在 |
| auto node destruction | 指向该 node 的 iterator/reference/pointer 全部随 storage lifetime 结束而失效；hook 自动摘链 |
| safe node destruction | 必须已经显式摘链；仍 linked 时是程序不变量错误 |

不要保存一个 list 的 `end()` 再假定它会随 splice/move/swap 的 nodes 到另一条 list。只有 node
iterator 保持稳定，sentinel iterator 始终属于具体 list 对象。

## 批量 callback

`consume_all(fn)` 先把 sentinel 置空，再逐项在 callback 前摘除当前 node。`fn` 必须
`noexcept`，可以销毁当前 node，或把当前 node 重链到任意合适的 list；不得销毁尚未访问的后继链
或正在执行的 list。它适合 ready/completion batch 的 drain。

`for_each(fn)` 不改变结构，并根据 callback 的异常声明传播异常；callback 不得摘链、销毁或重定位
当前/后继 node。普通只读/可写访问也可以直接 range-for。

若业务要交付“满足条件的最长前缀”，在业务模块中用 iterator 找到 `last`，完成自己的 owner/state
更新，再调用 range `splice(dst.end(), source, first, last)`。选择条件属于业务策略；intrusive list
只负责把已选定的连续区间 O(1) 重链，因此不会为某一种 timer/queue 形状增加组合 API。

## 不适合本类型的任务

- owning sequence、需要容器销毁元素：使用标准或 HYLE owning container；
- 始终 O(1) count、exact owner/contains：选择保存相应 metadata 的上层类型；
- O(1) counted list：当前没有同语义 HYLE 消费者，且会增加 list metadata 与每次 mutation 写；先确认
  count 是 membership 数而非业务/发布状态，通用需求直接选 Boost.Intrusive
  `constant_time_size<true>`；
- untyped `hook.swap_nodes(other_hook)`：两个 ready/timeout hook 具有同一 C++ hook 类型，裸 hook 入口
  无法阻止跨 membership 合同混用；使用绑定 `T` 与 member hook 的 `NodeList::swap_nodes(left,right)`；
- normal-link、base/function hook、shared-memory offset pointer 等完整策略矩阵：使用
  Boost.Intrusive；node 本身是否 polymorphic/standard-layout 不是 HYLE 的独立限制，但其直接 member
  hook 必须通过当前编译器的 data-member-pointer ABI gate；
- 单指针 intrusive slist/hlist：使用由真实状态机拥有的专用结构，当前没有公共 slist；
- 需要 sort/merge/reverse/unique/remove-if 等完整 `std::list` 风格算法：使用 Boost.Intrusive 或 owning
  标准容器；
- 需要内部同步、lock-free 或 reclamation：选择 sched/container 的并发机制。

本模块的 Native FFI 为 `N/A`。Pointer-to-member、C++ node lifetime 和 inline iterator 没有自然的
窄 C ABI；当前 source/static/shared 都是 C++ 交付，shared SDK 也不会产生
独立 `container_intrusive` DLL、用户 package component 或 raw-hook export。
