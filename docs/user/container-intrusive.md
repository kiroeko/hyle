# HYLE Container：侵入式容器怎么选

> 本页描述当前 C++23 用法。Intrusive container 不拥有 node；如果节点并非由 task/timer/pool/cache 等
> 稳定对象持有，优先选择 owning `std::vector`、`std::list` 或 HYLE Container。

## 接入

侵入式能力属于 Container 产品，源码、静态包和动态包都链接同一个 façade：

```cmake
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS container)
target_link_libraries(my_target PRIVATE hyle::container)
```

源码集成则在 `add_subdirectory(hyle)` 前设置 `HYLE_COMPONENTS=container`，再链接
`hyle::container`。不要直接选择内部 `container_intrusive` factor。完整包示例见
[`samples/package/container/intrusive/cpp`](../../samples/package/container/intrusive/cpp)。

## 先按需求选形状

| 你的需求 | 选择 | 不要误选 |
|---|---|---|
| 双向遍历、两端操作、任意连续 range O(1) 转移，node 可自行摘链 | `IntrusiveListHook` + `IntrusiveList` | auto-unlink 不是线程安全 reclamation |
| 同上，但只能由 owner 锁/状态机显式摘链 | `SafeIntrusiveListHook` + `SafeIntrusiveList` | Safe 不是“慢速检查版” |
| safe 双向 membership 且频繁读取 O(1) 数量 | `SafeIntrusiveListHook` + `CountedIntrusiveList` | 不要在 uncounted list 旁维护容易漂移的 shadow count |
| arena/FIFO/deferred chain，node 只放一个 pointer，还要 O(1) push-back | `IntrusiveForwardListHook` + `IntrusiveForwardList` | arbitrary node 没有 O(1) unlink；要带 predecessor |
| callback/bucket，head 只放一个 pointer，已知 node O(1) 注销 | auto/safe `IntrusiveHListHook` + `IntrusiveHList` | 它不是可排序的双向 sequence |

需要 owning storage、并发 publication/reclamation、intrusive tree/hash/heap、base hook 或 offset pointer 时，
这些类型都不合适；分别选择 owning/concurrent owner 或 Boost.Intrusive 的高级 policy。

## 双向 list

```cpp
#include <hyle/container/intrusive_list.h>

struct Request {
  int id;
  hyle::container::IntrusiveListHook ready_hook{};
  hyle::container::SafeIntrusiveListHook timeout_hook{};

  explicit Request(int value) noexcept : id(value) {}
  Request(const Request&) = delete;
  Request& operator=(const Request&) = delete;
};

using Ready =
    hyle::container::IntrusiveList<Request, &Request::ready_hook>;
using Timeouts =
    hyle::container::SafeIntrusiveList<Request, &Request::timeout_hook>;

Request first{1};
Request second{2};
Ready ready;
Timeouts timeouts;

ready.push_back(&first);
ready.push_back(&second);
timeouts.push_back(&first);       // 不同 hook，可同时属于两条链

while (Request* request = ready.pop_front()) {
  consume(*request);              // callback 前已经摘链
}
timeouts.remove(&first);          // safe hook：析构前显式退休
```

Auto hook 提供幂等 `hook.unlink()`；safe hook 没有该入口。两者均支持 front/back、push/pop、insert/erase、
`iterator_to`、single/whole/range `splice`、append、swap/move、clear、`consume_all`、`for_each`，以及
`reverse/remove/remove_if/unique/merge/sort`。默认 list 没有 `.size()`；冷路径用
`std::ranges::distance`，不要把 O(n) 计数伪装为 `sized_range`。

Auto list 的 `Ready::swap_nodes(&a,&b)` 只交换该 typed member 的 membership/位置，不移动 payload 或业务
owner/state/count；有这些镜像时调用方必须在同一外部事务里一起更新。Safe/count 不提供这个绕开 owner
状态机的入口。

## Counted list

```cpp
struct Waiter {
  int priority;
  hyle::container::SafeIntrusiveListHook hook{};
};
using WaitSet = hyle::container::CountedIntrusiveList<
    Waiter, &Waiter::hook>;

WaitSet pending;
Waiter a{2};
Waiter b{1};
pending.push_back(&a);
pending.push_back(&b);
assert(pending.size() == 2);       // O(1)
pending.sort([](const Waiter& x, const Waiter& y) {
  return x.priority < y.priority;
});
pending.clear();                   // node storage 不被销毁
```

Counted list 只增加 list 的一个 `size_t`；node 仍复用 safe 双向 hook。Single/whole transfer O(1)。跨 list
range splice 的普通重载会扫描 `[first,last)` 以更新两个 count，复杂度 O(k)：

```cpp
destination.splice(destination.end(), source, first, last);
```

如果 owner 已经可靠地知道 range 恰有 `n` 个元素，可使用 O(1) verified-count 重载：

```cpp
destination.splice(destination.end(), source, first, last, n);
```

错误的 `n` 是程序合同违反，不是可恢复输入。Auto-unlink 无法在不知道 list owner 时维护 count，所以没有
counted auto 类型。

## Forward list

```cpp
#include <hyle/container/intrusive_forward_list.h>

struct ArenaObject {
  int kind;
  hyle::container::IntrusiveForwardListHook chain_hook{};
};
using ObjectChain = hyle::container::IntrusiveForwardList<
    ArenaObject, &ArenaObject::chain_hook>;

ArenaObject first{1};
ArenaObject second{2};
ObjectChain batch;
batch.push_back(&first);           // cached tail，O(1)
batch.push_back(&second);

ObjectChain all;
all.append(batch);                 // O(1)，batch 变空
all.consume_all([](ArenaObject* object) noexcept {
  release_payload(*object);        // hook 已恢复 unlinked
});
```

Forward list 有 `before_begin`，位置 mutation 采用标准的 `*_after` 心智：`insert_after`、`erase_after`、
single/whole/range `splice_after`。已知 predecessor 的 single transfer 和 whole append O(1)；range 需要定位
尾节点，为 O(k)。它没有任意已知 node O(1) remove，也不自动析构摘链；node 析构前必须 pop/erase/clear。

它也提供 stable `sort/merge`、`reverse/remove/remove_if/unique`。List 缓存 tail，但 node 仍只有一个 link。

## Hlist

```cpp
#include <hyle/container/intrusive_hlist.h>

struct Callback {
  void (*fn)() noexcept;
  hyle::container::SafeIntrusiveHListHook hook{};
};
using CallbackSet = hyle::container::SafeIntrusiveHList<
    Callback, &Callback::hook>;

Callback callback{&on_cancel};
CallbackSet callbacks;
callbacks.push_front(&callback);

// 已经拿到 callback，不需要 predecessor scan：
callbacks.remove(&callback);       // O(1)
```

Hlist 的 list/head 只有一个 pointer；node 是 `next + previous-next` 两个 pointer。它支持 front、push-front、
known-position before/after insert、known-node erase/remove/replace、pop-front、clear/consume 和 swap/move。
Auto 版本允许 `hook.unlink()`；safe 版本必须经 list 退休。

Hlist 没有 back、reverse iterator、range splice、size、sort 或 merge。需要这些操作说明你的任务是 sequence，
应选择 forward 或双向 list，而不是给 hlist 补一套相互矛盾的 metadata。

## 生命周期、异常和常见坑

- Hook 必须是完整 node 的直接 member；linked 期间 node 地址稳定。不要把 linked node 放进会 relocate 的
  `std::vector<Node>`。
- 一个 hook 同一时刻只能属于一条 list；同时参与 ready/timeout 等 membership 时嵌入多个 hook。
- Hook copy/move construction 得到新的 unlinked hook；assignment 保留 destination membership。不要
  假定移动 node 会移动链身份。
- List 不销毁 node。List 可以先析构并清空 membership；safe/forward node 若先析构，必须已经显式摘链。
- `linked()` 只回答“是否在某条链”，不能回答 owner。错误 owner/iterator/range、重复 insertion 是 unchecked
  程序合同，不会返回 `Status`。
- `try_push_*` / `try_insert*` 只把 null 与 already-linked 当正常失败；它们不扫描 owner。
- `sort/merge/remove_if/unique` 的 predicate/comparator 可以抛异常。结构仍有效、所有 node 仍被某条 list
  拥有，但已完成的顺序变化不回滚。`consume_all` callback 必须 `noexcept`。
- 所有类型只保证 thread-compatible。Mutation、遍历、auto hook 析构和 list 析构必须由同一个外部同步
  协议保护；auto-unlink 不解决 race、ABA 或 UAF。

Pointer-to-member、inline iterator 与 caller-owned C++ lifetime 没有自然的窄 Native FFI。因此本能力的
Native FFI 是 `N/A`；若未来外语 runtime 需要 callback/timer，应由上层 opaque handle API 拥有 storage
和生命周期，而不是导出 raw hook layout。
