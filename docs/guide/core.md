# core：基础能力怎么选

core 大多是其他模块的地基。普通用户最常直接使用的是版本诊断；只有在写高性能数据结构、
plugin 或 stable-address 节点队列时才需要缓存与侵入链表入口。

| 任务 | 直接包含 | 入口 |
|---|---|---|
| header/runtime 版本诊断 | `hyle/core/version.h` | `kVersion*`、`version()` |
| cache 布局与访问 hint | `hyle/core/cache.h` | `k*InterferenceSize`、三种 prefetch |
| 调用方拥有的稳定节点队列 | `hyle/core/intrusive_list.h` | `IntrusiveListHook` / `IntrusiveList` |
| 维护 HYLE 或编写同层 kernel | `hyle/macros.h` | compiler/fatal contract；普通业务不要默认使用 |

这些头按任务拆分，没有必须 include 的 core umbrella；普通用户不需要为版本查询引入 intrusive
模板，也不需要为 cache hint 引入 platform runtime。

## 版本：编译期还是运行时

```cpp
#include "hyle/core/version.h"

static_assert(hyle::kVersion >= HYLE_VERSION_PACK(0, 1, 0));
std::printf("headers=%s runtime=%s\n", hyle::kVersionString,
            hyle::version_string());
if (hyle::version() != hyle::kVersion) {
  // 动态库/plugin 搜索路径可能装载了另一版 HYLE。
}
```

只编译当前源码时用 `kVersion*`；动态加载、plugin/FFI 边界诊断用 `version()`。版本号不承担
feature negotiation，预发布阶段也没有版本函数表。

## 缓存：布局常量还是运行时事实

```cpp
#include "hyle/core/cache.h"

struct alignas(hyle::core::kDestructiveInterferenceSize) HotCounter {
  std::atomic<std::uint64_t> value{0};
};
```

- 类型布局、共享内存结构、数组 stride：用 core 编译期常量，确保同一目标 ABI 稳定。
- 算法决定一次处理/flush 多少字节：用 `hyle::platform::cache_line_size()` 取得运行时事实。
- 随后还会反复读取同一 cache line：`prefetch_read(ptr)`。
- 大片流式读取、预计只消费一次：`prefetch_read_non_temporal(ptr)`；目标不支持时允许退化。
- 即将修改、希望 capable CPU 提前取得 exclusive ownership：`prefetch_write(ptr)`；它不是写入，
  也不能消除原子或锁。

三种 prefetch 都只放在 profile 证明有收益的位置。它们是 best-effort hint，不检查对象生命周期，
不建立地址有效性、内存序或线程可见性；过早、过晚或错误 locality 的 prefetch 反而会增加带宽
和 cache 压力。普通顺序遍历先相信硬件 prefetcher，不要默认手工插入。

Apple Silicon 的静态值是 128，当前其他支持目标为 64。SIMD 最大 alignment 是另一件事，
不要拿 cache-line 常量替代 `mem::kMaxSimdAlignment`。

## 侵入链表：什么时候值得用

当节点本来就有稳定生命周期，而且操作是频繁 enqueue/cancel/batch splice 时使用：

```cpp
#include "hyle/core/intrusive_list.h"

struct Request {
  int id;
  hyle::core::IntrusiveListHook ready_hook;
};
using ReadyList =
    hyle::core::IntrusiveList<Request, &Request::ready_hook>;

Request request{42};
ReadyList ready;
ready.push_back(&request);
Request* next = ready.pop_front();
```

它不拥有节点、不分配，push/pop/remove/append 都是 O(1)。节点析构或 list 析构会自动摘链；
但节点 linked 时不可移动地址。一个对象若同时进入 ready 与 timeout 两条链，需要两个独立
hook。跨线程并发需要外部同步。`push_back/front` 要求 node 非空且该 hook 当前未链接；输入
可能重复时改用返回 bool 的 `try_push_back/front`。

下列情况不要用它：希望容器拥有元素、需要随机访问/完整 STL iterator、节点地址会因 vector
扩容改变，或只是普通小集合。此时优先选 `SmallVector`、`PodVector`、`FlatHashMap` 或标准容器。

`remove(node)` 要求 node 非空且属于该 list；不知道是否 linked 时用幂等
`node.hook.unlink()`。`consume_all` 的 callback 执行前当前节点已经 unlinked，可以销毁或
加入另一条链；`for_each` callback 不得改变 membership。

## 宏的使用边界

`HYLE_FORCE_INLINE`、`HYLE_NOINLINE`、`HYLE_LIKELY`、`HYLE_NO_UNIQUE_ADDRESS`、
`HYLE_DISABLE_COPY*` 与 `HYLE_FATAL_ASSERT` 主要用于维护 HYLE 或编写同层扩展。业务输入错误
不要用 fatal assert；应走 error Status/Result。缓存、prefetch 已有类型化 API，不再使用旧的
裸宏。

写数据核时还有四类危险但有用的编译器合同：

- `HYLE_ASSUME(expr)`：Debug 诊断 false，Release 把 false 变成 undefined behavior；表达式必须
  无副作用，绝不能用于验证用户输入。
- `HYLE_RESTRICT`：只放在确实互不别名的 pointer declaration 上；承诺错误是 undefined behavior。
- `HYLE_FLATTEN`：只放在经过 profile 的深 wrapper/template 入口；要求 GCC/Clang 尽量内联函数体
  内的可内联调用。它可能显著增加 code size、伤害 I-cache，MSVC/未知编译器允许退为空。
- `HYLE_UNROLL(n)` / `HYLE_DEFAULT_UNROLL`：best-effort loop hint，只用于测过的 kernel；更多展开
  可能因 code size 和 I-cache 变慢。

这些形状分别对应 Abseil/Highway 的 assume、Highway restrict/flatten/unroll，并由 HYLE 的 SIMD/vqsort
真实消费；它们是 source/compiler seam，不是动态 ABI，也不是通用业务 API。

纯 C/FFI 用户见 [`c-core`](./c-core.md)。
