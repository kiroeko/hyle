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
#include <cstdio>

#include "hyle/core/version.h"

static_assert(hyle::kVersion >= HYLE_VERSION_PACK(0, 1, 0));
std::printf("headers=%s runtime=%s\n", hyle::kVersionString,
            hyle::version_string());
if (hyle::version() != hyle::kVersion) {
  // 动态库/plugin 搜索路径可能装载了另一版 HYLE。
}
```

只编译当前源码时用 `kVersion*`；动态加载、plugin/FFI 边界诊断用 `version()`。它比较的是
语义版本，不是 Git commit；同一语义版本的 source provenance 由发布 manifest/SBOM 记录。
版本号不承担 feature negotiation，预发布阶段也没有版本函数表。

## 缓存：布局常量还是运行时事实

```cpp
#include <atomic>
#include <cstdint>

#include "hyle/core/cache.h"

struct alignas(hyle::core::kDestructiveInterferenceSize) HotCounter {
  std::atomic<std::uint64_t> value{0};
};
```

- 让高频跨线程写状态互不干扰：`kDestructiveInterferenceSize` 是稳定 128B 上界。
- 把共同读取的小字段尽量放进一个 line：`kConstructiveInterferenceSize` 是稳定 64B 下界。
- 两者都是公共布局常量，不是“当前 CPU 的精确 cache-line size”。
- 算法决定一次处理/flush 多少字节：用 `hyle::platform::cache_line_size()` 取得运行时事实。
- 随后还会反复读取同一 cache line：`prefetch_read(ptr)`。
- 大片流式读取、预计只消费一次：`prefetch_read_non_temporal(ptr)`；目标不支持时允许退化。
- 即将修改、希望 capable CPU 提前取得 exclusive ownership：`prefetch_write(ptr)`；它不是写入，
  也不能消除原子或锁。编译器只能选择当前 translation unit ISA baseline 允许的 hint；若目标
  不支持 write-prefetch，可退化为 read-prefetch 或 no-op，不会越过 `-march` 强发指令。

三种 prefetch 都只放在 profile 证明有收益的位置。它们是 best-effort hint，不检查对象生命周期，
不建立地址有效性、内存序或线程可见性；过早、过晚或错误 locality 的 prefetch 反而会增加带宽
和 cache 压力。普通顺序遍历先相信硬件 prefetcher，不要默认手工插入。

128B destructive spacing 会多占空间，只有确实存在跨核写竞争的状态才使用；普通对象不要默认
整行 padding。SIMD 最大 alignment 是另一件事，不要拿 interference 常量替代
`mem::kMaxSimdAlignment`。C++17 以后普通 `new` 和标准 allocator 会为 over-aligned 类型选择
对应分配入口；若把这种类型放进自管 raw storage，仍须由 backing allocation 保证至少 128B
alignment，只有把类型声明成 `alignas` 并不能修正一个已经错位的地址。

这两个 bound 只认证 runtime coherency line 落在 `[64,128]` 的当前目标。platform 若在新
generic/RV64 机器报告更小或更大的值，精确探测仍然有效，但当前静态 layout 合同不成立；该
目标必须重新裁决常量/ABI，不能带着“支持”标签发布。

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
for (Request& item : ready) {
  item.id += 1;
}
Request* next = ready.pop_front();
```

批量交接和任意取消不需要另造 batch 类型：

```cpp
Request a{1}, b{2}, cancelled{3};
ReadyList ready;
ReadyList incoming;

ready.push_back(&a);
auto ticket = ready.insert(ready.end(), &cancelled);
ready.erase(ticket);                    // 返回 successor；cancelled 已摘链
incoming.push_back(&b);
ready.splice(ready.end(), incoming);    // O(1) 整批交接，incoming 变空

ReadyList deferred;
deferred.splice(deferred.end(), ready, ready.begin()); // O(1) 转移单节点
ready.swap(deferred);
```

它不拥有节点、不分配，push/pop/remove/insert/erase/splice/swap 都是 O(1)。支持双向 iterator、
range-for 与 reverse iteration；偶发计数用 `std::ranges::distance`（O(n)），没有每次修改都维护的
size/owner 元数据。节点析构或 list 析构会自动摘链。
copy/move 构造 hook 会得到新的 unlinked hook；对已有 hook 做 copy/move assignment 则刻意保留
destination 当前 membership，绝不从 source 复制 membership。节点 linked 时不可移动地址。
节点类型必须是 standard-layout；模板会机械拒绝不满足的类型，因为从 member hook 恢复外层对象
依赖该布局合同。
一个对象若同时进入 ready 与 timeout 两条链，需要两个独立 hook。跨线程并发需要外部同步。
`push_back/front` 要求 node 非空且该 hook 当前未链接；输入可能重复时改用返回 bool 的
`try_push_back/front`。`try_*` 与返回已摘节点的 `pop_*` 都是 must-use：前者不检查结果就不知道
操作是否发生，后者不接住指针就容易丢失调用方拥有的节点；若 membership 已由不变量证明，直接
用 unchecked `push_*`，不要调用 checked 入口再忽略结果。

下列情况不要用它：希望容器拥有元素、需要随机访问/常量时间 size、节点地址会因 vector
扩容改变，或只是普通小集合。此时优先选 `SmallVector`、`PodVector`、`FlatHashMap` 或标准容器。

`insert/erase/splice` 接受 conventional `const_iterator` position（普通 iterator 可隐式转换），
且它必须属于文档指定的 list；除被摘除的节点外，iterator 在插入、splice 与 swap 后仍指向同一
节点，但旧 list 的 `end()` 不能跨这些结构变更保存。遍历时删除当前节点，用
`it = list.erase(it)` 取得 successor。`remove(node)` 要求 node 非空且属于该 list；不知道是否 linked 时用幂等
`node.hook.unlink()`。`consume_all` 的 callback 执行前当前节点已经 unlinked，可以销毁或
加入另一条链，但 callback 必须是 `noexcept`。`for_each` callback 不得改变
membership/lifetime；它可以抛异常，此时遍历停止而链保持有效。`move_prefix_to` 的 predicate
与 `on_move` 必须 `noexcept`，且扫描时节点仍在 source 链中，不得摘链、销毁或重链；完整前缀
在回调完成后才以 O(1) 区间 splice 转移。

whole-list `splice(pos, source)`、`append(source)` 与 `move_prefix_to(destination, ...)` 要求两条
list 不是同一个对象；违反是程序 bug，未定义 `NDEBUG` 时会诊断并 abort，定义后属于 optimizer
合同违约。single-node `splice(pos, same_list, item)` 则明确支持同链重排，把节点移到自己或自己
successor 前是 no-op；`swap` 的 self 调用也安全。不要把 whole-list self-transfer 和这两个安全
特例混为一谈。

## 宏的使用边界

`HYLE_FORCE_INLINE`、`HYLE_NOINLINE`、`HYLE_NODISCARD`、`HYLE_LIKELY`、`HYLE_NO_UNIQUE_ADDRESS`、
`HYLE_DISABLE_COPY*` 与 `HYLE_FATAL_ASSERT` 主要用于维护 HYLE 或编写同层扩展。业务输入错误
不要用 fatal assert；应走 error Status/Result。缓存、prefetch 已有类型化 API，不再使用旧的
裸宏。

写数据核时还有四类危险但有用的编译器合同：

- `HYLE_ASSUME(expr)`：未定义 `NDEBUG` 时诊断 false，定义后把 false 变成 undefined behavior；表达式必须
  无副作用，绝不能用于验证用户输入。
- `HYLE_RESTRICT`：只放在确实互不别名的 pointer declaration 上；承诺错误是 undefined behavior。
- `HYLE_FLATTEN`：只放在经过 profile 的深 wrapper/template 入口；要求 GCC/Clang 尽量内联函数体
  内的可内联调用。它可能显著增加 code size、伤害 I-cache，MSVC/未知编译器允许退为空。
- `HYLE_UNROLL(n)` / `HYLE_DEFAULT_UNROLL`：best-effort loop hint，只用于测过的 kernel；更多展开
  可能因 code size 和 I-cache 变慢。Clang 的未优化或 sanitizer TU 会把它退为空，因为该编译器
  对无法完成的显式 loop transform 发 `-Wpass-failed`；hint 不能在消费者使用 `-Werror` 时变成
  correctness build failure。

`HYLE_NODISCARD` 放在必须检查返回值的函数声明前；它只产生编译期诊断。GCC/Clang 的
`HYLE_FORCE_INLINE` 在优化构建才强制 inline，`-O0` 退成普通 inline，让未优化构建保持可编译。
`HYLE_UNREACHABLE()` 只用于已经证明不可能到达的控制流：未定义 `NDEBUG` 时会先打印位置并
abort，定义后把抵达视为 undefined behavior。常规 CMake Debug/Release 会给出这两种组合；若用户
手工混用 `NDEBUG` 与优化开关，则分别按 translation unit 的宏定义和真实优化状态生效。

这些形状分别对应 Abseil/Highway 的 assume、Highway restrict/flatten/unroll，并由 HYLE 的 SIMD/vqsort
真实消费；它们是 source/compiler seam，不是动态 ABI，也不是通用业务 API。

纯 C/FFI 用户见 [`c-core`](./c-core.md)。
