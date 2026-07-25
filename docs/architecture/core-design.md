# core 模块架构

> 本页维护 core 的稳定设计。施工证据见
> [`core-module-closure`](../milestones/core-module-closure.md)，性能只见
> [`benchmarks-core`](../performance/benchmarks-core.md)，用户选型见
> [`guide/core`](../guide/core.md) 与 [`guide/c-core`](../guide/c-core.md)。

## 定位与边界

`core` 是零 HYLE 上游的源码与 ABI 根，承载四类能力：编译期/运行时版本、稳定缓存布局与
prefetch、编译器属性/合同宏、零分配侵入链表。`macros.h`、`visibility.h` 虽位于 `hyle/`
根目录，也属于 core 的实现边界；目录位置不能把它们变成无主资产。

```text
version_defs/cache_defs ──► C 与 C++ 公开头
version.h + hyle_core ────► runtime version
cache.h/macros/visibility ─► 全库 header seam
intrusive_list.h ─────────► time ─► sched/io
```

core 不依赖 error/platform/mem/time，也不创建资源，因此没有 Status 或 allocator seam。
运行时探测仍归 platform；core 只提供必须参与对象布局的编译期常量。

## 能力与真实场景

| 能力 | 入口 | 场景与形状 |
|---|---|---|
| header version | `kVersion*` / `HYLE_VERSION_*` | 编译期 feature/log 标记，无 DLL 调用 |
| linked runtime version | `version()` / `version_string()` | plugin、FFI、动态加载后诊断 header/runtime 是否匹配 |
| static cache layout | `core::k*InterferenceSize` / `HYLE_STATIC_CACHE_LINE_SIZE` | `alignas`、结构体 ABI、伪共享隔离；Apple Silicon 128B，其余当前支持目标 64B |
| runtime cache fact | `platform::cache_line_size()` | 算法按当前 CPU 精确取值，不参与稳定类型布局 |
| prefetch | `core::prefetch_read` | 已 profile 的读延迟隐藏；正确性不依赖 hint |
| compiler/fatal macros | `macros.h` / `visibility.h` | inline、branch、EBO、导出、不可恢复合同；主要供 HYLE 实现和高级用户 |
| intrusive batch/list | `IntrusiveListHook` + `IntrusiveList<T, Hook>` | stable-address、调用方拥有节点、O(1) arm/cancel/splice；time delivery/bucket 是生产消费者 |

版本定义只有一份公开宏源，`version.cc` 用 CMake project version 静态断言防止两处漂移。
`version_string()` 返回字面量静态存储，不分配、不使用 guard variable。

## IntrusiveList 合同

链表采用 Boost/Folly 同类的 circular sentinel + member auto-unlink：每个 hook 两个指针，列表
本身一个 sentinel，无 size 元数据、无分配。节点地址在 linked 期间稳定；一个 hook 同时只能
进一个 list，需要多重 membership 就嵌多个 hook。节点析构自动摘链，list 析构/`clear()`
会把所有 hook 复位；复制/移动 hook 不复制 membership。

`push_back/front` 与 Boost 的 Release 容器一样以“node 非空、该 hook 未链接”为前置条件；
输入可能重复时用返回 bool 的 `try_push_back/front`。`remove(node)` 同样以“非空且属于该
list”为前置条件，不存/查 owner；membership 可选时直接用幂等 `hook.unlink()`。time 已用
exact-owner 状态证明这些热路前置条件。类型必须是 standard-layout，member-to-parent 恢复
沿用 Boost `parent_from_member` 一类实现并 launder。

| 操作 | 复杂度 |
|---|---:|
| push/front/back/pop/remove/move-front | O(1) |
| append / list move | O(1) |
| clear / consume_all / for_each | O(n) |
| move_prefix | O(k) 扫描 + O(1) 区间重链 |

不加 O(1) `size()`：它会给每次修改增加常态元数据，而 time/batch 消费者不需要。需要 owning
元素、迭代器生态、随机访问或线程安全时应选 container 的相应类型，不把 core primitive
扩成另一个 Boost.Container。

## C/C++ 双面

C++ 使用 auto-unlink RAII 模板；C 头提供同算法的可见 hook/list 与 header-inline
push/pop/unlink/append，不产生 DLL 热路。C 无析构器，因此释放节点前必须显式 unlink，销毁
list 前必须 clear 或逐个 pop。缓存布局宏也由 C/C++ 共用 `cache_defs.h`，避免跨语言结构体
对齐不同。

动态 C ABI 只有 `hyle_version` 与 `hyle_version_string`；其余能力是编译期布局或纯指针模板/
inline 原语，导出函数反而会增加调用成本且不能改善 ABI 表达。

## 明确负空间

- runtime CPU/OS 信息归 platform；core 不重复 CPUID/sysctl。
- allocator、error、logging、clock 不进入零依赖根。
- 不提供拥有型 list、通用 iterator/ranges 或同步包装。
- 不使用 `std::hardware_destructive_interference_size` 参与公共布局：其值受编译参数/标准库
  影响，可能让不同 TU 得到不同 ABI；HYLE 用受支持平台的稳定常量，精确值另走 platform。
