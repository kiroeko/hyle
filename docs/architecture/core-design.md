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
version_defs/cache_defs/compiler_defs ──► C 与 C++ 公开头
version.h + hyle_core ────► runtime version
cache.h/macros/c/compiler/visibility ─► 全库 header seam
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
| cache hints | `prefetch_read` / `prefetch_read_non_temporal` / `prefetch_write` | 有复用读、一次性流式读、即将修改三种真实 memory-access intent；只在 profile 后使用 |
| compiler/fatal macros | `macros.h` / `hyle/c/compiler.h` / `visibility.h` | C/C++ restrict、assume、flatten、unroll 与 C++ inline/noinline、branch、EBO、导出、不可恢复合同；主要供 HYLE 实现和高级 kernel 用户 |
| intrusive batch/list | `IntrusiveListHook` + `IntrusiveList<T, Hook>` | stable-address、调用方拥有节点、O(1) arm/cancel/splice；time delivery/bucket 是生产消费者 |

### 语法与语义消费者账本

| core 职责 | 显式消费者 | 语义重复扫描结论 |
|---|---|---|
| version single source | root CMake、runtime C/C++、plugin/FFI 诊断 | 无第二份手写版本；CMake 从 defs 读取并由 runtime static_assert |
| static cache layout | container queues/maps/interner、mem pool、scheduler | false-sharing spacing 全部采用 core；SIMD/entry scratch 的 64B 是向量/对象 alignment，不属于 cache 重复 |
| prefetch intent | SwissTable、SIMD/vqsort | vqsort 私有 helper 已收束；没有其他 builtin/intrinsic 旁路 |
| compiler contract | SIMD/vqsort 及全库 fatal/inline 用户 | vqsort 的 assume/noinline/restrict/flatten/unroll 分支已映射到 core；无第二套同义宏 |
| intrusive stable-node list | time TimerQueue/CommonTimeoutQueue，sched 经 time 消费 | 无第二个同职责 intrusive doubly-list；container 的 owning/其他队列是不同产品 |

扫描必须按语义分类：`alignas(64)` 不能机械判成 cache 重复。AVX-512 lane/table、SIMD state 和
支持 64B over-aligned entry 的 scratch 是数据 alignment；只有表达 false-sharing spacing 的
StringInterner shard 属 core 职责，并已采用 destructive-interference 常量。

### 成熟库能力与形状对账

| 参考 | 借鉴的问题/形状 | HYLE 裁决 |
|---|---|---|
| oneTBB | compile/runtime version identity 与底层信息分层 | 版本单源；不引未发布阶段的函数表协商 |
| Abseil | local、NTA、for-write 三种 prefetch intent | C/C++ header-inline 对齐；不增加 cache residency 承诺 |
| Highway | noinline/assume/restrict/flatten/unroll 的 kernel compiler contract | 收为 core source seam；hint 无独立产品 benchmark |
| Boost/Folly intrusive | member hook、auto-unlink、无 owner/size 的 O(1) hot path | stable-node batch 采用；不扩成 owning/STL container |

这张表用于查漏和裁决算法/类型，不按参考库 feature 数量扩 core。runtime topology 留 platform，
owning container 留 container，error/clock/logging 留各自模块。

版本定义只有一份公开宏源；CMake 在 `project()` 前读取它，`version.cc` 再静态断言 runtime
编译定义，component package、loader identity、header 与 runtime 不存在第二份手写版本。
`version_string()` 返回字面量静态存储，不分配、不使用 guard variable。

三种 prefetch 对齐 Abseil 的 local/NTA/for-write 形状，在 GCC/Clang、MSVC x86 与 MSVC ARM64
分别映射 builtin/intrinsic；不支持时安全退化。C++ 与 C 的 force-inline、noinline、branch
hint、unreachable、assume、restrict、flatten、unroll 共用 `compiler_defs.h` 的工具链判断；
两种语言另各提供 release-surviving fatal assert。它们只表达编译器/程序合同，不
增加运行时状态。C++ 形状由 SIMD/vqsort 的 ported kernel 真实消费；纯 C 形状让外部 C kernel
取得同档 source contract，而不复制 compiler branches。assume/restrict 的错误承诺是调用方
bug；unroll 与 prefetch 都是可忽略的性能 hint。

cache-layout 语义消费者必须使用 core 常量，不能硬编码 64B；container 的 queue/map/interner、
mem pool 与 scheduler 状态据此在 Apple Silicon 使用 128B。SIMD lane scratch、AVX-512 table 和
“最多接受 64B aligned entry”的局部缓冲仍保留 `alignas(64)`：它们表达向量/对象 alignment，
不是 cache residency 或 false-sharing spacing，不能机械替换成 cache-line 常量。

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
list 前必须 clear 或逐个 pop。缓存布局宏由 C/C++ 共用 `cache_defs.h`，compiler contract 共用
`compiler_defs.h`，避免跨语言判断漂移；C 的 compiler 宏与三种 prefetch 同样 header-inline，
不引入动态 ABI 或热路调用。

动态 C ABI 只有 `hyle_version` 与 `hyle_version_string`；其余能力是编译期布局或纯指针模板/
inline 原语，导出函数反而会增加调用成本且不能改善 ABI 表达。

能力映射按用户任务而不是符号镜像：

| 能力 | C++ source API | C/FFI 产品面 | 边界理由 |
|---|---|---|---|
| compile/runtime version | constexpr + 两个 runtime 函数 | 同一 defs + 两个 C 符号 | plugin/loader 需要动态 identity |
| cache layout/prefetch | typed constexpr + inline 函数 | macro + inline 函数 | 布局跨语言一致，热路无 DLL hop |
| compiler contract | inline/noinline/branch/unreachable/assume/restrict/flatten/unroll + fatal | 同名 `HYLE_C_*` 集合 | 源码级 promise/hint；共用 backend，不存在有意义的动态调用 ABI |
| intrusive list | typed member hook + RAII/callback 便利 | 可见 hook/sentinel + inline 指针原语 | C 显式 lifetime；batch/遍历可同复杂度组合 |

因此“C 能力完整”指版本、cache intent、compiler contract、stable-node queue 等自然 C 用户
任务都有同档通路，不表示把 C++ 析构或模板 callback 逐符号翻译一遍。compiler contract 单独
放在 `hyle/c/compiler.h`，`hyle/c/core.h` 只为可发现性转 include；两者都不增加动态符号。

## API 易用性结论

- 默认旅程：按任务直接包含 version/cache/intrusive 单头；没有强迫所有用户引入 umbrella。
- 高级旅程：compiler contract 单独在 `macros.h`，危险 UB/代码体积合同不混进普通业务 API。
- 组合旅程：time 直接以 typed intrusive list 组合；container/mem/sched 直接消费 cache 常量，
  没有 C ABI 往返或 adapter 链。
- C 旅程：kernel-only 可单独 include compiler 头；版本为两个动态 identity 符号，其余热路
  header-inline；显式 init/unlink/clear 对应 C 生命周期，batch/遍历可由基础原语保持同一复杂度组合。

上述默认/高级/组合路径分别由 core C++ tests、SIMD/vqsort 消费编译和 pure-C smoke 机械覆盖；
shared/static 安装后还由 `samples/package/core/{c,cpp}` 只请求 `core` 组件独立复证。guide
给出入口选择与可复制 recipe。新增 source/compiler 便利必须先有真实扩展消费者，不能让
core 重新长成杂项 umbrella。

## 明确负空间

- runtime CPU/OS 信息归 platform；core 不重复 CPUID/sysctl。
- allocator、error、logging、clock 不进入零依赖根。
- 不提供拥有型 list、通用 iterator/ranges 或同步包装。
- 不提供显式 L1/L2/L3 cache 对象或 portable cache residency 保证；prefetch locality 只是 hint，
  runtime cache 拓扑仍归 platform。
- 不使用 `std::hardware_destructive_interference_size` 参与公共布局：其值受编译参数/标准库
  影响，可能让不同 TU 得到不同 ABI；HYLE 用受支持平台的稳定常量，精确值另走 platform。
