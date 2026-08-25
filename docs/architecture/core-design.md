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
| static cache layout | `core::k*InterferenceSize` / 两个 `HYLE_STATIC_*_INTERFERENCE_SIZE` | `alignas`、结构体 ABI；constructive 64B 下界，destructive 128B 上界 |
| runtime cache fact | `platform::cache_line_size()` | 算法按当前 CPU 精确取值，不参与稳定类型布局 |
| cache hints | `prefetch_read` / `prefetch_read_non_temporal` / `prefetch_write` | 有复用读、一次性流式读、即将修改三种真实 memory-access intent；只在 profile 后使用，且 compiler builtin 不越过 TU ISA baseline |
| compiler/fatal macros | `macros.h` / `hyle/c/compiler.h` / `visibility.h` | C/C++ must-use、restrict、assume、flatten、unroll、inline/noinline、branch、EBO、导出与不可恢复合同；主要供 HYLE 实现和高级 kernel 用户 |
| intrusive list/batch | `IntrusiveListHook` + `IntrusiveList<T, Hook>` | stable-address、调用方拥有节点、双向 range、O(1) arm/cancel/insert/erase/splice；time delivery/bucket 是生产消费者 |

按方法论展开后的场景合同如下；core 不分配资源，因此没有 allocator/OOM 路径：

| 场景 | 数据、ownership 与 lifetime | concurrency 与 failure | 规模/复杂度 |
|---|---|---|---|
| plugin/header-runtime identity | header 常量与 linked static string，调用方只借用、进程期有效 | 只读并发；版本不匹配由调用方诊断，不分配/抛错 | O(1)，冷路 |
| 公共 cache layout / prefetch | 调用方对象与地址；静态 bound 参与 ABI，runtime exact 归 platform | layout 可跨线程，hint 不建立同步；无效生命周期地址或错误 layout 认证是合同违约 | 编译期/O(1)，只在真实消费者承重 |
| kernel compiler contract | 调用方表达式、pointer 与 loop，零 runtime 状态 | TU-local；错误 assume/restrict/unreachable 是程序 bug，must-use 是诊断 | O(0) source seam；unroll/inline 只作 measured hint |
| stable-node queue/batch | 节点由调用方拥有且地址稳定；C++ hook/list RAII 摘链，C 显式 unlink/clear | 默认外部同步；checked insert 返回 bool，unchecked owner/membership 违约是 bug | 单节点/整批 O(1)，遍历/consume O(n)，不维护 size/owner |

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
| oneTBB | compile/runtime version identity、128B `max_nfs_size` 上界 | 版本单源；destructive 取稳定上界；不引未发布阶段的函数表协商 |
| Abseil | local、NTA、for-write 三种 prefetch intent | C/C++ header-inline 对齐；不增加 cache residency 承诺 |
| Highway | 优化构建 force-inline；Debug 普通 inline；noinline/assume/restrict/flatten/unroll | 收为 core source seam；hint 无独立产品 benchmark |
| Boost/Folly intrusive | member hook、auto-unlink、双向 iterator、无 owner/size 的 O(1) hot path | 保留一个紧凑通用 non-owning sequence；不扩成 owning/counted 容器家族 |

这张表用于查漏和裁决算法/类型，不按参考库 feature 数量扩 core。runtime topology 留 platform，
owning container 留 container，error/clock/logging 留各自模块。

本轮负空间也显式裁决：C++ alignment promise 已有 `std::assume_aligned`，SIMD aligned Load 又
直接表达 kernel 合同，没有纯 C 消费者要求另一套宏；noreturn/fallthrough 分别已有标准 C/C++
拼法，因此不包装。`lifetimebound` 对 `Status::message()` 和 container borrowed view/reference
是真实的零成本诊断候选，但它需要跨模块统一注解与 compile-diagnostic 门，留到 error/container
复审共同裁决，不在 core 为凑 feature 数量先放一个无人正确使用的宏。

版本定义只有一份公开宏源；CMake 在 `project()` 前读取它，`version.cc` 再静态断言 runtime
编译定义，component package、loader identity、header 与 runtime 不存在第二份手写版本。
`version_string()` 返回字面量静态存储，不分配、不使用 guard variable。

三种 prefetch 对齐 Abseil 的 local/NTA/for-write 形状，在 GCC/Clang、MSVC x86 与 MSVC ARM64
分别映射 builtin/intrinsic；不支持时安全退化。C++ 与 C 的 must-use、force-inline、noinline、branch
hint、unreachable、assume、restrict、flatten、unroll 共用 `compiler_defs.h` 的工具链判断；
两种语言另各提供 release-surviving fatal assert。它们只表达编译器/程序合同，不
增加运行时状态。C++ 形状由 SIMD/vqsort 的 ported kernel 真实消费；纯 C 形状让外部 C kernel
取得同档 source contract，而不复制 compiler branches。assume/restrict 的错误承诺是调用方
bug；未定义 `NDEBUG` 时 unreachable 会先诊断并 abort，定义后才成为 optimizer UB；unroll 与 prefetch
都是可忽略的性能 hint。GCC/Clang 的 force-inline 在 `-O0` 退为普通 inline，避免大量模板叶函数
拖垮未优化编译，优化构建仍保留 always-inline。Clang 在未优化/sanitizer TU 中不发 unroll
pragma，避免无法执行的 transform 通过 `-Wpass-failed` 破坏 `-Werror` consumer build；优化且
非 sanitizer 的 TU 仍保留同一 hint。通常 CMake Debug/Release 对应这些状态，但 source consumer
若自定义 `NDEBUG`、优化或 sanitizer flags，宏按其 translation unit 的真实状态选择。

cache-layout 语义消费者必须使用 core 常量，不能硬编码 64B；container 的 queue/map/interner、
mem pool 与 scheduler 状态据此使用 128B destructive upper bound。64B constructive lower bound
保证放入该 span 的数据在当前认证目标上不跨 L1 line；128B destructive spacing 覆盖当前认证的
64/128B cache-line/sector 家族，也避免 Rosetta 的编译期身份歧义。runtime exact 超出 `[64,128]`
的 generic/RV64 机器仍可由 platform 如实探测，但该二进制的静态 layout 认证必须失败，重新裁决
常量与 ABI 后才能发布。SIMD lane scratch、AVX-512 table 和
“最多接受 64B aligned entry”的局部缓冲仍保留 `alignas(64)`：它们表达向量/对象 alignment，
不是 cache residency 或 false-sharing spacing，不能机械替换成 cache-line 常量。

## IntrusiveList 合同

链表采用 Boost/Folly 同类的 circular sentinel + member auto-unlink：每个 hook 两个指针，列表
本身一个 sentinel，无 size 元数据、无分配。节点地址在 linked 期间稳定；一个 hook 同时只能
进一个 list，需要多重 membership 就嵌多个 hook。节点析构自动摘链，list 析构/`clear()`
会把所有 hook 复位；copy/move 构造 hook 得到 unlinked hook，copy/move assignment 则刻意
保留 destination 的既有 membership，均不导入 source membership。

`push_back/front` 与 Boost 的 Release 容器一样以“node 非空、该 hook 未链接”为前置条件；
输入可能重复时用返回 bool 的 `try_push_back/front`。`remove(node)` 同样以“非空且属于该
list”为前置条件，不存/查 owner；membership 可选时直接用幂等 `hook.unlink()`。time 已用
exact-owner 状态证明这些热路前置条件。类型必须是 standard-layout。C++ 标准没有任意
pointer-to-data-member 的逆运算，因此 member-to-parent 恢复显式依赖项目支持工具链的 ABI：
MSVC 的单 `int` data-member offset，或 GCC/Clang 的 Itanium `ptrdiff_t` offset；编译期 size
assert 对更复杂的 inheritance representation fail closed。实现用 `bit_cast` 解码并 launder，
不再通过 null-pointer member expression 制造形式 UB。

`consume_all` 先清空 sentinel，故只接受 `noexcept` callback；当前节点在 callback 前已摘链，
可安全销毁/重链。`for_each` 禁止改变节点 membership/lifetime，但允许异常传播，异常时链结构
保持不变。`move_prefix_to` 的 predicate/on_move 都必须 `noexcept`，扫描期间节点仍属于 source，
两者不得摘链、销毁或重链；全部 callback 完成后才 O(1) splice 整段。

`begin/end` 是无状态双向 iterator，支持 range-for 与 reverse traversal；insert/erase、单节点
splice 的 position/item 采用 mature-container 的 `const_iterator` 形状（iterator 可隐式转换），
整链 splice 和 swap 也不增加 list/node 元数据。除被 erase/unlink/销毁的节点外，节点
iterator 在插入、整链/单节点 splice 与 swap 后继续指向同一节点；旧 list 的 `end()` sentinel
不能跨 move/splice 保存。遍历中摘除当前节点时先取得 successor，或使用返回 successor 的
`erase(const_iterator)`。并发读写仍需外部同步。

| 操作 | 复杂度 |
|---|---:|
| begin/end/front/back/push/pop/remove/insert/erase | O(1) |
| single/whole-list splice、append、swap、list move | O(1) |
| iteration / clear / consume_all / for_each | O(n) |
| move_prefix | O(k) 扫描 + O(1) 区间重链 |

不加 O(1) `size()`：它会给每次修改增加常态元数据，而 time/batch 消费者不需要；偶发计数可用
`std::ranges::distance` 支付 O(n)。不存 owner，checked insert 只拒绝 NULL/already-linked，erase/
splice 的 list-membership 是前置条件。需要 owning 元素、随机访问、常量时间 size 或线程安全时
选 container/标准容器，不把 core primitive 扩成另一个 Boost.Container 家族。

## C/C++ 双面

C++ 使用 auto-unlink RAII 模板；C 头提供同算法的可见 hook/list、静态 initializer、const-safe
entry/traversal 与 header-inline push/pop/unlink/append，不产生 DLL 热路。C 无析构器，因此释放节点前必须显式 unlink，销毁
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
| compiler contract | must-use/inline/noinline/branch/unreachable/assume/restrict/flatten/unroll + fatal | 同名 `HYLE_C_*` 集合 | 源码级 promise/hint；共用 backend，不存在有意义的动态调用 ABI |
| intrusive list | typed member hook + RAII/range/callback | 可见 hook/sentinel + init/const traversal/insert-before/splice-before/inline 指针原语 | C 显式 lifetime；任意位置插入、batch 与遍历可同复杂度直达或组合 |

因此“C 能力完整”指版本、cache intent、compiler contract、stable-node queue 等自然 C 用户
任务都有同档通路，不表示把 C++ 析构或模板 callback 逐符号翻译一遍。compiler contract 单独
放在 `hyle/c/compiler.h`，`hyle/c/core.h` 只为可发现性转 include；两者都不增加动态符号。

## API 易用性结论

- 默认旅程：按任务直接包含 version/cache/intrusive 单头；没有强迫所有用户引入 umbrella。
- 高级旅程：compiler contract 单独在 `macros.h`，危险 UB/代码体积合同不混进普通业务 API。
- 组合旅程：time 直接以 typed intrusive list 组合；container/mem/sched 直接消费 cache 常量，
  没有 C ABI 往返或 adapter 链。
- C 旅程：kernel-only 可单独 include compiler 头；版本为两个动态 identity 符号，其余热路
  header-inline；静态/动态 init、const 遍历和显式 unlink/clear 对应 C 生命周期。

上述默认/高级/组合路径分别由 core C++ tests、SIMD/vqsort 消费编译和 pure-C smoke 机械覆盖；
shared/static 安装后还由 `samples/package/core/{c,cpp}` 只请求 `core` 组件独立复证。guide
给出入口选择与可复制 recipe。新增 source/compiler 便利必须先有真实扩展消费者，不能让
core 重新长成杂项 umbrella。

## 明确负空间

- runtime CPU/OS 信息归 platform；core 不重复 CPUID/sysctl。
- allocator、error、logging、clock 不进入零依赖根。
- 不提供拥有型/counted/owner-checked list、随机访问或同步包装。
- 不提供显式 L1/L2/L3 cache 对象或 portable cache residency 保证；prefetch locality 只是 hint，
  runtime cache 拓扑仍归 platform。
- 不使用 `std::hardware_destructive_interference_size` 参与公共布局：其值受编译参数/标准库
  影响，可能让不同 TU 得到不同 ABI；HYLE 用受支持平台的稳定常量，精确值另走 platform。
- 不把 Git revision 放入 core 版本 ABI：source archive、vendored/dirty tree 与 reproducible build
  没有统一 Git 语义；可选 source provenance 由 build/package manifest 或未来 opt-in build-info 承担。
