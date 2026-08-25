# core 模块架构

> 本页维护 core 的稳定设计。施工证据见
> [`core-module-closure`](../milestones/core-module-closure.md)，性能只见
> [`benchmarks-core`](../performance/benchmarks-core.md)，用户选型见
> [`guide/core`](../guide/core.md) 与 [`guide/c-core`](../guide/c-core.md)。

## 定位与边界

`core` 是零 HYLE 上游的源码根和最小 runtime identity ABI，承载四类能力：编译期/运行时版本、
稳定缓存布局与 prefetch、编译器属性/合同宏、零分配侵入链表。`macros.h`、`visibility.h`
虽位于 `hyle/` 根目录，也属于 core 的实现边界；目录位置不能把它们变成无主资产。

```text
version_defs/cache_defs/compiler_defs ─► C++ source / build and schema definitions
version.h + hyle_core ─────────────────► C++ linked runtime identity
c_core.cc + hyle_core ─────────────────► direct FFI runtime identity
cache.h/macros/visibility ─────────────► 全库 source seam
intrusive_list.h ──────────────────────► time ─► sched/io
```

core 不依赖 error/platform/mem/time，也不创建资源，因此没有 Status 或 allocator seam。
运行时 cache/CPU 探测由 platform 独占；core 只提供必须参与 native 对象布局的编译期常量。

## 能力与真实场景

| 能力 | C++ source / linked 入口 | 场景与形状 |
|---|---|---|
| header version | `kVersion*` / `HYLE_VERSION_*` | native 编译期 feature/log 标记，无 DLL 调用 |
| linked runtime identity | `version()` / `version_string()`；FFI ABI/product version C exports | managed binding 先协商 ABI，再诊断 native package/product identity |
| static cache layout | `core::k*InterferenceSize` + min/max | native `alignas` 与结构体 ABI；默认值按目标家族选择 64/128B |
| cache hints | 三种 `prefetch_*` | native TU 中的 local/NTA/for-write intent；只在 profile 后使用 |
| compiler/fatal macros | `macros.h` / `visibility.h` | must-use、restrict、assume、flatten、unroll、inline/noinline、branch、EBO、导出与不可恢复合同 |
| intrusive list/batch | `IntrusiveListHook` + `IntrusiveList<T, Hook>` | stable-address、调用方拥有节点、双向 range、O(1) arm/cancel/insert/erase/splice；time 是生产消费者 |

`platform::cache_line_size()` 是相邻 platform 模块的 C++ best-effort runtime observation，不是
core 能力。旧 `hyle_cpu_cache_line_size()` 只是黄色 FFI 候选：platform 复审可能保留，也可能因
没有真实外语任务裁决为 `FFI=N/A`。无论结果如何，查询失败回退 64，不能参与稳定类型布局或
安全隔离；core 不复制这个符号。

按方法论展开后的场景合同如下；core 不分配资源，因此没有 allocator/OOM 路径：

| 场景 | 数据、ownership 与 lifetime | concurrency 与 failure | 规模/复杂度 |
|---|---|---|---|
| managed binding/native runtime identity | binding 预期 ABI、header product 常量与 module-local static string；字符串只借用到 hyle_core unload | 只读并发；ABI 不兼容由 loader 拒绝，product 不匹配由调用方诊断 | O(1)，冷路 |
| 公共 cache layout / prefetch | 调用方对象与地址；target-family recommendation/min/max 参与 ABI，runtime observation 归 platform | layout 可跨线程，hint 不建立同步；无效生命周期地址，或把 fallback-able observation 冒充安全 layout，是合同违约 | 编译期/O(1)，只在真实消费者承重 |
| kernel compiler contract | 调用方表达式、pointer 与 loop，零 runtime 状态 | TU-local；错误 assume/restrict/unreachable 是程序 bug，must-use 是诊断 | source seam；unroll/inline 只作 measured hint |
| stable-node queue/batch | 节点由调用方拥有且地址稳定；C++ hook/list RAII 摘链 | 默认外部同步；checked insert 返回 bool，unchecked owner/membership 违约是 bug | 单节点/整批 O(1)，遍历/consume O(n)，不维护 size/owner |

### 语法与语义消费者账本

| core 职责 | 显式消费者 | 语义重复扫描结论 |
|---|---|---|
| version single source | root CMake、C++ linked API、direct FFI、plugin/loader 诊断 | product 与 FFI ABI 各有一份 defs；CMake/runtime 由 static_assert 锁定 |
| static cache layout | container queues/maps/interner、mem pool、scheduler | dense array 采用 target-family spacing，持续跨线程写的稀疏 control atomics 可显式采用 128B max；SIMD/entry scratch 的 64B 是向量/对象 alignment，不属于 cache 重复 |
| prefetch intent | SwissTable、SIMD/vqsort | vqsort 私有 helper 已收束；没有其他 builtin/intrinsic 旁路 |
| compiler contract | SIMD/vqsort 及全库 fatal/inline 用户 | vqsort 的 assume/noinline/restrict/flatten/unroll 分支已映射到 core；无第二套同义宏 |
| intrusive stable-node list | time TimerQueue/CommonTimeoutQueue，sched 经 time 消费 | 无第二个同职责 intrusive doubly-list；container 的 owning/其他队列是不同产品 |

扫描必须按语义分类：`alignas(64)` 不能机械判成 cache 重复。AVX-512 lane/table、SIMD state 和
支持 64B over-aligned entry 的 scratch 是数据 alignment；只有表达 false-sharing spacing 的
StringInterner shard 属 core 职责，并已采用 destructive-interference 常量。

### 成熟库能力与形状对账

| 参考 | 借鉴的问题/形状 | HYLE 裁决 |
|---|---|---|
| oneTBB | compile/runtime version identity、默认 cache-line 值与 128B `max_nfs_size` | 版本单源；默认 layout 按目标家族，最大隔离作为显式 opt-in；不引未发布阶段的函数表协商 |
| Abseil | local、NTA、for-write 三种 prefetch intent | C++ header-inline 形状对齐；不增加 cache residency 承诺，不另造 FFI symbol |
| Highway | 优化构建 force-inline；Debug 普通 inline；noinline/assume/restrict/flatten/unroll | 收为 core source seam；hint 无独立产品 benchmark |
| Boost/Folly intrusive | member hook、auto-unlink、双向 iterator、无 owner/size 的 O(1) hot path | 保留一个紧凑通用 non-owning sequence；不扩成 owning/counted 容器家族 |

这张表用于查漏和裁决算法/类型，不按参考库 feature 数量扩 core。runtime topology 留 platform，
owning container 留 container，error/clock/logging 留各自模块。

本轮负空间也显式裁决：C++ alignment promise 已有 `std::assume_aligned`，SIMD aligned Load 又
直接表达 kernel 合同；项目没有原生 C 产品需要另一套宏，managed FFI 也无法把 translation-unit
alignment promise 表达成有意义的动态调用。noreturn/fallthrough 已有标准 native 拼法，因此
不包装。`lifetimebound` 对 `Status::message()` 和 container borrowed view/reference
是真实的零成本诊断候选，但它需要跨模块统一注解与 compile-diagnostic 门，留到 error/container
复审共同裁决，不在 core 为凑 feature 数量先放一个无人正确使用的宏。

product version 与 FFI ABI version 都在 `version_defs.h` 单点定义。前者是 8/8/8 的 HYLE release
identity；后者是 16/16 的 binding protocol，major 变化允许破坏 ABI，minor 只允许 additive。
managed binding 必须先解析 `hyle_ffi_abi_version`，要求 major 相同且 runtime minor 不低于自己
生成时所需 minor，再解析其余 required symbols。CMake 在 `project()` 前读取 product version，
`version.cc` 再静态断言 runtime 编译定义，避免第二份手写 release identity。
`version_string()` 返回 `version.cc` 内部链接的静态存储，不分配、不使用 guard variable，也不会
被 consumer/另一 DSO 的 public inline header 常量抢占。指针只在承载 hyle_core 的动态模块仍
加载时有效；需要跨 `dlclose`/`FreeLibrary` 保存就先复制。

三种 prefetch 对齐 Abseil 的 local/NTA/for-write 形状，在 GCC/Clang、MSVC x86 与 MSVC ARM64
分别映射 builtin/intrinsic；不支持时安全退化。must-use、force-inline、noinline、branch hint、
unreachable、assume、restrict、flatten、unroll 共用 `compiler_defs.h` 的工具链判断，并由
`macros.h` 暴露 C++ source contract。它们只表达编译器/程序合同，不增加运行时状态，且由
SIMD/vqsort 的 ported kernel 真实消费。旧 `hyle/c/compiler.h` source helper 已移出 FFI schema；
它即使存在也不会构成 direct FFI。assume/restrict 的错误承诺是调用方
bug；未定义 `NDEBUG` 时 unreachable 会先诊断并 abort，定义后才成为 optimizer UB；unroll 与 prefetch
都是可忽略的性能 hint。GCC/Clang 的 force-inline 在 `-O0` 退为普通 inline，避免大量模板叶函数
拖垮未优化编译，优化构建仍保留 always-inline。Clang 在未优化/sanitizer TU 中不发 unroll
pragma，避免无法执行的 transform 通过 `-Wpass-failed` 破坏 `-Werror` consumer build；优化且
非 sanitizer 的 TU 仍保留同一 hint。通常 CMake Debug/Release 对应这些状态，但 source consumer
若自定义 `NDEBUG`、优化或 sanitizer flags，宏按其 translation unit 的真实状态选择。

cache-layout 语义消费者必须使用 core 常量，不能硬编码 64B。`kConstructiveInterferenceSize` /
`kDestructiveInterferenceSize` 是 target-family recommendation：x86-64 为 64B/64B constructive/
destructive，native Apple ARM64 为 128B/128B，ARM64EC 按 x64 ABI 为 64B/64B。尚未认证的
native Windows ARM64（`_M_ARM64`）、generic ARM64 以及其他 fallback/RV64 采用 conservative
64B/128B 组合，不宣称已获得平台认证。dense queue
cell、pool/map shard 采用默认值，避免在 x86 为跨平台统一 `sizeof`
永久付出双倍 footprint；被不同线程持续写入且数量很少的 control-plane atomics 可显式使用
`kMaxDestructiveInterferenceSize` 的 128B 隔离。`kMinConstructiveInterferenceSize` 固定为 64B，
用于确实需要跨已认证家族保守上限/下限的公共布局裁决。

这些常量都不是 runtime observation。算法的 flush/stride/blocking 可调用 platform 取得
best-effort 值，但查询失败会回退 64，不能把该值当成安全 destructive spacing。新目标必须把
runtime observation、对象 footprint 和真实争用一起复证，不能因为 fallback 能编译就宣称 layout 已认证。
SIMD lane scratch、AVX-512 table 和
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

## C++ source、native-C helper 与 direct FFI

C++ source surface 的语言基线是 C++23。Native FFI 的 C schema 保持 C17 可编译，用来描述
calling convention 和验证 ABI；项目没有原生 C 用户产品。安装态或 source target 会传播对应
compile feature，手工只取 C++ 头文件者必须自行建立相同 TU 合同。

core 的三层矩阵如下：

| 能力 | C++ source / linked API | native-C helper | direct Native FFI | 裁决 |
|---|---|---|---|---|
| compile/runtime identity | constexpr + `version()` / `version_string()` | 无行为 helper；仅 language-neutral expected-version defs | `hyle_ffi_abi_version` / `hyle_version` / `hyle_version_string` | ABI bootstrap 与 loader identity 有真实 runtime 价值 |
| static cache layout | 四个 typed constexpr | shared compile-time defs，不在 FFI schema | 无 | 只参与 native 对象布局；不是运行时事实 |
| prefetch intent | 三个 inline 函数 | 无公开 helper | 无 | 单地址 FFI hop 比 hint 更贵；需要效果时融合进所属模块 batch op |
| compiler/fatal contract | `HYLE_*` | 无公开 helper | 无 | translation-unit promise/hint，没有动态调用语义 |
| intrusive list | typed member hook + RAII/range/callback | 无公开 helper | 无 | 依赖稳定 native 地址与手工 lifetime，不是 managed 容器 ABI |

表中的 native-C helper 不属于 Java/.NET/Rust/Python 能力覆盖。core 已把旧 compiler/prefetch/
intrusive native-C source surface 从 `hyle/c/core.h` schema 移出；当前 schema 只保留 bootstrap
与 product identity。core 的 direct FFI 有意只有三个版本符号；这不是遗漏，而是其他四类能力
均没有合理的独立 runtime ABI。

### 为什么 prefetch 不导出

prefetch 的有效位置、距离、locality 与 ISA 都由拥有循环的 native translation unit 决定。把一个
地址从 Java/.NET/Python 逐次送过 FFI 再发 hint，边界成本与 pin/lifetime 风险通常远大于收益，
也无法建立 cache residency 或同步语义。若某个跨语言数据面确实受益，应由 container/simd/
compute 等拥有 workload 的模块在批处理导出内部 prefetch，而不是增加 `hyle_prefetch_*` 动态
符号。

### 为什么 intrusive list 不导出

透明 intrusive list 要求调用方在节点内嵌 hook、保持节点和 sentinel 地址稳定、理解 ownerless
membership，并在正确时序手工脱链。Java/.NET/Python 的 GC 和对象搬移不能自然满足这些合同；
Rust 虽能包装 native memory，也不能让面向四种语言的公共 ABI 退化为裸指针容器。若未来有真实
跨语言 queue/batch 场景，应在拥有任务的上层模块设计 opaque handle、library-owned storage 与
批量 enqueue/drain，而不是导出 core hook 操作。

旧 native-C intrusive helper 的设计还暴露了以下陷阱；这些也是它不应作为 managed ABI、且不应
重新混入 FFI schema 的直接证据：

- linked node 的地址必须稳定；
- **任何已初始化 list（即使为空）都不能搬址、赋值、`memcpy` 或按值返回**，因为 sentinel
  指向自身，非空时首尾 hook 还会反向指向它；只能在最终存储初始化，或用显式 splice 将节点
  转移到另一已初始化 list；
- `HYLE_INTRUSIVE_LIST_INIT(name)` 是按该对象地址建立自引用，复制初始化后的字节会悬挂；
- C 没有析构器，node 或 list 任一存储结束前都必须先把所有相关 hook 脱链；仅让 nodes 比 list
  活得久不能挽救已经失效的 sentinel；
- 不得对非空 live list 重新 `init`；`try_*` 只拒绝 NULL/double-link，不验证 position owner；
- `HYLE_INTRUSIVE_ENTRY*` 的 hook 必须非 NULL 且确实来自指定 member。

### runtime cache observation 的归属

core 的 cache 常量冻结 native layout recommendation；platform 才拥有当前机器的 best-effort
C++ observation。它是否需要 direct FFI 查询仍由 platform 的真实外语任务裁决，当前旧符号只是
候选。两者不可合并：runtime fallback 不能保证 false-sharing spacing，compile-time 常量也不应
伪装成硬件探测导出。新目标的 layout 认证与 platform probe 分别复证。

## API 与证据结论

- C++ 默认旅程：按任务包含 version/cache/intrusive 单头；
- C++ 高级旅程：compiler contract 单独在 `macros.h`，危险 UB/代码体积合同不混入普通业务 API；
- C++ 组合旅程：time 使用 typed intrusive list，container/mem/sched 使用 cache 常量，不绕 FFI；
- native-C helper：core 当前无公开 compiler/prefetch/list helper；旧 surface 的 C 测试不再计入；
- direct FFI：Java 与 .NET 从 shared artifact 先解析 `hyle_ffi_abi_version`，通过 major/minor
  握手后再解析两个 `hyle_version*` symbols，并验证 borrowed version string 的 module
  lifetime；未来若发布 Rust/Python binding，也应绑定同一组三个符号并先做同一 ABI 握手。

core C++ tests、SIMD/vqsort 消费编译、C schema smoke 与 direct managed smoke 必须分账。旧
历史 native-C package 示例即使能编译也只能证明 schema/source/package，不能代替 Java/.NET
动态加载；当前 [`samples/package/core/ffi_abi`](../../samples/package/core/ffi_abi) 是 conformance
fixture，direct FFI 证据则由 [`samples/ffi/core`](../../samples/ffi/core/README.md) 从 shared
artifact 解析三个真实 exports。
安装 SDK 保留统一 public header tree，隔离边界是 component/target closure，不是按目录裁剪。
当前通过状态与 exact candidate 只记录在 milestone；本页不把历史 smoke 冒充当前证书。

## 明确负空间

- runtime CPU/OS 信息归 platform；core 不重复 CPUID/sysctl。
- allocator、error、logging、clock 不进入零依赖根。
- 不提供拥有型/counted/owner-checked list、随机访问或同步包装。
- 不提供显式 L1/L2/L3 cache 对象或 portable cache residency 保证；prefetch locality 只是 hint，
  runtime cache 拓扑仍归 platform。
- 不使用 `std::hardware_destructive_interference_size` 参与公共布局：其值受编译参数/标准库
  影响，可能让不同 TU 得到不同 ABI；HYLE 用受支持平台的稳定常量，best-effort runtime
  observation 另走 platform。
- 不把 Git revision 放入 core 版本 ABI：source archive、vendored/dirty tree 与 reproducible build
  没有统一 Git 语义；可选 source provenance 由 build/package manifest 或未来 opt-in build-info 承担。
