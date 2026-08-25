# c-core：core 的纯 C / FFI 映射

> 场景选型先读 [`core.md`](./core.md)。本页只说明
> `include/hyle/c/core.h` 与 `include/hyle/c/compiler.h` 的精确 C 形状、所有权和坑。

公开 C source surface 的语言基线是 C17。通过 CMake 链接 `hyle::core` 时，exported target 会把
C17 以及 native MSVC 所需的 conforming preprocessor 合同传播给消费 target；整体源码嵌入或
手工只取头文件时，调用方必须自行用 C17 模式编译（native MSVC 同时启用
`/Zc:preprocessor`）。这不是 DLL ABI 要求，而是消费这些 header-inline 能力的翻译单元合同。

```cmake
find_package(Hyle 0.1 REQUIRED COMPONENTS core)
target_link_libraries(my_c_target PRIVATE hyle::core)
hyle_stage_runtime_dependencies(my_c_target)
```

安装 SDK 保留统一 public header tree；`COMPONENTS core` 约束的是 target/link closure，不表示按
目录裁剪成只剩 core 头。runtime staging helper 对 static/source 消费是安全 no-op，Windows
shared 消费则负责复制 DLL；三种分发形状的完整工程见 [`distribution`](./distribution.md)。

## 版本与缓存布局

| 名称 | 形状 | 说明 |
|---|---|---|
| `HYLE_VERSION_MAJOR/MINOR/PATCH` | compile-time macro | 当前 headers 版本 |
| `HYLE_VERSION_PACK(a,b,c)` / `HYLE_VERSION_PACKED` | macro | 8/8/8 packed version |
| `HYLE_VERSION_STRING` | string literal | 当前 headers 的 `major.minor.patch` |
| `hyle_version` | `() -> uint32_t` | 已链接 `hyle_core` 的 runtime packed version |
| `hyle_version_string` | `() -> const char*` | runtime 模块拥有的借用字符串，不 free；只在承载 hyle_core 的模块仍加载时有效 |
| `hyle_version_major/minor/patch` | header-inline | 拆 packed version，无 DLL hop |
| `HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE` | compile-time macro | target-family constructive recommendation；x86-64/ARM64EC 64B，native Apple ARM64 128B，未认证 native Windows/generic ARM64 为 conservative 64B |
| `HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE` | compile-time macro | target-family destructive recommendation；x86-64/ARM64EC 64B，native Apple ARM64 128B，未认证 native Windows/generic ARM64 为 conservative 128B |
| `HYLE_MIN_CONSTRUCTIVE_INTERFERENCE_SIZE` | compile-time macro | 已认证家族的 64B constructive minimum |
| `HYLE_MAX_DESTRUCTIVE_INTERFERENCE_SIZE` | compile-time macro | 稀疏 hot control state 可 opt-in 的 128B maximum isolation |
| `hyle_prefetch_read` | header-inline | 预计会复用的读，偏向本地 cache |
| `hyle_prefetch_read_non_temporal` | header-inline | 大片一次性/低复用读；不支持时可退化 |
| `hyle_prefetch_write` | header-inline | 即将修改的地址；capable target 可请求 exclusive ownership，不支持时按编译 ISA baseline 安全退化 |

```c
#include <stdio.h>

#include "hyle/c/core.h"

if (hyle_version() != HYLE_VERSION_PACKED) {
  /* plugin/动态库路径装载了不同于 headers 的 HYLE。 */
}
printf("HYLE %s\n", hyle_version_string());
```

若 loader 随后会 `dlclose`/`FreeLibrary` hyle_core，必须先把 runtime version string 复制到调用方
拥有的存储；卸载后继续持有该指针是悬空引用。

算法需要运行时 cache-line observation 时调用 platform 的 `hyle_cpu_cache_line_size()`；它是
best-effort probe，查询失败回退 64，不是精确拓扑 API，也不能用于 safety-critical padding。
不要用 runtime 值定义跨翻译单元 struct ABI。dense queue cell/pool/map shard 的 false-sharing
spacing 选 `HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE`；共同读取且希望紧凑在一个 target-family
span 内的数据才选 `HYLE_STATIC_CONSTRUCTIVE_INTERFERENCE_SIZE`。这样可避免 x86 dense 数组为跨家族
maximum 付双倍 footprint；数量少且被不同线程
持续写入的 control atomics 才显式使用 `HYLE_MAX_DESTRUCTIVE_INTERFERENCE_SIZE`。普通对象不要
为“看起来更快”默认 padding。

若用 `_Alignas` 声明 over-aligned C 类型，动态存储也必须来自满足该 target 实际 alignment 的
分配入口；普通 `malloc` 只承诺 `max_align_t`，不能因为目标类型声明了 `_Alignas` 就自动变成
over-aligned allocation。默认 macro 是编译目标家族的 ABI 选择，不是运行时自适应；新 ARM/
RV64 或超出认证范围的新目标必须重新认证 probe、layout/footprint/ABI 后才能作为受支持构建发布。
本次预发布升级把 x86 的 `HYLE_STATIC_DESTRUCTIVE_INTERFERENCE_SIZE` 从 128 改为 64；动态 C
符号集合没有变化，但任何使用这些 macro 定义 `_Alignas` struct 的 C source ABI 都会变化。
必须 clean rebuild 全部 C translation units/shims/plugins 和已安装 HYLE binary，不能混旧 header/
object；当前同为 0.1.0 的 runtime version 数字不会自动发现这种混用。

```c
for (size_t i = 0; i != count; ++i) {
  if (i + 16 < count) hyle_prefetch_read(items + i + 16);
  consume(items + i);
}
```

prefetch 不读取/写入 C abstract-machine 中的对象，不验证地址，也不建立 ownership、同步或
memory ordering。只有代表性 profile 证明等待内存时才使用；普通线性循环通常先交给硬件
prefetcher。`write` 仍要求调用方使用正确的原子/锁，`non_temporal` 也只是 locality hint，目标
允许忽略或降级。

## C kernel 的编译器合同

只写数据 kernel、不需要 version/list 时可单独 include `hyle/c/compiler.h`；`hyle/c/core.h`
也会包含它。全部宏都是 header-only source surface，不增加 DLL 符号或运行时状态：

| 名称 | 什么时候用 | 合同 |
|---|---|---|
| `HYLE_C_FORCE_INLINE` / `HYLE_C_NOINLINE` | 已测热叶函数 / 必须保持冷路或 frame 的函数定义 | 强/禁内联 hint；强内联仍受编译器合法性约束 |
| `HYLE_C_NODISCARD` | 必须检查/释放的状态或结果函数声明前 | GCC/Clang C17 产生 unused-result 诊断；MSVC C 记录 SAL `/analyze` 合同 |
| `HYLE_C_LIKELY(expr)` / `HYLE_C_UNLIKELY(expr)` | profile 证明概率明显偏斜的分支 | 表达式求值一次；未知编译器退为普通 bool |
| `HYLE_C_UNREACHABLE()` | 已由不变量证明绝不抵达的控制流末端 | 未定义 `NDEBUG` 时诊断并 abort；定义后抵达是 undefined behavior，不做输入验证 |
| `HYLE_C_FATAL_ASSERT(expr)` | 违反后进程不能安全继续的程序不变量 | Debug/Release 都诊断并 abort；业务失败改返回 error |
| `HYLE_C_RESTRICT` | 放在已证明互不别名的 pointer declaration 上 | 承诺错误是 undefined behavior；未知编译器安全退为空 |
| `HYLE_C_ASSUME(expr)` | 告诉优化器某个内部不变量恒真 | 表达式必须无副作用；未定义 `NDEBUG` 时 false 会诊断并 abort，定义后为 false 是 undefined behavior |
| `HYLE_C_FLATTEN` | 经过 profile 的深 wrapper/kernel 函数定义前 | GCC/Clang 尽量递归内联；可能扩大代码、伤害 I-cache；MSVC/未知编译器可忽略 |
| `HYLE_C_UNROLL(n)` | 已 profile 的固定展开 kernel loop 前 | best-effort hint；MSVC/未知编译器可忽略，不改变语义 |
| `HYLE_C_DEFAULT_UNROLL` | 需要工具链默认/保守展开而不指定因子时 | GCC 当前映射为 4，Clang 采用默认 unroll；同样只是 hint |

```c
#include <stddef.h>
#include "hyle/c/compiler.h"

int sum(const int* HYLE_C_RESTRICT values, size_t count) {
  int total = 0;
  HYLE_C_ASSUME(values != NULL); /* 这是内部前置条件，不是 NULL 检查。 */
  HYLE_C_UNROLL(4)
  for (size_t i = 0; i < count; ++i) total += values[i];
  return total;
}
```

业务输入验证必须写正常分支并返回 error，不能拿 `ASSUME` 代替；有副作用的表达式在是否定义
`NDEBUG` 的 TU 中可能有不同求值行为。`RESTRICT` 要覆盖 pointer 有效期内的全部访问，局部看起来不冲突
并不够。GCC/Clang 的 force-inline 在 `-O0` 退成普通 inline，避免未优化编译膨胀；优化构建仍
强制 inline。flatten/unroll 都可能增加 code size、伤害 I-cache，所以没有代表性 profile 时不加。

## 侵入链表类型

| 名称 | 说明 |
|---|---|
| `hyle_intrusive_list_hook` | 嵌入调用方节点的 prev/next；一个 hook 同时只能属于一条链 |
| `hyle_intrusive_list` | circular sentinel list；不拥有节点、不分配 |
| `HYLE_INTRUSIVE_HOOK_INIT` | hook 的静态/聚合初始化值 |
| `HYLE_INTRUSIVE_LIST_INIT(name)` | list 的自引用静态/自动初始化值，不必另调 init |
| `HYLE_INTRUSIVE_ENTRY(hook,type,member)` | 从非 NULL member hook 恢复外层标准布局 C struct |
| `HYLE_INTRUSIVE_CONST_ENTRY(hook,type,member)` | 从非 NULL const hook 恢复 const 外层 struct，不丢 const |

| inline 函数 | 合同 |
|---|---|
| `hyle_intrusive_hook_init` | 新/已脱链 hook 初始化；不可拿它强行重置 linked hook |
| `hyle_intrusive_hook_linked` | NULL/idle 为 false |
| `hyle_intrusive_hook_unlink` | 从当前链 O(1) 摘除；NULL/idle 幂等 |
| `hyle_intrusive_list_init` | 未使用 `HYLE_INTRUSIVE_LIST_INIT` 时初始化 fresh list；只能在首次使用前或已安全 clear 后调用 |
| `hyle_intrusive_list_empty` | NULL 视为空；已初始化 list 为 O(1) |
| `hyle_intrusive_list_insert_before` | O(1) unchecked 任意位置插入；position 为 NULL 表示尾部 |
| `hyle_intrusive_list_try_insert_before` | checked 插入；NULL list/hook 或 double-link 返回 0 |
| `hyle_intrusive_list_push_back/front` | O(1) 无分支；前置条件是 list/hook 非 NULL 且 hook 未链接 |
| `hyle_intrusive_list_try_push_back/front` | O(1) checked 入口；成功 1，NULL 或 double-link 为 0 |
| `hyle_intrusive_list_front/back` | 返回 hook；空/NULL 返回 NULL，不摘链 |
| `hyle_intrusive_list_front/back_const` | const-safe 首尾访问 |
| `hyle_intrusive_list_next/prev` | 正反向遍历；到 sentinel 后返回 NULL |
| `hyle_intrusive_list_next/prev_const` | const-safe 正反向遍历 |
| `hyle_intrusive_list_pop_front/back` | O(1) 摘链并返回 hook；空/NULL 返回 NULL |
| `hyle_intrusive_list_clear` | O(n) 摘除全部节点，不释放节点 |
| `hyle_intrusive_list_splice_before` | O(1) 把 source 整批移到 position 前；NULL position 表示尾部；NULL/self 返回 0，empty/success 返回 1；position owner 仍是 unchecked 前置条件 |
| `hyle_intrusive_list_append` | O(1) splice source 到 destination 尾；source 清空；非法/self 为 0 |

```c
typedef struct request {
  int id;
  hyle_intrusive_list_hook ready;
} request;

hyle_intrusive_list queue = HYLE_INTRUSIVE_LIST_INIT(queue);
request item = {42, HYLE_INTRUSIVE_HOOK_INIT};
hyle_intrusive_list_push_back(&queue, &item.ready);

hyle_intrusive_list_hook* hook = hyle_intrusive_list_pop_front(&queue);
if (hook != NULL) {
  request* next = HYLE_INTRUSIVE_ENTRY(hook, request, ready);
  consume(next);
}
```

`HYLE_INTRUSIVE_ENTRY*` 是 pointer arithmetic 宏，输入 NULL 不是“返回 NULL”，而是无效地址；
必须先检查 `front/back/pop` 的返回值。position 属于哪条 list 也不存 owner 元数据：unchecked
入口和 `splice_before` 都要求调用方保证 position 为 NULL 或属于 destination。

只读遍历不需要直接碰 sentinel，也不会丢掉 const：

```c
const hyle_intrusive_list* view = &queue;
const hyle_intrusive_list_hook* it = hyle_intrusive_list_front_const(view);
int id_sum = 0;
while (it != NULL) {
  const request* item = HYLE_INTRUSIVE_CONST_ENTRY(it, request, ready);
  id_sum += item->id;
  it = hyle_intrusive_list_next_const(view, it);
}
```

C 没有析构器：不得对仍含 live hooks 的非空 list 再调用 `hyle_intrusive_list_init`，否则旧节点会
被遗留在破损链中，后续 unlink 也会破坏结构。节点存储结束前必须 `hyle_intrusive_hook_unlink`，list 存储结束前必须 clear/
pop 完或确保节点仍更长寿。linked 期间节点地址不可变化；多重 membership 使用多个 hook；
并发访问由调用方同步。`try_push_*`、`try_insert_before`、`pop_*`、`splice_before` 与 `append` 都带
`HYLE_C_NODISCARD`：调用方必须
检查成功位或接住被摘节点；已知具体节点而只想摘链时直接调用 `hook_unlink`。所有 list 操作都是
header-inline，因此没有每次跨 DLL 的热路成本。

## C++ 能力到 C 的通路

| C++ 能力 | 纯 C 通路 | 是否同档 |
|---|---|---|
| hook/list RAII auto-unlink | 存储结束前显式 `hook_unlink`，list 结束前 `clear` | 同一 O(1) 摘链；析构语法是 C++ 特性 |
| `insert(pos,node)` / checked insert | `insert_before(list,pos,hook)` / `try_insert_before`；`pos == NULL` 表示 end | O(1)；checked 只拒绝 NULL/already-linked，不为 owner 增元数据 |
| `erase(pos)` | 先保存 `next(list,hook)`，再 `hook_unlink` | O(1)，保存值就是 successor |
| single-node `splice(pos,source,it)` | 先处理 `pos == it` 的同 list no-op，再 `hook_unlink(it)` + `insert_before(destination,pos,it)` | O(1)；C 不制造 iterator 类型 |
| whole-list `splice(pos,source)` | `hyle_intrusive_list_splice_before(destination,pos,source)` | O(1)，`pos == NULL` 表示 end |
| `append` / list move | `hyle_intrusive_list_append` | O(1)，等价于 tail `splice_before` |
| `swap` | 初始化临时 list，依次把 left→temp、right→left、temp→right；检查三次 `append` | O(1)；低频组合不扩新入口 |
| `consume_all` | 循环 `pop_front` 后调用 C 函数 | O(n)，当前 hook 在用户回调前已摘链 |
| range/`for_each` | `front[_const]` + `next[_const]`；若摘链先保存 `next` | O(n) |
| `move_prefix_to` | 按谓词循环 `front` + `pop_front` + `push_back` | O(k)，不需要 callback ABI 或额外分配 |

```c
hyle_intrusive_list_hook* hook;
while ((hook = hyle_intrusive_list_pop_front(&queue)) != NULL) {
  request* item = HYLE_INTRUSIVE_ENTRY(hook, request, ready);
  consume(item); /* 此时 ready 已经 unlinked，可以释放或重新入队。 */
}
```

`HYLE_C_*` 与对应的 C++ compiler macros 共用同一份 compiler backend，但保持语言
命名和 Debug 诊断独立；修工具链映射时不会出现 C/C++ 两套判断漂移。它们是给源码消费者的
产品面，不是可由 `dlopen`/FFI 调用的运行时 ABI。

## 导出形态与机械门

动态符号仍只有 `hyle_version`、`hyle_version_string`。这一区分很重要：本页其余入口是给 C
源码（以及能编译 C shim 的 FFI）使用的 source API，不是 Rust/Python 可直接按 DLL symbol 绑定的
动态 ABI。compiler contract、缓存常量、三种
prefetch、版本分解与 intrusive list 都是编译期/header-inline 能力，不进入导出表。
`tests/core/c_surface_symbols.txt` 锁动态导出；`c_core_test.c` 以 C 编译器覆盖 compiler macro
有效路径、版本、cache/prefetch、NULL/非法输入、空队列及完整链表生命周期；Debug death-test
覆盖 C assume 违约，header self-contained gate 从空翻译单元编译 C 头。

安装后最小工程见 `samples/package/core/c` 与 `samples/package/core/cpp`。它们只请求
`find_package(Hyle COMPONENTS core)` 并链接 `hyle::core`；严格验收必须在 Windows/Linux/macOS
分别以 shared/static package 配置、构建和运行，避免示例暗中依赖源码树 include 或其它 target。
当前 exact candidate 的通过状态只记录在 milestone，不能把历史 smoke 当作本轮证书。
