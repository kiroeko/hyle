# c-core：core 的纯 C / FFI 映射

> 场景选型先读 [`core.md`](./core.md)。本页只说明
> `include/hyle/c/core.h` 与 `include/hyle/c/compiler.h` 的精确 C 形状、所有权和坑。

## 版本与缓存布局

| 名称 | 形状 | 说明 |
|---|---|---|
| `HYLE_VERSION_MAJOR/MINOR/PATCH` | compile-time macro | 当前 headers 版本 |
| `HYLE_VERSION_PACK(a,b,c)` / `HYLE_VERSION_PACKED` | macro | 8/8/8 packed version |
| `HYLE_VERSION_STRING` | string literal | 当前 headers 的 `major.minor.patch` |
| `hyle_version` | `() -> uint32_t` | 已链接 `hyle_core` 的 runtime packed version |
| `hyle_version_string` | `() -> const char*` | runtime 字符串，静态存储，不 free |
| `hyle_version_major/minor/patch` | header-inline | 拆 packed version，无 DLL hop |
| `HYLE_STATIC_CACHE_LINE_SIZE` | compile-time macro | 公共结构布局：Apple Silicon 128，其余当前支持目标 64 |
| `hyle_prefetch_read` | header-inline | 预计会复用的读，偏向本地 cache |
| `hyle_prefetch_read_non_temporal` | header-inline | 大片一次性/低复用读；不支持时可退化 |
| `hyle_prefetch_write` | header-inline | 即将修改的地址；capable target 可请求 exclusive ownership，不支持时按编译 ISA baseline 安全退化 |

```c
#include "hyle/c/core.h"

if (hyle_version() != HYLE_VERSION_PACKED) {
  /* plugin/动态库路径装载了不同于 headers 的 HYLE。 */
}
printf("HYLE %s\n", hyle_version_string());
```

需要当前 CPU 的运行时精确 cache line 时调用 platform 的
`hyle_cpu_cache_line_size()`；不要用 runtime 值定义跨翻译单元 struct ABI。

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
| `HYLE_C_LIKELY(expr)` / `HYLE_C_UNLIKELY(expr)` | profile 证明概率明显偏斜的分支 | 表达式求值一次；未知编译器退为普通 bool |
| `HYLE_C_UNREACHABLE()` | 已由不变量证明绝不抵达的控制流末端 | 真正抵达是 undefined behavior，不做输入验证 |
| `HYLE_C_FATAL_ASSERT(expr)` | 违反后进程不能安全继续的程序不变量 | Debug/Release 都诊断并 abort；业务失败改返回 error |
| `HYLE_C_RESTRICT` | 放在已证明互不别名的 pointer declaration 上 | 承诺错误是 undefined behavior；未知编译器安全退为空 |
| `HYLE_C_ASSUME(expr)` | 告诉优化器某个内部不变量恒真 | 表达式必须无副作用；Debug 为 false 时诊断并 abort，Release 为 false 是 undefined behavior |
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

业务输入验证必须写正常分支并返回 error，不能拿 `ASSUME` 代替；有副作用的表达式在 Debug 与
Release 可能有不同求值行为。`RESTRICT` 要覆盖 pointer 有效期内的全部访问，局部看起来不冲突
并不够。flatten/unroll 都可能增加 code size、伤害 I-cache，所以没有代表性 profile 时不加。

## 侵入链表类型

| 名称 | 说明 |
|---|---|
| `hyle_intrusive_list_hook` | 嵌入调用方节点的 prev/next；一个 hook 同时只能属于一条链 |
| `hyle_intrusive_list` | circular sentinel list；不拥有节点、不分配 |
| `HYLE_INTRUSIVE_ENTRY(hook,type,member)` | 从 member hook 恢复外层标准布局 C struct |

| inline 函数 | 合同 |
|---|---|
| `hyle_intrusive_hook_init` | 新/已脱链 hook 初始化；不可拿它强行重置 linked hook |
| `hyle_intrusive_hook_linked` | NULL/idle 为 false |
| `hyle_intrusive_hook_unlink` | 从当前链 O(1) 摘除；NULL/idle 幂等 |
| `hyle_intrusive_list_init` | 使用 list 前必须调用 |
| `hyle_intrusive_list_empty` | NULL 视为空；已初始化 list 为 O(1) |
| `hyle_intrusive_list_push_back/front` | O(1) 无分支；前置条件是 list/hook 非 NULL 且 hook 未链接 |
| `hyle_intrusive_list_try_push_back/front` | O(1) checked 入口；成功 1，NULL 或 double-link 为 0 |
| `hyle_intrusive_list_front/back` | 返回 hook；空/NULL 返回 NULL，不摘链 |
| `hyle_intrusive_list_pop_front/back` | O(1) 摘链并返回 hook；空/NULL 返回 NULL |
| `hyle_intrusive_list_clear` | O(n) 摘除全部节点，不释放节点 |
| `hyle_intrusive_list_append` | O(1) splice source 到 destination 尾；source 清空；非法/self 为 0 |

```c
typedef struct request {
  int id;
  hyle_intrusive_list_hook ready;
} request;

hyle_intrusive_list queue;
request item = {42, {NULL, NULL}};
hyle_intrusive_list_init(&queue);
hyle_intrusive_list_push_back(&queue, &item.ready);

hyle_intrusive_list_hook* hook = hyle_intrusive_list_pop_front(&queue);
request* next = HYLE_INTRUSIVE_ENTRY(hook, request, ready);
```

C 没有析构器：节点存储结束前必须 `hyle_intrusive_hook_unlink`，list 存储结束前必须 clear/
pop 完或确保节点仍更长寿。linked 期间节点地址不可变化；多重 membership 使用多个 hook；
并发访问由调用方同步。所有 list 操作都是 header-inline，因此没有每次跨 DLL 的热路成本。

## C++ 能力到 C 的通路

| C++ 能力 | 纯 C 通路 | 是否同档 |
|---|---|---|
| hook/list RAII auto-unlink | 存储结束前显式 `hook_unlink`，list 结束前 `clear` | 同一 O(1) 摘链；析构语法是 C++ 特性 |
| `move_front_to` | `pop_front` 后 `push_back` | O(1) |
| `append` / list move | `hyle_intrusive_list_append` | O(1) |
| `consume_all` | 循环 `pop_front` 后调用 C 函数 | O(n)，当前 hook 在用户回调前已摘链 |
| `for_each` | 从 `sentinel.next` 遍历到 `&sentinel`；若摘链先保存 `next` | O(n) |
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

动态符号仍只有 `hyle_version`、`hyle_version_string`。compiler contract、缓存常量、三种
prefetch、版本分解与 intrusive list 都是编译期/header-inline 能力，不进入导出表。
`tests/core/c_surface_symbols.txt` 锁动态导出；`c_core_test.c` 以 C 编译器覆盖 compiler macro
有效路径、版本、cache/prefetch、NULL/非法输入、空队列及完整链表生命周期；Debug death-test
覆盖 C assume 违约，header self-contained gate 从空翻译单元编译 C 头。

安装后最小工程见 `samples/package/core/c` 与 `samples/package/core/cpp`。它们只请求
`find_package(Hyle COMPONENTS core)` 并链接 `hyle::core`，由 Windows/Linux/macOS 的
shared/static package smoke 分别配置、构建和运行，避免示例暗中依赖源码树 include 或其它组件。
