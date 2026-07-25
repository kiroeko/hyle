# c-core：core 的纯 C / FFI 映射

> 场景选型先读 [`core.md`](./core.md)。本页只说明
> `include/hyle/c/core.h` 的精确 C 形状、所有权和坑。

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

```c
#include "hyle/c/core.h"

if (hyle_version() != HYLE_VERSION_PACKED) {
  /* plugin/动态库路径装载了不同于 headers 的 HYLE。 */
}
printf("HYLE %s\n", hyle_version_string());
```

需要当前 CPU 的运行时精确 cache line 时调用 platform 的
`hyle_cpu_cache_line_size()`；不要用 runtime 值定义跨翻译单元 struct ABI。

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

## 导出形态与机械门

动态符号只有 `hyle_version`、`hyle_version_string`。缓存常量、版本分解与 intrusive list 都是
编译期/header-inline 能力，不进入导出表。`tests/core/c_surface_symbols.txt` 锁动态导出，
`c_core_test.c` 以 C 编译器覆盖版本、cache 和完整链表生命周期，header self-contained gate
单独编译 C 头。
