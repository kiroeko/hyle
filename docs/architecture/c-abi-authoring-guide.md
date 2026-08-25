# Native FFI ABI 施工参考

> **读者**：为 HYLE 模块新增或复审 Java/.NET/Rust/Python 动态绑定面的开发者。
> **配套**：why 与三层边界见
> [`c-abi-design.md`](./c-abi-design.md)；本页只规定 how 和验收。

## 0. 先确认这是 direct FFI 任务

HYLE 没有原生 C 客户。C 声明是 Native FFI ABI schema；目标是调用方不编译 C/C++ shim，
直接从 `.dll` / `.so` / `.dylib` 解析并调用符号。

先给每项能力填表：

| 能力 | Java/.NET/Rust/Python 任务 | 是否 runtime | direct FFI 形状 | 不导出理由 |
|---|---|---|---|---|
| capability | 谁为什么调用 | 是/否 | export / opaque / POD / callback / buffer / batch | 仅编译期、仅本地布局、无跨语言价值 |

裁决规则：

- 有真实跨语言 runtime 任务：必须有 exported symbol 通路；
- C++ RAII：改成 opaque handle + 明确 create/close/destroy；
- 模板：只有真实任务成立时才 runtime type erase，不能只给 typed macro；
- 高频循环：先设计 batch/fused 入口，不照搬逐元素函数；
- compiler hint、编译期 layout、source-only helper：可以明确不导出；
- `static inline`、macro、透明 C struct、纯 C recipe 都不能填入 direct FFI 列；
- 需要用户编译 shim 才能使用，视为 FFI 缺口，不是“已有 C API”。

如果整张表没有真实 runtime 任务，结论可以是“本模块 direct FFI = N/A”：保留自然的 C++ source
API，写清 non-export 理由，不创建空洞 parity symbols。下面的交付件和硬门只适用于至少承诺一项
direct FFI runtime 能力的模块。

## 1. 三份 surface 分开登记

| surface | 典型位置 | 验收 |
|---|---|---|
| C++ source API | `include/hyle/<mod>/*.h` | C++ sample/test、ownership 与组合 |
| native-C helper | 独立、明确标名的 source-helper 头或内部实现 | C self-contained/codegen test；不计 FFI |
| direct Native FFI ABI | C schema 中带 `HYLE_<MOD>_C_API` 的声明 | export manifest + 真实 managed binding |

公开文档和 review 表中禁止只写一个混合的 “C/FFI API”。`include/hyle/c/<mod>.h` 原则上只放
ABI schema；行为型 inline/macro helper 移到明确的 helper 层。迁移期仍共处时，也要以
section/comment 和 manifest 明确分界。

## 2. 一个模块的交付件

为模块 `<mod>` 建立或复审 direct FFI，至少包含：

1. `include/hyle/c/<mod>.h`：语言中立 ABI schema；
2. `src/<mod>/c_<mod>.cc`：C++ bridge；
3. 模块 CMake 接线和 shared-library export manifest；
4. C schema self-contained/layout test；
5. Java direct-load smoke；
6. .NET P/Invoke smoke；
7. `docs/guide/c-<mod>.md`：按导出入口记录 marshalling、ownership、threading 和 lifetime；
8. 若该模块随官方 Rust/Python binding 发布，再加入对应 package 的直接加载 smoke。

native-C helper 的 C test 可以保留，但它是额外的 source-helper 证据，不能替代第 5–7 项；
第 8 项适用时也不能替代。
static package smoke 也不能替代 shared artifact 的动态解析。

## 3. ABI schema 骨架

```c
#ifndef HYLE_C_<MOD>_H_
#define HYLE_C_<MOD>_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(HYLE_STATIC)
#  define HYLE_<MOD>_C_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(HYLE_<MOD>_BUILD)
#    define HYLE_<MOD>_C_API __declspec(dllexport)
#  else
#    define HYLE_<MOD>_C_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define HYLE_<MOD>_C_API __attribute__((visibility("default")))
#else
#  define HYLE_<MOD>_C_API
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  define HYLE_<MOD>_C_CALL __cdecl
#else
#  define HYLE_<MOD>_C_CALL
#endif

typedef struct hyle_foo_o hyle_foo_o;

HYLE_<MOD>_C_API hyle_status
HYLE_<MOD>_C_CALL hyle_foo_create(hyle_foo_o** out,
                                  const hyle_foo_options* options);
HYLE_<MOD>_C_API void HYLE_<MOD>_C_CALL hyle_foo_destroy(hyle_foo_o* foo);

#ifdef __cplusplus
}
#endif

#endif
```

要求：

- direct FFI 函数全部显式带导出宏；
- 使用 unmangled `extern "C"` 自由函数；
- calling convention 在 schema/build/binding 三处一致并有 smoke 锁定；不得依赖会被项目 flags
  改写的未记录默认值；
- schema 不泄漏 namespace、template、STL、reference、exception 或 C++ object layout；
- tag/flag 使用明确宽度整数 typedef + constants，不把 C enum 宽度放进 ABI；
- 不增加模块级函数指针表、`api_get` 或模块私有 ABI 版本；当前 pre-release core 用
  `hyle_ffi_abi_version`，其他实际承诺 direct FFI 的模块拟用
  `hyle_<module>_ffi_abi_version` 从**自己的 exact shared artifact**查询同一全局 ABI generation；
  FFI=`N/A` 的模块不制造 bootstrap；
- schema 中不新增行为型 `static inline`/macro；ABI tag/flag 常量可以是 declarative macro。
  迁移期仍保留的 helper 必须标为 `native-C helper: not exported, not FFI` 并登记移出裁决。

## 4. Export 是产品事实

每个模块维护期望 Native FFI 导出清单，并从实际 shared artifact 枚举 `hyle_*` FFI namespace 比对：

- Windows 检查 PE export；
- Linux 检查 ELF dynamic symbols；
- macOS 检查 Mach-O external symbols；
- FFI namespace 内额外导出和缺失导出都失败；同一库合法的 decorated C++ source-API exports 由
  C++ API/build contract 另管，不得误报成 FFI extra；
- 不能只 grep 头文件，也不能把静态库符号当动态 ABI。

Java/.NET smoke 必须通过普通 loader 从本轮待验的 exact artifact path 加载，不允许依赖系统搜索
路径碰巧找到同名旧库，也不允许链接测试程序时偷偷静态带入 bridge。
它们先只解析待验 artifact 的 bootstrap（core 为 `hyle_ffi_abi_version`，其他 FFI 模块为
`hyle_<module>_ffi_abi_version`），确认 major 相同且 runtime minor 不低于 binding 所需版本，
再解析本模块的 required symbols。core 的查询结果不能替另一 DLL/so/dylib 背书；一个 binding
使用多个模块时逐 artifact 验证，并只从同一 package/install root 装载闭包。

## 5. Opaque handle 与 managed lifetime

### 5.1 Bridge 形状

opaque control type 在匿名 namespace 外定义，内部显式拥有 C++ 对象：

```cpp
struct hyle_foo_o {
  hyle::mod::Foo object;
};
```

桥接规则：

- create 的 out 参数在前，进入时清空，成功才写入；
- destroy/close 的 NULL 语义明确；不要默认宣称幂等；
- 资源在哪个 allocator/module 创建，就由同一侧销毁；
- 不跨 DLL 让 Java/.NET/Python allocator 释放 HYLE 内存；
- bridge 显式构造/逐字段转换，绝不 reinterpret C++ 对象布局；
- C++ 异常绝不越过 `extern "C"`。

### 5.2 四种绑定生命周期

每个 owning handle 都要能自然包装成：

| 语言 | 正常资源形状 | 必须验证 |
|---|---|---|
| Java | `AutoCloseable` / FFM arena 绑定 wrapper | 显式 close，GC 只兜底；close 后不可再调用 |
| .NET | `SafeHandle` / `IDisposable` | `ReleaseHandle` 走正确 destroy；delegate 不被提前回收 |
| Rust | owning newtype + `Drop` | 不复制 ownership；borrow 与 thread trait 符合 native 合同 |
| Python | context manager + 显式 `close()` | finalizer 不承担正常时序；异常路径仍释放 |

若存在 borrowed child/view，必须声明它由哪个 owner 保活、哪些操作会使它失效。若有异步工作或
callback，destroy 前必须有可执行的 cancel/drain/join 顺序；“调用方自己小心”不是 managed
lifetime 合同。

## 6. POD 与 options 的施工纪律

公开 POD 至少在 bridge 中检查 standard-layout、trivially-copyable、size/alignment 和关键 offset。
可扩展输入：

- 首部使用 `uint32_t struct_size`、`uint32_t reserved`；
- `reserved != 0` 拒绝；
- `*_REQUIRED_SIZE` 只覆盖真正必需语义前缀；
- 首次读取 size 也按 byte/`memcpy`，不能先把短旧对象解引用成当前类型；
- 先构造当前默认值，再覆盖调用方完整提供的字段；
- pointer+count、`has_*`+value 等语义组合必须原子解码；
- 短于 required prefix 拒绝，未知未来尾部忽略；
- 不要求 `struct_size == sizeof(current)`。

小且永久冻结的输出 snapshot 可以值返回；预期扩展的输出使用 `struct_size + out*` 或新增查询。
每个字段还要检查 Java FFM/JNA、.NET `StructLayout`、Rust `repr(C)`、Python ctypes 是否能无歧义
描述。C 编译器接受某个 layout 不是充分条件。

## 7. Callback：先写 lifetime，再写 typedef

标准 callback 是 function pointer + context：

```c
typedef int32_t(HYLE_<MOD>_C_CALL *hyle_visit_fn)(const hyle_item* item,
                                                  void* context);
```

设计前必须决定：

- callback 是否只在调用栈内同步发生；
- 是否会保存，最迟到哪个 unregister/cancel/drain 点停止；
- 从哪个线程调用，是否并发，是否允许重入；
- 返回值如何表示 continue/stop/error；
- callback 内能否再次调用该 handle；
- destroy 与 callback race 如何收束。

bridge 若要把普通 C function pointer 交给 `noexcept` C++ callable，使用显式 `noexcept` trampoline，
不 cast function pointer。control 内若同时保存 callback state 和会在析构时回调的 C++ 对象，
按析构逆序安排字段。

managed smoke 必须：

- 保持 Java upcall stub / .NET delegate 的强引用；
- 可能从 native worker thread 回调时，用覆盖完整注册期的 Java shared arena 承载 upcall stub；
  confined arena 只用于明确同线程且同步的 callback；
- 在 managed trampoline 内捕获 Java `Throwable` / .NET exception，按既定 status/stop/record 通路
  报告，绝不让 managed unwind 穿过 native frame；
- 主动制造 GC 压力后再触发 callback；
- 覆盖 native worker thread 回调（若合同允许）；
- unregister/drain 后释放引用，并证明 destroy 后不再回调。

## 8. Buffer、字符串与 allocator ownership

每个 pointer 参数在 schema/guide 中写全：

- input/output/inout；
- element 数还是 byte 数；
- NULL 和 zero-length 组合；
- alignment；
- 同步借用还是会被保存；
- 返回 view 的失效点；
- owning buffer 的释放函数和所属模块。

优先使用 caller-owned buffer-fill：

```c
HYLE_<MOD>_C_API size_t HYLE_<MOD>_C_CALL
hyle_foo_name(const hyle_foo_o* foo, char* destination, size_t capacity);
```

明确返回值是 required size、written size，是否包含 NUL，以及截断规则。若必须由库分配，连同
pointer 返回长度并提供 `hyle_<mod>_buffer_destroy`；禁止跨 allocator family free。

对 managed heap 数据，调用期间 pin/copy 的选择由 binding 决定；native API 必须通过同步/保存
合同让选择成为可能。异步保存 caller buffer 时，需要显式 registration token 或 library-owned
copy，不能悄悄保存短命 pin。

## 9. Batch-first，而不是给逐项调用找 inline 借口

Java FFM/JNA、P/Invoke、ctypes 的边界成本会淹没单次 pointer 操作。数据面 API 默认检查：

- 能否接受 descriptor 数组 + count；
- 能否用一个 call 完成 submit/poll/drain；
- 是否支持 caller-provided output arrays；
- per-item status、完成数和 partial failure 如何返回；
- callback 是否可按 batch 交付；
- 是否避免每元素 owning handle。

逐项函数可作为控制面便利，但正常 workload 必须走 batch。prefetch、intrusive hook 操作、微小
算术等 native primitive 不因 header-inline 很快就值得导出；如果跨语言任务确实需要其效果，应把
它融合进拥有数据和循环的批处理函数。

### 9.1 Shared library 根生命周期

binding 必须选择并记录一种策略：让 HYLE shared libraries 保活到进程结束，或由根 wrapper 显式
管理 unload。后一种策略必须按 `停止新调用 → cancel/drain → unregister callback → close/free
所有 native owner → 放弃 borrowed view → 释放 managed arena/delegate → 逆依赖序卸载模块` 执行；
若 package 实际加载了 hyle_core，它在所有依赖者之后卸载。任何 handle、function pointer、upcall
stub、borrowed string/buffer 或 native worker 尚存时都不得卸载；close/free 函数本身也必须在库
仍加载时调用。

## 10. 错误边界

- 一般资源工厂返回 `hyle_status` + out；
- out 进入时清空；
- 成功 status 为零成本表示，错误的消息 ownership 清晰；
- 不使用模块私有 TLS last-error；
- NULL/计数作为正常耗尽时，不得与非法输入/系统失败混淆；
- batch 返回完成数并给出整体或逐项错误语义；
- 调用方数据非法在 bridge 校验并返回错误；foreign/double free、错误时序等程序 bug 才
  fail-fast。

所有 error path 都要由 Java/.NET 实际读取分类/消息并正确销毁 owning error state，不能只在 C
测试中比较数值。

## 11. 验证矩阵

### 11.1 Schema/helper 门

- C 与 C++ 空翻译单元分别包含 schema；
- `-fshort-enums` 等 ABI 扰动下公开 layout 不变；
- bridge layout/tag asserts；
- native-C helper 功能与 codegen（若存在）。

这些门只验证 schema 与第二层 helper。

### 11.2 Direct FFI 硬门

前三项适用于每个承诺 direct FFI 的模块；其余项按该模块实际导出的 shape 验收。不存在的
handle/POD/string/buffer/callback/batch 形状必须写 `N/A + 理由`，不得为让矩阵整齐而增加导出。

- 实际 shared artifact 的 FFI-subset export manifest 精确；
- 待验 artifact 的 FFI ABI major/minor 握手先于其余 required-symbol resolution，并覆盖不兼容
  拒绝路径；首个 binding 发布前协议可直接重做，发布后才保留该 binding 实际依赖的 bootstrap
  名、C calling convention 与 `uint32_t(void)` 签名；
- Java 和 .NET 都从文件路径直接加载，不经过用户编译 shim；
- create/use/error/close 全生命周期；
- POD options 与 snapshot；
- UTF-8/string buffer-fill 和截断；
- input/output buffer、zero length、alignment；
- callback rooting、线程、unregister/drain；
- batch 正常路径与 partial completion；
- 重复 close/close 后调用按公开合同处理。

若发布官方 Rust/Python binding，它们也必须直接加载同一 artifact，至少覆盖版本、一个 handle、
一个 buffer 和错误路径。尚未承诺该语言产品时登记 N/A；不能用 C smoke 代替 Java/.NET 硬门。

### 11.3 平台门

- Windows：真实 DLL + calling convention + export + 最小动态调用；
- Linux：真实 ELF `.so` + export + 最小动态调用；
- macOS：真实 `.dylib` + export + 最小动态调用；
- 模块收口至少一台宿主完成 Java + .NET 双 managed smoke；正式发布某个 binding package 时，
  该 package 声明支持的每个平台都必须运行自己的 consumer；
- 位宽、UTF-8、loader search path 和 dependent-library staging 按发布包真跑；
- shared FFI 证据与 static/C++ package 证据分开。

## 12. Guide 必须回答什么

`docs/guide/c-<mod>.md` 的 “c” 表示 C ABI schema，不表示原生 C 用户产品。对每个 direct export
写清：

- Java/.NET 的类型映射；已承诺的其他 binding 追加自己的映射；
- handle ownership 与 managed wrapper；
- pointer/length、字符串编码、buffer free；
- callback rooting、线程和停止时序；
- batch 正常路径；
- status/error ownership；
- 从哪个 shared artifact 解析哪个 symbol。

native-C helper 只能在独立小节标明 “not exported / not direct FFI”；不要再用纯 C recipe 证明
跨语言能力等价。编译期能力不导出时，直接说明其只影响 native translation unit，以及 managed
用户为何不需要对应 symbol。

## 13. 收口 checklist

- [ ] 每项能力有 Java/.NET/Rust/Python runtime 场景或明确的 compile-time non-export 理由；
- [ ] C++ source、native-C helper、direct FFI 三份矩阵分开；
- [ ] 每个 direct FFI 入口都有真实导出宏、bridge 实现和 export manifest 项；
- [ ] binding 先检查待调用 artifact 的 bootstrap；core 当前用 `hyle_ffi_abi_version`，其他 FFI
      模块用 `hyle_<module>_ffi_abi_version` 返回同一全局 ABI generation，未复制私有版本号或函数表；
- [ ] 多 artifact binding 逐个握手并只从一个 package/install root 加载，未用 core 结果替别的
      artifact 背书；
- [ ] 无 STL/template/reference/exception/C++ object layout 泄漏；
- [ ] 实际 exported shape 中适用的 opaque handle create/close/destroy、borrow、线程和 async drain
      合同完整；不存在则记 N/A；
- [ ] 实际含 POD/options 时遵守 fixed-width、layout asserts、`struct_size` 演进规则，否则记 N/A；
- [ ] 实际含 callback 时，其保存、GC rooting、线程、重入、unregister/drain 已裁决，否则记 N/A；
- [ ] 实际含 buffer/string 时，其方向、长度、编码、ownership、失效和 free 完整，否则记 N/A；
- [ ] 实际有数据面时提供 batch-first 入口；逐项 FFI hop 未冒充正常性能路径，否则记 N/A；
- [ ] 实际有可恢复错误时，可恢复/fail-fast 边界清楚且 owning status 在 managed binding 中正确释放；
- [ ] 纯 C/header-inline/macro/recipe 证据未计入 direct FFI；
- [ ] 若模块承诺 direct FFI，Java + .NET 真实动态加载 smoke 通过；否则模块 FFI 明确 N/A；
- [ ] 已承诺的 Rust/Python binding smoke 通过；尚未承诺时明确 N/A；
- [ ] Windows/Linux/macOS shared artifact、导出、最小动态调用与依赖 staging 真跑；
- [ ] guide 按 symbol 记录 Java/.NET 映射、managed lifetime、buffer/callback/batch/error。
