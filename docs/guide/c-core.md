# c-core：Native FFI bootstrap 与版本合同

> 场景选型先读 [`core.md`](./core.md)。本页面向 Java FFM、.NET P/Invoke、Rust
> `extern "C"`/dynamic loader 和 Python `ctypes`/`cffi` ABI mode。调用方不需要、也不应
> 编译 C/C++ shim。

## 先看产品边界

core 的三层 surface 是：

| 层 | 当前 core 形状 | 是否 direct FFI |
|---|---|---|
| C++ source API | version constexpr/函数、cache layout/prefetch、compiler contract、`IntrusiveList` | 否，由 C++23 translation unit 编译 |
| native C source helper | **不提供** | 否；HYLE 没有独立的原生 C 客户产品面 |
| Native FFI ABI | `hyle_core` shared library 的 3 个 unmangled 符号 | 是 |

`include/hyle/c/core.h` 是 Native FFI **ABI schema**，不是给原生 C 业务编程提供的
source helper 头。它能被 C 编译器或 binding generator 解析，只是为了锁定语言中立的
calling convention 与整数宽度；“头能由 C 编译”不表示 HYLE 维护 native C API。

## 动态导出：3 个符号

```c
uint32_t hyle_ffi_abi_version(void);
uint32_t hyle_version(void);
const char* hyle_version_string(void);
```

三个入口都使用 C calling convention；Windows 上显式为 `cdecl`。因此 .NET 应声明
`CallConvCdecl`，Python 在 Windows 使用 `ctypes.CDLL` 而不是 `WinDLL`；Java FFM 与 Rust
`extern "C"` 使用平台 C ABI。

| 符号 | 作用 | 编码/所有权 |
|---|---|---|
| `hyle_ffi_abi_version` | binding 查询当前 `hyle_core` artifact 所属的全局 ABI generation | 16/16 packed `major/minor` |
| `hyle_version` | 已加载 HYLE 产品版本 | 8/8/8 packed `major/minor/patch` |
| `hyle_version_string` | 人读产品版本 | `hyle_core` 拥有的 UTF-8/ASCII、NUL-terminated 借用字符串 |

`HYLE_FFI_ABI_VERSION*` 与 `HYLE_VERSION*` 是 schema/header 常量，不是可由
`dlsym`/`GetProcAddress` 解析的符号。每个外语 binding 应把它生成时所需的
FFI ABI major/minor 作为自身常量，再与 runtime 查询比较。

## 加载与兼容流程

binding 的正确启动顺序是：

1. 按确切路径加载 `hyle_core` shared library；
2. 首先只解析并调用 `hyle_ffi_abi_version`；
3. 检查 runtime major 与 binding major 相同，runtime minor 大于或等于 binding 所需 minor；
4. 通过后才解析该 binding 的其余 required symbols；
5. 需要日志、包匹配或问题诊断时，再读产品版本。

FFI ABI 是未来跨语言调用合同，但**现在没有已发布 binding，因而没有兼容对象**。首个 FFI 产品
发布前，`hyle_ffi_abi_version` 的名字、签名、版本值乃至整个方案都可以直接调整，不保留无人使用
的旧入口。某个 binding 正式发布后，它实际依赖的 bootstrap 才须继续可发现，让旧 binding 能
读出并拒绝不兼容 generation；普通 symbol 的 major/minor 演进规则也从那一刻开始生效。产品版本
是 HYLE release identity，两者有意独立。不要用产品版本代替 ABI
握手，也不要在 ABI 握手前乐观地解析新模块符号。若老库连
`hyle_ffi_abi_version` 都没有，binding 必须将其视为不兼容，不得回退猜测。

常规 shared 产物名为 Windows `hyle_core.dll`、Linux `libhyle_core.so`、macOS
`libhyle_core.dylib`（安装包可同时有带版本的文件名/链接）。生产 binding 优先由
应用的部署层传入 canonical/absolute path，避免 loader 意外拾取另一份同名库。
Java/.NET/Rust/Python host 不使用 `target_link_libraries`；CMake runtime staging helper 只服务
CMake 构建的 native executable。完整交付边界见 [`distribution`](./distribution.md)。

## 外语类型映射

| 语言/机制 | 两个 `uint32_t` 查询 | `hyle_version_string` |
|---|---|---|
| Java FFM | 32-bit integer carrier，按 unsigned 位模式解读 | address，立即复制 NUL-terminated 字符串 |
| .NET P/Invoke | `UInt32` | `IntPtr`，立即 marshal/copy |
| Rust | `unsafe extern "C" fn() -> u32` | `unsafe extern "C" fn() -> *const c_char` |
| Python `ctypes` | `restype = c_uint32` | `restype = c_void_p`，再用 `string_at` 复制 |

Python 直接加载示例（`REQUIRED_ABI` 由 binding 生成时的 schema 决定）：

```python
import ctypes
import pathlib
import sys

REQUIRED_ABI = (1 << 16) | 0
library = pathlib.Path(sys.argv[1]).resolve(strict=True)
core = ctypes.CDLL(str(library))

core.hyle_ffi_abi_version.argtypes = []
core.hyle_ffi_abi_version.restype = ctypes.c_uint32

runtime_abi = core.hyle_ffi_abi_version()
required_major, required_minor = REQUIRED_ABI >> 16, REQUIRED_ABI & 0xFFFF
runtime_major, runtime_minor = runtime_abi >> 16, runtime_abi & 0xFFFF
if runtime_major != required_major or runtime_minor < required_minor:
    raise RuntimeError(f"incompatible HYLE FFI ABI: 0x{runtime_abi:08x}")

# ABI 握手通过后才解析其余 required symbols。
core.hyle_version.argtypes = []
core.hyle_version.restype = ctypes.c_uint32
core.hyle_version_string.argtypes = []
core.hyle_version_string.restype = ctypes.c_void_p

product = core.hyle_version()
product_tuple = ((product >> 16) & 0xFF,
                 (product >> 8) & 0xFF,
                 product & 0xFF)
product_text_pointer = core.hyle_version_string()
if not product_text_pointer:
    raise RuntimeError("hyle_version_string returned null")
product_text = ctypes.string_at(product_text_pointer).decode("ascii")
```

三个函数都无参数、不分配，可并发调用。`hyle_version_string` 的指针不得 free，
且只在承载 `hyle_core` 的模块仍加载时有效；在 `dlclose`/`FreeLibrary` 前必须复制。

## core 其余能力为什么不导出

| C++ source 能力 | direct FFI 状态 | 理由/外语路径 |
|---|---|---|
| header product/ABI 常量 | 不导出为变量 | binding 内嵌 required ABI；握手只调用 `hyle_ffi_abi_version`，两个产品版本查询仅用于可选诊断 |
| cache-layout constexpr | 无符号 | 是 C++ 对象布局合同；外语不应镜像 HYLE 内部类型布局 |
| 三种 prefetch | 无符号 | 必须由调用 TU 编译器按 ISA 发指令；单次 FFI hop 会破坏它的价值 |
| compiler/fatal contract | 无符号 | 本质是 C++ 调用方的编译期/优化器语义 |
| `IntrusiveList` | 无符号 | 细粒度 pointer primitive；managed/moving heap 不满足 stable-address hook 合同，外语使用宿主容器 |

不要从旧头或历史代码重建 transparent intrusive-list binding。那种 C list 的 sentinel
自引用：**任何已初始化 list，即使为空，也不能复制、`memcpy`、按值返回或搬到新地址**；
非空时节点还会反向指向旧 sentinel。当前 core schema 已移除这类 helper；若外语确有队列任务，
应由拥有数据的上层模块提供 opaque handle 与 batch operation，而不是镜像裸 hook。

若外语算法需要当前机器的 best-effort cache-line observation，这属于 **platform** 的真实任务
候选，不是 core 导出。platform 当前 FFI 仍为黄色；若复审后保留该动态能力，binding 必须先从
exact `hyle_platform` artifact 调用届时选定的 `hyle_platform_ffi_abi_version`，不能用 core 的查询
结果替 platform DLL/so/dylib 背书。该观察值即使未来导出，失败时也只能回退 64，不能用于公共
ABI 布局或 safety-critical false-sharing padding。

## 示例与机械证据

这些 probe 是 ABI 可绑定性与发布验收工具，不是已经发布的 NuGet/JAR/Rust/Python binding
package；生产级 binding 还需各自的高层 wrapper、artifact 选择和所声明平台的发布矩阵。

- [`samples/ffi/core`](../../samples/ffi/core/README.md) 用 .NET 8 `LibraryImport` 和 Java 25 FFM
  加载同一个指定 shared artifact，验证 ABI bootstrap 与两个产品版本查询；随后各自加载只导出
  不兼容 ABI major 的 test-only fixture，证明拒绝发生在其余 symbol resolution 之前。
- [`scripts/run-core-ffi-smoke.py`](../../scripts/run-core-ffi-smoke.py)
  `--library <absolute-path> --incompatible-library <fixture-path>` 运行这两个正/负
  managed-runtime probe；
  它们不编译 C shim，也不搜索模糊库名。
- [`tests/core/c_surface_symbols.txt`](../../tests/core/c_surface_symbols.txt) 锁定三个动态导出，
  shared artifact 多出/少出都失败。
- [`c_core_test.c`](../../tests/core/c_core_test.c) 与
  [`samples/package/core/ffi_abi`](../../samples/package/core/ffi_abi) 只是用 C 编译器表达的 ABI schema/
  installed-package conformance fixture，不是 native C 产品 sample。
