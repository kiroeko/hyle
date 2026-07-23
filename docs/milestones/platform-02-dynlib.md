# platform-02：dynlib 动态库加载

| 字段 | 值 |
|---|---|
| **阶段** | P1 |
| **模块** | platform/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；platform +8 case）|

> **范围说明**:roadmap platform-02 是 "virtual_memory + dynlib"。本里程碑只做 **dynlib**(自包含、解锁 compute/simd 变体 DLL 加载);**virtual_memory 另立 platform-02b**(mem-02 mimalloc 自带 VM,HYLE VM 优先级低、按需再做);**path/fs/env/os_info → platform-03**。
> **cloc**：**生产 158 行**(dynlib.h 39 + dynlib.cc 119) + **测试 73 行**。

## clay API diff + 改进

- **覆盖**:open/sym(→symbol/symbol_as)/close/error/extension。
- **改进①(RAII)**:clay 手动 open/close(忘 close 漏 OS 句柄);HYLE `DynLib` 析构自动 close,move-only 转移所有权。
- **改进②(错误，历史实现)**:clay error 走 `char*(Allocator*)`;初版 HYLE 用 thread-local 256B buffer + `dynlib_last_error()`。
- **改进③(typed)**:`symbol_as<Fn>(name)` 类型化查符号(clay 只 void*)。
- **POSIX 语义**:`RTLD_NOW|RTLD_LOCAL`——NOW 在 load 时解析全符号(缺符号当场失败,非调用时崩)、LOCAL 不把库符号泄进全局。
- **gap 登记**:~~clay `dynlib_get(path)`~~ → ✅ **`DynLib::get(path)`**(2026-07-17):Win `GetModuleHandle`(非拥有)/POSIX `dlopen(RTLD_NOLOAD)`(拥有一次 close);未加载则失败不加载。

## 对抗式自审（六类）

- **① 正确性**:owns_ 区分 open(拥有,close 时 free)vs current(进程镜像,不 free);move 转移后源 invalid;dlsym 前 `dlerror()` 清旧错误(区分 "符号值为 NULL 但存在" vs "不存在"——经典 dlsym 陷阱);double close 安全。
- **② 性能**:加载期一次性操作,非热路径;symbol_as inline reinterpret_cast 零开销。
- **③ 规范**:.cc 用 windows.h/dlfcn.h(内部不出头);公开头只宏 + 类声明;RAII;错误 thread-local 无分配。
- **④ clay 覆盖**:open/**get**/sym/close/error/extension 全覆盖 + RAII/typed/无分配错误三改进。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:RAII 类比 clay vtable+手动管理简洁。
- **平台真相**:Win LoadLibrary + Linux dlopen **真编测**;mac 路径(dylib/dlopen 同 POSIX)**已写未验**(无机器),测试 lib 名按平台 #if。

## 交付物

- `include/hyle/platform/dynlib.h`（DynLib RAII + dynlib_extension；当前错误形状见下方追记）
- `src/platform/dynlib.cc`（Win LoadLibrary / POSIX dlopen）+ CMake(Linux 链 `${CMAKE_DL_LIBS}`)
- `tests/platform/dynlib_test.cc`（8）

## 参考

- clay `sys/os.h` dynlib 段(345-399)。conventions §10（变体 DLL + dlopen + score 选优）。

## 2026-07-23 追记：错误接口重收口

本页正文保留 platform-02 当时的任务事实。未发布 ABI 在
[`platform-error-convergence`](./platform-error-convergence.md) 中改为 C++ `Result`、
C `hyle_status` + 出参；模块私有 `dynlib_last_error` 已删除。当前稳定接口只看
architecture 与 guide，不从历史任务正文推断。
