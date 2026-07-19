# close-01：对外纯 `extern "C"` C-ABI shim（发布给用户的唯一稳定接缝）

| 字段 | 值 |
|---|---|
| **阶段** | P6（收口） |
| **模块** | 收口 / 全模块 |
| **Tier** | 边界 |
| **LOC 估** | 2-3k |
| **骨架?** | 否 |
| **状态** | 🔶 **部分落地**：error + platform + mem + time + simd C-ABI ✅（time/SIMD 2026-07-19）；全库跨编译器 / FFI 冒烟仍待 |

> **本文件的目的**：把"发布后用户必须用稳定 C ABI"这件事**白纸黑字钉死**，避免后面忘了或被"反正 C++ 头能用"糊弄过去。实现细节临到开工再细化，但**产品铁律 + 验收门槛现在就定**。

## 产品铁律（用户拍板，2026-06）

**HYLE 发布给外部使用者绑定的、唯一承诺 ABI 稳定的接缝 = 纯 `extern "C"` C 头 shim。** 推论：

1. **C++ 头（`include/hyle/**/*.h`，除 `include/hyle/c/`）不是发布产品面**。它们服务内部实现、
   同工具链源码开发和 expert kernel，**不承诺跨编译器/跨 STL/跨语言稳定**。核心仓库也不另建
   `std::vector`、C#、Java 或 Rust 容器 facade；这些薄适配由消费方或独立 binding 项目拥有。
   今天 samples 直接使用 C++ 头属开发期集成，**不等于** charter §6 承诺的稳定 C ABI。
2. **未做 close-01 之前，HYLE 不具备"可发布给 FFI 消费方"的资格**。Python ctypes / Rust bindgen / TS N-API 绑的必须是 C shim，不是现在的 C++ 头。**发布说明里必须讲清这条**，别让用户拿 C++ `.dll` 去跨语言绑然后踩 mangling。
3. **内部 C++23 的 ABI 风险由 C 头完全隔离**（charter §85）——这是内部敢用 C++23 的前提，也是 close-01 存在的全部理由。

## 设计（charter §6 / conventions §4，已定）

- **对外契约**：纯 `extern "C"` 自由函数 + 不透明句柄（opaque `_o`）
  create/destroy 配对 + 小型 POD/显式 allocator 注入。函数指针只在能力本身
  是回调/vtable 时出现；没有模块级 `xxx_api_t`/`api_get` 表。
- **零异常跨边界**：错误经 `Status` 值返回，绝不让 C++ 异常跨 C 接缝（UB）。
- **分配责任锁在 DLL 内**（防 CRT 堆不匹配），禁客户端 `new/delete` HYLE 对象。

## 具体事务（开工时细化，先占位）

- [x] 每个对外模块封一层 `include/hyle/c/<mod>.h`（或等价单独命名空间），**纯 C、可被 C 编译器 include**，与内部 C++ 头物理隔离。**目录约定已坐实**：`include/hyle/c/` 起手。
- [x] **样板 error**（2026-07-17，2026-07-19 复审）：`include/hyle/c/error.h` + 文档 [`c-error.md`](../guide/c-error.md)。Status 全码/所有权/格式化与诊断 handler 均可达；C/C++ scoped handler 共用严格 LIFO 还原门，乱序不再可能重发悬空回调。
- [x] **样板 platform**（2026-07-17，2026-07-19 扩面复审）：`include/hyle/c/platform.h` + `c_platform.cc` + 纯 C `hyle_platform_c_test`。覆盖 wait/Parker、Thread、DynLib、VM/VmRegion、hugepage/NUMA、完整 fs/env/fd、MappedFile、CPU/cache/topology；补 `hyle_platform_last_error`、thread opts `struct_size`、具名 flags/perms。纯 C 机械覆盖 114/114 入口。文档：[`c-platform.md`](../guide/c-platform.md)。
- [x] **样板 mem**（2026-07-17，2026-07-19 复审）：`include/hyle/c/mem.h` + `c_mem.cc` + 纯 C `hyle_mem_c_test`。面：`hyle_allocator` vtable、system allocator、MemoryArena / StackAllocator / Arena / GrowablePool / HugeRegion、进程级 hugepage/NUMA、typed 池宏；资源失败均有 last-error，成功清 stale。**有意仅 C++**：模板构造层（ObjectPool/GrowablePool/pmr）、有状态调试类（AllocStats/LeakTracker/StatsAllocator）、Allocator typed 层。纯 C 机械覆盖 51/51 入口。文档：[`c-mem.md`](../guide/c-mem.md)。
- [x] **time**（2026-07-19 收口）：`include/hyle/c/time.h` + `c_time.cc` + 纯 C
      `hyle_time_c_test`。覆盖高精度/opt-in fast 时钟、饱和 deadline、sleep、opaque
      `TimerEntry`、8×64 `TimerWheel`、`CommonTimeoutQueue`、visitor 批量到期回调和
      recoverable last-error；34 个动态符号 + 4 个 header-inline 便利入口，不冻结
      C++ 类布局，不导出 STL/模板。文档：
      [`c-time.md`](../guide/c-time.md)。
- error/platform/mem 五维复审（功能/下游利用、红队实现、测试/性能、ABI 覆盖、用户文档）见 [`core-foundation-c-abi-audit.md`](./core-foundation-c-abi-audit.md)；time 见 [`time-module-closure.md`](./time-module-closure.md)。
- **架构定调（2026-07-18）**：删除三模块的 `hyle_*_api` 函数指针表 + `api_get` + 版本门/版本宏——纯 `extern "C"` 自由函数导出（对齐 ggml 核心）。版本协商是"分发独立编译插件"才需要的件（clay 零、ggml 仅 backend 插件处），HYLE 单一 `.so`、未发布不背此负担。C++→C ABI 的类型擦除手法、三层架构、有状态调试类留 C++ 等决策见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md)。
- [x] **SIMD**：`include/hyle/c/simd.h` + `c_simd.cc` + [`c-simd.md`](../guide/c-simd.md)；
      纯 C 预处理入口 **247/247** 被测试引用，动态库导出 **244/244**（另 3 个 typed header-inline）。
- [x] SIMD 是首批语义样板之一：新增 `include/hyle/c/simd.h`；add/dot/math/scan/dtype/sort/
      bitpack 等无状态接口只用 pointer/count/POD，随机数状态使用
      `hyle_simd_random_o` make/destroy/fill，不冻结 C++ `RandomState` 布局。
- [x] SIMD C 头提供固定宽度 enum/POD、尺寸与对齐断言；按架构决策直接
      导出自由函数，不重新引入 `hyle_simd_api_t`/版本门，C shim 只转发现有
      dispatcher/kernel。
- [x] `samples/02_simd/main.c` + `hyle_simd_c_test` 是 **纯 C consumer**（C 编译器，
      只 include C shim），证明无 mangling/STL/template 泄漏。
- [ ] 目录约定坐实：`include/hyle/c/` = 稳定且受支持的发布面；`include/hyle/<mod>/` = 内部/
      同工具链 source surface，不作为发布契约宣传。
- [ ] conventions §2 的 `decls.h`（前向声明 + 导出宏）一并落地，卡住"公开头不泄漏内部类型"。
- [x] close-01 完成 SIMD shim 时同步更新 `docs/guide/simd-usage.md`：把普通示例换成纯 C，
      补 `struct_size`、调用方数组所有权与 opaque 生命周期，并保留普通动态、exact library、
      fixed TU 三层选择说明。

## 验收标准（"算完成"的硬指标）

- **纯 C 可消费**：一个用 **C 编译器**（非 C++）编译的 TU，`#include` C shim 头并调用 `make/调用/destroy` 全程通过——证明无 mangling、无模板、无 STL 泄漏。
- **跨编译器**：MSVC 编的 HYLE `.dll` 被 GCC/Clang 编的消费者链接调用通过（反之亦然）——这是 C++ 头做不到、C ABI 才有的能力，必须实测。
- **FFI 冒烟**：TS（N-API/WASM）+ Python（ctypes）各调一个 HYLE 函数跑通（roadmap P6 门槛）。
- **异常隔离**：故障注入测试确认无 C++ 异常逃逸到 C 边界。
- **依赖**：各对外模块 C++ 实现已稳定（C shim 是薄转发层，底层没定型就封等于返工——故排在 P6）。

## 运行时依赖 + 安装/分发（2026-07-05 登记，属 close 阶段，非各功能里程碑）

**问题来源**：io-08 TLS 引入第一个**外部动态运行时依赖**（OpenSSL 的 `libssl.so`/`libssl-3.dll`）。这暴露一个此前没有的关注点——HYLE 发布给用户时，**用户机器上怎么拿到 HYLE 依赖的第三方动态库**。这不是某个功能里程碑（io-08 只管把 TLS 机制做通）的事，是**统一的打包/分发问题**，钉在这里：

- **现状**：HYLE **完全没有 `install()` / `install(EXPORT)` / `Config.cmake` 机制**——目前只本地构建 + samples 直接链 `build/bin`。发布分发从零开始，属 P6。
- **两个维度别混**（用户提出）：
  1. **开发者构建 HYLE**：`find_package(OpenSSL)` 找到即可（Linux 系统库零配置 / Windows `OPENSSL_ROOT_DIR` 或 vcpkg）。这层各功能里程碑自己解决（io-08 已定 `find_package` 可选依赖）。
  2. **部署 HYLE 的最终用户**：用户机器要有 HYLE 依赖的运行时库。**动态链**（HYLE 默认，安全库随系统 CVE 更新）意味着 `libssl.so.3`/`libssl-3.dll` 必须在用户机——Linux 发行版几乎都预装 OpenSSL 运行时（不用 HYLE 带）；**Windows 不一定有，close 阶段的 `install()` 应能把依赖 DLL 随 HYLE 制品一起安装**（`install(IMPORTED_RUNTIME_ARTIFACTS OpenSSL::SSL ...)` 或 `$<TARGET_RUNTIME_DLLS>` copy）。
- **close 阶段要做（登记，开工时细化）**：
  - [ ] `install(TARGETS hyle_* EXPORT HyleTargets ...)` + `install(EXPORT ...)` + `HyleConfig.cmake`（+ `find_dependency(OpenSSL)` 等）——让下游 `find_package(hyle)` 可用。
  - [ ] 运行时依赖 DLL 分发策略：Windows `install($<TARGET_RUNTIME_DLLS>)` 带上 OpenSSL 等；Linux 依赖系统库（文档声明最低版本），不打包。
  - [ ] **静态 vs 动态的分发抉择**：动态=依赖用户机有库但制品小、安全库可独立更新；静态=单一制品无依赖但 CVE 要重编 + 体积 + 静态链 OpenSSL 的符号/初始化坑。默认动态；提供 `HYLE_STATIC` 全静态构建选项时，第三方也尽量静态链成单一制品。
  - [ ] 每个可选依赖（OpenSSL/将来的 ggml 等）都遵循"不开该能力=零依赖"（io-08 已立此范式：TLS 独立目标，不要 TLS 就不背 OpenSSL）。
- **为什么归 close 而非 io-08**：install/export/依赖分发是**一次性统一做**的收口件（对所有模块 + 所有第三方依赖），塞进单个功能里程碑会把它撑成打包里程碑，且要等对外 C-ABI 面（本文档主体）定型后才有意义。io-08 只需保证"TLS 可选 + 动态链干净 + 给 install 留好接口"。
