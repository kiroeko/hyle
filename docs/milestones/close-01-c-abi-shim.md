# close-01：独立 `extern "C"` C/FFI 产品面

> 本页是跨模块 C/FFI 施工任务，不是当前 API 参考。稳定原则见
> [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md)，逐符号用法见
> [`../guide/`](../guide/README.md)。

| 字段 | 值 |
|---|---|
| **阶段** | P6（收口） |
| **模块** | 收口 / 全模块 |
| **Tier** | 边界 |
| **LOC 估** | 2-3k |
| **骨架?** | 否 |
| **状态** | 🔶 **部分落地**：core + error + platform + mem + time + obs + util + simd + container + sched + io C-ABI ✅；install/export ✅；compute/graph、math 边界与全库跨编译器 / FFI 冒烟仍待 |

> **本文件的目的**：跟踪 C/FFI 产品面的跨模块实施与验收。HYLE 是 C++ 库，原生 C++
> 头是 C++ 用户的首选 source API；C ABI 另行设计，用于 C 和跨编译器/跨语言绑定。

## 产品铁律（用户拍板，2026-06）

**纯 `extern "C"` C 头是唯一承诺跨编译器/跨 STL/跨语言二进制稳定的接缝，
但不是唯一产品 API。** 推论：

1. **C++ 头（`include/hyle/**/*.h`，除 `include/hyle/c/`）是原生 C++ 产品面**。它们服务
   同工具链源码开发和 expert kernel，提供模板/RAII/PMR，但**不承诺跨编译器/跨 STL ABI 稳定**。核心仓库也不另建
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

- [x] core/error/platform/mem/time/simd/util/obs/container/**sched**/io 已封
  `include/hyle/c/<mod>.h`，纯 C 编译器可 include，与内部 C++ 头物理隔离；
  compute/graph 仍待各自模块收口，math 先做产品边界裁决。
- [x] **样板 error**（2026-07-17，2026-07-23 重审）：`include/hyle/c/error.h` + 文档
  [`c-error.md`](../guide/c-error.md)。Status 全码、所有权、上下文、buffer 格式化与传播均可达。
  早期诊断 handler 已在 reclosure 中删除：日志归 `obs`，合同失败归 fail-fast，不再混入
  可恢复错误 C ABI。
- [x] **样板 platform**（2026-07-17，2026-07-26 NUMA 复开）：`include/hyle/c/platform.h` + `c_platform.cc` + 纯 C `hyle_platform_c_test`。覆盖 wait/Parker、Thread、DynLib、VM/VmRegion、hugepage/NUMA（actual nodes/capability/selected interleave）、完整 fs/env/fd、MappedFile（含 advice/residency）、CPU/cache/topology/per-CPU 能力；私有 `hyle_platform_last_error` 已删除并统一共享 Status。当前 **120/120** 动态声明、纯 C 引用与导出机械对齐。文档：[`c-platform.md`](../guide/c-platform.md)。
- [x] **样板 mem**（2026-07-23 重审）：`include/hyle/c/mem.h` + `c_mem.cc` + 纯 C
  `hyle_mem_c_test`。覆盖 system allocator/collect、Quota/Stats、Region/FixedRegion/FixedPool、
  hard `SharedPool` + 显式 `SharedPoolCache`、Stack/PooledRegion/GrowablePool、HugeRegion 与
  typed raw-slot 宏；Stats 用 POD out-snapshot + bounded leak copy。当前 post-error-convergence
  **103** 个动态导出由 header、
  纯 C consumer 与 DLL/ELF manifest 同步。能力边界和剩余 parity 审计见
  [`mem-final`](./mem-close-plan.md)，逐符号文档见 [`c-mem.md`](../guide/c-mem.md)。
- [x] **util**（2026-07-19 全面复审）：`include/hyle/c/util.h` + `c_util.cc` + 纯 C smoke。RNG 用稳定 POD；Ref/ScopeGuard/Registry/SpinLock 用 opaque handle + callback/ctx 类型擦除；demangle/help 用 buffer-fill。机械覆盖 46/46 可调用入口，其中动态导出 44 个、header-inline 2 个；三平台导出及 C/C++ 测试通过。仅 `type_name<T>`/`type_hash<T>` 因依赖编译器静态类型而有意留 C++。文档：[`c-util.md`](../guide/c-util.md)，复审：[`util-module-closure.md`](./util-module-closure.md)。
- [x] **container**（2026-07-19 收口）：`include/hyle/c/container.h` +
      `c_container.cc`，覆盖字符串/路径、StringInterner、runtime-erased POD
      vector/flat Swiss map、五类 pointer-payload queue 和 append-only vector。
      纯 C 测试引用 124/124 callable names；109/109 动态符号，另 15 个
      header-inline 便利入口。allocator/OOM/alias/collision/并发均有红队。文档：
      [`c-container.md`](../guide/c-container.md)，复审：
      [`container-module-closure.md`](./container-module-closure.md)。
- [x] **time**（2026-07-25 重新收口）：`include/hyle/c/time.h` + `c_time.cc` + 纯 C
      `hyle_time_c_test`。覆盖高精度/opt-in fast 时钟、饱和 deadline、sleep、opaque
      timer、exact-ns radix general queue、`CommonTimeoutQueue`、visitor 批量交付、caller-storage
      bulk control 和 owning `hyle_status`；41 个动态符号 + 4 个 header-inline 便利入口，不冻结
      C++ 类布局，不导出 STL/模板，也不引入 time→mem。文档：
      [`c-time.md`](../guide/c-time.md)。
- [x] **obs**（2026-07-25 严格复审）：`include/hyle/c/obs.h` + `c_obs.cc` + 纯 C
      `hyle_obs_c_test`。覆盖结构化 record、level、quiescent opaque sink、optional flush、
      默认 sink，以及 OFF 时稳定 no-op / ON 时接单一 Tracy client 的 runtime trace；
      23 个动态入口 + 5 个 header-inline level 便利入口。可恢复失败统一使用 owning
      `hyle_status`，不再有 obs 私有 TLS last-error。C++ 原生 trace 宏仍保留编译期
      零成本能力。文档：
      [`c-obs.md`](../guide/c-obs.md)，复审：
      [`obs-module-closure.md`](./obs-module-closure.md)。
- error/platform/mem 五维复审（功能/下游利用、红队实现、测试/性能、ABI 覆盖、用户文档）见 [`core-foundation-c-abi-audit.md`](./core-foundation-c-abi-audit.md)；time 见 [`time-module-closure.md`](./time-module-closure.md)，util 见 [`util-module-closure.md`](./util-module-closure.md)，container 见 [`container-module-closure.md`](./container-module-closure.md)。
- **架构定调（2026-07-18，2026-07-23 修订）**：纯 `extern "C"` 自由函数导出；版本协商
  只在独立插件边界需要。C++ 原生面与 C/FFI 面独立审查；有状态能力不能仅因“是类”而从
  C 排除。稳定原则见 [`../architecture/c-abi-design.md`](../architecture/c-abi-design.md)。
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

## C++ 形状到 C 能力的用户通路复核

“头里有多少函数”只证明符号覆盖，不证明 C/FFI 用户能完成同一件事。2026-07-20
按 authoring guide §4.1 重新检查已收口模块：每个不能直接导出的模板、RAII 或
源码宏，guide 必须给出稳定原语映射、所有权/完成时序，并在非平凡组合处给可复制
recipe。当前入口如下：

| 模块 | C++ 形状的 C/FFI 通路 | 用户文档 | 判定 |
|---|---|---|---|
| error | `Result<T>`→`hyle_status` + 分支/return-if-error；作用域 handler→push/pop token | `c-error` §4.6/§5 | 完整 |
| platform | RAII resource→opaque create/destroy；typed symbol→`void*` + caller cast；遍历模板→visitor | `c-platform` §3.8/§4 | 完整 |
| mem | typed pool→runtime size/align + typed 宏；RAII allocator→vtable + 显式 free；stats/leak→opaque decorator + POD snapshot/copy-out inventory；selected NUMA→pointer/count/chunk owner | `c-mem` §3.9/§4 | ✅ 严格复证；109/109 声明、纯 C 引用与动态导出一致，C++-only 只剩模板/RAII/raw fiber stack 等语言专属形状 |
| time | `Instant`→`uint64_t`；侵入节点/source concept→opaque timer/source + visitor | `c-time` §2–§6 | 完整 |
| simd | ISA-dependent vector/D-tag 不冻结；普通用户能力由 whole-buffer pointer/count API 动态派发 | `c-simd` 全文 + `simd-usage` | 完整 |
| util | `Ref<T>`/scope RAII→type-erased ref/guard；静态 type metadata 无运行时等价物 | `c-util` 对应各节 | 完整 |
| obs | format/RAII trace→调用方格式化 + record/runtime zone begin/end；sink→opaque handle | `c-obs` §1–§5 | 完整 |
| container | typed vector/map→runtime-erased POD 容器；拥有型结果→caller buffer；payload 生命周期显式 | `c-container` 全文 | 完整 |
| sched | `Task<T>`/Join→result context + oneshot；when_all→WaitGroup；when_any→winner + oneshot + cancel | `c-sched` §9（可复制 recipe） | 完整 |
| io | proactor/coroutine→token+completion；RAII handle→调用方 native handle；multishot→flags+显式归还；shmem→opaque handle | `c-io` | 底层机制完整；HTTP/TLS 高层 callback/state-machine facade 待补 |

复核的红线是：未来任何 guide 若只能写“C 没模板，所以不提供”，且又没有上述
稳定组合通路，就必须把模块从“C ABI 完整”退回“有覆盖缺口”，直到补
POD/opaque/callback 接口。

## 验收标准（"算完成"的硬指标）

- **纯 C 可消费**：一个用 **C 编译器**（非 C++）编译的 TU，`#include` C shim 头并调用 `make/调用/destroy` 全程通过——证明无 mangling、无模板、无 STL 泄漏。
- **跨编译器**：MSVC 编的 HYLE `.dll` 被 GCC/Clang 编的消费者链接调用通过（反之亦然）——这是 C++ 头做不到、C ABI 才有的能力，必须实测。
- **FFI 冒烟**：TS（N-API/WASM）+ Python（ctypes）各调一个 HYLE 函数跑通（roadmap P6 门槛）。
- **异常隔离**：故障注入测试确认无 C++ 异常逃逸到 C 边界。
- **依赖**：各对外模块 C++ 实现已稳定（C shim 是薄转发层，底层没定型就封等于返工——故排在 P6）。

## 运行时依赖 + 安装/分发（2026-07-05 登记；2026-07-29 已落地）

**原问题**：io-08 TLS 首次把外部动态运行时依赖带进产品，暴露出本地能链接不等于 SDK 能迁移。

**当前结果**：`HYLE_INSTALL` 已生成 relocatable `HyleConfig.cmake`/export targets；headers+shared、
headers+static 与 clean tracked-source 三种形状均有独立 consumer/迁移 smoke。static 包传播
mimalloc/nghttp2/ggml/volk 及所需 OpenSSL/CUDA/Threads 闭包，shared HYLE 模块使用 sibling-relative
loader path；Windows consumer 可用 `hyle_stage_runtime_dependencies()` staging HYLE DLL。许可证按
实际 target 闭包安装。实现与验证证据见
[`build-package-system-closure.md`](./build-package-system-closure.md)。

**仍未完成的发布层工作**：真实语言 FFI、跨编译器 C ABI、可选外部动态 runtime 的发行策略，以及
包含全部 fetched source 的离线依赖 bundle。它们不能再被表述成“install/export 尚未实现”。
