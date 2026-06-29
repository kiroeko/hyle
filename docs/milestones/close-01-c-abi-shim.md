# close-01：对外纯 `extern "C"` C-ABI shim（发布给用户的唯一稳定接缝）

| 字段 | 值 |
|---|---|
| **阶段** | P6（收口） |
| **模块** | 收口 / 全模块 |
| **Tier** | 边界 |
| **LOC 估** | 2-3k |
| **骨架?** | 否 |
| **状态** | 📋 计划（**承诺已锁定，实现待排期**） |

> **本文件的目的**：把"发布后用户必须用稳定 C ABI"这件事**白纸黑字钉死**，避免后面忘了或被"反正 C++ 头能用"糊弄过去。实现细节临到开工再细化，但**产品铁律 + 验收门槛现在就定**。

## 产品铁律（用户拍板，2026-06）

**HYLE 发布给外部使用者绑定的、唯一承诺 ABI 稳定的接缝 = 纯 `extern "C"` C 头 shim。** 推论：

1. **C++ 公开头（`include/hyle/**/*.h`）不是发布 ABI 面**。它们是"同一 C++ 工具链下的便利头"，**不承诺跨编译器/跨 STL/跨语言稳定**（模板/`std::expected`/类布局都会泄漏）。今天 samples 用它们集成属"C++ ABI 跨 DLL"，**不等于** charter §6 承诺的稳定 C ABI。
2. **未做 close-01 之前，HYLE 不具备"可发布给 FFI 消费方"的资格**。Python ctypes / Rust bindgen / TS N-API 绑的必须是 C shim，不是现在的 C++ 头。**发布说明里必须讲清这条**，别让用户拿 C++ `.dll` 去跨语言绑然后踩 mangling。
3. **内部 C++23 的 ABI 风险由 C 头完全隔离**（charter §85）——这是内部敢用 C++23 的前提，也是 close-01 存在的全部理由。

## 设计（charter §6 / conventions §4，已定）

- **对外契约**：不透明句柄（opaque `_o`）+ `make_X()/destroy_X()` 配对 + 值类型 POD 结构体 + 显式 `Allocator*` 注入 + 函数指针 vtable `xxx_api_t`（designated-init 填表）。
- **零异常跨边界**：错误经 `Status` 值返回，绝不让 C++ 异常跨 C 接缝（UB）。
- **分配责任锁在 DLL 内**（防 CRT 堆不匹配），禁客户端 `new/delete` HYLE 对象。

## 具体事务（开工时细化，先占位）

- [ ] 每个对外模块封一层 `include/hyle/c/<mod>.h`（或等价单独命名空间），**纯 C、可被 C 编译器 include**，与 C++ 便利头物理隔离。
- [ ] 先做样板：拿**最接近 C ABI 的模块试封第一层**（`mem` 的 `Allocator` 已是 vtable + 不透明 state，最贴；或 `container` SwissTable 的 type-erased 核）。
- [ ] `samples/` 加一个 **纯 C consumer**（用 C 编译器编，只 include C shim），证明跨语言可绑。
- [ ] 目录约定坐实：`include/hyle/c/` = 稳定 C ABI（承诺）；`include/hyle/<mod>/` = C++ 便利头（不承诺跨编译器）。
- [ ] conventions §2 的 `decls.h`（前向声明 + 导出宏）一并落地，卡住"公开头不泄漏内部类型"。

## 验收标准（"算完成"的硬指标）

- **纯 C 可消费**：一个用 **C 编译器**（非 C++）编译的 TU，`#include` C shim 头并调用 `make/调用/destroy` 全程通过——证明无 mangling、无模板、无 STL 泄漏。
- **跨编译器**：MSVC 编的 HYLE `.dll` 被 GCC/Clang 编的消费者链接调用通过（反之亦然）——这是 C++ 头做不到、C ABI 才有的能力，必须实测。
- **FFI 冒烟**：TS（N-API/WASM）+ Python（ctypes）各调一个 HYLE 函数跑通（roadmap P6 门槛）。
- **异常隔离**：故障注入测试确认无 C++ 异常逃逸到 C 边界。
- **依赖**：各对外模块 C++ 实现已稳定（C shim 是薄转发层，底层没定型就封等于返工——故排在 P6）。

## 参考

- charter §6（对外沟通/FFI 决策）、§5（模块公开/内部二分、ABI 承诺只在 C 接缝）。
- conventions §4（C-ABI 边界范式）、§2（decls.h include 纪律）。
- usage-guide §1.5（当前诚实标注：纯 C-ABI shim 是 P6 未做，现为 C++ ABI 跨 DLL）。
