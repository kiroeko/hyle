# error-01：Status + std::expected + ASSIGN_OR_RAISE 宏

| 字段 | 值 |
|---|---|
| **阶段** | P0 |
| **模块** | error/ |
| **Tier** | 2 |
| **LOC 估** | 1-2k |
| **骨架?** | ✅ |
| **状态** | ✅ 完成（commit 8f3fb40，MSVC 19.50 Debug+Release 双配置，22 case 全过）|

> **cloc 留档（`cloc include/hyle/error src/error tests/error`）**：**456 行** = 163 头（status.h+result.h）+ 292 cc（status.cc 生产 + status_test.cc 测试）+ 1 CMake。低于估 1-2k 属设计使然（靠 std::expected，密度在 X-macro 与 23 测试里），**非 stub**。

> **实现注记**：Status 用 **clay 式指针标签**——公开头只有一个 `uintptr_t rep_`（8 字节，`static_assert` 钉死）：`0`=OK、`(code<<2)|1`=内联码无消息（零分配）、低 2 位清零的指针=堆 `StatusRep`（仅有消息时 malloc 一次，header-before-data 内联消息字节 + 手写原子 refcount）。**公开表面零 std 拥有型**（满足 coding-conventions 行 22 + clay 自有字符串先例；起因：用户指出高性能库不该泄 std::string）。
> `ToString` 改 **buffer-fill**（`size_t ToString(char*,size_t)`，snprintf 式返回需长），不返回 `std::string`——彻底移除公开头的 std 拥有型；要碎片就用 `status_code_name()`/`message()`（都是 `string_view`）自组。
> 传播宏用 `detail::FailReturn`（双向隐式转换 `Status`/`Result<T>`）+ `__COUNTER__` 唯一临时变量，规避 MSVC 无 GNU 语句表达式。`HYLE_FATAL_ASSERT` 早在 macros.h（00 步）；软断言 `HYLE_ASSERT` **不在本步**——它要走可安装诊断 handler 通道，故落在 error-02（`error_handler.h`），result.h 这层不定义（否则与 error_handler.h 双定义、按 include 顺序分叉行为）。

## 前置依赖

- 硬前置：`00-scaffolding`（CMake/测试/宏骨架）。
- 决策依据：`engineering-charter.md` §2 错误模型（零异常 + std::expected + 保留 ASSIGN_OR_RAISE）；`coding-conventions.md` §5 错误处理（X-macro 状态码、零开销 OK 路径）。

## 目的

万物用错误处理。零异常 infra 必须先有统一的 `Status`/`expected` 与传播宏，否则每个模块各造一套。这步定下 HYLE 的错误"词汇"。

## 在架构中的位置

依赖图 `error/` 节点（零上游、被所有模块依赖）。是 Tier 2 基质，但必须最先有。

## 具体事务

- [ ] `include/hyle/error/status.h`：`StatusCode`（X-macro `HYLE_FOREACH_STATUS_CODE` 生成 enum + 工厂 + `IsXxx()`）；`Status` 紧凑表示（OK 态零分配，参考 clay 指针标签内联但简化）。
- [ ] `include/hyle/error/result.h`：`Result<T>` = `std::expected<T, Status>` 的 HYLE 别名 + 约定。
- [ ] 传播宏 `HYLE_RETURN_NOT_OK` / `HYLE_ASSIGN_OR_RAISE`（Rust `?` 式早返回，标准无等价）。
- [ ] `HYLE_FATAL_ASSERT`（硬断言，已在 macros.h/00 步）。
- [ ] 不做：`HYLE_ASSERT`（软断言，求值为 bool、release 不消失）——它要经诊断 handler 通道，随 error_handler vtable 一起落 error-02，不在这层。
- [ ] 不做：error_handler vtable（跨 DLL 错误上报）留后续；只做进程内错误模型。

## 交付物

- status.h + result.h + 传播宏 + 测试。

## 验收标准

- **编译**：三编译器（需 C++23 `std::expected`）；低工具链有 polyfill 或明确报错。
- **测试**：覆盖接近 100%（Tier 2 偏高，错误模型是地基）——每个状态码、OK/Err 路径、宏的早返回、嵌套传播。
- **零开销验证**：OK 路径基准——`Result<int>` 返回 OK 不应有堆分配（对标裸返回）。
- **硬指标**：一个示范函数链用 `HYLE_ASSIGN_OR_RAISE` 串 3 层、错误正确冒泡。

## 参考

- clay `utils/status.h`（X-macro 码表、指针标签内联）、`utils/result.h`（ASSIGN_OR_RAISE）——继承思路，核心存储换 std::expected。
- `engineering-charter.md` §2、`coding-conventions.md` §5。
