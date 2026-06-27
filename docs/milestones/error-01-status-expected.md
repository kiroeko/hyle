# error-01：Status + std::expected + ASSIGN_OR_RAISE 宏

| 字段 | 值 |
|---|---|
| **阶段** | P0 |
| **模块** | error/ |
| **Tier** | 2 |
| **LOC 估** | 1-2k |
| **骨架?** | ✅ |
| **状态** | 未开始 |

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
- [ ] `HYLE_ASSERT`/`HYLE_FATAL_ASSERT`（求值为 bool、release 不消失，参考 clay）。
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
