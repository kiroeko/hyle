# error-02：error_handler vtable 跨 DLL 错误上报

| 字段 | 值 |
|---|---|
| **阶段** | P0 |
| **模块** | error/ |
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；error 23→29 test）|

> **范围**：吸纳 clay `sys/error.h`+`.cc`（222+143）。`error_handler_t` 可插拔诊断 sink（C-ABI vtable）——人类可读的**诊断通道**（"把这条消息送到开发者能看到的地方"），区别于 Status/expected 的机器可读**返回值通道**。补齐 charter §5 缺的 `HYLE_ASSERT`。**P0 error 层收尾**。
> **cloc**：**生产 127 行**(error_handler.h 59 + .cc 68) + **测试 74 行**。

## clay API diff + 改进

- **覆盖**：clay `error_handler_t`(inst+errorf+fatal) + `error_api_t`(log/def/create_record_handler) + `CLAY_ERROR/ASSERT/FATAL_ASSERT` 宏簇 + record handler。
- **改进①(vtable 非变参,ABI 安全)**：clay `errorf/fatal` 是**变参函数指针**——varargs 过跨 DLL 函数指针是 ABI 隐患(调用约定/寄存器-vs-栈规则跨编译器不一)。HYLE 调用侧 snprintf 成型,vtable 收 `const char*`;两个近重复函数指针收成 `report()+Severity`。
- **改进②(干净 fatal)**：clay 默认 fatal 干 `rand()%1 + *(int*)0=10 + exit(-1)`(故意空写触发调试器)——脏且 UB。HYLE 派发器在 handler 返回后 `HYLE_TRAP()`:handler 仍可先记录/弹框,但崩溃由派发保证(任何 handler 都崩),fail-fast 干净。
- **改进③(补 HYLE_ASSERT)**：charter §5 要求的"求值 bool + release 不消失 + 失败记录非 UB"此前缺失(只有 FATAL_ASSERT)。现补:`if(!HYLE_ASSERT(p)) return ...;` 可恢复;真测求值一次(无双副作用)、release 语义。
- **改进④(record handler 无分配)**：clay record 用 carray<char*>+Allocator 存每条;HYLE `ErrorRecord` counts[3]+last[256] 内联,零堆分配,可选 backing 转发。
- **gap 登记**：`error_api_t->def/log` 全局默认 → HYLE `error_default_handler()/error_set_default_handler()`(atomic 可运行时换)。**便利宏 ✅(2026-07-17)**：`HYLE_NOT_IMPLEMENTED` / `HYLE_DEPRECATED_WARNING`（**HYLE 命名**，非 clay 前缀）。

## 对抗式自审（六类）

- **① 正确性**：handler swap 用 atomic acq_rel(运行时换 sink 与 reporter 竞争干净);ErrorRecord severity idx<3 边界;last_ 截断 NUL;HYLE_ASSERT 求值一次真测;install/restore 嵌套真测;backing 转发真测;vsnprintf 截断安全。
- **② 性能**：诊断非热路径。OK 路径(HYLE_ASSERT cond 真)零开销(HYLE_LIKELY + 三元短路,不进 error_report);report 一次 atomic acquire load。
- **③ 规范**：vtable+不透明 inst+`_o/_t`(§4);非变参 ABI 安全;导出宏在方法/ErrorRecord(有非 POD 成员→放类含 C4251 考量但 ErrorRecord 跨 DLL 整体用,标类级);无 std 拥有型出头;无 iostream(fprintf stderr)。`HYLE_ERROR_API` 用 `#ifndef` 守护(与 status.h 同模块多头)。
- **④ clay 覆盖**：handler/record/宏簇/默认 sink 全覆盖 + 四改进;def/log→atomic 默认;NOT_IMPLEMENTED 等便利宏 gap 登记。
- **⑤ 残留脚手架**：无新增。
- **⑥ 过度复杂**：vtable 单 report 比 clay 双函数指针简洁;ErrorRecord 定长比 clay carray 简单。
- **"比 std 快"**：N/A——std 无可插拔诊断 sink(`<cassert>` assert 在 NDEBUG 直接消失、不可路由、不可恢复)。这是 std 没有的**能力**(C-ABI 跨 DLL 错误路由 + release-surviving 可恢复断言)。

## 交付物

- `include/hyle/error/error_handler.h`（Severity + error_handler_t vtable + error_report + ErrorRecord + HYLE_WARN/ERROR/FATAL/ASSERT 宏）
- `src/error/error_handler.cc`（stderr 默认 sink + atomic 默认 + ErrorRecord）
- `tests/error/error_handler_test.cc`（6）
- CMake：error 模块加 error_handler.cc。

## 参考

- clay `sys/error.h`+`src/clay/sys/error.cc`。conventions §4(C-ABI vtable)、§5(HYLE_ASSERT)。charter §5(fail-fast)。
