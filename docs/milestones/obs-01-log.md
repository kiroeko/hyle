# obs-01：结构化日志产出（log sink registry）

| 字段 | 值 |
|---|---|
| **阶段** | P0（obs 模块首步） |
| **模块** | obs/（**本里程碑新建模块**） |
| **状态** | ✅ 完成（Win x64 MSVC 19.50 + Linux x64 GCC14.2 双平台 11/11）|

> **范围**：吸纳 clay `sys/log.h`+`.cc`——level 门控 + sink 注册表 + 默认 stderr 彩色 sink。这是 obs/ 的"**日志产出**"半边(hyle-scope：obs 产结构化日志,聚合/采集是运维栈,不在此)。instrumentation hook(计数器/trace point)是 obs 后续步。
> **cloc**：**生产 132 行**(log.h 47 + log.cc 85) + **测试 103 行**。

## 模块定位（obs-01 = log，不是 instrumentation hook）

scaffolding 占位把 obs-01 写成"instrumentation hook(编译期 no-op)"，coverage-map 把 `sys/log` 映射到 "obs-01/platform"。二者是 obs/ 的两件事：**hook(计数器/trace)** 与 **log 产出**。clay 只有 log（无 instrumentation 框架），所以 **obs-01 吸纳 = log 产出**;hook 是 obs 后续净新增步（hyle-scope：T2 第一天嵌热路径）。本里程碑**新建 `src/obs` 模块**（P0 层，error 之后），落 log。

## clay `logger_api_t` → HYLE diff

| clay | HYLE | 备注 |
|---|---|---|
| `log_level_e`(TRACE/DEBUG/INFO/WARN/ERROR) | ✅ `enum class LogLevel`(+`Off` 静默) | enum class 强类型 |
| `logger_t{inst, log(inst,level,msg)}` | ✅ `LogSink{state, write(state,level,msg)}` | vtable sink,**非变参** |
| `add_logger`/`remove_logger` | ✅ `add_sink`/`remove_sink` | 身份去重 + swap-remove |
| `print(level,msg)` | ✅ `log_message(level,msg)` | |
| `printf(level,fmt,...)` **变参 fn-ptr** | ✅ `log_fmt<>()` 模板(头内,`std::format`) | **见下:去变参** |
| `set_level` | ✅ `set_level` + `level()`/`should_log()` | atomic 无锁门 |
| `default_logger`(printf ANSI) | ✅ `default_sink()`(fprintf(stderr) ANSI + Win OutputDebugString) | |
| `CLAY_LOG/INFO/WARN/...` 宏 | ✅ `HYLE_LOG_TRACE/DEBUG/INFO/WARN/ERROR` | should_log 短路 |

**clay log.cc 全 177 行已读 → 真实现**（64-slot sink 数组、va_list+SYS_vsnprintf、ANSI 彩色 default）。无美化空函数。

## 三处设计决策（对抗 clay 的 ABI/规范坑）

1. **sink 写函数非变参**（error-02 同款教训）：clay vtable 是 `printf(level, fmt, ...)`——**变参函数指针跨 C 接缝是 ABI 隐患**（调用约定/栈布局不跨编译器稳定）。HYLE sink 是 `write(state, level, const char* msg)` 收**已格式化的串**。
2. **格式化 caller-side `std::format`**（conventions §8）：头内 `log_fmt<>` 模板用 `std::format_to_n` 在调用方栈缓冲(1024B,无分配)格式化,**只 `const char*` 过 seam**。编译期 `{}` 查错替 clay 运行期 va_list。`<format>` 是值语义非流,合规。
3. **诊断走 fprintf(stderr) 非 std 流**（§1 铁律）：default sink 用 `std::fprintf(stderr,...)`,绝不 `<iostream>`。

## 并发模型（clay 是全局裸数组,无同步）

- **level 门 lock-free**：`std::atomic<LogLevel>` relaxed load,`should_log` 热路径零锁。
- **sink 注册表**：`std::mutex` 仅护 add/remove 与 log_message 里的**快照**;sink 的实际 `write()` I/O 在**锁外**跑（绝不持锁做可能阻塞的 I/O,也防 sink 重入死锁）。clay 全局裸数组多线程 add/log 是 data race;HYLE 修。
- default sink 首次触碰注册表时装一次（`g_initialized` 标记）;`remove_sink(default_sink())` 后不再自动重装（clay "auto-added,可显式移除" 契约）。

## 对抗式自审（六类）

- **① 正确性**:11 测——到达 sink/level 门丢弃/should_log/Off 全静默/log_fmt caller-side 格式化/门控时跳过格式化/宏路由 level/多 sink 广播/add 幂等/remove 停投递/默认 level=Info。双平台 11/11。
- **② 性能**:热路径 should_log 是 relaxed atomic 单 load(无锁);门控时 log_fmt 直接 return 不格式化;快照后锁外 I/O。
- **③ 规范**:非变参 vtable(§4 ABI)、std::format(§8)、fprintf 非流(§1)、HYLE_OBS_API 三态;头只 include format/macros/visibility。
- **④ clay 覆盖**:logger_api_t 逐项 diff(上表);变参 printf 显式降级为非变参+caller format(登记原因);instrumentation hook 标 obs 后续。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:64-slot 定长数组(同 clay,无动态分配);swap-remove;无投机的异步/ring buffer(hyle-scope:异步刷盘"借设计不借库",留后续按需)。
- **跨平台**:fprintf+ANSI 双平台;OutputDebugStringA 仅 Win 分支;atomic/mutex 是 std。

## 交付物

- `include/hyle/obs/log.h`(LogLevel + LogSink + add/remove_sink/set_level/level/should_log/log_message/default_sink + log_fmt 模板 + HYLE_LOG_* 宏)
- `src/obs/log.cc`(atomic level + mutex-snapshot 注册表 + default stderr ANSI sink)
- `src/obs/CMakeLists.txt`(新模块) + 根 CMakeLists 接线(P0,error 后)
- `tests/obs/log_test.cc`(11)

## 参考

- clay `sys/log.h`+`log.cc`(vtable logger registry)。conventions §1(禁 std 流/用 fprintf)、§4(非变参 vtable ABI)、§8(std::format)。error-02(同款去变参 fn-ptr 教训)。hyle-scope(obs 产日志,聚合是运维)。
