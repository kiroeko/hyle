# obs 模块收口（2026-07-19）

obs 的功能、红队正确性、下游 trace 接入、测试/性能、纯 C ABI 和用户文档已完成
一次全面收口。设计信源见
[`../architecture/obs-design.md`](../architecture/obs-design.md)，使用指南见
[`../guide/c-obs.md`](../guide/c-obs.md)。

## 1. 功能与边界

| 面 | 收口结果 |
|---|---|
| 日志 record | level/message + key/value fields + source file/function/line + 可选 Unix ns |
| 格式化 | C++ `std::format_to_n` 1 KiB 栈缓冲；宏在格式化前门控并捕获调用点 |
| sink | 64 槽同步 fan-out、默认 TTY 彩色 stderr、自定义 C++/C sink |
| 生命周期 | quiescent unregister；跨线程旧 callback 排空；callback 自摘安全；跨摘拒绝 |
| trace | zone/plot/counter/fiber/frame/message/alloc/free/thread name |
| 下游 | mem/sched/io/compute 均有真实 hook；OFF 构建全部消失 |
| 明确不做 | 日志队列/滚动/上传/聚合/告警、trace viewer；属于 sink/运维层 |

log 没有被 HYLE 各原语主动调用是有意边界，不是“下游未利用”：基础库不应擅自向
宿主输出业务日志。机器可读失败继续走 error/status；需要记录时由调用方主动发
record。trace 才是应由基础库内部布置的性能观测面，因此已接入四个热路径模块。

## 2. 红队发现与修复

| 问题 | 修复 |
|---|---|
| 旧 mutex-snapshot 允许 remove 返回后旧 callback 继续使用 state，存在 UAF 窗口 | slot active reference + retiring 状态 + atomic wait，remove 成功后保证其他 callback 排空 |
| callback 跨摘同一快照中待调用 sink 会等待当前线程自己 | callback 内只允许摘当前 sink；跨摘在改变 registry 前拒绝 |
| C opaque handle 自摘后 trampoline 可能继续访问已 delete wrapper | 进外部 callback 前复制 callback/state，返回后不再碰 wrapper |
| 静态 TracyClient 分别链入多 DLL，产生多个 profiler singleton | Tracy target 跟随 `BUILD_SHARED_LIBS`；共享 HYLE 只加载一个 shared client |
| 默认 stderr 多次 `fprintf` 会在线程间穿插，且重定向仍输出 ANSI | 默认 sink 独立锁；`isatty`/`_isatty` 后才着色 |
| trace message 误用 literal-only 入口接动态串 | 动态 message 复制长度；另留 literal 快路径 |
| 所有匿名 Fiber 共用一个 Tracy 身份 | 每个 Fiber 保存稳定唯一 trace name |
| C query 可能留下旧 last-error，invalid should-log 无诊断 | 所有成功 query 清 stale；非法 record level 设置线程局部错误 |

## 3. C ABI

`include/hyle/c/obs.h` 是纯 C 发布面：

- 12 个日志/registry 动态入口；
- 11 个 runtime trace 动态入口；
- 5 个 header-inline level 便利函数；
- 2 个 POD（field/record）、1 个 opaque sink、1 个 opaque-sized zone token。

没有模块 api 表、版本协商、STL、C++ 异常、C++ reference 或 Tracy 类型。输入错误
通过线程局部 `hyle_obs_last_error` 报告；正常 gate/no-sink 返回 0 但不制造错误。
纯 C 测试覆盖全部便利函数、record 扩展大小、错误、sink 生命周期和 runtime trace。

有意仅 C++：`std::format` 模板、编译期零成本 trace 宏、native `LogSink` 值类型。
这些能力在 C 侧分别由“语言侧格式化 + POD emit”、runtime trace、opaque sink 覆盖。

## 4. 验收证据

| 环境 | 结果 |
|---|---|
| Windows x64 MSVC Release | obs C++/pure-C 通过；下游 mem/sched/io/compute 回归通过 |
| WSL2 GCC 14 Release | obs C++/pure-C 通过 |
| WSL2 GCC 14 ASan/UBSan/TSan | obs C++/pure-C 全通过 |
| Apple M4 AppleClang Release | obs C++/pure-C 通过 |
| Tracy ON（Windows/WSL/M4） | obs + mem/sched/io/compute 通过 |
| Tracy ON + static HYLE（WSL GCC14） | obs C++/pure-C 通过；单 archive 在最终程序解析 |
| shared Tracy 依赖 | PE/ELF/Mach-O 均确认各模块只依赖一个 `hyle_tracy_client` |

Release 三平台代表值（nanobench，无 I/O）：

| 场景 | Windows | WSL | Apple M4 |
|---|---:|---:|---:|
| gated `HYLE_LOG_INFO` | 2.03–2.11 ns | 1.05–1.09 ns | 0.91–0.92 ns |
| C++ preformatted + 1 sink | 22.48 ns | 13.07 ns | 8.85 ns |
| C ABI preformatted + 1 C sink | 23.93 ns | 16.03 ns | 10.68 ns |
| `std::format` + safe fan-out | 81.9–84.6 ns | 69.5–71.4 ns | 50.9–53.5 ns |

C ABI 对 native preformatted seam 的增量约 1.5/3.0/1.8 ns。emit 相比旧版多出的
active-reference 成本换来了可证明的 quiescent unregister；真实 sink I/O 通常远大于
这十余纳秒。gated 热路径曾因 helper 形状回退到 Windows 约 3.5 ns，最终通过宏前置
gate 恢复到约 2.1 ns。

## 5. 收口结论

obs 现在具备基础库所需的完整“结构化日志产出 + profiling trace 产出”能力，
生命周期、C ABI 和三平台行为均已闭环。后续只有两类合理扩展：

- 新模块接入已有 trace 原语；
- 经真实消费者/profile 证明需要后，再新增 record 字段或 sink 类型。

异步日志系统、运维聚合和自造 profiler 不应重新塞回 obs。
