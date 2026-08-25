# obs-01：结构化日志产出

> **历史任务提示（2026-08-25）：**本页保留 2026-07 的施工形状；当前 obs 已改用 owning
> `hyle_status`、caller-storage sink 并扩为 29 个动态入口。现行设计/用法与认证分别只看
> [`obs-design`](../../../architecture/obs-design.md)、[`c-obs`](../../../architecture/native-ffi-policy.md) 和
> [`pre-release-readiness`](../../status.md)。

| 字段 | 值 |
|---|---|
| **阶段** | P0 |
| **模块** | obs |
| **状态** | 🗃️ 2026-07-19 历史复审；当时三平台 + ASan/UBSan/TSan + C ABI |

obs-01 从 clay `sys/log` 起步；该轮实现已经不是“level + 字符串 callback”的薄
移植。当前完整形状与维护规则见
[`../architecture/obs-design.md`](../../../architecture/obs-design.md)，本轮验收见
[`obs-module-closure.md`](obs-module-closure.md)。

## 该轮交付能力

- `LogRecord`：level、message、key/value fields、source file/function/line、可选
  Unix timestamp；
- `HYLE_LOG_*`：调用方 `std::format_to_n` 固定栈缓冲，gate 失败不求值参数并捕获
  caller source location；
- 64 槽固定 registry；callback 在 logger 线程同步执行，registry 锁外 fan-out；
- quiescent remove：成功返回后其他线程无旧 callback，state 可释放；
- callback 自摘安全；跨摘其他 sink 明确拒绝，避免同一 snapshot 自等待死锁；
- 默认 stderr sink：线程内 record 不穿插，只在 TTY 着色，Windows 同步 debugger
  output；
- 当时的纯 C ABI：POD record + opaque sink + last-error + 五个 level 便利函数；last-error
  后来已由共享 owning `hyle_status` 取代，不能按本行实现新绑定。

## 相对 clay 的关键修正

| clay | HYLE |
|---|---|
| variadic logger function pointer | 完成字符串/record 的非变参 callback |
| 全局裸数组，多线程 add/log data race | mutex snapshot + per-slot active reference |
| remove 后旧 snapshot 仍可能碰已释放 state | quiescent unregister |
| printf ANSI 到所有 stderr | 仅 TTY 着色 |
| 字符串消息 | 可扩展结构化 POD record |
| 无稳定 C 发布面 | `include/hyle/c/obs.h` + 纯 C consumer |

## 边界

obs 是“产出”原语，不是日志服务。异步缓冲、文件轮转、网络传输、聚合、检索和告警
由 sink/运维层实现。HYLE 内部原语不主动输出日常日志；error/status 是机器可读失败
通道，应用决定何时把诊断变成 record。

trace/counter 属于 obs-02；该轮记录中它也已接入 mem/sched/io/compute。两条通路的当前
统一架构见 obs-design，认证状态不由本页决定。

## 验收

- C++ 测试覆盖 level、格式化、结构化字段、容量、并发 churn、remove 等待、自摘、
  跨摘拒绝、默认 sink 状态；
- 纯 C 测试覆盖完整日志/C ABI 生命周期和错误面；
- Windows MSVC、WSL GCC14、Apple M4 AppleClang 通过；
- WSL ASan、UBSan、TSan 通过；
- gated/preformatted/formatted 三层基准进入 `hyle_misc_bench`，历史结果见
  [`../performance/README.md`](../../evidence/performance/README.md) 的 obs 行及其 freshness gate。
