# obs 选型与 C++ 使用指南

> **`provisional / NOT_REVIEWED_V2`。** obs 当前只提供 C++ 产品面，可通过完整源码、headers + static libraries 或
> headers + shared libraries 使用。本页不描述 C API 或跨语言调用层。

obs 有两条彼此独立的通路：**log** 是用户主动产出的同步诊断 record，**trace** 是
profiling build 才存在的纳秒级时间线。它们都不是 error 的替代品，也不是一套日志
运维系统。

## 1. 先决定该用什么

| 需求 | 选择 | 原因 |
|---|---|---|
| 函数失败，要让调用方分支/重试/传播 | `hyle::Status` / `hyle::Result<T>` | 这是机器可读控制流；不要靠解析日志 |
| 调用方主动记录一条普通诊断 | `HYLE_LOG_*` | level 先门控，未通过时参数不求值；通过后捕获 source location |
| 已有字符串或 byte slice，不想再格式化 | `log_message(std::string_view)` | 同步借用 exact bytes，不扫描、不复制，可保留内嵌 NUL |
| 有字段、外部时间戳或自定义 source | `log_record` | 明确的 borrowed structured record |
| 作用域内接进现有 logger/队列/文件/遥测 | `SinkRegistration` + `LogSink` | RAII 注销；复制、排队、轮转、上传归 sink |
| 进程期或 callback 内特殊控制 registry | `add_sink/remove_sink` | 高级显式路径；调用方自行证明配对和 state 生命周期 |
| 看热路径 zone、fiber、内存和 counter 时间线 | `HYLE_TRACE_*` | 默认编译期消失；shared 或单最终 image 的 static 构建汇入一个 Tracy client |

如果应用已经使用 Abseil/Folly/spdlog 等日志系统，通常不要再建第二套日志配置树：注册
一个 HYLE sink，把 record 适配进现有系统即可。分层 category、异步 writer、文件滚动、
上传、查询和告警留给那个上层系统。obs 的目标是提供轻、可安全注销的底层 producer seam。

## 2. 普通日志

```cpp
#include <utility>

#include "hyle/obs/log.h"

using hyle::obs::LogLevel;

hyle::obs::set_level(LogLevel::Info);
HYLE_LOG_INFO("loaded {} kernels", kernel_count);
HYLE_LOG_WARN("queue {} is {}% full", queue_name, occupancy);
```

`HYLE_LOG_*` 是常用入口。宏先做一次 relaxed level check，所以被 gate 的参数不会求值，
也不会进入 `std::format`。通过时使用 1 KiB 栈缓冲；过长消息以
`...[truncated]` 结尾。大 payload 不应塞进普通日志：需要完整保留时直接构造 record，或交给
应用拥有型日志系统。

发布构建若要把低等级调用连参数求值和 obs 符号引用一起删除，在 include 之前设置每个
translation unit 的编译期 floor：

```cpp
#define HYLE_OBS_COMPILED_MIN_LEVEL 2  // 0 Trace ... 4 Error, 5 Off
#include "hyle/obs/log.h"
```

这不是库 target 的 ABI 选项，也不改变其他 translation unit；它只裁剪
`HYLE_LOG_*` 宏。运行期仍用 `set_level` 动态调节剩余等级。并发收紧 level 时，已经通过
唯一一次 relaxed gate 的 record 可以完成本次同步投递，这是配置的 eventual 语义。

已有文本或结构化字段时不必绕经 format：

```cpp
hyle::obs::LogField fields[] = {
    {"request_id", "a7f2"},
    {"device", "gpu0"},
};
hyle::obs::LogRecord record{
    .level = LogLevel::Warn,
    .message = "submission delayed",
    .fields = fields,
    .field_count = std::size(fields),
    .timestamp_ns = unix_time_ns,  // 0 表示调用方未提供
};
bool delivered = hyle::obs::log_record(record);
```

message 是 exact byte view；例如 `log_message(level, std::string_view(data, size))` 不需要
NUL 结尾并保留内嵌 NUL。`const char*` 便利重载会寻找结尾 NUL。record、字段和所有字符串
都只在同步调用期间借用。`false` 表示被 gate 或没有 sink，不是错误。悬空 view、非空
`field_count` 配空 fields 或空 field 字符串指针等是 C++ 合同违反，会
fail-fast；从不可信来源构造 record 的上层代码必须先验证输入，再进入 obs 核心。

默认 sink 写 stderr。只有终端才加 HYLE 自己的 level 色码；source/message/field 中的换行、
TAB、ESC 等控制字符会转义，所以一条 record 不会伪造额外物理行或注入 ANSI 序列。

## 3. 接入自己的 sink

```cpp
struct Backend {
  static void write(void* state,
                    const hyle::obs::LogRecord& record) noexcept {
    auto& self = *static_cast<Backend*>(state);
    self.enqueue_copy(record);  // 异步使用必须在 callback 内复制 borrowed 数据
  }

  static void flush(void* state) noexcept {
    static_cast<Backend*>(state)->drain();
  }
};

Backend backend;  // 必须比 registration 先构造、后析构
auto registration = hyle::obs::SinkRegistration::create(
    hyle::obs::LogSink{&backend, &Backend::write, &Backend::flush});
if (!registration) return hyle::make_unexpected(std::move(registration.error()));

// ... 多线程产生日志；离开作用域时 quiescent unregister ...
hyle::obs::flush_sinks();
```

`Result` 内的 registration 是 move-only，析构自动注销；`reset()` 幂等，失败时仍保留
ownership，便于离开 callback
后重试。不要把 state 声明在 registration 之后，不要对同一 membership 再手工
`remove_sink`。需要进程期注册或特殊 callback 控制时才用显式 add/remove，并保证每条退出
路径都配对。

callback 由 producer 线程同步调用，可能在多个线程并发，必须 thread-safe、`noexcept`，
也不能长期阻塞 producer。`flush` 可与 `write` 并发。需要异步行为时，sink 在 callback
里复制 record 后放进自己的队列；obs 不暗中启动线程。

`remove_sink` 是 quiescent unregister。callback 可以摘除自己，但 state 仍要活到本次
callback 返回；callback 内摘除另一个 sink 会返回 `FailedPrecondition`，因为目标可能已在
同一 dispatch snapshot 中等待当前线程调用。sink 内再次产生日志不会重入注册 sink 图：
默认 stderr 仍启用时 nested record 直接落 stderr，否则报告未投递。这个规则避免诊断
backend 递归到栈溢出。

不要在一个 sink callback 内析构另一个仍注册的 `SinkRegistration`：它的 quiescent remove
会被拒绝，而析构无法返回错误，因此按生命周期合同 fail-fast。应调用 `reset()` 获取 Status，
并把跨 sink 注销调度到 callback 外。

add/remove 是冷路径并返回共享 `Status`：空 callback、重复、缺失、容量耗尽和 callback
跨摘可以区分。registry 固定最多 64 个 sink；emit 不分配，也不拿 mutation mutex。

## 4. Trace：只在该看的 build 中付费

```cpp
#include "hyle/obs/trace.h"

void reap() {
  HYLE_TRACE_ZONE_NAMED("app.reap");
  HYLE_TRACE_COUNTER("app.completions", completed_total);
}
```

默认 `HYLE_TRACE=OFF` 时宏参数不求值、没有函数调用、没有后端依赖。开启 Tracy：

```sh
cmake -S . -B build-trace \
  -DHYLE_TRACE=ON \
  -DHYLE_TRACY_DIR=/path/to/tracy
```

zone 用作用域配对；fiber enter/leave 必须跟逻辑执行上下文配对。plot/frame/pool/fiber 的
身份名要用字符串字面量或其他长期稳定存储，不能传临时 `std::string::c_str()`。动态
message 会复制文本。共享 HYLE 的所有模块连接同一个 `hyle_tracy_client`，不能在各 DLL
再各编一份 TracyClient。trace-on static SDK 只支持所有 HYLE archive 最终进入一个 link image；
多个 DLL/plugin 各自静态链接会得到多个互不相认的 Tracy singleton，不属于受支持形状。此类 host
必须使用 shared `hyle_obs`，不能把“每个 image 一个”描述成 process-unique。

极热的 C++ 源码路径应使用上述编译期宏。HYLE 内部的 mem/sched/io/compute 也走同一条
native trace seam。

## 5. 常见误用

- 不要在库失败时“打日志然后返回成功”；失败走 error，是否记录由调用方决定。
- 不要假设 sink callback 串行；需要串行化就在 sink 内做。
- 不要在 callback 返回后保存 borrowed record 指针；异步 sink 必须复制。
- 不要把 `record.message.data()` 当 C string；它按 `size()` 定界并可能包含 NUL。
- 不要让 `SinkRegistration` 比其 callback state 活得更久，或同时手工 remove 同一 sink。
- 不要把 obs 当信号处理器日志：mutex、stdio、format 和任意 callback 都不是 async-signal-safe。
- 不要为每个模块创建一套 logger/category 树；已有应用 logger 时用一个 adapter sink。
- 不要把 runtime trace no-op 宣传成关闭态零指令；只有 `HYLE_TRACE_*` 宏具有这个合同。

设计和并发细节见 [`../architecture/obs-design.md`](../architecture/obs-design.md)，点值与
方法见 [`../performance/benchmarks-obs.md`](../implementation/evidence/performance/benchmarks-obs.md)。
