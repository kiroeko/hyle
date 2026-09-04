# obs 选型与 C++ 使用指南

obs 当前只提供 C++ 产品面。本页适用于 C++23 source/static/shared 三种交付，不描述 C API 或
跨语言调用层；Native FFI 当前为 `N/A`。三种集成方式、组件选择与工具链合同见
[`distribution`](distribution.md)。

obs 有两条彼此独立的通路：**log** 是用户主动产出的同步诊断 record，**trace** 是
profiling build 才存在的纳秒级时间线。它们都不是 error 的替代品，也不是一套日志
运维系统。

## 1. 先决定该用什么

| 需求 | 选择 | 原因 |
|---|---|---|
| 函数失败，要让调用方分支/重试/传播 | `hyle::Status` / `hyle::Result<T>` | 这是机器可读控制流；不要靠解析日志 |
| 调用方主动记录一条普通诊断 | `HYLE_LOG_*` | level 先门控，未通过时参数不求值；通过后捕获 source location |
| 重复 warning/progress 需要防刷屏 | `HYLE_LOG_*_EVERY_N/FIRST_N/EVERY` | 每个调用点独立状态；被抑制时不格式化 |
| 多个 request/slot 各自维护进度周期 | `LogIntervalLimiter` | 状态跟随对象，而不是把所有实例混入一个调用点 static |
| 已有字符串或 byte slice，不想再格式化 | `log_message(std::string_view)` | 同步借用 exact bytes，不扫描、不复制，可保留内嵌 NUL |
| 有 category/event、typed 属性、关联 ID、外部时间戳或自定义 source | `log_record` | 明确的 borrowed structured event |
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

已有文本或结构化事件时不必绕经 format。下面是推理 slot progress 的正常形状：

```cpp
hyle::obs::LogAttribute attributes[] = {
    {"model", "qwen"},
    {"backend", "gpu0"},
    {"tokens_per_second", 72.5},
    {"queue_depth", std::uint32_t{3}},
};
hyle::obs::LogAttribute correlation[] = {
    {"request_id", std::uint64_t{0xa7f2}},
    {"slot", std::uint32_t{2}},
};
hyle::obs::LogRecord record{
    .level = LogLevel::Warn,
    .category = "inference.decode",
    .event = "slot.progress",
    .message = "decode progress",
    .attributes = attributes,
    .correlation_attributes = correlation,
    .source = hyle::obs::LogSourceLocation::from(
        std::source_location::current()),
    .timestamp = hyle::time::unix_timestamp(1'700'000'000, 123),
};
bool delivered = hyle::obs::log_record(record);
```

`LogValue` 接受 exact `string_view`/字符串字面量、借用的 `std::basic_string<char>` 左值、≤64-bit
signed/unsigned integer、`float/double` 和 `bool`。临时 `std::string` 被拒绝，避免 record dispatch 前
view 已经悬空；enum 必须由调用方明确决定转成整数还是名字，`long double` 不会被静默降精度；sink 先看
`kind()`，再调用对应的 `as_string/as_signed_integer/as_unsigned_integer/as_floating_point/as_boolean`，
错 kind 是 programmer contract violation。attributes 与 correlation attributes 都保序，允许重复或空
key；前者是事件事实，后者是 request/slot/trace 等跨记录 identity，Obs 不复制、索引或解释 key。

category、event、message、key 和 string value 都是 exact byte view；例如
`log_message(level, std::string_view(data, size))` 不需要 NUL 结尾并保留内嵌 NUL。
`log_message(const char*)` 便利重载会寻找结尾 NUL；`LogValue` 有意不接收任意 raw pointer，已知长度的
字符指针应显式传 `string_view`。`std::string` 等拥有者必须以活过本次调用的左值传入。record、span 和
所有 view 都只在同步调用期间借用；悬空或
无效 view 是 C++ 前置条件违反。timestamp 未设置才表示 absent，Unix epoch `{0,0}` 和 pre-epoch 都是
有效值；`nanoseconds >= 1'000'000'000` 会 fail-fast。Obs 不自动读钟，只有调用方选择 timestamp 才付费。
`false` 表示被 gate 或没有 sink，不是错误。从不可信来源构造
record 的上层代码必须先验证 pointer/length，再进入 obs 核心。`true` 只表示至少一个同步 sink
callback 被调用，不是文件落盘、网络送达或 `stderr` I/O 成功的 durability receipt；需要这类保证时
由上层 sink 自己的 flush/ack 协议负责。

用户自定义 `std::formatter` 可能抛异常；obs 会在这个外来 adapter 边界把异常变成静态
`[log formatting failed]` record，诊断失败不会改变业务控制流。HYLE invariant 仍然 fail-fast，
不会被这条边界吞掉。

默认 sink 写 stderr。只有终端才加 HYLE 自己的 level 色码；source/message/field 中的换行、
TAB、ESC 等控制字符会转义，所以一条 record 不会伪造额外物理行或注入 ANSI 序列。
typed 数值使用固定栈缓冲渲染；correlation key 带 `correlation.` 前缀，pre-epoch timestamp 分成
`timestamp.seconds` 与 `timestamp.nanoseconds`，避免错误的小数解释。

`Status` 不常驻每条 record。只有错误事件需要时，按宿主 schema 组合普通 typed attributes：
`status.code` 放稳定 code name，`status.message` 放确认可外发的消息，原生错误再加
`status.native_domain` 和 signed `status.native_code`。业务控制流仍传播 `Status/Result`；日志只是边界
诊断，不能通过解析 message 恢复错误类型。

```cpp
const hyle::Status status = get_model();
if (!status.ok() && status.has_native_error()) {
  const hyle::NativeError native = status.native_error();
  hyle::obs::LogAttribute attributes[] = {
      {"status.code", hyle::status_code_name(status.code())},
      {"status.message", status.message()}, // 只在确认可外发时放入
      {"status.native_domain", hyle::native_error_domain_name(native.domain)},
      {"status.native_code", native.code},
  };
  hyle::obs::log_record({
      .level = hyle::obs::LogLevel::Error,
      .category = "model.loader",
      .event = "model.load_failed",
      .attributes = attributes,
  });
}
```

没有 native error 时不要伪造 `None/0`；只提供 `status.code`，有实际 message 且符合脱敏
策略时再加 `status.message`。

## 3. 条件、采样和推理进度

常用调用点控制保持与基础宏相同的 level 和 source-location 语义：

```cpp
#include <chrono>

using namespace std::chrono_literals;

HYLE_LOG_WARN_IF(queue_depth > limit, "queue depth={}", queue_depth);
HYLE_LOG_WARN_EVERY_N(1024, "allocator pressure={}", pressure);
HYLE_LOG_WARN_EVERY_N_EXACT(1024, "exact sample={}", sample_id);
HYLE_LOG_ERROR_FIRST_N(8, "device {} is unavailable", device_id);
HYLE_LOG_DEBUG_EVERY_POW_2("retry count={}", retries);
HYLE_LOG_INFO_EVERY(3s, "model load is {}% complete", percent);
```

五种 level 都提供 `_IF`、`_EVERY_N`、`_EVERY_N_EXACT`、`_FIRST_N`、
`_EVERY_POW_2` 和 `_EVERY`。编译期 floor、运行期 level、条件/状态和 format 参数严格按这个顺序
短路；level 关闭期间不会推进计数或读取时钟。组合条件无需另一套宏矩阵：
`if (large_packet) HYLE_LOG_WARN_EVERY_N(10, ...)` 中条件为 false 的调用自然不计数。

`EVERY_N` 和 `EVERY_POW_2` 是线程安全的低争用**近似**序列：并发线程可能拿到相同计数，从而偶尔
多记或少记，适合诊断采样。必须严格按全局第 N 次时使用 `EVERY_N_EXACT`，它每次支付 relaxed atomic
RMW。`FIRST_N` 使用 relaxed CAS 重试，并发调用合计绝不会超过 N 条；无竞争时是常数成本，竞争下
单次调用的重试次数不承诺有界。实现不引入产品 mutex；底层 64-bit atomic 是否 lock-free 服从目标
标准库。N<=0 不记录也不推进；无符号计数极罕见
wrap 后从新序列开始。

一个宏展开只有一个隐藏 static state。如果一个函数同时服务多个推理 request/slot，把状态放进对象：

```cpp
#include <chrono>
#include <cstdint>

#include "hyle/obs/log.h"

struct DecodeSlot {
  std::uint64_t id;
  double tokens_per_second;
  hyle::obs::LogIntervalLimiter progress;
};

const auto now = std::chrono::steady_clock::now();  // 一轮调度只读一次
for (DecodeSlot& slot : active_slots) {
  if (slot.progress.should_log_at(now, std::chrono::seconds(3))) {
    HYLE_LOG_INFO("slot {}: {:.2f} token/s", slot.id, slot.tokens_per_second);
  }
}
```

需要更低分辨率、更便宜的 consumer，也可把
`hyle::time::coarse_monotonic_ns()` 和同单位 interval 传给整数 `should_log_at`；Obs 本身不因此依赖
Time 的读钟实现；Obs 的 Time 组件依赖来自公开 wall timestamp value。首个 check 通过，之后一个
interval 内最多一个并发调用通过。时间回退会抑制到追上上次值；
只能传 monotonic clock，不能传 Unix/realtime。interval<=0 表示关闭限流、每次通过。`reset()` 用于
请求生命周期重新开始；它本身无数据竞争，但与 check 并发可能额外放行一次，所以应在 slot 停止后调用。

显式状态还包括 `LogEveryNState`、`LogEveryNExactState`、`LogFirstNState` 和
`LogEveryPow2State`；当消息需要带累计次数、状态需要随对象移动，或宏不适合控制流时直接用它们的
`should_log`、`count` 和 `reset`。这些对象的 move/reset 都是生命周期控制，不是运行中迁移协议：必须
先停止或外部同步并发 check，再移动、复用或销毁状态。

## 4. 接入自己的 sink

```cpp
struct Backend {
  void write(const hyle::obs::LogRecord& record) noexcept {
    enqueue_copy(record);  // 异步使用必须在 callback 内复制 borrowed 数据
  }

  void flush() noexcept { drain(); }
};

Backend backend;  // 必须比 registration 先构造、后析构
auto registration = hyle::obs::SinkRegistration::create(backend);
if (!registration) return hyle::make_unexpected(std::move(registration.error()));

// 作为宿主 logger 的唯一 adapter 时，在启动 producer 线程前完成交接；
// 否则同一 record 还会同时写 HYLE 默认 stderr。
auto disabled = hyle::obs::set_default_sink_enabled(false);
if (!disabled.ok()) {
  auto rollback = registration->reset();
  if (!rollback.ok()) return hyle::make_unexpected(std::move(rollback));
  return hyle::make_unexpected(std::move(disabled));
}

// ... 多线程产生日志；离开作用域时 quiescent unregister ...
hyle::obs::flush_sinks();
```

`Result` 内的 registration 是 move-only，析构自动注销；`reset()` 幂等，失败时仍保留
ownership，便于离开 callback
后重试。不要把 state 声明在 registration 之后，不要对同一 membership 再手工
`remove_sink`。需要进程期注册或特殊 callback 控制时才用显式 add/remove，并保证每条退出
路径都配对。普通 C++ adapter 直接提供 `write(record) noexcept`，可选提供 `flush() noexcept`；
`LogSink::from(backend)` 会生成零分配 trampoline。只有高级框架已经拥有 function-table 时才手工
构造 `LogSink{state, write, flush}`。

上例是“宿主 logger 接管”的正常形状。若确实希望 HYLE stderr 与宿主同时收到日志，可以有意保留
default sink。临时接管在停止 producer 后按相反生命周期恢复：先
`set_default_sink_enabled(true)`，再 `registration.reset()`，并处理两个 Status；不要靠析构猜测是否还要
恢复默认输出。注册后关闭 default 可避免无 sink 窗口，前提是启动阶段尚无并发 producer。

callback 由 producer OS thread 同步调用，可能在多个线程并发，必须 thread-safe、`noexcept`，
也不能长期阻塞 producer。callback 必须在同一 OS thread 上执行到返回；不要在其中 yield/suspend
stackful fiber，也不要迁移后再返回。普通 OS 抢占没有问题。`flush` 可与 `write` 并发并遵守相同的
no-suspension 规则。需要异步行为时，sink 在 callback 里复制 record 后放进自己的队列；obs 不暗中
启动线程。

`remove_sink` 是 quiescent unregister。callback 可以摘除自己，但 state 仍要活到本次
callback 返回；callback 内摘除另一个 sink 会返回 `FailedPrecondition`，因为目标可能已在
同一 dispatch snapshot 中等待当前线程调用。sink 内再次产生日志不会重入注册 sink 图：
默认 stderr 仍启用时 nested record 直接落 stderr，否则报告未投递。这个规则避免诊断
backend 递归到栈溢出。

同一个 membership 的控制操作应由一个 owner 串行执行。两个线程同时调用 `remove_sink` 时，只有
返回成功的调用持有“旧 callback 已排空”的 quiescence receipt；另一个线程可能得到 `NotFound`，
不能据此提前销毁仍由成功调用者负责排空的 callback state。普通代码使用 move-only
`SinkRegistration` 即可自然满足这个 owner 规则。

不要在一个 sink callback 内析构另一个仍注册的 `SinkRegistration`：它的 quiescent remove
会被拒绝，而析构无法返回错误，因此按生命周期合同 fail-fast。应调用 `reset()` 获取 Status，
并把跨 sink 注销调度到 callback 外。

add/remove 是冷路径并返回共享 `Status`：空 callback、重复、缺失、容量耗尽和 callback
跨摘可以区分。registry 固定最多 64 个 sink；emit 不分配，也不拿 mutation mutex。

## 5. Trace：只在该看的 build 中付费

```cpp
#include "hyle/obs/trace.h"

void reap() {
  HYLE_TRACE_ZONE_NAMED("app.reap");
  HYLE_TRACE_ZONE_TEXT(request_name);
  HYLE_TRACE_ZONE_VALUE(work_items);
  HYLE_TRACE_COUNTER("app.completions", completed_total);
}
```

默认 `HYLE_TRACE=OFF` 时宏参数不求值、没有函数调用、没有后端依赖。开启 Tracy：

```sh
cmake -S . -B build-trace \
  -DHYLE_TRACE=ON
```

首次在线 configure 会从中央 lock 自动准备 Tracy 到共享源码 cache。只有父工程或逐项离线介质需要
显式注入源码时，才设置 `-DHYLE_TRACY_DIR=/path/to/exact-tracy`；该目录仍必须通过完整源码树与
许可证校验，不能用来绕过 lock。

Tracy 默认监听全部网络接口，适合受信局域网中的 viewer。只允许本机 viewer 时，在启动被观测程序前
设置 `TRACY_ONLY_LOCALHOST=1`；HYLE 的 trace-on CTest 已只对测试子进程设置它，因此不会为了测试申请
Windows 防火墙入站权限。Tracy v0.14.1 的这个 runtime 变量只限制 listener，并不关闭 UDP discovery
broadcast；网络策略禁止任何 discovery traffic 时应保持 `HYLE_TRACE=OFF`。不要在不受信网络上暴露
默认 listener。

zone 用作用域配对；`ZONE_TEXT/NAME/VALUE/COLOR` 只注释当前 lexical zone，动态 text/name 会在
宏返回前复制。fiber enter/leave 必须跟逻辑执行上下文配对。plot/frame/pool/fiber 的
身份名要用字符串字面量或其他长期稳定存储，不能传临时 `std::string::c_str()`。动态
message 会复制文本；动态文本按 Tracy 的 16-bit 长度边界安全截断。完整 `uint64_t` counter 会
规范化成 backend 可表达的数值，不会静默 wrap 成负数。共享 HYLE 的所有模块连接同一个
`hyle_tracy_client`，不能在各 DLL
再各编一份 TracyClient。trace-on static SDK 只支持所有 HYLE archive 最终进入一个 link image；
多个 DLL/plugin 各自静态链接会得到多个互不相认的 Tracy singleton，不属于受支持形状。此类 host
必须使用 shared `hyle_obs`，不能把“每个 image 一个”描述成 process-unique。

极热的 C++ 源码路径应使用上述编译期宏。HYLE 内部的 mem/sched/io/compute 也走同一条
native trace seam。

## 6. 常见误用

- 不要在库失败时“打日志然后返回成功”；失败走 error，是否记录由调用方决定。
- 不要假设 sink callback 串行；需要串行化就在 sink 内做。
- 不要在 sink callback 内 yield/suspend stackful fiber 或迁移 OS thread；复制并 enqueue 后先返回。
- 不要在 callback 返回后保存 borrowed record 指针；异步 sink 必须复制。
- 不要把 `record.message.data()` 当 C string；它按 `size()` 定界并可能包含 NUL。
- 不要把 `LogValue` 的 string 当 owning storage，或按错误 kind 读取；异步 adapter 必须连同两个 attribute span 一起复制。
- 不要用空 timestamp 或整数 0 猜测“未提供”；检查 `record.timestamp.has_value()`，epoch 0 是真实时间。
- 不要把 correlation identity 混入普通 attributes 后再靠 key 前缀猜；使用独立 `correlation_attributes` span。
- 不要把 enum 或 `long double` 隐式塞进 `LogValue`；明确选择稳定整数/name，或先做有意的精度转换。
- 不要让 `SinkRegistration` 比其 callback state 活得更久，或同时手工 remove 同一 sink。
- 不要把 obs 当信号处理器日志：mutex、stdio、format 和任意 callback 都不是 async-signal-safe。
- 不要为每个模块创建一套 logger/category 树；已有应用 logger 时用一个 adapter sink。
- 不要把 call-site interval static 用作多 request/slot 的独立进度状态；把 limiter 放进对象。
- 不要把近似 `EVERY_N` 当严格审计计数；需要精确频率时选 `_EVERY_N_EXACT`。
- 不要给 interval limiter 传 realtime/Unix 时间；回拨会按合同持续抑制。
- 不要把 runtime trace no-op 宣传成关闭态零指令；只有 `HYLE_TRACE_*` 宏具有这个合同。

设计和并发细节见 [`../architecture/obs-design.md`](../architecture/obs-design.md)，点值与
方法见 [`../performance/benchmarks-obs.md`](../implementation/evidence/performance/benchmarks-obs.md)。
