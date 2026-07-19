# obs 纯 C ABI 使用指南

`include/hyle/c/obs.h` 是 obs 模块的稳定 C/FFI 接缝。它覆盖同步结构化日志、
可安全注销的 sink，以及显式 trace 产出；不暴露 `std::format`、Tracy 类型、
C++ 回调或 STL。

obs 只负责“产出”。异步队列、文件滚动、网络上报、聚合、查询和告警属于 sink
或运维系统，不由基础库重复实现。

## 1. 最小日志调用

```c
#include "hyle/c/obs.h"

if (!hyle_obs_set_level(HYLE_OBS_LOG_INFO)) {
  report(hyle_obs_last_error());
}
hyle_obs_log_info("server started");
hyle_obs_log_error("request failed");
```

`hyle_obs_set_level`、`hyle_obs_get_level` 和 `hyle_obs_should_log` 控制全局门槛。
`HYLE_OBS_LOG_OFF` 只能作为门槛，不能作为一条 record 的 level。`hyle_obs_log`
及五个 header-inline level 便利函数返回 1 表示至少一个 sink 收到了消息；门控
丢弃、无 sink 或输入错误返回 0。只有输入错误会设置
`hyle_obs_last_error()`，正常门控不是错误。

## 2. 结构化 record

```c
hyle_obs_log_field fields[] = {
    {"request_id", "a7f2"},
    {"peer", "10.0.0.8"},
};
hyle_obs_log_record record = HYLE_OBS_LOG_RECORD_INIT;
record.level = HYLE_OBS_LOG_WARN;
record.message = "slow request";
record.fields = fields;
record.field_count = 2;
record.file = __FILE__;
record.function = __func__;
record.line = __LINE__;
record.timestamp_ns = unix_time_ns; /* 可选；0 表示未提供 */

hyle_obs_emit_record(&record);
```

`struct_size` 必须由 `HYLE_OBS_LOG_RECORD_INIT` 初始化。库只读取
`struct_size` 覆盖的字段，因此未来可以在结构尾追加字段。`message`、field
的 key/value 是 NUL 结尾 UTF-8；`file`/`function` 可为 NULL。

所有指针均为借用：emit 是同步调用，函数返回后库不保存 record、字段或字符串。
需要延迟写盘/发网的 sink 必须在 callback 内自行复制。

## 3. 自定义 sink 与生命周期

```c
struct writer {
  FILE* file;
};

static void write_record(void* state,
                         const hyle_obs_log_record* record) {
  struct writer* writer = (struct writer*)state;
  fprintf(writer->file, "%d %s\n", (int)record->level, record->message);
}

struct writer writer = {fopen("app.log", "a")};
hyle_obs_sink* sink = hyle_obs_sink_create(write_record, &writer);
if (!sink) report(hyle_obs_last_error());

/* 多线程日志…… */

if (!hyle_obs_sink_destroy(sink)) {
  fprintf(stderr, "%s\n", hyle_obs_last_error());
}
fclose(writer.file);         /* 此后释放 state 安全 */
```

`hyle_obs_sink_create` 成功后立即注册；registry 满或 callback 为 NULL 时返回
NULL。`hyle_obs_sink_destroy(NULL)` 成功且安全。

关键并发保证：destroy 成功返回时，其他线程不可能仍在该 callback 中，所以随后
释放 state 不会与旧 snapshot 形成 UAF。callback 也可以销毁自己的 handle，不会
死锁；但当前 callback 自己显然仍在执行，其 state 必须活到 callback 返回。一个
callback 不得销毁另一个 sink：目标可能仍在同一次分发快照中；函数会返回 0、保留
目标 handle，并设置 `hyle_obs_last_error()`，调用方应在 callback 外重试。
每个非 NULL handle 只能成功 destroy 一次；destroy 不能与另一个线程对同一 handle
的 destroy 并发。若 callback 自己 destroy，外部所有者必须把该 handle 视为已消费。

callback 由产生日志的线程同步执行，可能多线程并发、不得抛异常跨 ABI。阻塞 sink
会阻塞对应 logger；需要异步行为时由 sink 自己把 record 复制进队列。

默认 sink 用 stderr，同步 flush；只有 stderr 连接 TTY 时才着色，重定向到文件/CI
不会包含 ANSI escape。以下入口管理和查询 registry：

| 入口 | 用途 |
|---|---|
| `hyle_obs_sink_create` / `hyle_obs_sink_destroy` | 注册 opaque sink；安全摘除 |
| `hyle_obs_sink_count` / `hyle_obs_sink_capacity` | 当前数量与固定上限 |
| `hyle_obs_default_sink_enabled` | 默认 stderr sink 是否存在 |
| `hyle_obs_set_default_sink_enabled` | 幂等启用/移除默认 sink；返回是否达到目标状态 |

## 4. Trace：原生零成本面与 C/FFI 面

HYLE 默认以 `HYLE_TRACE=OFF` 构建。纯 C/FFI 可以始终链接下列函数；关闭构建中
它们是稳定 no-op：

```c
if (hyle_obs_trace_enabled()) {
  hyle_obs_trace_zone zone = HYLE_OBS_TRACE_ZONE_BEGIN("parse");
  hyle_obs_trace_counter("requests", request_count);
  hyle_obs_trace_plot("queue_depth", queue_depth);
  hyle_obs_trace_message("batch ready");
  hyle_obs_trace_zone_text(zone, "json");
  hyle_obs_trace_zone_end(zone);
}
```

| 入口 | 用途 |
|---|---|
| `hyle_obs_trace_enabled` | 库是否编入 trace 后端 |
| `hyle_obs_trace_zone_begin` / `hyle_obs_trace_zone_end` | 动态 source-location zone |
| `HYLE_OBS_TRACE_ZONE_BEGIN` | 自动传 `__FILE__/__func__/__LINE__` 的 header 宏 |
| `hyle_obs_trace_zone_text` | 给活动 zone 附加动态文本 |
| `hyle_obs_trace_plot` / `hyle_obs_trace_counter` | 浮点序列 / 整数累计值 |
| `hyle_obs_trace_message` | 动态消息；后端复制文本 |
| `hyle_obs_trace_frame` | frame mark；非 NULL 名必须长期稳定 |
| `hyle_obs_trace_alloc` / `hyle_obs_trace_free` | 内存归因；pool 名必须长期稳定且成对一致 |
| `hyle_obs_trace_thread_name` | 命名当前 OS 线程 |

zone 必须在同一逻辑执行上下文按栈式顺序结束，不能把 token 当持久化句柄或跨进程
传输。plot/frame/pool 的非 NULL 名由 Tracy 按稳定身份使用，应传字符串字面量或
进程期存储。

C 入口有一次 ABI 函数调用，不能宣称“关闭态零指令”。HYLE 自身的纳秒热路径用
`hyle/obs/trace.h` 的编译期宏：关闭时参数不求值、无调用、无指令；开启时转发同
一个 Tracy 后端。构建方式：

```sh
cmake -S . -B build-trace \
  -DHYLE_TRACE=ON \
  -DHYLE_TRACY_DIR=/path/to/tracy
```

共享库构建只生成一个 `hyle_tracy_client` 动态库，所有 HYLE 模块导入它；不能把
TracyClient 静态复制进每个模块，否则会产生多个 profiler singleton。

## 5. 完整导出清单与有意仅 C++ 项

日志导出：

- `hyle_obs_last_error`
- `hyle_obs_set_level` / `hyle_obs_get_level` / `hyle_obs_should_log`
- `hyle_obs_emit_record` / `hyle_obs_log`
- `hyle_obs_sink_create` / `hyle_obs_sink_destroy`
- `hyle_obs_sink_count` / `hyle_obs_sink_capacity`
- `hyle_obs_default_sink_enabled` / `hyle_obs_set_default_sink_enabled`

header-inline 日志便利入口：

- `hyle_obs_log_trace` / `hyle_obs_log_debug` / `hyle_obs_log_info`
- `hyle_obs_log_warn` / `hyle_obs_log_error`

trace 导出：

- `hyle_obs_trace_enabled`
- `hyle_obs_trace_zone_begin` / `hyle_obs_trace_zone_end` /
  `hyle_obs_trace_zone_text`
- `hyle_obs_trace_plot` / `hyle_obs_trace_counter` /
  `hyle_obs_trace_message` / `hyle_obs_trace_frame`
- `hyle_obs_trace_alloc` / `hyle_obs_trace_free` /
  `hyle_obs_trace_thread_name`

有意仅 C++：

| C++ 面 | 原因 / C 通路 |
|---|---|
| `log_fmt`、`HYLE_LOG_*` | `std::format` 模板不能跨 C ABI；C/FFI 调用方在本语言格式化后用 `hyle_obs_log`，或构造 record |
| `HYLE_TRACE_*` 原生宏 | 编译期消失和 RAII/source-location 是源码级能力；C/FFI 用上述 runtime trace 函数 |
| `LogSink` 值类型 | C ABI 用 opaque `hyle_obs_sink`，让 DLL 内管理 callback 注销与生命周期 |

没有模块级 api 表、`api_get` 或 ABI 版本宏；发布面是普通 `extern "C"` 自由函数。
