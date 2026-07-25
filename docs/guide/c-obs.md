# obs 纯 C / FFI 精确映射

先读 [`obs.md`](./obs.md) 决定该用 error、log 还是 trace。本页只解释
`include/hyle/c/obs.h` 怎样映射同一能力，以及 Status、out parameter、callback 和
borrowed 数据的精确合同。C surface 在首发前已经机械审查，但这不是提前作出的兼容承诺。

## 1. Status 与最小日志调用

recoverable failure 统一使用 error 模块的 owning `hyle_status`，obs 不再维护私有
thread-local last-error：

```c
#include "hyle/c/obs.h"

hyle_status configure_log(void) {
  HYLE_C_RETURN_IF_ERROR(hyle_obs_set_level(HYLE_OBS_LOG_INFO));
  HYLE_C_RETURN_IF_ERROR(hyle_obs_log_info("server started"));
  return (hyle_status){0};
}
```

转移/传播非 OK status 后不要再 free；在本函数消费它时调用 `hyle_status_free`。OK status
不拥有错误对象，free 仍安全。level 便利函数只返回 Status；需要区分“成功投递”与
“正常 gate/no sink”时使用完整入口：

```c
int emitted = 0;
hyle_status status =
    hyle_obs_log(&emitted, HYLE_OBS_LOG_INFO, "ready");
if (!hyle_status_is_ok(status)) {
  report(hyle_status_message_cstr(status));
  hyle_status_free(&status);
} else if (!emitted) {
  /* 被 level gate，或当前没有 sink；不是错误。 */
}
```

`hyle_obs_set_level` 接受 TRACE..OFF；OFF 只是 threshold，不能作为 record level。
`hyle_obs_should_log` 是无失败 query，非法 record level 直接返回 0。所有带 out parameter
的入口先清空输出再做工作。

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
record.line = (uint32_t)__LINE__;
record.timestamp_ns = unix_time_ns; /* 可选；0 表示调用方未提供 */

int emitted = 0;
hyle_status status = hyle_obs_emit_record(&emitted, &record);
```

必须用 `HYLE_OBS_LOG_RECORD_INIT` 设置 `struct_size`。库只读取 `struct_size` 已覆盖的尾字段，
因此较短的旧布局仍可被 checked 地解释；非法大小、level、NULL message/field 会返回
`HYLE_STATUS_INVALID_ARGUMENT`。

message、field key/value 是 NUL 结尾 UTF-8；file/function 可为 NULL。所有指针只在同步
emit/callback 期间借用，库不会保存。异步 sink 必须在 callback 内复制。C/FFI 在本语言
侧格式化后调用 `hyle_obs_log`，或直接填 POD record；C ABI 不重复一套 C++ format DSL。

## 3. Opaque sink、flush 与注销保证

```c
struct writer { FILE* file; };

static void write_record(void* state,
                         const hyle_obs_log_record* record) {
  struct writer* writer = (struct writer*)state;
  fprintf(writer->file, "%d %s\n", (int)record->level, record->message);
}

static void flush_writer(void* state) {
  fflush(((struct writer*)state)->file);
}

static hyle_status run_with_file_sink(void) {
  struct writer writer = {fopen("app.log", "a")};
  hyle_obs_sink* sink = NULL;
  HYLE_C_RETURN_IF_ERROR(
      hyle_obs_sink_create(&sink, write_record, flush_writer, &writer));

  /* 多线程产生日志…… */
  hyle_obs_flush_sinks();
  HYLE_C_RETURN_IF_ERROR(hyle_obs_sink_destroy(sink));
  fclose(writer.file); /* destroy 成功后释放 state 安全 */
  return (hyle_status){0};
}
```

create 成功后立即注册；write 为 NULL、registry 满或分配失败通过共享 Status 区分。
`hyle_obs_sink_destroy(NULL)` 是 OK。destroy 成功返回时，其他线程不可能仍在旧 callback
中；同一非 NULL handle 只能消费一次，也不能由两个线程并发 destroy。

callback 可以销毁自己的 handle。opaque trampoline 在进入外部 callback 前复制函数与
state，返回后不再读已释放 wrapper；但 state 自然仍须活到 callback 返回。callback 内销毁
另一个 sink 返回 `HYLE_STATUS_FAILED_PRECONDITION` 且不消费目标 handle，调用方应在所有
sink callback 外重试。

write/flush 由 producer 线程同步调用，可能多线程并发；不得让异常跨 ABI。flush 可与
write 并发，`hyle_obs_flush_sinks` 返回实际调用的非 NULL flush callback 数。sink 内再次
产生日志不会重入注册 sink 图；默认 stderr 仍启用时 nested record 直接落 stderr，否则
正常未投递。异步 queue、文件滚动、上传和 backpressure 由 sink/应用负责。

registry 入口：

| 入口 | 语义 |
|---|---|
| `hyle_obs_sink_create` / `hyle_obs_sink_destroy` | Status + opaque ownership；quiescent unregister |
| `hyle_obs_sink_count` / `hyle_obs_sink_capacity` | 当前数量 / 固定 64 槽上限 |
| `hyle_obs_flush_sinks` | 同步 flush 所有支持它的 sink |
| `hyle_obs_default_sink_enabled` | 默认 stderr sink 当前是否注册 |
| `hyle_obs_set_default_sink_enabled` | 幂等达到目标状态，失败返回 Status |

默认 stderr sink 每条写后 flush，只在 TTY 加自己的颜色，并转义 source/message/field 的控制字符，
使每条 record 保持一个物理行。它适合最低限度诊断；高吞吐持久化请接应用 sink。

## 4. Trace：runtime C seam

HYLE 默认以 `HYLE_TRACE=OFF` 构建。C/FFI 可以始终链接 runtime trace 函数；关闭构建中
它们是 no-op：

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

| 入口 | 用途 / 生命周期 |
|---|---|
| `hyle_obs_trace_enabled` | 库是否编入 trace 后端 |
| `hyle_obs_trace_zone_begin/end/text` | 动态 source-location zone；同一逻辑上下文栈式配对 |
| `HYLE_OBS_TRACE_ZONE_BEGIN` | header 宏自动传 file/function/line |
| `hyle_obs_trace_plot/counter` | 浮点序列 / 调用方维护的整数累计值；name 长期稳定 |
| `hyle_obs_trace_message` | 动态消息；开启时后端复制文本 |
| `hyle_obs_trace_frame` | frame mark；非 NULL name 长期稳定 |
| `hyle_obs_trace_alloc/free` | 内存归因；pointer 与 pool name 成对，pool name 长期稳定 |
| `hyle_obs_trace_thread_name` | 命名当前 OS 线程 |

NULL message/name/pointer 在有意义的入口上安全忽略；NULL frame name 表示无名 frame，NULL
pool name 表示默认内存域。zone token 只在当前进程/build 有意义，不能持久化或跨进程传输。

C runtime seam 即使关闭仍有一次 ABI 调用，所以不能宣称“关闭态零指令”。同工具链 C++
源码的纳秒热路径使用 `hyle/obs/trace.h` 的 `HYLE_TRACE_*` 宏：关闭时参数不求值；开启时
所有模块汇入同一个 Tracy client。

## 5. 导出覆盖与仅 C++ 能力

动态日志/registry 导出共 12 个：

- `hyle_obs_set_level` / `hyle_obs_get_level` / `hyle_obs_should_log`
- `hyle_obs_emit_record` / `hyle_obs_log`
- `hyle_obs_sink_create` / `hyle_obs_sink_destroy`
- `hyle_obs_sink_count` / `hyle_obs_sink_capacity` / `hyle_obs_flush_sinks`
- `hyle_obs_default_sink_enabled` / `hyle_obs_set_default_sink_enabled`

runtime trace 导出共 11 个：enabled、zone begin/end/text、plot、counter、message、frame、
alloc、free、thread name。另有 5 个 header-inline level 便利函数。manifest、纯 C consumer
与 DLL/.so 导出由 `hyle_obs_c_surface_test` 机械保持一致。

有意仅 C++ 的部分只有语言/编译期形状，而不是功能缺口：

| C++ 面 | C/FFI 组合 |
|---|---|
| `std::format` + `HYLE_LOG_*` | 在调用语言格式化，再用 `hyle_obs_log` / POD record |
| 编译期消失的 `HYLE_TRACE_*` | runtime trace 自由函数；接受一次 ABI no-op 成本 |
| `LogSink` 值类型 | opaque `hyle_obs_sink` 管理 trampoline 与安全注销 |

并发实现与模块边界见 [`../architecture/obs-design.md`](../architecture/obs-design.md)。
