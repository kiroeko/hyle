# obs 设计：同步日志产出 + 编译期追踪

obs 有两条相互独立的产出通路：

- **log** 面向运行中长期存在的诊断记录。它是同步、level-gated、结构化 fan-out；
- **trace** 面向 profiling build 的纳秒级时间线。它默认编译期消失，开启时接 Tracy。

HYLE 不实现日志文件滚动、异步上传、聚合、查询、告警或 profiler viewer。这些是
sink/运维系统的职责；obs 只提供基础库应有的安全产出接缝。

## 1. 总体架构

```text
                  同工具链 / 源码用户                    C / FFI 用户
               include/hyle/obs/*.h                  include/hyle/c/obs.h
                         |                                   |
             +-----------+-----------+              +--------+--------+
             |                       |              |                 |
        HYLE_LOG_* / log_record   HYLE_TRACE_*   hyle_obs_log_*   hyle_obs_trace_*
             |                       |              |                 |
             |                       |              |       HYLE_TRACE=OFF: 函数 no-op
             |                       |              |       HYLE_TRACE=ON : Tracy C seam
             |                       |              |                 |
             +------ structured log core -----------+                 |
                         |                                             |
              level gate + registry snapshot                          |
              + per-sink active reference                             |
                         |                                             |
          +--------------+----------------+                            |
          |              |                |                            |
     default stderr   native C++ sink   opaque C sink trampoline       |
          |                               |                            |
          +---------- callback 外部消费 --+                            |
                                                                       |
                         HYLE_TRACE=ON                                  |
        native macros ----------------+---------------- runtime C ABI --+
                                     |
                           one shared hyle_tracy_client
                                     |
                +----------+----------+---------+----------+
                |          |          |         |          |
               mem       sched       io      compute    app/FFI
```

共享库构建里 `hyle_tracy_client` 必须也是共享库。若将静态
`TracyClient.cpp` 分别链进 `hyle_mem.dll`、`hyle_sched.dll` 等模块，会制造多个
profiler singleton，时间线与 fiber 身份被拆开。静态 HYLE 则仍用一个 archive，
最终链接器只解析一份。

## 2. 日志数据流与生命周期

`LogRecord` / `hyle_obs_log_record` 的字符串和字段都是同步借用：

```text
caller builds record
        |
        v
threshold gate -- dropped --> return false / 0（正常，不是错误）
        |
        v
registry lock: snapshot live slots + acquire active references
        |
        v
unlock, invoke callbacks synchronously
        |
        v
release active references, wake unregister waiter
```

registry 固定 64 槽，不在 emit 热路径分配。锁只覆盖 live-slot 快照；任何 sink
callback、格式化和 I/O 都不持 registry 锁。每个快照对 slot 持 active reference，
因此 `remove_sink` / `hyle_obs_sink_destroy` 成功返回后，其他线程不可能仍在旧
callback 中，调用方可安全释放 state。

callback 可以摘除自己。实现先把 callback/state 复制到局部变量，外部 callback
返回后不再访问 opaque wrapper；本次 callback 的 state 仍须活到自己返回。callback
内跨摘另一个 sink 会被拒绝：目标可能已被同一 dispatch 快照持有且尚未调用，等待
它退出会等待当前线程自己，形成死锁。

默认 sink 用 stderr；只有 stderr 真连 TTY 时才输出 ANSI 色码，重定向到文件/CI
不会污染 escape sequence。默认 sink 自带串行锁，避免多线程 record 的多个
`fprintf` 片段互相穿插。

## 3. 两种日志调用面

`HYLE_LOG_*` 是 C++ 便利面：宏先做一次门控，未通过时参数不求值、不进入
`std::format`；通过后在调用方 1 KiB 栈缓冲格式化，并附 file/function/line。
`log_record` 则适合已有结构化字段或已经格式化好的消息。

稳定发布面是 `include/hyle/c/obs.h`：

- POD record/field；
- opaque sink create/destroy；
- 普通 `extern "C"` level、emit、registry 函数；
- header-inline 的五个 level 便利函数。

C ABI 不暴露 `std::format`、C++ reference、STL、Tracy 类型或 C++ 类布局。各语言
在本语言侧格式化后调用 `hyle_obs_log`，或构造 POD record。

HYLE 的底层模块**不应为了证明 obs 被使用而主动打日常日志**：库无法替应用决定
日志策略，非请求输出也会污染宿主进程。下游利用发生在 trace hook；error/status
仍是机器可读失败通道，log 是调用方主动选择的诊断通道。

## 4. Trace 双面与零成本边界

原生 `HYLE_TRACE_*` 宏用于 HYLE 热路径和同工具链源码用户：

- 默认 OFF：宏参数不求值，无函数调用、无后端依赖；
- ON：zone/plot/counter/fiber/frame/message/memory/thread-name 转发 Tracy。

C/FFI runtime trace 函数必须始终存在，OFF 时是函数级 no-op。它们有一次 ABI 调用，
所以不能宣传为“关闭态零指令”；极热 C/C++ 源码路径应使用原生宏。

当前真实下游：

| 模块 | 埋点 |
|---|---|
| mem | system allocate/free/reallocate 与 named pool 归因 |
| sched | worker thread 名、fiber enter/leave、timer fire counter |
| io | completion reap zone/counter |
| compute | GPU poller thread 名、poll zone/completion counter |

Fiber trace 名存放在 Fiber 自身的稳定进程期存储中，不能把临时字符串指针交给
Tracy。plot/frame/pool 名同理应使用字面量或其他稳定存储。

## 5. 维护硬门

改日志 registry 或 C trampoline 时必须通过：

1. 正常 C++ 测试 + 纯 C consumer；
2. add/remove/log 并发压力与 remove-waits、自摘、跨摘拒绝测试；
3. WSL ASan、UBSan、TSan；
4. 三平台 Release gated/preformatted/formatted 基准；
5. 动态库导出与 `include/hyle/c/obs.h` / C 测试 / guide 机械覆盖核对。

改 trace/CMake 或新增下游 hook 时必须通过：

1. OFF 构建参数不求值测试，且产品库不依赖 Tracy；
2. ON 构建 obs + 被埋点模块测试；
3. 共享构建的每个模块只依赖同一个 `hyle_tracy_client`；
4. 名称和 zone token 的存储期/配对审计；
5. Windows、Linux、macOS 至少各跑一次 ON 构建。

性能数与仪器见
[`../performance/benchmarks-mem-container.md`](../performance/benchmarks-mem-container.md)，
模块收口证据见
[`../milestones/obs-module-closure.md`](../milestones/obs-module-closure.md)。
