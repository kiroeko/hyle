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
              level gate + immutable snapshot                         |
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
threshold gate -- dropped --> return false / emitted=0（正常，不是错误）
        |
        v
short reader epoch: copy immutable published snapshot
                    + acquire per-sink active references
        |
        v
invoke callbacks synchronously（producer 不拿 registry mutex）
        |
        v
release active references, wake unregister waiter
```

registry 固定 64 槽，不在 emit 热路径分配。add/remove 在 mutation mutex 下改 live
slot，再发布两代 in-place immutable snapshot；producer 只做原子 reader epoch、复制
snapshot 和 active-reference，不拿 registry mutex。snapshot reader-count 与多 waiter-count、
active refcount 与 remover waiter bit 分别位于一个原子状态字，release 的一次 RMW 同时决定
是否唤醒；不存在独立 waiter flag 的 missed-wake 窗口。remove 在发布不含目标的新一代后
排空旧 snapshot reader 与目标 active callback，因此 `remove_sink` /
`hyle_obs_sink_destroy` 成功返回后，其他线程不可能仍在旧 callback 中，调用方可安全
释放 state。两代 snapshot 复用前先排空短 reader epoch，既没有无界 retired-list，也
没有 emit 分配。

callback 可以摘除自己。实现先把 callback/state 复制到局部变量，外部 callback
返回后不再访问 opaque wrapper；本次 callback 的 state 仍须活到自己返回。callback
内跨摘另一个 sink 会被拒绝：目标可能已被同一 dispatch 快照持有且尚未调用，等待
它退出会等待当前线程自己，形成死锁。

注册 sink 内再次产生日志不会重入注册 sink 图，否则一个诊断 callback 就能无限递归。
默认 stderr sink 仍启用时，nested record 直接落 stderr；默认 sink 已关闭时返回
“没有 sink 收到”。`flush_sinks` 同步调用每个可选 flush callback，并使用相同的
递归抑制与 active-reference 生命周期。这个形状对应 Abseil 的 registered-sink
recursion guard 与 Flush、Folly 的同步 handler + flush 合同。

registry 采用 process-lifetime `NoDestructor`：它第一次真正使用时构造但不参与静态
析构。这样比 registry 更晚析构的全局对象仍可记录或查询，不会访问已经销毁的 mutex。

默认 sink 用 stderr；只有 stderr 真连 TTY 时才输出 ANSI 色码，重定向到文件/CI
不会污染 escape sequence。默认 sink 自带串行锁，避免多线程 record 的多个
`fprintf` 片段互相穿插。source/message/field 中的换行、TAB、ESC 和其他 ASCII control 会
转义成可见文本，使一条 record 始终只占一个物理行，也不能借用户数据注入终端控制码。

## 3. 两种日志调用面

`HYLE_LOG_*` 是 C++ 便利面：宏先做一次门控，未通过时参数不求值、不进入
`std::format`；通过后在调用方 1 KiB 栈缓冲格式化，并附 file/function/line。超长文本
显式以 `...[truncated]` 结尾，避免静默截断；需要完整大 payload 时应直接构造 record，
或让应用自己的日志系统管理拥有型缓冲。
`log_record` 则适合已有结构化字段或已经格式化好的消息。

经本轮审查的 C/FFI 发布面是 `include/hyle/c/obs.h`（首发前仍可按产品需要调整，不把
“已审查”误写成兼容承诺）：

- POD record/field；
- opaque sink create/destroy；
- 普通 `extern "C"` level、emit、registry 函数；
- header-inline 的五个 level 便利函数。

C ABI 不暴露 `std::format`、C++ reference、STL、Tracy 类型或 C++ 类布局。各语言
在本语言侧格式化后调用 `hyle_obs_log`，或构造 POD record。

可恢复失败统一返回 owning `hyle_status`：输入错误、重复/不存在、registry 满和 callback
内跨摘分别保留共享 error taxonomy；gate/no-sink 是 OK + `out_emitted=0`。obs 不再维护
私有 TLS `last_error`。

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
[`../performance/benchmarks-obs.md`](../performance/benchmarks-obs.md)，
模块收口证据见
[`../milestones/obs-module-closure.md`](../milestones/obs-module-closure.md)。
