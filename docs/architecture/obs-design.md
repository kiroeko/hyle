# obs 设计：同步日志产出 + 编译期追踪

本页定义 obs 的长期产品边界和设计理由。当前轮次与未通过的门只看
[`module-status.json`](../implementation/module-status.json)；当前公开用法只看
[`user/obs`](../user/obs.md)。obs 只提供 C++23 source/static/shared，Native FFI 当前为 `N/A`。

obs 有两条相互独立的产出通路：

- **log** 面向运行中长期存在的诊断记录。它是同步、level-gated、结构化 fan-out；
- **trace** 面向 profiling build 的纳秒级时间线。它默认编译期消失，开启时接 Tracy。

HYLE 不实现日志文件滚动、异步上传、聚合、查询、告警或 profiler viewer。这些是
sink/运维系统的职责；obs 只提供基础库应有的安全产出接缝。

## 1. 总体架构

```text
                    C++23 source/static/shared
                      include/hyle/obs/*.h
                               |
                    +----------+----------+
                    |                     |
               HYLE_LOG_* / log_record  HYLE_TRACE_*
                    |                     |
             structured log core     HYLE_TRACE=ON
                    |                     |
          level gate + immutable     one shared
          snapshot + active refs     hyle_tracy_client
                    |                     |
          +---------+---------+       mem / sched / io / compute
          |                   |
     default stderr     native C++ sink
```

共享库构建里 `hyle_tracy_client` 必须也是共享库。若将静态
`TracyClient.cpp` 分别链进 `hyle_mem.dll`、`hyle_sched.dll` 等模块，会制造多个
profiler singleton，时间线与 fiber 身份被拆开。静态 HYLE 只在所有 archive 进入**同一个
最终 link image**（通常是一个 executable）时由最终链接器解析一份 client；如果 host 的多个
DLL/plugin 各自静态链接 trace-on SDK，它们必然各有 Tracy singleton，当前不支持把这些 image
伪装成 process-unique。该部署形状必须改用 shared `hyle_obs`/`hyle_tracy_client`。独立 package
配置无法从一个 image 的 target graph 看见另一个 plugin，因此这条边界以 artifact/consumer
形状审计和文档守门，不宣称存在不可实现的跨构建自动检测。

## 2. 日志数据流与生命周期

`LogRecord` 的消息和字段都是同步借用。message 与 field key/value 都使用 exact
`std::string_view`，已有 slice 不必复制或扫描，也能保留内嵌 NUL；字段集合以
`std::span<const LogField>` 表达。可选 source file/function 来自 `source_location`，仍是 NUL 结尾的
稳定字符串。调用方必须让全部借用数据活过同步 dispatch；sink 需要异步保存时自行复制。

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
排空旧 snapshot reader 与目标 active callback，因此 `remove_sink` 成功返回后，其他线程不可能
仍在旧 callback 中，调用方可安全
释放 state。两代 snapshot 复用前先排空短 reader epoch，既没有无界 retired-list，也
没有 emit 分配。

C++ 默认用 move-only `SinkRegistration` 持有一次 registry membership。它的析构执行
quiescent unregister，消除 early return/异常路径遗漏 remove 后留下悬空 callback 的风险；
需要显式处理失败或 callback 内复杂注销时才直接使用 `add_sink/remove_sink`。RAII 对象只是
控制面所有权，不进入每条 record，也不给 sink/record 增加常态元数据。

callback 可以摘除自己。实现先把 callback/state 复制到局部变量，外部 callback
返回后不再访问 opaque wrapper；本次 callback 的 state 仍须活到自己返回。callback
内跨摘另一个 sink 会被拒绝：目标可能已被同一 dispatch 快照持有且尚未调用，等待
它退出会等待当前线程自己，形成死锁。

callback 是 producer OS thread 上同步、执行到返回的 borrowed 边界。实现的 recursion frame 存在
OS TLS 中，因此 callback 不能主动 yield/suspend stackful fiber，也不能在返回前迁移到另一 OS
thread：即使稍后回到原线程，同线程上穿插运行的其他 fiber 也会继承错误的 nested 状态。普通 OS
抢占不改变 thread identity，属于支持行为。异步 sink 必须在 callback 内复制 record 并 enqueue 后
立即返回；不在每次 emit 上增加 thread-id 检查或 fiber metadata 来补救违反前置条件的 backend。

注册 sink 内再次产生日志不会重入注册 sink 图，否则一个诊断 callback 就能无限递归。
默认 stderr sink 仍启用时，nested record 直接落 stderr；默认 sink 已关闭时返回
“没有 sink 收到”。`flush_sinks` 同步调用每个可选 flush callback，并使用相同的
递归抑制与 active-reference 生命周期。这个形状对应 Abseil 的 registered-sink
recursion guard 与 Flush、Folly 的同步 handler + flush 合同。

registry 采用 process-lifetime `NoDestructor`：它第一次真正使用时构造但不参与静态
析构。这样比 registry 更晚析构的全局对象仍可记录或查询，不会访问已经销毁的 mutex。

默认 sink 用 stderr；只有 stderr 真连 TTY 时才输出 ANSI 色码，重定向到文件/CI
不会污染 escape sequence。默认 sink 自带串行锁，避免多线程 record 的多个
`fprintf` 片段互相穿插。source/message/field 中的换行、TAB、NUL、ESC 和其他 ASCII control 会
转义成可见文本，使一条 record 始终只占一个物理行，也不能借用户数据注入终端控制码。
Windows debugger mirror 输出同一份 level/source/message/fields，并用固定 4 KiB 缓冲；过长
记录显式写入 `...[truncated]`，不会静默截断或为诊断路径分配。

## 3. 两种日志调用面

`HYLE_LOG_*` 是 C++ 便利面：调用方可在包含头之前定义
`HYLE_OBS_COMPILED_MIN_LEVEL=0..5`，把低于编译期 floor 的调用及参数求值完全删除；保留的
宏再做**一次** relaxed runtime 门控，未通过时参数不求值、不进入 `std::format`。通过后在
调用方 1 KiB 栈缓冲格式化并直接 dispatch，不重复读 threshold，并附
file/function/line。编译期 floor 是每个 translation unit 的消费策略，不是 HYLE binary
ABI 或进程全局配置；动态 level 仍由 `set_level` 控制。并发修改 runtime level 是 eventual
configuration：已经通过一次门控的 record 可以完成本次同步 dispatch。超长文本
显式以 `...[truncated]` 结尾，避免静默截断；需要完整大 payload 时应直接构造 record，
或让应用自己的日志系统管理拥有型缓冲。
`log_record` 则适合已有结构化字段，`log_message(std::string_view)` 适合已有 byte view；
`const char*` 便利重载会先扫描 NUL，长度已知时直接使用 view。

obs 当前只交付 C++23 source/static/shared。BUILD-03 已删除早期 log/trace/sink C schema、bridge、
动态导出和 fixture；Native FFI 为 `N/A`，没有 C ABI layout、callback 或 handle 兼容承诺。
将来只有已命名的 foreign-runtime consumer 通过全局
[Native FFI 准入策略](native-ffi-policy.md)后，才按任务另行设计窄的 runtime 类型擦除边界。

HYLE 的底层模块**不应为了证明 obs 被使用而主动打日常日志**：库无法替应用决定
日志策略，非请求输出也会污染宿主进程。下游利用发生在 trace hook；error/status
仍是机器可读失败通道，log 是调用方主动选择的诊断通道。

## 4. 依赖、消费者与负空间

`hyle_obs` 在中央 component graph 中 `PUBLIC` 依赖 `core` 与 `error`：core 提供 compiler、fatal
invariant 与 visibility 基础，sink 注册的可恢复失败使用 error 的 `Status/Result`。默认
stderr/debugger writer 直接拥有 TTY 和 debugger OS 交互，因为
platform 没有此抽象、obs 与 platform 同处底层且这正是 sink 的 I/O 职责，不能为消除一个
局部 syscall 制造 `obs -> platform` 依赖。trace gate 是 header/CMake 横切 seam，不让
mem/sched/io/compute 反向链接 log runtime。
typed C++ sink 只是借用调用方 backend，registry 也使用固定进程期存储，因此无需为了分配器引入
`obs -> mem`。异步拥有型复制与队列由 application adapter 自己选择 allocator。

语法消费者扫描没有发现 HYLE 生产模块调用 runtime `HYLE_LOG_*` 或注册 `LogSink`；这是
基础库不擅自输出日志的正确负空间。语义扫描中的 util fatal diagnostic、core fatal macro
属于不可恢复合同失败，io benchmark stats 只属于显式 benchmark 输出，均不应迁移成普通
log。trace 的真实消费者仍是 mem/sched/io/compute。

与 Abseil/Folly 对账后保留同步借用 record、同步 handler/flush、递归抑制、进程期 registry
和作用域注册形状；没有真实消费者支持 category hierarchy、per-sink routing policy、异步
线程/队列、文件轮转、上传、聚合/查询/告警、fatal 控制流、owning async record 或 profiler
viewer。这些能力应在 adapter/application 层组合，不给所有基础 record 添加 category、线程
ID、refcount 或拥有型字符串。需要时间戳、线程 ID 等 enrich 时，同步 adapter sink 可在
复制入上层系统时补齐。

## 5. Trace 双面与零成本边界

原生 `HYLE_TRACE_*` 宏用于 HYLE 热路径和同工具链源码用户：

- 默认 OFF：宏参数不求值，无函数调用、无后端依赖；
- ON：zone/当前 zone 的 text/name/value/color、plot/counter、fiber/frame、message、memory 与
  thread-name 转发 Tracy。

旧 runtime trace bridge 已由 BUILD-03 删除。极热 C++ 源码路径只使用原生宏；OFF 时参数不求值，
也没有函数级 no-op 调用。
Tracy copied text 使用 16-bit byte count，且上游只在 debug assert 长度；adapter 必须先做
bounded scan。动态 message、zone text 与 zone name 截到 65534 bytes，不能在 release 把
full `strlen` 与 truncated allocation length 同时交给后端。source location 属于同步 log record，
当前 native trace seam 不提供另一套 source-location bridge。

当前真实下游：

| 模块 | 埋点 |
|---|---|
| mem | system allocate/free/reallocate 与 named pool 归因 |
| sched | worker thread 名、fiber enter/leave、timer fire counter |
| io | completion reap zone/counter |
| compute | GPU poller thread 名、poll zone/completion counter |

Fiber trace 名存放在 Fiber 自身的稳定进程期存储中，不能把临时字符串指针交给
Tracy。plot/frame/pool 名同理应使用字面量或其他稳定存储。

## 6. 维护硬门

改日志 registry 时必须通过：

1. 正常 C++ 测试 + 编译期 floor codegen；
2. add/remove/log 并发压力与 remove-waits、自摘、跨摘拒绝测试；
3. WSL ASan、UBSan、TSan；
4. W1-WIN `closure-perf-v1` 的冻结正常 workload；其他平台做 correctness/交付，不池化点值；
5. Native FFI 保持 `N/A`；若出现已命名的 foreign-runtime 任务，先按全局准入策略独立立项；
6. installed source/shared/static package 中只请求 `COMPONENTS obs` 的 C++23 consumer。

改 trace/CMake 或新增下游 hook 时必须通过：

1. OFF 构建参数不求值测试，且产品库不依赖 Tracy；
2. ON 构建 obs + 被埋点模块测试，并用超长 message/zone text/source components 验证截断；
3. 共享构建的每个模块只依赖同一个 `hyle_tracy_client`；
4. 名称和 zone token 的存储期/配对审计；
5. Windows、Linux、macOS 各跑适用的 ON correctness/build；性能只按登记机器合同裁决。

trace-on 自动测试只需要本机 listener，CTest 因而给 Obs test process 设置 Tracy 上游的 runtime
`TRACY_ONLY_LOCALHOST=1`。该变量不写入产品 target，也不改变普通 trace-on 程序的网络策略。Tracy
v0.14.1 的 runtime 开关只限制 listener；compile-time 同名宏才同时改变 discovery broadcast 地址，
所以维护文档不能把测试环境变量描述为“完全禁网”。

性能数与仪器见
[`../performance/benchmarks-obs.md`](../implementation/evidence/performance/benchmarks-obs.md)，
当前模块任务见
[`obs-v2-resume`](../implementation/tasks/obs/obs-v2-resume.md)。
