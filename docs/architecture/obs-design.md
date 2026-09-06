# obs 设计：同步日志 + 数值指标 + 编译期追踪

本页定义 obs 的长期产品边界和设计理由。当前轮次与未通过的门只看
[`module-status.json`](../implementation/module-status.json)；当前公开用法只看
[`user/obs`](../user/obs.md)。obs 只提供 C++23 source/static/shared，Native FFI 当前为 `N/A`。

obs 有三条相互独立的产出通路：

- **log** 面向运行中长期存在的诊断记录。它是同步、level-gated、结构化 fan-out；
- **metrics** 面向 production 热路径上的 counter、gauge 与固定桶 histogram。它只聚合数值，不拥有
  名字、标签、registry 或 exporter；
- **trace** 面向 profiling build 的纳秒级时间线。它默认编译期消失，开启时接 Tracy。

HYLE 不实现日志文件滚动、异步上传、动态指标 registry、wire exporter、查询、告警或 profiler viewer。这些是
sink/运维系统的职责；obs 只提供基础库应有的安全产出接缝。

## 1. 总体架构

```text
                    C++23 source/static/shared
                      include/hyle/obs/*.h
                               |
              +-------------+-------------+
              |             |             |
        log / log_record   metrics    HYLE_TRACE_*
              |             |             |
      synchronous sinks   caller-owned  HYLE_TRACE=ON
      + mem_reclaim       numeric state  one Tracy client
              |             |             |
       stderr / adapter   host exporter  mem / sched / io / compute
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

`LogRecord` 是同步借用的 logger-neutral event。category、event、message、attribute key、string value
与 source file/function 都是 exact `std::string_view`，已有 slice 不必复制或扫描，也能保留内嵌 NUL；
普通属性和 correlation 属性分别用两个 `std::span<const LogAttribute>`，保序并允许重复/空 key。
`LogValue` 是 trivially-copyable tagged scalar，覆盖 string、signed/unsigned 64-bit integer、double 与
bool，不引入 variant allocation 或 runtime schema；拥有型 `std::basic_string<char>` 只接受左值借用，
临时值在编译期拒绝。source 用 `LogSourceLocation` 表达，既可从
`std::source_location` 捕获，也可由 logger bridge 显式构造；wall timestamp 是可选的 canonical
`time::UnixTimestamp`，因此 absent、Unix epoch 0 和 pre-epoch 不再混淆。调用方必须让全部借用数据活过
同步 dispatch；sink 需要异步保存时自行复制。

```text
caller builds record
        |
        v
threshold gate -- dropped --> return false / emitted=0（正常，不是错误）
        |
        v
short mem_reclaim guard: copy immutable published snapshot
                    + acquire per-sink active references
        |
        v
invoke callbacks synchronously（producer 不拿 registry mutex）
        |
        v
release active references, wake unregister waiter
```

registry 固定 64 槽，不在 emit 热路径分配。add/remove 在 mutation mutex 下改 live slot，再发布两代
in-place immutable snapshot；producer 用独立 `mem::ReclamationDomain` 的短 guard 保护“读取 published
pointer、复制 snapshot、取得全部 per-sink active refs”这一小段，不拿 registry mutex。writer 在覆写
inactive snapshot 前等一个 grace period；remove 发布不含目标的新一代后再等一个 grace period，关闭
旧 snapshot 到 active ref 的 hand-off race，然后才排空目标 callback。guard **不覆盖 callback**：任意
sink 的执行期仍由该 slot 的 active count + waiter bit 所有，避免 stalled sink 把整个 reclamation domain
绑死，也保留 self-remove/cross-remove 语义。`remove_sink` 成功返回后，其他线程不可能仍在旧 callback
中，调用方可安全释放 state。这样复用了已冻结的 epoch primitive，又不把两个不同生命周期强行合并。

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

默认 sink 用 stderr；只有 stderr 真连 TTY 时才输出 ANSI 色码，重定向到文件/CI 不会污染 escape
sequence。默认 sink 自带串行锁，避免多线程 record 的多个 I/O 片段互相穿插。identity、source、
message、key 与 string value 中的换行、TAB、NUL、ESC 和其他 ASCII control 都转义成可见文本；整数、
浮点与 bool 用固定栈缓冲的 `to_chars` 渲染，不分配。correlation key 以 `correlation.` 前缀区分；时间戳
输出独立的 `timestamp.seconds`/`timestamp.nanoseconds`，不会把 canonical pre-epoch pair 错写成误导性
小数。一条 record 始终只占一个物理行，也不能借用户数据注入终端控制码。Windows debugger mirror
输出同一语义并用固定 4 KiB 缓冲；过长记录显式写入 `...[truncated]`。

## 3. 日志调用面与控制状态

`HYLE_LOG_*` 是 C++ 便利面：调用方可在包含头之前定义
`HYLE_OBS_COMPILED_MIN_LEVEL=0..5`，把低于编译期 floor 的调用及参数求值完全删除；保留的
宏再做**一次** relaxed runtime 门控，未通过时参数不求值、不进入 `std::format`。通过后在
调用方 1 KiB 栈缓冲格式化并直接 dispatch，不重复读 threshold，并附
file/function/line。编译期 floor 是每个 translation unit 的消费策略，不是 HYLE binary
ABI 或进程全局配置；动态 level 仍由 `set_level` 控制。并发修改 runtime level 是 eventual
configuration：已经通过一次门控的 record 可以完成本次同步 dispatch。超长文本
显式以 `...[truncated]` 结尾，避免静默截断；需要完整大 payload 时应直接构造 record，
或让应用自己的日志系统管理拥有型缓冲。
`log_record` 则适合已有 typed event descriptors，`log_message(std::string_view)` 适合已有 byte view；
`const char*` 便利重载会先扫描 NUL，长度已知时直接使用 view。

重复诊断有两种状态所有权，不能用一种 static 勉强覆盖：

1. 普通调用点宏在展开处持有隐藏 static state，并保持编译期 floor -> runtime level -> condition/
   control -> format 的短路次序。`EVERY_N`/`EVERY_POW_2` 采用 Abseil/Folly 一致的 relaxed
   load/store 近似计数，避免共享调用点上的 RMW 争用；`EVERY_N_EXACT` 提供明确的 relaxed RMW
   替代，`FIRST_N` 用 relaxed CAS 重试保证并发总量不超过 N。无竞争路径是常数成本；竞争时单次
   调用的重试次数不承诺有界。HYLE 不引入产品 mutex；底层原子是否 lock-free 仍服从目标标准库，
   已签目标上的 64-bit atomic 是否 lock-free 由 candidate 验证，不写死成所有平台承诺。
2. 推理 request/slot、设备或长任务把 `LogIntervalLimiter` 作为自己的 8-byte 状态。caller-supplied
   monotonic time 让一次 scheduler tick 的 clock read 在多个 slot 间共享；便利 overload 才自行读取
   `steady_clock`。无竞争边界通常由一次 relaxed CAS 裁决；竞争时允许重试，因此同一状态并发最多
   放行一条，但不把单次调用误写成 wait-free 或严格最坏 O(1)。

这两个面复用同一组公开 state 类型；宏只是 source location 与隐藏生命周期的便利层。控制状态不进入
`LogRecord`、sink snapshot 或每条已发出记录，不给未使用限流的消费者增加 layout/registry 成本。
interval 以“上次放行时间”而非“下个 deadline”计算差值，避免 `now + interval` 溢出；编码值 0 表示
从未放行。时间回退安全地抑制到追平，interval<=0 明确表示不限制。`reset` 是生命周期控制操作，与
check 并发虽保持内存安全但可能多放一条，因此 consumer 在 slot 静止时调用；显式状态的 move 和析构
也要求相同的外部生命周期同步。

obs 当前只交付 C++23 source/static/shared。BUILD-03 已删除早期 log/trace/sink C schema、bridge、
动态导出和 fixture；Native FFI 为 `N/A`，没有 C ABI layout、callback 或 handle 兼容承诺。
将来只有已命名的 foreign-runtime consumer 通过全局
[Native FFI 准入策略](native-ffi-policy.md)后，才按任务另行设计窄的 runtime 类型擦除边界。

HYLE 的底层模块**不应为了证明 obs 被使用而主动打日常日志**：库无法替应用决定
日志策略，非请求输出也会污染宿主进程。下游利用发生在 trace hook；error/status
仍是机器可读失败通道，log 是调用方主动选择的诊断通道。

## 4. 数值指标的数据面

`metrics.h` 提供两种明确的 ownership 档位，而不是以 `fast/light` 命名两套含糊 API：

- `Counter`、`Gauge`、`FixedHistogram` 由一个 scheduler/worker owner 更新；它们是普通 scalar/array，
  不分配、不加锁、不读钟。多个 worker 各自保存一份后，用 `HistogramSnapshot::merge` 在 scrape 或
  iteration 边界合并，是默认的低争用路线；
- `ConcurrentCounter`、`ConcurrentGauge`、`ConcurrentFixedHistogram` 只用于多个 worker 必须立即共享
  同一状态的场景。它们要求所选原子类型 always-lock-free，并只使用 relaxed atomic；这保证指标更新
  不会被误认为 application synchronization。

固定桶在 `FixedBuckets::create` 时一次性验证为有限、严格递增。每个 sample 落入第一个大于等于它的
有限上界，另有隐式 `+Inf` tail；update 用二分查找，复杂度 O(log B)，snapshot/merge/桶上界分位数
查询为 O(B)。`quantile_upper_bound` 返回包含目标 rank 的桶上界，它不是精确样本 quantile，也不是
近似 sketch。并发 snapshot 在 writer 活跃时是弱一致的，在 writer quiescence 后精确；reset、move 和
destruction 都是 caller 管理的 quiescent 生命周期操作。

这些类型不保存 metric name、label、timestamp、registry membership 或 exporter pointer。宿主把自然
C++ member 与自己的 schema 绑定，在 scrape 时读取/合并 snapshot，再编码 Prometheus、OpenTelemetry
或其他格式。这样 producer 热路不支付 string/hash/cardinality/clock 成本，也不把服务框架策略固化进
基础数值类型。

## 5. 依赖、消费者与负空间

`hyle_obs` 在中央 component graph 中 `PUBLIC` 依赖 `core`、`error` 与 `time`：core 提供 compiler、fatal
invariant 与 visibility 基础，sink 注册的可恢复失败使用 error 的 `Status/Result`，公开 record 直接
命名 `time::UnixTimestamp`。它 `PRIVATE` 依赖 `mem_reclaim`，只复用 registry 短 snapshot grace period；
该实现依赖不泄漏到公开 header。默认
stderr/debugger writer 直接拥有 TTY 和 debugger OS 交互，因为
platform 没有此抽象、obs 与 platform 同处底层且这正是 sink 的 I/O 职责，不能为消除一个
局部 syscall 制造 `obs -> platform` 依赖。trace gate 是 header/CMake 横切 seam，不让
mem/sched/io/compute 反向链接 log runtime。
typed C++ sink 只是借用调用方 backend，registry 也使用固定进程期存储，因此无需依赖完整 Mem 产品或
分配器；异步拥有型复制与队列由 application adapter 自己选择 allocator。区间控制仍使用标准
`steady_clock` 类型并提供 caller-supplied monotonic-ns seam；Time 依赖来自公开 wall timestamp value，
不是让 Obs 自动读钟。推理引擎可以把同域的 `monotonic_now()` 或 `coarse_monotonic_ns()` 传入显式
limiter，但普通 message path 不读取 wall/monotonic clock，也不自动捕获 thread/context。
指标 header 使用 Core 的 invariant/compiler 合同与 Error 的可恢复构造/merge 结果；标准 array、atomic
与 binary search 已是同合同最小机制，不为复用而增加 Container、Mem、Platform、SIMD 或新的 Time
依赖。Obs 既有 log surface 已经需要公开 Time，因此中央 component edge 不变。

语法消费者扫描没有发现 HYLE 生产模块调用 runtime `HYLE_LOG_*` 或注册 `LogSink`；这是
基础库不擅自输出日志的正确负空间。语义扫描中的 util fatal diagnostic、core fatal macro
属于不可恢复合同失败，io benchmark stats 只属于显式 benchmark 输出，均不应迁移成普通
log。trace 的真实消费者仍是 mem/sched/io/compute。

与 Abseil/Folly 对账后保留同步借用 record、同步 handler/flush、递归抑制、进程期 registry、作用域
注册与调用点采样，并补齐 logger-neutral category/event、typed scalar、correlation、显式 source 与
typed timestamp。当前 llama.cpp server 的 model-load state callback 使用三个 stage-owned throttle，
按首条、每 200 ms 与末条提交 `stages/current/value`；slot 日志则仍把 task/rate/queue/status 拼进文本。
两者共同证明正常 product shape 是一个宿主 adapter 加对象持有的 interval state，不是 4/16 个独立
原子 sink，也给出上述 typed fields 的真实来源。合成的固定 slot/tick replay 只能作机制诊断，不得
冒充某个版本 llama.cpp 的精确消费。没有真实消费者支持 category 配置树、per-sink
filter policy、异步线程/队列、文件轮转、上传、聚合/查询/告警、fatal 控制流、owning async record、
自动 task context 或 profiler viewer。这些能力应在 adapter/application 层组合，不给所有基础 record
添加 thread ID、refcount 或拥有型字符串。

## 6. Trace 双面与零成本边界

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

## 7. 维护硬门

改日志 registry 时必须通过：

1. 正常 C++ 测试 + 编译期 floor codegen；
2. add/remove/log 并发压力与 remove-waits、自摘、跨摘拒绝测试；
3. WSL ASan、UBSan、TSan；
4. `closure-perf-v1` 在登记裁决机上的冻结正常 workload；其他平台做 correctness/交付，不池化点值；
5. Native FFI 保持 `N/A`；若出现已命名的 foreign-runtime 任务，先按全局准入策略独立立项；
6. installed source/shared/static package 中只请求 `COMPONENTS obs` 的 C++23 consumer。

改 trace/CMake 或新增下游 hook 时必须通过：

1. OFF 构建参数不求值测试，且产品库不依赖 Tracy；
2. ON 构建 obs + 被埋点模块测试，并用超长 message/zone text/source components 验证截断；
3. 共享构建的每个模块只依赖同一个 `hyle_tracy_client`；
4. 名称和 zone token 的存储期/配对审计；
5. Windows、Linux、macOS 各跑适用的 ON correctness/build；性能只按登记机器合同裁决。

改 metrics 时必须通过：

1. bounds、tail、multiplicity、sum/count、merge、quantile、reset 与无效输入 correctness；
2. owner-thread 和 concurrent 两个档位分别验证，并发 writer 停止后的 snapshot 必须精确；
3. P0 复核无 allocation/mutex/clock/syscall、复杂度不退化；P1 只与相同 ownership、原子序列、输入验证、
   runtime bounds 和计费范围的直接机制比较，不拿 owner-thread 路线打 concurrent peer；
4. selected Obs 的 source/static/shared consumer 都能直接包含 `metrics.h`，且不产生 Native FFI/C symbol；
5. metric schema、label、exporter 与时间窗口保持宿主所有，不因 benchmark 或某个 serving 框架进入类型。

trace-on 自动测试只需要本机 listener，CTest 因而给 Obs test process 设置 Tracy 上游的 runtime
`TRACY_ONLY_LOCALHOST=1`。该变量不写入产品 target，也不改变普通 trace-on 程序的网络策略。锁定
Tracy revision 的 runtime 开关只限制 listener；compile-time 同名宏才同时改变 discovery broadcast
地址，所以维护文档不能把测试环境变量描述为“完全禁网”。具体 revision 属外部依赖账本。

当前 closure record 与工程性能证据从
[`implementation/status`](../implementation/status.md) 进入；历史 candidate、task 与数字不在架构页
重复。
