# obs 设计：同步日志产出 + 编译期追踪

> **V2 状态：`NOT_REVIEWED`。** 本页暂作既有设计输入，不代表当前功能、性能或 API 证书。
> 当前 Native FFI 为 `N/A`；正文旧 C/FFI surface 与验收叙事待 obs 轮次重写。

> **Native FFI 状态校正（2026-08-25）**：HYLE 没有 native-C 客户，也没有已发布的 managed
> binding。本文关于 C/C++、C/FFI 或 C 用户的旧表述只记录待复审的 ABI 候选，不是当前跨语言
> 证书或兼容承诺；首发前可破坏性调整。先从真实 foreign-runtime 任务裁决 FFI 是否为 `N/A`；
> 只有保留 direct FFI 时，obs 才必须通过 Java FFM + .NET P/Invoke、真实导出、sink/callback
> 生命周期与 batch 旅程后冻结该边界。

obs 有两条相互独立的产出通路：

- **log** 面向运行中长期存在的诊断记录。它是同步、level-gated、结构化 fan-out；
- **trace** 面向 profiling build 的纳秒级时间线。它默认编译期消失，开启时接 Tracy。

HYLE 不实现日志文件滚动、异步上传、聚合、查询、告警或 profiler viewer。这些是
sink/运维系统的职责；obs 只提供基础库应有的安全产出接缝。

## 1. 总体架构

```text
                  同工具链 / 源码用户             foreign-runtime 任务候选
               include/hyle/obs/*.h             include/hyle/c/obs.h（未认证）
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
     default stderr   native C++ sink   opaque ABI sink candidate      |
          |                               |                            |
          +---------- callback 外部消费 --+                            |
                                                                       |
                         HYLE_TRACE=ON                                  |
        native macros ----------------+----------- runtime ABI candidate --+
                                     |
                           one shared hyle_tracy_client
                                     |
                +----------+----------+---------+----------+
                |          |          |         |          |
               mem       sched       io      compute    foreign host
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

`LogRecord` / `hyle_obs_log_record` 的消息、字符串和字段都是同步借用。消息是
`std::string_view` / `pointer + exact byte count`，因此已有 slice 不必复制或扫描，也能保留
内嵌 NUL；field key/value 与可选 source 仍是 NUL 结尾字符串：

C raw record 的 `struct_size` 表示可读前缀。其 semantic required prefix 直接包含
level/message/message_size；record 不是带全套业务默认的 options，所以不增加会移动热路径
payload 的 reserved header word。公开 emit 边界用 `const void *record_bytes`，避免把真实短对象或
未对齐输入伪装成已满足当前布局的 typed pointer；普通 `&record` 仍可直接传入。适配器先建完整
默认 record，再用 `memcpy` 按完整字段读取；
fields/count 作为一个 pointer+count 单元，optional 短尾保持默认，未知未来尾部不访问。这样
真实短对象和未对齐 FFI bytes 不会因把它们解引用成当前完整 C struct 而越界或触发未对齐 UB。
调用方仍须让 `min(struct_size, sizeof(hyle_obs_log_record))` 字节真实可读；raw-byte 边界消除的是
typed-object/对齐假设，不可能替调用方验证一个虚报长度的裸指针。

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

C++ 默认用 move-only `SinkRegistration` 持有一次 registry membership。它的析构执行
quiescent unregister，消除 early return/异常路径遗漏 remove 后留下悬空 callback 的风险；
需要显式处理失败或 callback 内复杂注销时才直接使用 `add_sink/remove_sink`。RAII 对象只是
控制面所有权，不进入每条 record，也不给 sink/record 增加常态元数据。C 侧 opaque
`hyle_obs_sink` 有两种创建形状：`init` 在调用方提供的 queried size/alignment storage 中
构造且不分配，`create` 明确是 process C++ heap convenience；两者都用 destroy 表达同一
quiescent ownership。external storage 的 destroy 结束对象生命周期但不释放 storage。

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
`const char*` 便利重载和 C 的 `hyle_obs_log` 会先扫描 NUL，长度已知时用 view / `_n` 入口。

现有 `include/hyle/c/obs.h` 是待任务裁决的 C-syntax ABI 候选，不是发布面。HYLE 当前没有
native-C 产品或已发布 managed binding；先决定真实 foreign-runtime log/trace/sink 任务是否成立，
可以整块 `N/A`。若保留 direct FFI，下面只是进入 Java/.NET 复审的候选能力：

- POD record/field；
- opaque sink caller-storage init、system-heap create 与共同 destroy；
- 普通 `extern "C"` level、emit、registry 函数；
- `hyle_obs_log_n` 与五个 `_n` header-inline helper，避免 slice 扫描并保留内嵌 NUL；
- 五个 Status-expression `HYLE_OBS_EMIT_*` 宏，遵守同一个消费者编译期 floor；
- 原有五个 NUL 字符串 level 便利函数；
- 始终可链接的 runtime trace free functions，包括与 C++ hook 同合同的 logical-fiber
  enter/leave；稳定 pointer identity/lifetime 留在调用方，不穿 ABI 暴露 Tracy/scheduler 类型。

候选 ABI 不暴露 `std::format`、C++ reference、STL、Tracy 类型或 C++ 类布局。各语言
在本语言侧格式化后调用 `hyle_obs_log_n`，或构造 POD record。`LogField` 在 C++ 面复用
C POD 的两个 pointer 布局，只为 callback 零复制共享 record 字段；HYLE C++ 调用仍直接走
native implementation，不绕 C ABI。
跨导出边界按值传递并嵌入 record 的 `hyle_obs_log_level` 是明确的 `uint32_t` typedef，
命名 level 使用常量而非 ABI 宽度未定的 `typedef enum`；纯 C gate 以 `-fshort-enums`
反向证明编译器开关不会改变当前 schema layout，但不证明 managed binding 或兼容承诺。

可恢复失败统一返回 owning `hyle_status`：输入错误、重复/不存在、registry 满和 callback
内跨摘分别保留共享 error taxonomy；gate/no-sink 是 OK + `out_emitted=0`。obs 不再维护
私有 TLS `last_error`。

HYLE 的底层模块**不应为了证明 obs 被使用而主动打日常日志**：库无法替应用决定
日志策略，非请求输出也会污染宿主进程。下游利用发生在 trace hook；error/status
仍是机器可读失败通道，log 是调用方主动选择的诊断通道。

## 4. 依赖、消费者与负空间

`hyle_obs` 的唯一 HYLE runtime 上游是 `hyle_error`；sink 注册的可恢复失败使用共享
`Status/Result`。默认 stderr/debugger writer 直接拥有 TTY 和 debugger OS 交互，因为
platform 没有此抽象、obs 与 platform 同处底层且这正是 sink 的 I/O 职责，不能为消除一个
局部 syscall 制造 `obs -> platform` 依赖。trace gate 是 header/CMake 横切 seam，不让
mem/sched/io/compute 反向链接 log runtime。
opaque C sink 的 allocation-domain 控制由 size/alignment query + caller storage 提供，因此
无需为了可替换分配器引入 `obs -> mem`；保留的 `create` 是显式 system-heap convenience，
不应在受限/插件 ABI 边界伪装成调用方 allocator ownership。

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
- ON：zone/plot/counter/fiber/frame/message/memory/thread-name 转发 Tracy。

当前候选 runtime trace 函数在 OFF 时仍存在并做函数级 no-op；这只是现有实现事实。若真实
foreign-runtime 任务不成立，可以在首发前删除并记 `FFI=N/A`；若保留，它们有一次 ABI 调用，
不能宣传为“关闭态零指令”，并必须通过 Java/.NET direct smoke。极热 C++ 源码路径使用原生宏。
Tracy copied text 使用 16-bit byte count，且上游只在 debug assert 长度；adapter 必须先做
bounded scan。message/zone text 截到 65534 bytes，动态 source-location 的
name/file/function 各截到 21841 bytes，使 header + 三段 payload 总长始终可表示，不能在
release 把 full `strlen` 与 truncated allocation length 同时交给后端。

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

改日志 registry 或候选 ABI trampoline 时必须通过：

1. 正常 C++ 测试 + 编译期 floor codegen；纯 C fixture（含 caller-storage sink）只验证
   schema/compiler/export hygiene；
2. add/remove/log 并发压力与 remove-waits、自摘、跨摘拒绝测试；
3. WSL ASan、UBSan、TSan；
4. 三平台 Release gated/preformatted/formatted 基准；
5. 先裁决真实 foreign-runtime 任务；`N/A` 时记录理由且不制造 parity export；
6. 保留 direct FFI 时，动态库导出与 `include/hyle/c/obs.h`/fixture/guide 机械对账，并由 Java
   FFM + .NET P/Invoke 从精确 shared library 验证 sink/callback rooting、remove/drain 和 trace OFF/ON；
7. installed shared/static package 中只请求 `COMPONENTS obs` 的 C++ consumer；static 不是 managed
   FFI 载体。

改 trace/CMake 或新增下游 hook 时必须通过：

1. OFF 构建参数不求值测试，且产品库不依赖 Tracy；
2. ON 构建 obs + 被埋点模块测试，并用超长 message/zone text/source components 验证截断；
3. 共享构建的每个模块只依赖同一个 `hyle_tracy_client`；
4. 名称和 zone token 的存储期/配对审计；
5. Windows、Linux、macOS 至少各跑一次 ON 构建。

性能数与仪器见
[`../performance/benchmarks-obs.md`](../implementation/evidence/performance/benchmarks-obs.md)，
模块收口证据见
[`../milestones/obs-module-closure.md`](../implementation/tasks/obs/obs-module-closure.md)。
