# Obs：同步记录、数值指标与编译期追踪

Obs 提供安全产生观测数据的机制，不拥有服务的日志策略、后台队列、exporter 或告警系统。它分为
独立的 log、metrics 与 trace 三条路；使用示例见 [Obs 指南](../user/obs.md)。

| 路线 | 所有权与成本 | 外部宿主承担 |
|---|---|---|
| log | level-gated、同步借用 record、同步 fan-out | 异步复制/队列、轮转、上传、持久化 |
| metrics | caller-owned counter/gauge/fixed histogram | 名字、labels、registry、scrape/export |
| trace | 默认编译消失，开启后使用同一 Tracy backend | profiler 部署、收集与可视化 |

## 同步借用 record 不成为拥有型事件框架

LogRecord 的 category、event、message、source 和字符串值使用 exact string_view，属性与 correlation
使用独立 span，保序并允许重复 key。LogValue 是标量 tagged value，不引入每条 variant allocation。
wall timestamp 可选并复用 Time 的 UnixTimestamp，absent、epoch 0 与 pre-epoch 保持区别。

调用者保证全部 backing 活过同步 dispatch；sink 若要异步保存就自行复制。传入 record 不自动抓取
thread/task context 或读取时钟。owning string 只接受可安全借用的形状，不能把临时字符串交给异步宿主
后假装原 record 负责寿命。

数据流是 level gate、取得 published sink snapshot/active references、同步调用、释放 references。
emit 不持 registry mutation mutex，不分配 sink list；慢 callback 的时间仍由 producer 支付。
“同步成功”只表示 callback 返回，不代表日志已落盘。

## 为什么 snapshot grace period 与 callback drain 分开

registry 使用固定槽和两代 immutable snapshot。producer 在 Mem ReclamationDomain 的短 read guard
内复制 snapshot 并取得每个 sink 的 active reference，随后退出 guard，再执行任意 callback。
writer 在复用旧 snapshot 前等待 grace period；remove 发布新 snapshot 后再次关闭 reader→active-ref
handoff，最后 drain 目标 callback。

guard 不包住任意 sink 执行，避免一个 stalled callback 绑住整个 reader domain；callback lifetime
由 active count 独立覆盖。remove 成功返回后，其他线程不再执行旧 callback，宿主才能释放其 state。
SinkRegistration 是 move-only 控制面 RAII membership，帮助早退/异常时正确注销，不进入每条 record。

callback 可自摘，但自己的 state 必须活到当前 callback 返回。callback 内跨摘其他 sink 会被拒绝：
目标可能已被同一 snapshot 取得 active reference 却尚未调用，等待它会等待当前线程自己。

callback 在 producer OS thread 同步执行，必须 noexcept 并在同一线程返回。递归跟踪使用 OS TLS，
因此 callback 不能主动 yield/suspend stackful fiber 或迁移 OS thread；普通 OS 抢占不违反该合同。
异步 adapter 应复制、enqueue 然后立即返回，不能让底层 emit 为所有用户支付 fiber 跟踪成本。

## 递归、默认 sink 与进程寿命

sink 内再次 log 不重入注册 sink 图：default stderr 启用时直接交给它，否则报告没有 sink 收到。
flush_sinks 使用相同递归抑制与 active-reference 生命周期；flush 的持久性由各 backend 定义。

registry 首次真正使用时初始化且不参与静态析构，避免较晚析构的全局对象访问已销毁 mutex。
这不使动态卸载的 callback 代码永久有效，宿主仍须先注销并 drain。

默认 stderr sink 串行化一条 record 的多个片段，仅 TTY 输出 ANSI 色码。用户文本里的换行、NUL、ESC
等控制字符转义为可见形式，防止注入额外物理行或终端控制。标量固定栈缓冲渲染；有限 debugger
mirror 对超长输入明确标记 truncated，不伪称完整输出。

## 宏门控与限流状态

每 TU 可设 `HYLE_OBS_COMPILED_MIN_LEVEL`，低于 floor 的调用连参数求值一起消失。保留的宏再做一次
runtime gate，未通过不 format、不计算参数；通过后使用有界栈缓冲并附 source，超长文本显式截断。
完整大 payload 可直接借用 LogRecord，由宿主管理拥有型存储。

runtime level 是 eventual configuration，已经通过 gate 的 record 可以完成当前 dispatch。
编译期 floor 不改变 Obs binary ABI，也不是进程全局开关。

调用点 EVERY_N/EVERY_POW_2 使用低争用近似计数；明确 exact 入口才支付 RMW，FIRST_N 通过 CAS
维持并发总量上限。竞争下不承诺 wait-free 或严格有界重试。请求/设备等对象可显式持有
LogIntervalLimiter，并传入共用 monotonic tick，避免每个 slot 自行读钟。reset/move/destruction 是
所属 owner 的生命周期操作，不是隐藏全局 registry policy。

## 数值指标先按 owner 分型

Counter、Gauge、FixedHistogram 由单个 owner 更新，不分配、不读钟、不加锁。多个 worker 各自保留
一份，在 scrape/iteration 边界合并，是默认低争用路线。

ConcurrentCounter/Gauge/FixedHistogram 用于确需共享的数值，要求对应 atomic always-lock-free，
采用 relaxed 操作。指标更新不是应用数据同步 fence；不能因观察到 counter 变化就读取未发布 payload。

固定桶在创建时验证为有限、严格递增；sample 进入第一个大于等于它的上界，并有隐式 +Inf tail。
update O(log B)，snapshot/merge/桶上界 quantile 为 O(B)。quantile_upper_bound 返回桶上界，不是
精确样本 quantile 或 sketch。并发 histogram snapshot 在 writer 活跃时弱一致，静止后才精确；
多字段 reset 也不能伪装成与 update 原子隔离的事务。

指标不保存 name、label、timestamp、exporter pointer 或 registry membership。宿主将自然 C++ member
绑定自己的 schema，scrape 时合并并编码。这样热更新不承担字符串 hash、cardinality 或服务框架成本。

## Trace 关闭时消失，开启时仍有真实寿命

HYLE_TRACE 默认 OFF：参数不求值，无 backend 调用与依赖。开启后，native macros 将 zone、plot、
fiber/frame、message、memory、thread-name 等转给 Tracy，不再提供另一层常态 runtime no-op bridge。

共享 Hyle 各模块必须链接同一个 shared Tracy client；若多个 plugin 各自静态链接 trace-on SDK，
各 image 会有各自 singleton，不能假装全进程唯一。静态路线只有进入同一最终 link image 才自然
汇合成一份 backend；跨构建的 plugin 拓扑不能靠单个 CMake target graph 自动证明。

复制 text 的长度先在 adapter 做有界处理，避免 backend 有限长度与实际扫描/分配不一致。静态 name、
frame/plot/pool 标识和 fiber 名还要满足 Tracy 对存储期的要求，不能交临时指针。trace-on 不意味着
free-for-all 的生命周期；name、zone 配对和 callback 代码仍各有 owner。

## 依赖与扩展边界

Obs 公开依赖 Core/Error/Time，来自 compiler/invariant、sink 注册失败和 UnixTimestamp；私有使用
mem_reclaim 保护短 snapshot，不把完整 Mem 或 Container 类型带进 record。stderr/debugger 输出是
sink 自身职责，不为一个局部 syscall 新建第二套平台抽象。trace 是跨切 seam，其他模块使用它不
等于必须反向链接 log runtime。

底层库不为证明 Obs 被消费而主动打印日常日志。机器失败通过 Status/Result，fatal 通过 Core，
人读日志由真正知道政策的调用者选择。异步日志、文件轮转、context 自动传播、指标 schema/exporter
与 profiler viewer 在宿主层组合；当前交付 C++ source/static/shared，无额外 C ABI。

修改时重点核对 remove 的 quiescence、自摘/跨摘、递归、借用 backing、trace OFF 消失与 shared
backend 唯一性。数值路线比较相同 ownership/atomic/桶合同，不能用单 owner 值击败 concurrent
实现后宣称同等。完整接口见[公开头](../../include/hyle/obs/)。
