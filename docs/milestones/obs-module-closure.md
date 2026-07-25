# obs 模块严格收口（2026-07-25）

obs 已按“实际使用而非 benchmark 形状”的当前合同完成复审。它是**实现体量小、横向影响面
大**的底层模块：产品面只有同步结构化日志 producer 与编译期 trace seam；生命周期、
C ABI、构建后端和 mem/sched/io/compute 埋点会穿过全库。

稳定设计见 [`../architecture/obs-design.md`](../architecture/obs-design.md)，用户选型见
[`../guide/obs.md`](../guide/obs.md)，C/FFI 精确映射见
[`../guide/c-obs.md`](../guide/c-obs.md)，性能点值只见
[`../performance/benchmarks-obs.md`](../performance/benchmarks-obs.md)。

## 1. 功能面与负空间

| 真实场景 | 最终能力 / 选择 |
|---|---|
| 机器可读失败、重试、传播 | 继续使用 error 的 `Status` / `Result`；日志不能替代控制流 |
| 普通诊断文本 | level-gated `HYLE_LOG_*`；未通过时参数不求值 |
| 已格式化或结构化诊断 | borrowed `LogRecord`：message、fields、source、可选 timestamp |
| 接已有 logger/队列/文件/遥测 | 同步 `LogSink{state, write, flush}` adapter；callback 内复制后可异步 |
| 安全动态卸载 sink | quiescent unregister；返回后其他线程不再使用旧 callback/state |
| profiling 时间线 | OFF 编译期消失、ON 汇入一个 Tracy client 的 native trace seam |
| C / FFI | POD record、opaque sink、共享 `hyle_status`、runtime trace seam |

明确不做：category 配置树、per-sink routing policy、异步线程/队列、文件滚动、上传、聚合、
查询、告警和 profiler viewer。已有 Abseil/Folly/spdlog 的应用应注册一个 adapter sink，
而不是让 HYLE 再造第二套日志框架。category 若只是业务字段可直接作为 record field；只有
真实消费者证明需要 producer-side routing 才重开。GPU timeline 也应由 compute 的真实
后端合同驱动，不先投机增加 obs API。

## 2. 成熟库对账与本轮架构修正

| 参考 | 借鉴 | 没有照搬的部分 |
|---|---|---|
| Abseil `LogSink` | 同步 sink、Flush、registered-sink recursion suppression、process-lifetime registry | 完整 LogEntry/config/category 元数据不属于轻量底层 seam |
| Folly logging | 同步 handler + flush、默认终端控制字符防护、应用层 category 思路 | async writer/category tree 留给现有应用 logger |
| Tracy | 编译期 no-op 宏、单一 backend client、zone/counter/memory/fiber 形状 | 不自造 viewer/protocol |

本轮发现旧 registry 为 producer 复制 snapshot 时持 mutation mutex；它虽然通过旧 benchmark，
却会把多 producer 正常路径串在控制面锁上。最终改为两份原地 immutable snapshot：mutation
串行发布，producer 用短 reader epoch 复制已发布代并取得 per-slot active reference。emit
不分配、不拿 mutation mutex；remove 发布新代后排空旧 reader/callback，保留可证明的
quiescent unregister。

其他修正：

- registry 改为 first-use、永不析构，晚期静态析构日志/查询不触碰已销毁 mutex；
- 高频 add/remove/log 重复压力发现独立 active/waiting 原子有极低概率 missed wake，且
  外部 remover 与 last releaser 都收尾会让旧 helper 跨入固定槽下一代；active 改为
  waiter-bit + refcount 单状态字，自摘才由本 callback 延迟收尾。snapshot 也用
  reader-count + 多 waiter-count 单状态字，release 同一次 RMW 决定唤醒；
- registered sink 内递归日志不再重入 sink 图；默认 stderr 开启时只直落 default，否则
  正常返回未投递；flush 使用同一抑制和生命周期合同；
- 默认 stderr 把换行、TAB、ESC 与 ASCII control 转义为可见文本，一条 record 保持一个
  物理行，避免日志/终端控制注入；
- 1 KiB format 缓冲显式以 `...[truncated]` 结尾，不再静默截断；
- `LogSink` 增加可选 flush；add/remove/default-control 改为共享 `Status`，保留
  InvalidArgument/AlreadyExists/NotFound/ResourceExhausted/FailedPrecondition 原因；
- obs runtime target 正确公开依赖 error；HYLE C++ 内部仍直接走 native API，不绕 C ABI。

## 3. C++ 与 C ABI 易用性

C++ 默认路径是 `HYLE_LOG_*`，已有文本/字段走 `log_message`/`log_record`，接宿主日志走
`LogSink`，profiling 走 `HYLE_TRACE_*`。guide 明确 borrowed lifetime、callback 并发、
自摘/跨摘、format 截断和 trace name 存储期，不要求用户从 benchmark 反推用法。

C ABI 删除未经审查的 obs 私有 TLS last-error，全部可恢复失败统一返回 owning
`hyle_status`，out parameter 在工作前清空；gate/no-sink 是 OK + `out_emitted=0`。record
保持纯 C POD，sink 为 opaque ownership handle，并提供 write/optional flush。23 个
`extern "C"` 符号通过 declaration 清单、纯 C reference 与 shared DLL export 三重机械
核对；header-inline 五个 level helper 复用同一 Status seam。首发前仍可调整 ABI，这里认证
的是能力完整和形状易用，不虚构兼容承诺。

## 4. 正确性、依赖与三平台

| 闸门 | 结果 |
|---|---|
| Windows MSVC Release | obs header/C++/pure-C/C-surface 4/4 |
| WSL2 GCC 14 Release | 同组 4/4 |
| Apple M4 AppleClang Release | 同组 4/4 |
| WSL ASan / UBSan | 同组各 4/4 |
| WSL TSan | 27 个 C++ case + pure-C 通过；并发 churn 额外 100 轮；WSL 地址布局需 `setarch x86_64 -R` |
| retirement/reuse 重复压力 | Windows、M4 各 500 轮；修前约几十轮可复现 hang，修后通过 |
| Trace ON，真实消费者 | Windows/WSL/M4 均构建 obs+mem+sched+io+compute，并通过 obs 4/4 |
| C surface | 23/23 declaration、纯 C reference、DLL export |

Trace ON 三平台仍由一个 `hyle_tracy_client` target 承载。Windows 对所有受影响组件打开
`HYLE_WERROR=ON` 会在本轮未修改的 SIMD 代码触发既有 unused/unreachable warning；关闭
WERROR 后 obs 与全部真实 trace 消费者构建/测试通过。该 warning debt 归 SIMD 严格复审，
不通过顺手改未审模块来伪装 obs 结论。

依赖图上 obs runtime 只新增正确的 `obs -> error`；native trace 是 header/CMake 横切 seam，
不是让 error/platform/mem/time 反向二进制依赖 obs。基础库内部不擅自输出日常 log 是边界，
不是 adoption 漏洞；mem/sched/io/compute 使用 trace 才是正确的内部消费。

## 5. 性能与 SOTA 判定

三平台 Release、15 个独立运行、中位数、sink 无 I/O：

| 场景（ns/op） | Windows | WSL2 | M4 utility |
|---|---:|---:|---:|
| gated macro | 2.007 | 0.916 | 0.906 |
| native preformatted，1 sink | 39.328 | 34.488 | 25.001 |
| structured 4 fields | 38.690 | 34.464 | 24.267 |
| `std::format` 宏 | 92.150 | 85.761 | 64.356 |
| C Status seam | 41.406 | 37.020 | 27.237 |
| 8 producer，per emit | 95.171 | 64.011 | 123.902 |

复杂度为 gate O(1)、emit/flush O(S)、mutation O(S≤64)。Windows 同 workload 的旧
exclusive-mutex A/B：单 producer约 43.9 ns、8 producer约 205 ns；immutable 版约
40 ns/92–100 ns，多 producer 改进约 2 倍且不改变公开 API。最终 wait-state 修正后 8
producer 为 95.2 ns，仍在同档。M4 同时跑 interactive/
utility，因无硬 P/E affinity 支持只报告 QoS；短 normal 行即使 utility 仍可高波动，不能把
heterogeneous scheduling 噪声写成核心性能。

Abseil 可选行约 439 ns gated、892 ns stream+ToSinkOnly，但它承载完整 LogEntry/config
元数据，不是同合同胜负。当前认证的是：算法复杂度与成熟同步 registry 同档、正常路径无
已知锁/分配缺口、C seam 只多约 2–3 ns；没有同 metadata/ownership 合同的 peer 时不宣称
“几十倍 SOTA”。

## 6. 严格收口结论与重开条件

obs 的功能面、复杂度、C++/C 形状、依赖、guide 和三平台证据已经满足当前严格合同，可以
停止反复重构。只有以下证据可重开：

- 真实应用需要 producer-side category/per-sink filtering，且 adapter field 不够；
- 真正的 GPU/backend profiling 消费者要求现有 trace seam 无法表达的事件；
- 同 quiescent ownership/metadata 合同的成熟 peer 证明复杂度档或正常路径存在实质差距；
- correctness/security/ABI、平台或 Tracy backend 变化破坏当前合同。

异步日志系统、运维能力、纯粹为了赢 benchmark 的 record 元数据或亚纳秒波动不能单独重开。
