# obs-runtime 模块严格复证（2026-07-29）

> **历史证据提示（2026-08-25）：**本页的绿色结论绑定文中日期/commit，不是 Tracy 0.14.1
> 与新 C sink/record 合同后 working tree 的当前证书。现行状态只看
> [`pre-release-readiness`](../../status.md)。

obs-runtime 已按当前 [`module-closure-methodology`](../../../developer/review/module-closure-methodology.md)
重新打开并完成复证。runtime/API 实现绑定 `4d31bcf`，性能采样合同绑定 `06cec5a`；它是
**实现体量小、横向影响面
大**的底层模块：产品面只有同步结构化日志 producer 与编译期 trace seam；生命周期、
C ABI、构建后端和 mem/sched/io/compute 埋点会穿过全库。

稳定设计见 [`../architecture/obs-design.md`](../../../architecture/obs-design.md)，用户选型见
[`../guide/obs.md`](../../../user/obs.md)，C/FFI 精确映射见
[`../guide/c-obs.md`](../../../architecture/native-ffi-policy.md)，性能点值只见
[`../performance/benchmarks-obs.md`](../../evidence/performance/benchmarks-obs.md)。

## 0. 审查单元与消费者账本

| 项 | 本轮边界 | 结论 |
|---|---|---|
| C++ runtime surface | `hyle/obs/log.h`、`src/obs/log.cc`、`hyle_obs` target | 纳入；默认/高级/组合三条路径复写 |
| C / FFI surface | `hyle/c/obs.h`、`c_obs.cc`、24-symbol manifest | 纳入；纯 C、DLL/.so、shared/static installed consumer |
| trace seam | `trace.h`、Tracy target、mem/sched/io/compute hooks | 边界回归；本轮未改算法/后端，2026-07-25 证据保留 |
| 上游 | error / OS default-sink calls | `obs -> error` 正确；TTY/debugger I/O 归 sink，不制造 `obs -> platform` |
| 下游 | HYLE 生产模块、samples、tests、bench | runtime log 是用户 product root；生产模块没有被迫打日志 |

语法扫描覆盖 `HYLE_LOG_*`、`LogSink`、C ABI 调用、target link 与 trace hooks。生产代码中没有
runtime log 调用；trace 的真实下游仍是 mem/sched/io/compute。语义重复扫描另查了直接
stderr/printf、fatal macro、局部 status/logger：util 的 fatal diagnostic 与 core fatal macro
属于合同失败，io 的 stats 是显式 benchmark 输出，都不是绕过 obs。没有发现下游复制一套
普通 runtime logger，也没有错误地从 HYLE C++ 内部绕经 C ABI。

外部用户任务重新写为：普通格式化诊断、已有 byte slice、结构化借用 record、作用域 adapter
sink、进程期高级 registry、C/FFI checked emit、profiling trace。每个公开入口均能映射到其中
一项；category hierarchy、异步 writer、rotation/upload/query、owning async record 和 profiler
viewer 没有真实基础库消费者，继续作为负空间。

## 1. 功能面与负空间

| 真实场景 | 最终能力 / 选择 |
|---|---|
| 机器可读失败、重试、传播 | 继续使用 error 的 `Status` / `Result`；日志不能替代控制流 |
| 普通诊断文本 | level-gated `HYLE_LOG_*`；未通过时参数不求值 |
| 已格式化或结构化诊断 | exact-byte borrowed `LogRecord`：message、fields、source、可选 timestamp |
| 接已有 logger/队列/文件/遥测 | `SinkRegistration` RAII + 同步 `LogSink` adapter；callback 内复制后可异步 |
| 安全动态卸载 sink | quiescent unregister；返回后其他线程不再使用旧 callback/state |
| profiling 时间线 | OFF 编译期消失、ON 汇入一个 Tracy client 的 native trace seam |
| C / FFI | POD record、opaque sink、共享 `hyle_status`、runtime trace seam |
| 发布构建删除低等级调用 | per-TU `HYLE_OBS_COMPILED_MIN_LEVEL`；裁剪后参数和 obs 符号引用均消失 |

明确不做：category 配置树、per-sink routing policy、异步线程/队列、文件滚动、上传、聚合、
查询、告警和 profiler viewer。已有 Abseil/Folly/spdlog 的应用应注册一个 adapter sink，
而不是让 HYLE 再造第二套日志框架。category 若只是业务字段可直接作为 record field；只有
真实消费者证明需要 producer-side routing 才重开。GPU timeline 也应由 compute 的真实
后端合同驱动，不先投机增加 obs API。

## 2. 成熟库对账与本轮架构修正

| 参考 | 借鉴 | 没有照搬的部分 |
|---|---|---|
| Abseil `LogSink` | 同步 sink、Flush、registered-sink recursion suppression、process-lifetime registry | 完整 LogEntry/config/category 元数据不属于轻量底层 seam |
| Folly logging | 同步 handler + flush、长度感知 LogMessage、默认终端控制字符防护、应用层 category 思路 | async writer/category tree 留给现有应用 logger |
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

2026-07-29 重开轮另发现并关闭六项消费者合同缺口：

- C++ 注册原来只能手工 add/remove，early return 容易留下悬空 callback；新增 move-only
  `SinkRegistration`，析构执行 quiescent unregister，显式 reset 失败时保留 ownership；
- message 从 NUL pointer 改为 `string_view` / exact byte count，slice 不复制不扫描并保留内嵌
  NUL；NUL convenience 的 O(M) 扫描被明确登记；
- format macro 从 gate 后再进入 `log_record` 的二次 atomic load 改为一次 runtime gate；
- C++/C 都补 per-TU compile floor，裁剪等级不求值参数；C 仍返回可组合的 OK Status literal；
- C 新增 `_n` 动态入口和五个 `_n` inline helper，补齐 length-aware FFI；动态 surface 23→24；
- Windows debugger mirror 原来只写 message、1 KiB 静默截断且未转义；现与 stderr 同逻辑
  record、控制字符安全，固定 4 KiB 并显式 truncation marker。

这些修正没有给每条 record 添加 category、owner map、refcount、TLS cache 或诊断统计；新增
RAII 只存在于控制面。registry 数据结构与复杂度没有为 benchmark 改形状。

## 3. C++ 与 C ABI 易用性

C++ 默认路径是 `HYLE_LOG_*`，已有文本/字段走 `log_message`/`log_record`，接宿主日志默认走
`SinkRegistration + LogSink`，profiling 走 `HYLE_TRACE_*`。guide 明确 borrowed lifetime、callback 并发、
自摘/跨摘、format 截断和 trace name 存储期，不要求用户从 benchmark 反推用法。

C ABI 删除未经审查的 obs 私有 TLS last-error，全部可恢复失败统一返回 owning
`hyle_status`，out parameter 在工作前清空；gate/no-sink 是 OK + `out_emitted=0`。record
保持纯 C POD，sink 为 opaque ownership handle，并提供 write/optional flush。24 个
`extern "C"` 符号通过 declaration 清单、纯 C reference 与 shared DLL export 三重机械
核对；10 个 header-inline level helper 与 5 个 compile-floor 宏复用同一 Status seam。
首发前仍可调整 ABI；这里认证的是能力完整和形状易用，不虚构兼容承诺。

## 4. 正确性、依赖与三平台

| 闸门 | 结果 |
|---|---|
| Windows MSVC 19.50 Release | error+obs 9/9；obs 自身 5/5 |
| WSL2 GCC 14.2 Release | error+obs 9/9；obs 自身 5/5 |
| Apple M4 AppleClang 21 Release | error+obs 9/9；obs 自身 5/5 |
| WSL ASan / UBSan / TSan | 三种配置各 9/9；TSan 用 `setarch x86_64 -R` |
| retirement/reuse 重复压力 | Windows、M4 各 500 轮；修前约几十轮可复现 hang，修后通过 |
| Trace ON，真实消费者 | Windows/WSL/M4 均构建 obs+mem+sched+io+compute，并通过 obs 4/4 |
| C surface | 24/24 declaration、纯 C reference、DLL/.so export |
| installed package | Windows/WSL/M4 的 shared + static；obs-only C 与 C++ probe 各 1/1 |
| compile floor codegen | 三平台参数不求值；dumpbin/nm 无被裁剪 obs 未定义符号 |

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
| runtime gated macro | 2.160 | 0.910 | 1.100 |
| native preformatted view，1 sink | 42.070 | 38.970 | 23.910 |
| structured 4 fields | 38.450 | 34.500 | 23.560 |
| `std::format` 宏 | 89.190 | 87.260 | 63.210 |
| C NUL / exact-length Status seam | 44.610 / 44.430 | 40.230 / 39.790 | 26.790 / 27.090 |
| 8 producer，per emit | 93.690 | 62.950 | 124.390 |

复杂度为编译裁剪 O(0)、runtime gate O(1)、exact-length emit/flush O(S)、NUL convenience
O(M+S)、mutation O(S≤64)。Windows 同 workload 的旧
exclusive-mutex A/B：单 producer约 43.9 ns、8 producer约 205 ns；immutable 版约
40 ns/92–100 ns，多 producer 改进约 2 倍且不改变公开 API。最终 wait-state 修正后 8
producer 当前为 93.7 ns，仍在同档。M4 同时跑 interactive/utility，因无硬 P/E affinity
支持只报告 QoS；normal workload 加长 epoch 后 spread 0.5–5.1%，而约 1 ns gate 仍会被异构
调度放大到 63.6% spread，因此只用复杂度/codegen 认证 gate，不把最好点写成核心性能。

Abseil 历史可选行约 439 ns gated、892 ns stream+ToSinkOnly，但它承载完整 LogEntry/config
元数据，不是同合同胜负。当前认证的是：算法复杂度与成熟同步 registry 同档、正常路径无
已知锁/分配缺口；没有同 metadata/ownership 合同的 peer 时不宣称
“几十倍 SOTA”。

## 6. 严格收口结论与重开条件

obs 的功能面、复杂度、C++/C 形状、依赖、guide 和三平台证据已经满足当前严格合同，可以
停止反复重构。只有以下证据可重开：

- 真实应用需要 producer-side category/per-sink filtering，且 adapter field 不够；
- 真正的 GPU/backend profiling 消费者要求现有 trace seam 无法表达的事件；
- 同 quiescent ownership/metadata 合同的成熟 peer 证明复杂度档或正常路径存在实质差距；
- correctness/security/ABI、平台或 Tracy backend 变化破坏当前合同。

异步日志系统、运维能力、纯粹为了赢 benchmark 的 record 元数据或亚纳秒波动不能单独重开。

## 7. 本轮 LOC

固定 `cloc` 口径（实现提交 `06cec5a` 工作树）：

| 口径 | code 行 |
|---|---:|
| 全库 tracked（含 docs / in-tree third-party） | 277,692 |
| HYLE 自实现含测试（`include/hyle + src + tests`） | 135,321 |
| HYLE 自实现不含测试（`include/hyle + src`） | 92,891 |
| obs 当前实现（`include/hyle/obs + include/hyle/c/obs.h + src/obs`） | 1,250 |
