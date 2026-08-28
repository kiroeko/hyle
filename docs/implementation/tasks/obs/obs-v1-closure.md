# obs-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`SCOPED`。本文是 obs 当前轮唯一 closure card 与施工记录；既有 obs task、architecture、
> guide、旧 C/FFI 叙事和历史 benchmark 只作审计输入，不继承旧“已收口”或 SOTA 结论。机械状态只看
> [`module-status.json`](../../module-status.json)。在 capability map 与有限 `FIX` 集进入 `AUDITED`
> 前，不修改 obs 产品实现。

## Closure card（`SCOPED`）

- component / owner / round ID：`obs` / observability owner / `obs-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`f305b9d3fed87a065d300145e5d9e8de3f5a77a0` /
  `ee1d875638ec95e2228d0b5bb54e5e3dc1c54369`
- 进入原因与 demand/defect ID：`OBS-CAP-01`（按基础库完整性标准重新枚举 log/trace 能力族）、
  `OBS-NORTHSTAR-01`（本地推理引擎的请求/算子诊断与 allocation/fiber/io/device-poller 时间线）、
  `OBS-API-01`（同步 borrowed record、sink owner、format/trace 便利面的自然 C++ 形状）、
  `OBS-LIFETIME-01`（并发 fan-out、递归、quiescent unregister 与进程 teardown）、
  `OBS-PERF-01`（旧点值未进入当前统一仪器且完整 Abseil/Folly record 不是同合同 peer）、
  `OBS-DOC-01`（architecture/guide 仍标 `NOT_REVIEWED`，旧 task 仍叙述已删除的 C ABI）、
  `OBS-DELIVERY-01`（Tracy 0.14.1 升级后的 OFF/ON 与 source/static/shared 复签）。
- 真实 consumer task / normal workload：
  - 推理引擎把已存在的 request/operator/backend 诊断按同步 borrowed record 交给一个宿主 adapter
    sink；adapter 在 callback 内决定复制/排队，obs 不拥有异步线程、文件或上传策略；
  - 多 worker 同时产出预格式化 record，控制面偶尔注册/注销 adapter；注销成功后 host 可以立即
    卸载 callback state；
  - mem allocation、sched fiber/timer、io completion 与 compute GPU-poller 的当前
    `HYLE_TRACE_*` 埋点在 profiling build 汇入同一个 Tracy client；默认 build 完全消失；
  - 工程计算留出使用相同 log record/sink 与 zone/counter/memory/thread/fiber 机制，不引入模型、
    token、dashboard 或服务协议类型。
- 范围：`include/hyle/obs/**`、`src/obs/**` 的同步 structured-log producer、level/compile gate、
  default sink、sink registry/RAII/flush、native trace macro seam；全部 public surface 反向追踪；
  `obs PUBLIC core error` 与当前 trace consumers；Tracy owner/build/package 投影；obs-only
  source/static/shared C++23 delivery；architecture/user/performance 文档与有限统一 P1 replay。
- 非目标：异步 queue/thread、文件轮转、网络 export、聚合/查询/告警、metrics registry、dashboard、
  crash/signal-safe logging、fatal/check 控制流、完整 category/config tree、OpenTelemetry protocol、
  profiler viewer、GPU timestamp/context API、业务采样/命名策略、Native FFI/C API、稳定 C++ ABI、
  下游模块整体收口和 Formal 发布证书。
- 直接依赖 / 直接消费者：`PUBLIC core error`；log runtime 当前由外部 C++ 用户/sample 主动消费，
  HYLE 底层模块不擅自输出日常日志；trace 的真实源码消费者为 `mem`、`sched`、`io`、`compute`。
  trace 是受检的 header/CMake 横切 seam，不制造这些模块到 obs log runtime 的伪二进制依赖。
- tier 与理由：`P1`。runtime level gate、同步 emit/fan-out 与 trace event 可位于高频路径；sink mutation、
  default stderr、flush、配置和 teardown 是 P0 控制面。关闭态 trace/compile floor 是 codegen/交付门，
  不能制造“0.00 ns”计时排名。
- material margin / resource budget：W1-WIN 同合同 baseline/direct-Tracy replay 使用 5% NI；runtime
  gate 至多一次 relaxed threshold load且参数不求值；exact-byte emit 为 O(S)、S≤64，不做 heap
  allocation、不拿 mutation mutex、不扫描 message；mutation O(S) 且只在控制面；默认 sink 不因安全
  转义引入 heap；OFF trace 零参数求值、零符号、零 backend，ON seam 不比直接 Tracy 增加 material
  event 成本。吞吐不能抵偿 callback tail、unregister hang、stack/record 常态膨胀或多 singleton。
- 总调查预算 / 性能预算：从 `SCOPED` 起总计最多 120 分钟，其中统一性能取证最多 45 分钟、最终
  红队最多 20 分钟；到时只能 `CLOSED`、具名 `BLOCKED`，或发现产品变化后按协议退回阶段，不能继续
  发明实验。
- 最多复现、机制诊断、产品修改次数：每个 material 异常一次反序/复现、一次 codegen/profile/
  contention 机制检查、最多一个同时具备机制与 consumer 钥匙的产品修改及一次重跑；不得修改中央
  统计语义或发明第五种 benchmark 姿势。
- machine profile：`W1-WIN` 为 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 承担适用
  correctness/sanitizer/OS/default-sink/trace-on/交付复核，不跨平台池化点值。Apple 大小核只按已登记
  profile/QoS 观察，不用最好 placement 签胜负。
- instrument profile：P1 为 `closure-perf-v1`；correctness/delivery 为
  `closure-correctness-v1` 与当前 component/package contract。
- strong peers（绑定当前本地 clean source）：
  - synchronous log entry/sink/flush/recursion/process lifetime：Abseil
    `81c993af4fb88b3a51829a796df02251fa27102f` 与 Folly
    `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9`；它们用于能力、lifetime、复杂度和负空间裁决，
    完整 category/config/LogEntry 的点值不冒充同合同排名；
  - compile-time profiling seam/backend lifecycle：Tracy v0.14.1
    `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c`；相同 event 可作 direct same-contract peer。
- 无同合同 peer时的预算及禁止声明：logging 正常 replay 对 frozen baseline
  `f305b9d3` 做 5% regression NI，并守一 sink exact emit ≤100 ns、format emit ≤150 ns、8 producer
  ≤150 ns/emit 的 W1-WIN 预声明上限；同时以 strong-peer source 审计证明复杂度同档。该证据只支持
  “当前轻量同步合同无 material regression/已知复杂度缺口”，不得宣称比完整 Abseil/Folly 快若干倍、
  “obs 全面 SOTA”或跨平台固定纳秒数。
- correctness gates：normal；empty/zero/embedded-NUL/format truncation；invalid level/field/sink、容量
  与 waiter/refcount 边界；duplicate/absent/full/retiring；move/reset/destructor/static teardown；并发
  emit/mutation/flush、remove-waits、自摘/跨摘、nested record；default-sink control-byte/TTY/debugger；
  stress；ASan/UBSan/TSan；trace OFF 参数/codegen 与 ON 真 Tracy backend；public-header self-containment。
- delivery gates：中央 `obs PUBLIC core error` 与 include-owner/target 对账；Tracy pin/license/offline/
  shared-one-client/static-one-final-image；obs-only source embedding、relocated static/shared C++23
  consumer；trace-OFF package 无 Tracy，trace-ON package 与 mem/sched/io/compute seam；无旧 C schema、
  bridge、fixture 或 Native FFI symbol。
- performance gates：冻结四个互不池化的 family：
  1. `compile/runtime-gate`：compile floor/trace OFF 走 codegen O(0)，runtime drop 走一次 threshold load；
  2. `single-adapter-emit`：exact byte + structured 4-field + format macro 向一个无 I/O同步 adapter sink；
  3. `multi-producer-fanout`：8 producers 与 1/4/16 sink，报告 per emit/callback、资源与 callback tail，
     mutation 只作并发 correctness/control replay；
  4. `trace-native-seam`：counter 与 named zone 对直接 Tracy 同 event 的 codegen和5% NI。
- 预期输出：capability-family map、全部 public-surface trace、有限 FIX 集、owner/complexity/resource 表、
  correctness/stress/consumer证据、统一 P1 bundle、三种交付、更新 architecture/user/performance、独立
  红队与四项 LOC。
- 停止条件 / 可接受限制：全部 `MUST` 为 `COMPLETE` 或完整 `ROUTED`，四个 family 的适用工程门
  `PASS`，无 correctness/security/delivery blocker 即停止；完整日志平台、GPU profiler、multi-image
  static singleton、Native FFI 与 Formal 可保持明确负空间。若功能 `MUST`、并发生命周期、统一仪器
  资格或 P1 NI 不能在预算内裁决，进入具名 `BLOCKED`，不无限研究。

## Capability-family coverage map

待 `AUDITED`：先从北极星/成熟来源自顶向下枚举，再把 `log.h`、`trace.h` 的全部入口逐项反向映射。

## Capability / gap ledger

待 `AUDITED`：覆盖状态与 `KEEP/FIX/BORROW/DEFER/REJECT` 分列，并冻结有限 `FIX` 集。

## Candidate

尚未冻结。

## Verification manifest

尚未执行。

## 独立红队

尚未执行。

## Closure decision

尚未裁决。
