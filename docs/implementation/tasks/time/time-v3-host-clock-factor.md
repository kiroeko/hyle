# Time v3：host-clock 质因数与可注入时间

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`time` / Time / `time-v3-host-clock-factor` / `REOPEN`；
- baseline：`25b13f5a18a0164a5fa81d01c4ef5d1a32f85cf4`；前轮：`time-v2`；
- trigger：`TIME-FACTORIZATION-2026-09-02`；
- `TIME3-R1`：已接受的 Stage C 证明 host clock/value 与 caller-owned timer node 是两个独立消费者、依赖
  和成本闭环。现有 `time -> container_intrusive` 迫使只需要 `UnixTimestamp` 的 Platform 取得 timer
  storage 依赖，满足“component dependency/delivery shape changed”的重开条件；
- `TIME3-R2`：能力账本 `TIME-A-17` 已由 Seastar manual clock、Abseil simulated clock 与确定性
  scheduler/timeout 测试共同准入，但当前 HYLE deadline helpers 固定读取全局 clock，只能给 queue 显式
  传 `now`，属于已确认 `MISSING`；
- `TIME3-R3`：time-v2 candidate 后，Time 实现已迁移到新的 Core environment contract，当前源码 identity
  不再等于旧 candidate；本轮以当前 baseline 重新绑定受影响的依赖、交付和 codegen，不伪装旧证据覆盖；
- 最新参考身份（2026-09-02 clean fast-forward）：llama.cpp
  `b96806d96061049a5b574269b049bf6241d63d46`（随仓 ggml tree
  `bc886da1a6c9d69660b0043d407200015667d609`）、standalone ggml
  `d4716378882593333721eb33f153144b6885caf2`、Seastar
  `991cf9a908482a3f11ca2fc34a1b3970a06d8622`、Abseil
  `cc05267c15127e41d497ddfc985328e59992ea53`；
- 初步受影响面：host clock/value/conversion/sleep 与 injectable/manual clock 的公开自然形状；从 Time
  迁出 timer headers/source ownership；Platform/Sched/IO 的直接边；`hyle::time` façade 的双 factor
  聚合；correctness、P1、三平台三交付、architecture/user 文档；
- 初步 tier：`P1`。monotonic/profiling/coarse reads 与 timer lifecycle 都可能处于 runtime 热路；本轮
  只在 API freeze 后分别冻结 host-clock 与 timer-queue 正常 workload，不用 clock-call 亚纳秒点值制造
  接口；
- 暂定非目标：device clock、timezone/tzdb/locale、协议文本、callback/executor、周期/跨线程 timer、
  cancellation propagation、Sched/IO 重写、Native FFI/C API、稳定 C++ ABI、公共仪器改造与 Formal。

当前只登记足以击穿旧结论的新事实；closure card、完整 public trace、有限 `FIX` 集和 API freeze 尚未
完成，因此不授权产品代码修改。

## Closure card（`SCOPED`）

- component / owner / round / kind：`time` / Time / `time-v3-host-clock-factor` / `REOPEN`；
- baseline：`25b13f5a18a0164a5fa81d01c4ef5d1a32f85cf4`；trigger：
  `TIME-FACTORIZATION-2026-09-02`，证据即上节 `TIME3-R1..R3`；
- 真实 consumer / normal workload：Platform 文件时间值、Sched/IO deadline 与批次读钟、Obs 时间戳；
  北极星旅程是 llama.cpp/ggml 在加载、采样和 backend 阶段的 elapsed 计费，以及不等待
  真实时间的 scheduler/timeout 确定性测试；
- 范围：host monotonic/realtime/coarse/profiling/CPU clocks，duration/deadline/`Instant`，Unix/OS
  conversion，blocking thread sleep；新增可编译期注入的 system/manual monotonic provider 与可选的
  小型借用型 runtime reference；从本 factor 移除 timer-node storage 所有权，但保持用户级
  `hyle::time` product facade；
- 非目标：timer callback/executor、periodic/fixed-rate 策略、跨线程 arm/cancel、fiber waiter 唤醒、
  device clocks/host-device correlation、tzdb/calendar/locale/协议文本、Sched/IO 状态机重写、Native FFI/C API、
  稳定 C++ ABI、公共性能仪器改造和 Formal certification；
- 直接依赖：完成后 `time -> PUBLIC core`，标准库/compiler builtin/OS clock API 是机制合同，
  不是第三方产品依赖；直接消费者：`timer_queue`、Platform、Sched、IO，Obs 只按值语义选择读钟而
  不强制建立产品边；
- tier：`P1`。默认 monotonic/profiling 读钟可位于 inference/runtime 高频路径；手动时钟和类型
  擦除只在显式选择时付费；
- normal performance shape：一个 phase/event-loop 批次仅采样一次 host monotonic time，完成
  begin/end elapsed 和饱和 deadline 算术；primary 对照直接 `std::chrono::steady_clock`/等价 OS 路线，
  ggml `ggml_time_us` 只作真实旅程来源而不把微秒精度冒充同合同 peer；holdout 为一次显式
  runtime clock-reference 采样加 deadline 计算；
- 常态成本合同：默认/templated system route 无 allocation、lock、virtual dispatch、registry 或新 metadata，
  不多于底层钟本身的读取；`ManualClock` 每次 now/set/advance 至多一个 64-bit atomic 操作，
  不拥有 waiter；显式 runtime reference 为两字大小、一次间接调用、无分配，生命周期由 owning
  runtime 保证；
- material margin / resource budget：W1-WIN 同合同 primary `5%`；默认路线代码形状不得多出
  间接调用/分支/常态存储，注入路线不得分配、系统调用或管理 waiter；
- 调查/性能预算：本 host-factor 轮总预算 3 小时，性能取证 45 分钟；异常后最多一次
  复现/反序、一次 codegen/profile 机制检查、一个产品修改与一次重跑；
- machine / instrument：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 primary 固定 W1-WIN +
  `closure-perf-v1`，Formal 仅 `PENDING/formal-release-v1`；
- strong peers：Seastar `991cf9a908482a3f11ca2fc34a1b3970a06d8622`
  `manual_clock.hh/lowres_clock.hh`；Abseil `cc05267c15127e41d497ddfc985328e59992ea53`
  `clock_interface.h/simulated_clock.h`。默认 host clock 还受 C++23 chrono 和各 OS clock/sleep 合同约束；
- provenance：已有 clock/deadline 为 `REAL_CONSUMER`，ggml/llama phase timing 为
  `NORTH_STAR_JOURNEY`，injectable/manual clock 为两个成熟 peer 与确定性 runtime 共同形成的
  `PROVISIONAL_ARCHETYPE`；
- correctness gates：normal、zero/past、boundary/overflow、invalid programmer contract、move/destroy/lifetime、
  concurrent manual reads/advance、property/stress、ASan/UBSan；OOM 对零分配 clock value `NOT_APPLICABLE`，
  cancellation/close、real ISA/backend 对 host factor `NOT_APPLICABLE`；
- delivery gates：public-header self-containment/macro composition，Time 直接消费者编译，Windows/WSL/macOS，
  source embedding + relocated installed static/shared C++ consumer，旧 C/Native FFI surface 反证；
- 预期输出：只依赖 Core 的 host-time factor，完整且自然的 injectable/manual clock API，
  `timer_queue` 所有权迁出所需的已冻结 host API，P1/correctness/delivery 证据，同步 architecture/user/task；
- 停止条件：有限 `FIX` 完成后按 candidate 裁决；新 timezone/device/executor 需求只路由，
  相邻 Sched/IO 状态机问题记 backlog；仪器不合格则 `BLOCKED`，不在 Time 任务里扩建仪器。

`SCOPED` 只冻结上述合同；当前仍不授权产品实现。必须先完成 capability map、全部
public-surface 反向追踪、依赖/交付影响与有限 `FIX` 集，并进入 `AUDITED`。
