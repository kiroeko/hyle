# obs v2 收口恢复记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本页是 `obs-v2` 的 `RESUME` 施工记录。权威状态是
> [`module-status.json`](../../module-status.json)；已终止的
> [`obs-v1`](obs-v1-closure.md) 只提供冻结候选、已通过门和 blocker 事实，不在本轮回写。

## Closure card（`AUDITED`）

- component / owner / round ID：`obs` / observability owner / `obs-v2`；task kind：`RESUME`。
- 前一 round / blocker：`obs-v1` / 公共性能仪器不能为冻结的 8-producer × 1/4/16-sink workload
  提供合格 N-worker verdict。
- 前一 resume condition：中央工具须在 fresh W1-WIN 通过自身测试，并对不变 workload 提供
  N-worker placement、双顺序、A/A、5%/10% delta、timer-floor、n=15→31 与 deep replay。
- 恢复证据：工具任务
  [`perf-multicore-topology-v1`](../performance/perf-multicore-topology-v1.md) 已 `COMPLETE`；clean
  `8a704f7a22026a36cbc40a5efbf4a73c131ad50c` qualification 与独立 verifier 均通过。仪器给出的产品
  结果为：1-sink 两项 NI `PASS`；4/16-sink 部分 stratum 在 n=31 后仍 `UNRESOLVED`，聚合
  `INCONCLUSIVE`。这解除工具 blocker，但不自动让 Obs 性能通过。
- baseline commit：`a3fc0f833f605fefb89fe904a8acaf716a4feaf8`；复用前一产品候选
  `2c784d9d6a925529995c6a5582d9bc373d91535d` / tree
  `bff512b2fd644b3a88bce55be419dadad43c6a89` 作为审计输入，不预先宣布仍为本轮 candidate。
- 进入原因与 demand/defect ID：`OBS-V2-PERF-01`（合格中央仪器给出多 sink 统计未决）、
  `OBS-V2-SINK-FIBER-01`（同步 sink callback 的 fiber yield/migration 负边界未公开）、
  `OBS-V2-TRACY-TEST-01`（trace-on 测试默认 Tracy listener/broadcast 触发 Windows 网络权限提示，
  测试与用户指南未区分 local-only 和受信网络 profiler）。
- 真实 consumer task / normal workload：推理 worker 同步提交 borrowed record 到宿主 adapter；adapter
  在 callback 返回前消费或复制 record，不能在 callback 内挂起 stackful fiber；profiling build 可由
  本机 Tracy viewer 连接，自动测试只需 loopback；冻结性能任务仍是 8 个常驻 producer 对 1/4/16 个
  同步 relaxed-atomic sink 的 fan-out 与 256-emit chunk tail。
- 范围：仅恢复并绑定 obs-v1 未变化能力/候选；裁决上述三个 ID；更新公开 callback 契约、trace-on
  测试环境、architecture/user/performance/task 证据；执行一次冻结性能机制诊断，只有同时取得机制与
  consumer materiality 才允许一个产品实现修改和一次完整重跑。
- 非目标：新增 log/trace 能力、改变 record/layout/registry/callback/chunk/同步语义或 5% margin；
  async logging、文件/网络 export、category/metrics/OTLP/GPU policy；修改中央统计、追加 n=31 样本、
  新 benchmark 姿势、Native FFI/C API、稳定 C++ ABI、Formal、多平台性能池化。
- 直接依赖 / 直接消费者：`obs PUBLIC core error`；trace seam consumers 为 mem/sched/io/compute；
  本轮不改变依赖图，也不把 HYLE 底层 failure 自动写日志。
- tier / material margin：`P1`；W1-WIN、`closure-perf-v1`、冻结 baseline `f305b9d3`、5% NI。
  emit 保持 O(S)、S≤64、零 heap/热路 mutex；callback 数是不可消除的渐进下界。同步 callback 的
  no-yield/same-thread-return 是 lifetime 前置条件，不增加热路检查或 metadata。
- 总调查预算 / 性能预算：最多 90 分钟，其中统计结果读取与唯一机制诊断最多 25 分钟、最终红队
  15 分钟；已完成 qualification 不重复采样。只有产品实现实际改变才允许一轮 fresh contract 重跑。
- 最多复现、机制诊断、产品修改次数：资格化 replay 已是唯一正常复现；允许一次 source/codegen/
  contention 机制诊断、至多一个双钥匙产品假设及一次完整重跑；无机制钥匙立即保留
  `INCONCLUSIVE`，不得换顺序、地址、worker、chunk、迭代数或统计规则。
- machine / instrument：P1 为 `W1-WIN` / `closure-perf-v1`；Windows trace OFF/ON 做 focused
  correctness；只有公开头或产品语义变化时才重绑 W1-WSL/M1-MAC 与三种交付，未运行项不得写 PASS。
- strong peers：Abseil `81c993af4fb88b3a51829a796df02251fa27102f`、Folly
  `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 只用于同步 sink/lifetime 形状；Tracy v0.14.1
  `30997d5ca6bb632cc10807a1da8a6d3de0aeeb3c` 是 listener/network 行为 owner；性能 peer 仍只为
  immutable HYLE baseline `f305b9d3`。
- correctness gates：callback normal/nested/remove/flush 与新 no-yield contract 文档对账；trace OFF/ON
  tests，trace-on test child 必须显式 local-only 且不改变用户 runtime 默认值；公开头 self-contained。
- delivery gates：若产品 identity 不变，复核 exact product diff 与既有 source/static/shared closure；
  若公开头或实现改变，则重新运行三种 C++23 journey。无 Native FFI journey。
- performance gates：绑定 qualified manifest、controls、raw deep replay 与六项 workload；执行唯一机制
  检查。1-sink PASS 不能抵消 4/16-sink `UNRESOLVED`，中位数接近 1 不能冒充 NI PASS。
- 预期输出：有限 capability/gap ledger 与 `FIX` 集；focused tests/docs；机制结论；更新的 Obs 性能
  证据与六问；terminal `CLOSED` 或具名 `BLOCKED`、四项 LOC。
- 停止条件 / 可接受限制：只有所有适用门含聚合 P1 都 `PASS` 才关闭。若一次机制诊断没有支持安全的
  material 产品修复，或重跑后仍未决，则在完成 correctness/usability 缺口后以“qualified n=31
  statistical unresolved”为新 blocker 停止；不把统计不确定性改写成产品失败或 benchmark 优化任务。

## Pre-audit inputs

以下只是进入 `AUDITED` 前要裁决的输入，不是已授权实现：

| ID | observed fact | candidate disposition |
|---|---|---|
| `OBS-V2-PERF-01` | qualified controls/deep replay PASS；4/16-sink 部分 98.9% CI 穿过 5% NI 下界，median 大体接近 1 | `DIAGNOSE` once；无双钥匙则 `BLOCKED` |
| `OBS-V2-SINK-FIBER-01` | registry recursion frame 是 OS TLS；callback yield/migrate 会让 frame 留在错误线程 | 冻结 synchronous same-thread return / no-yield 前置条件；文档和测试边界 |
| `OBS-V2-TRACY-TEST-01` | trace-on test artifact 默认启动 Tracy 网络发现/listener，Windows 会请求网络权限 | test process 使用 `TRACY_ONLY_LOCALHOST=1`；guide 说明本地/受信网络选择，不全局改产品默认 |

`BLOCKED -> SCOPED` 前态快照：
`C:\hyle-scratch\obs-v2\transitions\module-status-before-scoped.json`，SHA-256
`7BD67B957782BE9D6A609EDB96BDD0C3A0D42F1A9B658A592102BDB2D2F7A869`。

## Capability / gap ledger（`AUDITED`）

| ID | requirement / observed fact | coverage | frozen action / verification |
|---|---|---|---|
| `OBS-V2-IDENTITY-01` | v1 产品候选的公开 surface、registry、snapshot 与 callback 算法在恢复前没有漂移 | `COMPLETE` | `KEEP`：实现前 `include/hyle/obs`、`src/obs` 相对 `2c784d9d` exact diff 为空；本轮不授权算法改动 |
| `OBS-V2-PERF-01` | 合格的 8-P-core、1/4/16-sink 双顺序 n=31 replay 必须按冻结 5% NI 合同裁决 | `COMPLETE evidence / UNRESOLVED gate` | `KEEP + BLOCK`：绑定 qualified bundle 与唯一机制结论；不得追加样本、改 workload/统计或改产品追分 |
| `OBS-V2-SINK-FIBER-01` | 同步 borrowed callback 必须在调用它的 producer OS thread 上执行到返回；stackful fiber 不得在 callback 内 yield/suspend/migrate | `PARTIAL` | `DOC/API CONTRACT FIX`：补 public header、architecture 与 guide；不增加热路 thread-id 检查或 metadata |
| `OBS-V2-TRACY-TEST-01` | trace-on CTest 不应以 Tracy 默认 wildcard listener 请求 Windows 网络权限 | `PARTIAL` | `TEST/DOC FIX`：仅给 trace-on Obs test process 设置 `TRACY_ONLY_LOCALHOST=1`；guide 区分 loopback listener 与受信网络 viewer，不改变产品默认值 |
| `OBS-V2-CORRECTNESS-01` | 原有 normal/nested/remove/flush、trace OFF/ON 语义不能回退 | `COMPLETE but affected` | `KEEP + VERIFY`：focused trace OFF/ON tests、公开头 self-contained；no-yield 是调用方前置条件，不制造虚假 yield test |
| `OBS-V2-DELIVERY-01` | C++23 source/static/shared 与 selected `obs` package 仍须成立；Native FFI 继续 `N/A` | `COMPLETE but affected` | `KEEP + VERIFY`：公开头/CTest CMake 改动后重跑三种 journey；不新增导出符号、依赖或 option |

冻结 `FIX` 集只有 `OBS-V2-SINK-FIBER-01` 与 `OBS-V2-TRACY-TEST-01`。性能、registry、record layout、
callback 语义、中央仪器、依赖图和交付 surface 均不在产品修改集；新发现必须退回 `SCOPED/AUDITED`，
不能边实现边扩张。

### Audit result

- clean W1 qualification 位于
  `C:\hyle-scratch\obs-nworker-qualified-evidence-8a704f7a`。manifest / summary / controls /
  raw-log-index SHA-256 分别为 `3B4538580989A9665423E5C02E2C1B587E2FB75071D7223F9B71784731B05662`、
  `02DD48AC65558B78C4C2F21961D62F2F9FE74671528F4CDA3BF46B6DABFFDC72`、
  `80A8157E9BC3B6E0DF67F06BE63968EE3004A773DD36C9B7429E68169A92B758`、
  `6788B19471669AA2E74118C9AB6FC0A2E7A871346248BB1CE3192E5FC6CC4275`。qualification、A/A、
  5%/10% positive control、timer floor、placement、artifact/build/accounting 与 deep replay 全部
  `PASS`；1-sink 两项 NI `PASS`，4/16-sink 仍有 strata `UNRESOLVED`。
- 唯一机制检查确认两版 `SinkSnapshot`、active-reference、callback loop 与 release 路径的源码算法一致；
  MSVC 反汇编只在当前 exact-view `LogRecord` 构造处多见用于 optional fields 的 `xorps/movups`
  零初始化。该固定每 emit 成本在 1-sink 已通过，随 sink 增多应被摊薄，不能解释 4/16-sink 方差扩大；
  两个 artifact 的绝对分布都出现相同快慢模态，adapter 的 relaxed-atomic cache-line handoff 是双方共有
  的争用。没有同时满足“可复现产品机制 + 真实 consumer materiality”的双钥匙，因此不授权产品修复。
- Abseil/Folly peer 都把 sink/handler 定义为 producer 上同步、并发、返回后才继续的 borrowed callback。
  结合 HYLE 使用 OS TLS recursion frame 的实现可推得：OS 抢占本身安全，但 callback 主动挂起 stackful
  fiber 会让同一 OS thread 上的其他 fiber 继承错误 recursion state，迁移后还会在错误 thread TLS 上
  恢复。因此 no-yield/same-thread-return 是公开前置条件，不是 benchmark 特化。
- Tracy v0.14.1 默认监听所有网络接口；其 runtime `TRACY_ONLY_LOCALHOST=1` 只把 listener 限制到
  loopback，compile-time `TRACY_ONLY_LOCALHOST` 才同时改变 discovery broadcast 地址。CTest 的缺口是
  wildcard listener 导致的权限提示，故测试进程使用上游 runtime 开关；guide 必须避免把它误写成
  “完全禁网”，并保留用户在受信网络上连接 viewer 的默认选择。

`SCOPED -> AUDITED` 前态 registry 保存于
`C:\hyle-scratch\obs-v2\transitions\module-status-before-audited.json`，SHA-256
`6FE8422D23B5E75CE310C0AE2961F0D77D57E862FEB552018A30C61D9E5A1030`。
