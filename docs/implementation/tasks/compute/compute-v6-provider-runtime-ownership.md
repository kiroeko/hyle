# Compute v6：Provider runtime ownership 与 delegated completion

## Closure card

- component / owner：`compute` / Compute；round=`compute-v6-provider-runtime-ownership`；
- baseline commit：`0ce95e2d2f40280163714165d27c2434c5b57771`；[上一轮 v5](compute-v5-backend-command-spine.md)
  的候选 `db8d1d335c84fb89675f6be2c2350efc6e35b271` 只作为现状 baseline，不再签发架构完成；
- trigger 与证据：已接受的三轴资源模型要求 Graph 在同一 storage domain 上使用多条 execution lane；
  MLX 等 runtime 自有 allocator/stream/event，而现有 `ProviderPreparedVTable::submit` 和 Graph region SPI
  只能接收 HYLE `Queue&`。把 delegated runtime 假装成 shared Queue 会隐藏 stream、同步和完成边界；
- 正常 consumer 任务：direct/Graph 用户在 shared-runtime provider 上继续向显式 HYLE Queue 录入；Graph
  也能把一个完整 region 交给自有 runtime 的 provider，并取得真实可 poll/wait/cancel、可保活执行 owner
  的 HYLE Completion，所有 boundary copy、host wait 与不支持的 interop 都可见；
- 主产品纵切 / 具名来源：Apple MLX compiled region 与原生 Metal queue/resource 组合；MLX
  `229f5b430df7926743c5b6ac62068cae2ebc8978` 的 `async_eval`、array `Event` 和 stream event wait/signal
  证明私有 timeline 可等待，但 API、错误和线程约束必须由独立 provider adapter 承担；
- 正交反偏置纵切：一个不依赖 MLX 类型的测试 delegated runtime，以独立 worker/event 异步完成 region，
  验证 completion、abandon、错误、取消与 Graph 聚合，而不把 Metal 特例写进 Compute core；
- 跨模块 public journey：Mem 拥有 Buffer/resource，Compute 拥有 runtime-mode、Completion 和通用 owner
  bridge，具体 provider 拥有其私有 runtime 与 typed interop，Graph 拥有 island placement、dependency 和
  final aggregation；
- scope：给 prepared op/region 冻结 shared/delegated runtime mode；保留既有 Queue sequence fast path；增加
  delegated submit/Completion 内部 SPI、准确 ownership/failure/cancel/drain facts 和 report 输入；迁移已有
  native/oneDNN provider 为显式 shared mode；验证三种 C++ 交付；
- non-goals：交付 MLX/Metal/Vulkan provider、横向补充 operation family 或目录式 native route、模型/KV/
  batching/serving、公开万能 Event/Stream、Native FFI、隐式线程池、把 host wait 冒充异步 interop。允许
  为 C60--C64 写最小私有 probe adapter/kernel；它不自动成为公共算子能力，验证后必须裁决保留或删除；
- 直接依赖 / 直接消费者：依赖 Core/Error/Mem/compute_ops 和现有 Compute runtime；直接消费者是 Graph、
  compute_provider_native、compute_provider_onednn、graph_provider_onednn 与未来独立 delegated provider；
- 性能 tier：`P0`。shared steady path 的 layout、分支与间接调用不得增加；delegated path 每 region 一次
  预绑定 submit 和一个真实 Completion，不按 node/element 路由。真实 MLX/Metal P1 留给其独立 factor；
- 适用机器 / 仪器 / material margin：Windows 运行 shared/mock-delegated correctness、P0/codegen 和三交付；
  Linux/macOS 验证异常转换、lifetime 与交付。没有真实 provider candidate 前不运行 P1；
- 调查与纠正预算：通用 SPI 最多两轮；shared provider 迁移一轮；mock delegated lifecycle 一轮；
- 预声明验证：invalid/incomplete vtable、shared/delegated callback 互斥、sync rejection、async success/error、
  partial admission、completion abandon、late terminal、cancel unsupported/supported、owner teardown、OOM、
  concurrent invocation、Context close/drain、已有 provider 回归、逐头、sanitizer、source/static/shared、P0；
- stop / blocker：若 MLX 或第二个 delegated runtime 无法通过该 seam 保留真实 owner 和完成语义，不能把
  mock 通过写成架构资格；记录精确 provider blocker并回到 `AUDITED`，不扩大算子目录。

## 审计结论与 API freeze

### 质因数与现状

- execution lane 仍是 Compute Queue 的核心语义，与 Device/Context/Completion 共用 owner、依赖和测试，
  不拆 `compute_lane` factor；
- delegated bridge 是 provider contract 的内部公共缝，也不形成可独立消费产品；每个真正 delegated
  runtime 仍以独立 provider factor 拥有依赖、构建开关、许可证和资格；
- `compute_sched` 保持 `QUEUED`，本轮不借 provider ownership 启动 CPU worker bridge；
- 现有 `Completion` 已能承载外部异步状态、non-blocking abandon 和 lifetime attachment；现有 backend
  `wait_completion` 能在识别 completion identity 时录入 device wait，因此不新增第二套 public event。

### 能力与形状审计

| 能力 | 层级 | 当前 | 动作 |
|---|---|---|---|
| shared Queue enqueue | 原子机制 | 完整且已是热路径 | KEEP |
| provider runtime mode | 原子机制 | 缺失；所有 provider 被假定为 shared | FIX |
| delegated submit -> Completion | 原子机制 | 缺失 | FIX |
| delegated execution owner | 原子机制 | Completion attachment 可复用，provider SPI 未表达 | FIX |
| cross-lane dependency import/export | 复合功能 | Compute completion identity 已存在；具体 interop 由 backend/provider typed bridge 实现 | KEEP/FIX |
| hidden host wait | 应用取舍 | 不可接受 | REJECT |
| public generic runtime/plugin ABI | 应用策略 | 无消费者 | REJECT |
| operation catalog 扩张 | 独立能力 | 与本轮 seam 无关 | DEFER |

冻结内部形状为一个带版本的 runtime-mode discriminator 和互斥执行入口：shared artifact 只提供
`enqueue(..., QueueSequence&) -> EnqueueResult`；delegated artifact 只提供
`submit(..., dependencies) -> Result<Completion>`。两者都由 invocation owner 保活到终态。dependency
资格在 cold replay preparation 中按精确 endpoint identity 决定；steady submit 只消费已选 route。无法
device-wait 的边只允许由 Graph 依显式 policy 在 admission 前 host-wait，provider 不得偷偷等待。

### 有限 FIX ledger

| ID | 问题 | 允许改动 | 验收 |
|---|---|---|---|
| C60 | prepared provider 无 runtime mode | 内部 enum/properties、构造验证和 report facts | 非法组合拒绝；shared 默认不增加热分支 |
| C61 | provider/region 只能 `submit(Queue&)` | 增加互斥 delegated callback，返回真实 Completion | mock async success/error/abandon/cancel 全通过 |
| C62 | delegated dependency 无冷路径资格 | 版本化 endpoint/dependency capability 与预绑定 route | steady 不搜索；unsupported 明示 |
| C63 | owner/failure/drain 未完整传递 | 复用 Completion lifetime，迁移 provider state teardown | pending handle 销毁不 wait、不 UAF |
| C64 | existing providers 未声明 shared | native/oneDNN/Graph region 机械迁移 | 行为、layout 和 codegen 不回归 |
| C65 | 构建/文档 | 更新内部合同、用户限制和三交付 | source/static/shared 同语义 |

除 C60--C65 外不允许扩张算子 schema/catalog、placement 算法、公开 FFI 或无关模块。为证明本轮 seam
必需的最小私有 adapter/kernel 可以实现，但必须在 task 中回指具体 probe，且不能借此签发新的公共算子
能力。正式 MLX 实现属于后续独立 factor，不进入本轮。

## 七问（AUDITED）

1. 第三方路径：本体不新增第三方；MLX revision 只证明 seam，真正依赖由未来 factor 拥有。
2. 功能完整：shared 与 delegated 两种真实 ownership 均可表达；模型策略和算子 breadth 明确在外。
3. 内部复用：复用 Error、Mem、Completion、QueueSequence，不另造 event/allocator/threadpool。
4. 性能：shared path P0 零新增成本；delegated 每 region 一次预绑定边界，P1 由真实 provider 签。
5. 交付边界：纯 C++ 内部 type erasure；Native FFI N/A。
6. 构建交付：Compute core 不链接 MLX；具体 provider 才拥有依赖和开关。
7. 易用文档：普通 direct 用户继续只见 Queue/PreparedOperation；Graph 用户从 report 看到 island/host wait。

本卡已完成范围与有限 FIX 审计，`compute` 依法从旧 `CANDIDATE` 返回 `AUDITED`。完成 C60--C65 前不得
再次形成 candidate。
