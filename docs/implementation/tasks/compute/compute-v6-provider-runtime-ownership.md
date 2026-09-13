# Compute v6：Provider runtime ownership 与 resource contract

## Closure card

- component / owner：`compute` / Compute；round=`compute-v6-provider-runtime-ownership`；
- baseline commit：`0ce95e2d2f40280163714165d27c2434c5b57771`；[上一轮 v5](compute-v5-backend-command-spine.md)
  的候选 `db8d1d335c84fb89675f6be2c2350efc6e35b271` 只作为现状 baseline，不再签发架构完成；
- trigger 与证据：Graph 需要 execution target、storage domain 与 execution lane 三轴；MLX 等 runtime 自有
  allocator/stream/event，而现有 `ProviderPreparedVTable::submit` 只接受 HYLE `Queue&`。同时 provider
  prepare 只看 TensorDesc，resource compatibility 直到 submit 才由 backend/具体 provider 检查，会让
  Graph 无法在冷路径正确选择 no-copy/provider/copy route；
- 正常 consumer 任务：direct/Graph 用户在 shared-runtime provider 上向显式 HYLE Queue 录入；Graph 也能
  把完整 region 交给自有 runtime 的 provider，并取得真实可 poll/wait/cancel、可保活 owner 的 Completion。
  两条路线都在 admission 前固定 binding/resource、dependency、workspace 和 failure 合同；
- 主产品纵切 / 具名来源：Apple MLX compiled region 与原生 Metal resource/queue 组合。MLX 的 stream、
  thread-affinity、async completion/error 不能伪装成 HYLE Queue；正式 adapter 必须隔离其 runtime owner；
- 正交反偏置纵切：一个不依赖 MLX 类型的 mock delegated runtime，以独立 worker/event 异步完成 region；
  CPU native/oneDNN shared route 保持直接 QueueSequence 短路径；
- 具名参考：ggml `e71b80510c84` 分开 buffer type、backend stream 与 event；ONNX Runtime EP 先给 capability/
  allocator、CUDA EP 可分别接 user stream/external allocator；IREE HAL 分开 device/allocator/queue affinity；
  ExecuTorch delegate 拥有内部 tensor，只向外暴露 boundary；
- 跨模块 public journey：Mem 拥有 Buffer/resource/coherence identity，Compute 拥有 Context/Queue、provider
  runtime mode、binding requirement、Completion 和 owner bridge，Graph 拥有 target/storage/lane placement、
  dependency 与 final aggregation；具体 provider 拥有私有 runtime 与 typed interop；
- scope：给 op/region proposal 增加 resource/binding requirements；对 selected exact route prepare；冻结
  shared/delegated runtime mode；保留 QueueSequence fast path；增加 delegated submit/Completion SPI、准确
  ownership/failure/cancel/drain facts；迁移 native/oneDNN/Graph region；验证三种 C++ 交付；
- non-goals：交付 MLX/Metal/Vulkan provider、横向补 operation family/catalog、模型/KV/batching/serving、
  public 万能 Event/Stream、Native FFI、隐式线程池、把 host wait 冒充异步 interop。允许为 C60--C67 写
  最小私有 probe adapter/kernel；它不自动成为公共算子能力；
- 直接依赖 / 直接消费者：依赖 Core/Error/Mem/compute_ops 和现有 Compute runtime；直接消费者是 Graph、
  compute_provider_native、compute_provider_onednn、graph_provider_onednn 与未来 delegated provider；
- 性能 tier：`P0`。shared steady path 的 layout、分支与间接调用不得增加；resource proposal/validation 只在
  prepare/bind/Graph compile；delegated 每 region 一次预绑定 submit 和一个真实 Completion，不按 node/
  element 路由。真实 MLX/Metal P1 留给独立 factor；
- 适用机器 / 仪器 / material margin：Windows 运行 shared/mock-delegated correctness、P0/codegen 和三交付；
  Linux/macOS 验证异常转换、lifetime 与交付。没有真实 provider candidate 前不运行 P1；
- 调查与纠正预算：resource contract 两轮；通用 runtime SPI 两轮；shared provider 迁移一轮；mock delegated
  lifecycle 一轮；
- 预声明验证：invalid/incomplete vtable、proposal/prepare resource mismatch、one resource 多 target、
  shared/delegated callback 互斥、sync rejection、async success/error、partial admission、completion abandon、
  late terminal、cancel unsupported/supported、owner teardown、OOM、concurrent invocation、Context close/drain、
  既有 provider 回归、逐头、sanitizer、source/static/shared、P0；
- stop / blocker：若 MLX 或第二个 delegated runtime 无法通过该 seam 保留真实 owner/完成语义，不能把 mock
  通过写成架构资格；若 provider requirement 无法在 submit 前判定，退回 resource seam，不以隐藏 copy 修补。

## 审计纠正（2026-09-13，首个产品改动前）

初版 v6 只处理 shared/delegated execution ownership，没有把 provider storage requirement 纳入冷选择；这与
三轴资源模型矛盾。此时尚未在 v6 授权下修改产品代码，所以本次在同一 trigger 内纠正审计和有限 FIX，
不保留错误内部 SPI，也不产生兼容义务。

## 能力与形状审计

| 能力 | 层级 | 当前 | 动作 |
|---|---|---|---|
| shared Queue enqueue | 原子机制 | 完整且已是热路径 | KEEP |
| provider binding/resource proposal | 原子机制 | 缺失；prepare 只见 TensorDesc | FIX |
| selected exact-route prepare | 复合功能 | 缺失；resource 可能到 submit 才拒绝 | FIX |
| provider runtime mode | 原子机制 | 缺失；所有 provider 被假定为 shared | FIX |
| delegated submit -> Completion | 原子机制 | 缺失 | FIX |
| delegated execution owner | 原子机制 | Completion attachment 可复用，provider SPI 未表达 | FIX |
| cross-lane dependency import/export | 复合功能 | Compute completion identity 已存在；typed bridge 不完整 | KEEP/FIX |
| hidden allocation/copy/host wait | 应用取舍 | 不可接受 | REJECT |
| public generic plugin ABI | 应用策略 | 无消费者 | REJECT |
| operation catalog 扩张 | 独立能力 | 与本轮 seam 无关 | DEFER |

provider proposal 必须对每个 input/output/workspace/packed constant 给出 alignment、required memory properties、
可接受 exact domain/resource 或 provider-private bridge，以及 shared/delegated mode。Graph 用它选择 target/storage
和 copy；selected artifact 再承诺该 exact route。Direct resource-aware prepare 使用同一合同；普通 convenience
可从实际 TensorView 推导 resource。bind 可以做防误用验证，但 steady submit 不搜索 provider、不重新解释
requirement，也不第一次发现不兼容。

shared artifact 只提供 `enqueue(..., QueueSequence&) -> EnqueueResult`；delegated artifact 只提供
`submit(..., dependencies) -> Result<Completion>`。两者都由 invocation owner 保活到终态。dependency 资格
在 cold replay preparation 中按 exact endpoint identity 决定；无法 device-wait 的边只允许由 Graph 依显式
policy 在 admission 前 host-wait，provider 不得偷偷等待。

## 有限 FIX ledger

| ID | 问题 | 允许改动 | 验收 |
|---|---|---|---|
| C60 | provider proposal 不报告 binding/resource requirement | 版本化 internal requirement 与完整验证 | Graph 可在 prepare 前选择 storage/copy |
| C61 | selected artifact 未绑定 exact resource route | prepare request/record 固化 target、resource 与 workspace owner | steady submit 不因 resource 拒绝或 fallback |
| C62 | direct prepare 无 resource-aware 路线 | 提供自然 resource-aware prepare；旧 desc-only 形状可删除/降 convenience | 用户无需理解 provider SPI，错误发生在 cold path |
| C63 | prepared provider 无 runtime mode | internal enum/properties、构造验证和 report facts | 非法 callback 组合拒绝；shared 无热分支 |
| C64 | provider/region 只能 `submit(Queue&)` | 增加互斥 delegated callback，返回真实 Completion | mock async success/error/abandon/cancel 全通过 |
| C65 | delegated dependency/owner/failure/drain 未表达 | endpoint capability、Completion lifetime 与 teardown 迁移 | unsupported 明示；pending handle 不 wait、不 UAF |
| C66 | existing providers 未显式 shared/resource 合同 | native/oneDNN/Graph region 机械迁移 | 行为、layout、route 和 codegen 不回归 |
| C67 | 构建/文档 | 更新内部合同、用户限制和三交付 | source/static/shared 同语义 |

除 C60--C67 外不允许扩张 operation schema/catalog、Graph placement 算法、公开 FFI 或无关模块。最小私有
probe 必须回指具体验收，验证后裁决 `KEEP_PRIVATE/DELETE`。正式 MLX 实现属于后续独立 factor。

## 七问（AUDITED）

1. 第三方：本体不新增第三方；锁定源码/官方文档只证明 seam，真正依赖由具体 provider factor 拥有。
2. 功能：resource-aware shared route 与 delegated ownership 都可表达；模型策略和算子 breadth 明确在外。
3. 复用：复用 Error、Mem resource identity、Completion、QueueSequence，不另造 event/allocator/threadpool。
4. 性能：proposal/validation 全在冷路径；shared path P0 零新增 steady 成本；delegated 每 region 一次边界。
5. 交付：纯 C++ internal type erasure；Native FFI N/A。
6. 构建：Compute core 不链接 MLX；具体 provider 才拥有依赖、版本、许可证和开关。
7. 易用：direct 用户提供实际 TensorView/resource 后 prepare；Graph 用户从 report 看 island、copy 和 host wait。

本卡已在任何 v6 产品修改前完成审计纠正；`compute` 保持 `AUDITED`。完成 C60--C67 前不得形成 candidate，
也不得横向填充算子目录。
