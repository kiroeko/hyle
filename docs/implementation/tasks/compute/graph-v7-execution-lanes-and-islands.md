# Graph v7：Execution target、storage、lane 与 delegated island

## Closure card

- component / owner：`graph` / Graph；round=`graph-v7-execution-lanes-and-islands`；
- baseline commit：`0ce95e2d2f40280163714165d27c2434c5b57771`；[旧 v6](graph-v6-runtime-specialization-placement.md)
  candidate `db8d1d335c84fb89675f6be2c2350efc6e35b271` 是待迁移实现，不再证明新资源合同；
- trigger 与证据：v6 的 `ExecutionDomain` 把 Context 与 workspace resource 合成一个 identity，NodePlacement、
  ValuePlacement 和 QueueBinding 又共用该 identity。当前实现据此为每个 `(value, domain)` 建 physical value，
  会把“换执行者”“换 allocator”和“换 Queue”都误作跨域 copy。v7 初次审计虽增加 lane，仍未拆开
  execution target 与 storage，且把线性 liveness 放在多 lane schedule 之前；
- 正常 consumer 任务：调用方提供 execution targets、storage domains 与可用 lanes；Graph 联合选择
  provider/target/storage，编译 copy、schedule、happens-before 与 workspace，Invocation 再把 shared lanes
  绑定到 exact Queue，并把 delegated island 的 Completion 聚合为一个真实终态；
- 主产品纵切：一个 CUDA context 同时访问 pinned/device/managed resource，在同一 device storage 上运行
  H2D transfer lane → compute lane → D2H transfer lane，重复 prepare/capture/submit，无假 physical value、
  隐藏 host wait 或不安全 workspace alias；
- 正交反偏置纵切：CPU 单 target/storage/Queue 保持短路径；mock delegated region 与 shared Queue segment
  混合，验证 island boundary、completion 和失败聚合不依赖 CUDA 类型；
- 具名来源：ggml/llama.cpp `e71b80510c84` 分开 buffer type/backend stream/event/scheduler；vLLM
  `51da0ca66c80` 真实使用 compute/communication/copy stream；IREE HAL 分开 device、allocator、queue affinity、
  command buffer/fence；ONNX Runtime EP 分开 capability/allocator/stream；TensorRT 明确 auxiliary streams 会
  减少 activation reuse；ExecuTorch delegate 只把 boundary tensor 交给外层 memory planner；
- 跨模块 public journey：Compute 提供 Context/Queue/Completion/provider runtime 与 binding requirement；Mem
  提供 resource/coherence identity 和 lifetime packing；Graph 拥有 target/storage placement、lane schedule、
  happens-before、boundary 和 report；Sched 与推理引擎分别拥有 CPU worker policy 和请求/KV/batching policy；
- scope：以 `ExecutionTargetId`、`StorageDomainId`、`ExecutionLaneId` 取代多义 `ExecutionDomainId`；resource-aware
  provider proposal/prepare；按 lane 建 dependency；按 happens-before 规划 workspace；显式并发/内存取舍；
  delegated island 和内部 typed region seam；迁移现有 Graph 产品、直接消费者、三交付和文档；
- non-goals：横向补 operation catalog/native route、任意 `if/while`、模型/KV/batching/sampling、自动线程池、
  timing oracle、公开第三个 Plan 产品、正式 MLX/Metal/Vulkan provider、Native FFI。允许为 G70--G80 写最小
  私有 probe command/provider；它只证明结构 seam，不自动成为公共算子能力；
- 直接依赖 / 消费者：依赖 Core/Error/Compute/Mem/compute_ops，私有使用 Container/Util-core；直接消费者
  是 Compute/Graph 用户、CUDA probe、oneDNN region 与未来 MLX/collective provider；
- 性能 tier：`P0`。单 target/storage/lane steady path 不增加搜索、分配或每-node Completion；多 lane compile
  是冷路径有界分析，steady submit 只走预绑定 segment/dependency。真实 CUDA/MLX candidate 后才运行 P1/P2；
- 适用机器 / 仪器 / material margin：Windows/RTX 4070 验证真 CUDA lane/capture；Windows mock/CPU、WSL、
  两台 Apple 验证通用语义、sanitizer 和三交付。没有真实 provider candidate 前不运行 P1；
- 调查与纠正预算：三轴 API 两轮；provider resource contract 两轮；schedule/workspace 两轮；delegated/
  dependency 两轮；每个真 backend probe 一轮；
- 预声明验证：zero/one/multi target/storage/lane；一个 target 多 resource、一个 resource 多 target；duplicate/
  foreign/missing Queue/resource；不同 target 可直访同一 storage 时无 copy；same-storage cross-lane dependency；
  compute/transfer/communication role；capture 与 role 正交；happens-before safe reuse、unordered no-alias、显式
  serialization/budget failure；provider resource rejection 在 cold path；delegated async success/error/cancel/
  abandon；OOM、并发 Invocation、逐头、sanitizer、source/static/shared、真 CUDA probe和 P0；
- stop / blocker：如果正式 backend 不能让 target/storage/lane 分离且不隐藏 copy/wait，或 partial-order
  workspace 无法在有限预算内安全保守退化，则退回该 seam；mock 通过不得冒充 Apple/CUDA 资格。

## 审计纠正（2026-09-13，首个产品改动前）

初版 v7 task 把 lane 从 domain 中拆出，但仍留下 `ExecutionDomain = Context + workspace_resource`，并错误地
先做线性 liveness、后做 lane schedule。此时尚未在 v7 授权下修改产品代码，所以本次直接纠正同一 task 的
审计结论与有限 FIX，不产生兼容义务，也不把错误 freeze 当历史产品事实。

## API / architecture freeze

概念上的普通形状为：

```cpp
StorageDomain storage[]{
    {.resource = device_resource, .workspace_budget_bytes = SIZE_MAX},
    {.resource = pinned_resource, .workspace_budget_bytes = SIZE_MAX},
};
ExecutionTarget targets[]{
    {.context = &cuda_context, .default_storage = StorageDomainId{0}},
};
ExecutionLane lanes[]{
    {.target = ExecutionTargetId{0}, .role = ExecutionLaneRole::kCompute},
    {.target = ExecutionTargetId{0}, .role = ExecutionLaneRole::kTransfer},
};
auto compiled = HYLE_TRY(graph::compile(
    graph,
    CompileEnvironment{.storage_domains = storage,
                       .targets = targets,
                       .lanes = lanes},
    CompileOptions{}));

QueueBinding bindings[]{
    {.lane = ExecutionLaneId{0}, .queue = &compute_queue},
    {.lane = ExecutionLaneId{1}, .queue = &copy_queue},
};
HYLE_TRY(invocation.prepare_replay(bindings, ReplayOptions{}));
auto done = HYLE_TRY(invocation.submit(bindings));
```

- `NodePlacement` 使用 target，`ValuePlacement` 使用 storage，`QueueBinding` 使用 lane；report 对三者分别枚举；
- `StorageDomain` 是 Graph ordinal + exact canonical `MemoryResourceRef` + optional hard workspace budget；Mem
  `MemoryDomainId` 继续表示 coherence/accessibility，不复用为 Graph 下标；
- 每个 target 有自然 `default_storage`，但 target/resource compatibility 必须由 backend 与 provider 逐 binding
  冷验证。target 改变不自动 copy；
- lane 绑定 target，role 只有 `compute/transfer/communication`。capture 只由 ReplayMode/capability 表达；
- lanes 为空时每个 active target 建一条 compute lane。delegated island 的 synthetic lane 不要求用户提供
  QueueBinding，但其 endpoint、dependency 和 Completion owner 必须进入 report；
- provider 先 proposal resource requirements，再对 exact selected route prepare。Graph 不接受“首次 submit
  才发现 resource 不支持”；
- `WorkspaceReusePolicy::kExistingOrderOnly` 是默认值；只有显式 `kAllowSerialization` 可为硬 budget 添加
  ordering edge。无论哪档，unordered lifetimes 不 alias，新增 edge/peak bytes 都可查询。

## 能力与有限 FIX

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| G70 | `ExecutionDomain` 合并 Context/resource | 增加 TargetId/StorageDomainId 与独立环境记录，删除多义 identity | one target 多 storage、one storage 多 target |
| G71 | node/value/default placement 混在一轴 | node→target、value→storage；target 仅持 default storage | target 变化不制造 physical value/copy |
| G72 | provider resource constraint 到 submit 才可见 | 消费 Compute proposal/binding requirement；exact route 冷 prepare | unsupported route 可改选 provider/copy，steady 不拒绝 |
| G73 | 每 `(value, old-domain)` 无条件复制 | physical value 按真实 storage residence 建模，copy 按访问/requirement 插入 | shared/unified/peer-access 纵切无假 copy |
| G74 | QueueBinding 按 domain | 增加 LaneId/Lane/role，QueueBinding 与 segment 改按 lane | invalid/duplicate/foreign 精确失败；默认短路径自然 |
| G75 | capture 被当作 lane role | role 只保留 compute/transfer/communication；ReplayMode 独立 | 多 lane capture fork/join 与非 capture 同一 lane 图 |
| G76 | liveness 早于 schedule、线性 interval 对并发不安全 | 先 lane schedule/HB，后 safe reuse；unknown relation 不 alias | 并发压力/TSan/地址校验无 overlap corruption |
| G77 | 内存与并发没有显式取舍 | hard budget + ExistingOrderOnly/AllowSerialization；有界重排与报告 | 超预算失败或显式加边，不隐藏同步 |
| G78 | dependency/finalization 只识别 domain Queue | 按 lane/endpoint 预绑定 device/ordered/host-wait route并聚合 prefix | 无每-step Completion；host wait 可拒绝 |
| G79 | report 混淆执行、存储、lane 和 provider owner | 分列 target/storage/lane、HB/serialization、copy/island facts | 用户可解释每次 copy、wait、peak 与 owner |
| G80 | region 只能 shared Queue enqueue | 消费 Compute delegated SPI，生成 provider-owned island segment | mock async lifetime/error/cancel/abandon 全通过 |
| G81 | flat IR、构建与旧 v6 文档 | 保留私有 typed region boundary；迁移三交付/guide，删除 domain=一切说法 | 当前 DAG 无控制流/热成本；source/static/shared 一致 |

除 G70--G81 外不允许扩张 operation schema/catalog、模型 runtime、公开 FFI 或无关模块。最小 probe 验证后
逐项 `KEEP_PRIVATE/DELETE`；不能用 probe 数量提高算子完成度。

## 实现与资格化事实（candidate `7718b163`）

- G70--G75 已进入产品实现：公开环境分别声明 execution target、storage domain 与 execution lane；physical
  value 只按真实 storage residence 建模，target/lane 改变本身不再制造 copy；compute/transfer/communication
  role 与 capture policy 分离；
- operation 与 region 都先取得 provider-neutral binding requirement，再由 Graph 为每个输入、输出及 caller
  workspace 选择 exact resource，最后只 prepare/compile 该路线。provider-owned workspace 不进入 Graph lifetime
  plan；融合路线没有可行资源时，在未接受工作前显式退回 operation fallback；
- schedule 已先于 lifetime planning；workspace alias 只发生在 partial-order 证明可比较的 chain 内，跨 lane
  unordered lifetime 即使共享同一 resource 也不会因拓扑序偶然复用；旧单 target `prepare_steps` 路径已删除；
- G78/G80 的 Windows 纵切已经贯通：delegated operation/region 使用无需 QueueBinding 的 synthetic lane；
  shared↔delegated 与 delegated↔delegated 依赖在冷路径按 exact endpoint 选择 ordered、non-blocking import 或
  可拒绝 host wait；取消、异步失败、丢弃 pending handle、独立 delegated sinks 与整次 Invocation/workspace
  保活均有定向覆盖。shared segment 仍走原 QueueSequence，delegated 每 island 只返回一个真实 Completion；
- G77 已进入产品实现：默认 `ExistingOrderOnly` 在独立 lane 的 workspace 峰值超出 storage hard budget 时
  失败；显式 `AllowSerialization` 才会用稳定 step identity 添加无环顺序边、重新调度并重新运行 lifetime
  planner。新增边数、最终 peak 和 segment dependency 均可查询；预算本来足够或低于单 chain 峰值的边界
  有定向测试；
- Windows 当前 Graph/oneDNN/CPU 七组定向目标通过，并覆盖 exact region input/output/workspace、真实 boundary
  copy、provider-owned workspace、resource-route fallback 与 budget serialization；
- 2026-09-14 的 Linux 资格回放在 Ubuntu 24.04/GCC 14 上通过 `hyle_graph_test`、
  `hyle_graph_executable_test`、`hyle_compute_test`、`hyle_compute_ops_test`、
  `hyle_compute_backend_cpu_test` 与 `hyle_compute_provider_native_test` 的 ASan、UBSan 和 TSan；TSan 使用
  `setarch x86_64 -R` 取得可用 shadow layout，未报告 data race；
- Windows `SelectedComponentSmoke(graph)` 已通过 source embedding、安装并重定位后的 shared/static C++
  consumer，以及 shared export hygiene（14 个 HYLE library、1088 个导出符号）。这证明 G81 的 Windows
  三交付形状，不替代 Apple 交付或 CUDA factor 的独立资格；
- 同一 `4d674113` git archive 已在 MacBook Air 与 Mac mini（macOS 26.6.2、AppleClang 21、ARM64）完成
  六组 Compute/Graph 定向测试；两机的 `SelectedComponentSmoke(graph)` 也都通过 source embedding、
  安装并重定位后的 shared/static consumer。ARM 产品自然选择 native provider，oneDNN/CUDA 均未污染
  portable Graph closure；
- shared-only P0 结构复核通过：普通三节点单 lane 仍为一个 shared segment、一次 QueueSequence
  finalization、零 delegated/serialization/dependency，并由既有 metadata allocator 探针覆盖 steady submit
  零 Graph allocation；owner 判别按 segment 而非 node。旧 v2 P1 不换 revision 冒充当前证据；
- RTX 4070/CUDA 13.3 真机纵切通过：一个 CUDA target 的 compute/transfer lanes 共用一个 device storage，
  未产生 segment 的 compute lane 不要求 Queue；transfer lane 的两个 copy segments 能 cold capture、以
  hybrid plan 重放。独立 backend probe 证明同 Context 两条 stream 以 CUDA event 非阻塞导入 Completion，
  延迟 producer 写入可由 consumer stream 正确观察；
- CUDA Graph P2 纵切已迁移到 target/storage/lane v7 合同；同一 4 MiB CUDA→CPU Unary→CUDA 语义先完成
  功能回放，随后由下述冻结合同正式签发；
- CUDA factor 的 bounded loss/race probes 已通过：typed asynchronous terminal failure 与并发
  wait/abandon/drain/reaper 均有真机回放；Graph 产品根的 CUDA-on source embedding、安装并重定位后的
  shared/static SDK 也全部通过，三种 consumer 均实际运行 CUDA async fill，而非只检查符号；
- CUDA factor 当前 revision 的 P1 已为 `PARITY/PASS`。Graph 当前 `7718b163` 的唯一正式 P2 也在 W1-WIN
  于 `n=15` 停止：同一 4 MiB CUDA→CPU Unary→CUDA 完整语义的八个 strata ratio=
  `1.24305--1.26042`，全部 `WIN/PASS`，综合 exact=`WIN`、NI=`PASS`；所有 controls 与 identity 门通过。
  retained bundle 见
  [`graph-cuda-pipeline-p2-v1-7718b163-w1-win`](../../../../evidence/performance/graph-cuda-pipeline-p2-v1-7718b163-w1-win/)；
- 一般 multi-stream capture 与 CUDA operation provider 不冒充本轮基础资格。

## 七问（CLOSED）

1. 第三方：Graph 本体不链接第三方；上述版本只提供设计/consumer 证据，具体 provider 各自拥有依赖。
2. 功能：三轴、资源约束、partial-order workspace、single-lane convenience 与 delegated island 覆盖推理和
   renderer 两条纵切；模型策略和任意控制流明确在外。
3. 复用：复用 Mem resource/lifetime、Compute Queue/Completion/provider，不私造 allocator/event/threadpool。
4. 性能：single lane 无新增热搜索；multi-lane reuse 先保证 HB，冷搜索有硬预算，steady 按 segment/lane。
5. 交付：纯 C++；旧 callable/C ABI 不恢复，Native FFI N/A。
6. 构建：Graph 本体无 vendor 链接；backend/provider 独立 factor，三种交付使用同一组件图。
7. 易用：普通用户只给 Context/resource/Queue；高级用户显式 targets/storage/lanes/budget，并从 report 看全部
   copy、wait、serialization 和 island。

## 终局红队、限制与重开条件

- red team：所有公开 entry 回映用户任务；旧 domain identity 已从当前 public/user/architecture surface 清除；
  Graph 不私造 allocator/event/threadpool，不依赖 Sched，不在 steady submit 搜索 provider 或分配 Graph
  metadata。workspace reuse 只发生在 happens-before 可证路径，host wait 是 cold-selected report fact；
- performance：P0 单 lane 是一个 segment/QueueSequence、零 Graph allocation；正式 CUDA P2 在八个 strata
  全部 `WIN/PASS`，但只签发该完整跨域用户任务，不外推为 leaf kernel 或任意 DAG 胜率；
- 七门全部 `PASS`；候选是 `7718b163946104989c9a000f19fd01f454211d36`；
- 限制：当前 DAG 没有任意 `if/while`、模型/KV/batching/sampling 策略；capture 只承诺 shared single-queue
  segment；delegated mock 不冒充正式 MLX，CUDA runtime 不冒充 CUDA operation provider；
- 重开：真实 provider 证明三轴、dependency 或 island seam 无法表达；可复现 schedule/workspace/lifetime
  错误；公开 Graph 合同变化；或合格同合同证据证明 material plan/replay 税；
- LOC：candidate 上 `graph` implementation `8,454` 行；自实现不含测试全库 `116,292` 行。

`graph` v7 本轮 `CLOSED`。后续算子与 provider 只扩 typed op/region，不以此重开基础 plan/runtime。
