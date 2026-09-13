# Graph v7：Execution lane 与 delegated provider island

## Closure card

- component / owner：`graph` / Graph；round=`graph-v7-execution-lanes-and-islands`；
- baseline commit：`0ce95e2d2f40280163714165d27c2434c5b57771`；[旧 v6](graph-v6-runtime-specialization-placement.md)
  candidate `db8d1d335c84fb89675f6be2c2350efc6e35b271` 是待迁移实现，不再证明新资源合同；
- trigger 与证据：v6 的 `QueueBinding` 以 `ExecutionDomainId` 为键，`prepare_replay` 要求每个 active domain
  恰好一个 Queue，并拒绝重复 domain；这无法表达同一 CUDA allocation domain 上 copy/compute 多 stream，
  只能通过复制 domain 伪造，进而污染 workspace、value identity 与 liveness；同时所有 region 被假定能
  录入 HYLE Queue，不能诚实承载 delegated runtime；
- 正常 consumer 任务：调用方明确提供一组 storage domains 和 execution lanes；Graph 在约束内选择
  placement 与 lane schedule，编译 copy/compute/dependency，Invocation 把 shared lanes 绑定到 exact Queue，
  并把 delegated island 的真实 Completion 聚合为一个终态；
- 主产品纵切：单 CUDA device/resource 上 H2D copy lane → compute lane → D2H copy lane，重复 prepare/
  capture/submit，无隐藏 host wait；
- 正交反偏置纵切：CPU 单 domain/单 Queue 保持短路径；mock delegated region 与 shared Queue segment 混合，
  验证 island 边界和 final aggregation不依赖 CUDA；
- 跨模块 public journey：Compute 提供 Queue/Completion/provider runtime mode；Mem 提供 resource/lifetime；
  Graph 拥有 domain placement、lane schedule、boundary/dependency 和 report；Sched 不进入 Graph 基础产品；
- scope：增加 `ExecutionLaneId` 与 compile-time lane declarations；把 QueueBinding/report/segments 从 domain
  route 迁为 lane route；实现同 domain 多 lane dependency/capture；消费 Compute delegated contract并形成
  island segment；为未来 structured region 保留内部 typed boundary，但不增加公开控制流；
- non-goals：横向补 operation catalog/native route、通用 `if/while`、模型/KV/batching/sampling、自动线程池、
  provider timing oracle、公开第三个 Plan 产品、正式 MLX/Metal/Vulkan provider、Native FFI。允许为
  G70--G76 实现最小私有 probe command/provider；它只证明结构 seam，不自动成为公共能力；
- 直接依赖 / 消费者：依赖 Core/Error/Compute/Mem/compute_ops，私有使用 Container/Util-core；直接消费者
  是 Compute/Graph 用户、CUDA probe、oneDNN region 与未来 MLX provider；
- 性能 tier：`P0` 先证明单 lane 热路径不增加搜索/分配且 finalization 为 O(active lanes)，多 lane plan 为
  O(V+E+L)；真实 CUDA candidate 后才运行同合同 P1/P2，material margin 5%；
- 适用机器：Windows/RTX 4070 为真 CUDA lane/capture；Windows mock/CPU、WSL、两台 Apple 验证通用语义和
  三交付；MLX 真资格在独立 provider card；
- 调查与纠正预算：公开 lane API 两轮，plan migration 两轮，dependency/aggregation 两轮；
- 预声明验证：zero/one/multi lane、duplicate/foreign/missing Queue、same-domain cross-lane dependency、
  same Queue bound to several lanes、copy/compute overlap、device wait/host-wait reject、capture identity change、
  partial failure、late error、cancel/abandon、workspace/liveness 不因 lane 复制、OOM、并发 Invocation、旧
  single-domain convenience、逐头、sanitizer、三交付、真 CUDA probe、P0及适用 P1/P2；
- stop / blocker：如果 CUDA 真机不能用 lane 模型表达无 host round-trip 的依赖，退回该 seam；若 delegated
  provider 只能隐藏等待，则保持明确 unsupported/BLOCKED，不能用 mock 宣称 Apple 资格。

## 审计结论与 API freeze

### 质因数裁决

`ExecutionLane` 与 Graph plan/Invocation 共用存储、调度、report 和直接消费者，不拆 factor；`graph_plan`
仍没有独立公开 owner。具体 CUDA/MLX/Metal provider/backend 保持独立 factor。`compute_sched` 不因 lane
存在而启动：CPU worker 数、P/E 核和 NUMA 是 Sched/Platform 组合策略，不是 Graph lane layout。

### 冻结公开形状

```cpp
ExecutionDomain domains[]{
    {.context = &cuda_context, .workspace_resource = device_resource},
};
ExecutionLane lanes[]{
    {.domain = ExecutionDomainId{0}, .role = ExecutionLaneRole::kCompute},
    {.domain = ExecutionDomainId{0}, .role = ExecutionLaneRole::kTransfer},
};
auto compiled = HYLE_TRY(graph::compile(
    graph, CompileEnvironment{.domains = domains, .lanes = lanes}, options));

QueueBinding bindings[]{
    {.lane = ExecutionLaneId{0}, .queue = &compute_queue},
    {.lane = ExecutionLaneId{1}, .queue = &copy_queue},
};
HYLE_TRY(invocation.prepare_replay(bindings, replay_options));
auto done = HYLE_TRY(invocation.submit(bindings));
```

- `ExecutionDomainId` 只表示 storage/context locality；`ExecutionLaneId` 是本次 CompileEnvironment 的稳定
  ordinal，绑定一个 domain 和通用 role hint。role 只帮助确定性 cold scheduling，不承诺 vendor engine；
- lanes 为空时为每个 domain 隐式建立一条 `kCompute` lane，保持普通单域 API；显式 lanes 必须覆盖所有
  将执行 step 的 domain；
- `QueueBinding` 以 lane 为键。多个 lane 可绑定不同 Queue；若多个 lane 刻意绑定同一 Queue，prepare 会
  合并物理 admission/finalization并保持逻辑依赖；
- step、segment、dependency 和 replay route 都保存 lane，value/physical instance/workspace 仍只保存 domain；
- delegated lane 不要求 QueueBinding，其 endpoint 和 Completion owner 来自 provider invocation；是否可与
  shared lane device-wait 在 cold preparation 判定，不支持时只产生可拒绝的显式 host wait；
- lane selection 是确定性有界结构启发式。默认 operation 使用 domain 的 compute lane，boundary copy 优先
  transfer lane；只有 dependency 和 backend capability 证明可重叠时才形成并行 segments。

### 能力与有限 FIX

| ID | 缺口 | 允许改动 | 验收 |
|---|---|---|---|
| G70 | domain 同时承担 storage 与 queue identity | 新增 LaneId/Lane/role 与验证 | 同 domain 多 lane，workspace 不复制 |
| G71 | QueueBinding 按 domain | 改为 lane binding并保留自然默认 convenience | invalid/duplicate/foreign 精确失败 |
| G72 | scheduler/segment 按 domain | domain placement 后独立 lane assignment | 确定性、O(V+E+L)、无热搜索 |
| G73 | dependency/capture 只识别 domain Queue | 按 lane 构图、预绑定 route、聚合 physical Queue prefix | device wait 或显式 host wait，无隐藏同步 |
| G74 | report 混淆存储和执行 | 增加 lane/endpoint/runtime-mode facts | domain/lane/copy/island 可分别查询 |
| G75 | region 只能 shared Queue enqueue | 消费 Compute delegated SPI，生成 island segment | mock async lifetime/error/cancel 全通过 |
| G76 | flat IR 会堵死未来 region | 私有 typed region boundary；v1 只承载 partition | 当前 DAG/API 不增加控制流或热成本 |
| G77 | 构建、guide 与旧 v6 迁移 | 更新三交付和用户选型；删除 domain=queue 说法 | source/static/shared 一致 |

只允许 G70--G77。`compute_ops` 的 7 个未实现 route和 attention/catalog 横向扩张都在基础四探针完成前
冻结；确实为某条探针必需的最小私有 kernel/adapter 可以实现，验证后逐项 KEEP/PRIVATE/DELETE，不能用
这些 probe 提高算子完成度或掩盖 runtime 问题。

## 七问（AUDITED）

1. 第三方：Graph 不直接链接第三方；CUDA/MLX 等由独立 factor 拥有版本和许可证。
2. 功能：同 domain 多 lane、single-lane convenience 和 delegated island 覆盖推理与 renderer 两条纵切。
3. 复用：复用 Mem resource/lifetime、Compute Queue/Completion/provider，不私造 stream/event/pool。
4. 性能：domain 与 lane 分离不进入 per-element；single lane 无新增热搜索，多 lane event 数按 segment/lane。
5. 交付：纯 C++；旧 callable/C ABI 不恢复，Native FFI N/A。
6. 构建：Graph 本体无 vendor 链接；具体 backend/provider 是可选 factor，三种交付同图。
7. 易用：默认一 domain 一 lane；高级用户显式给 lanes/queues，并从 report 看所有 copy/wait/island。

本卡已完成范围与有限 FIX 审计，`graph` 依法从旧 `CANDIDATE` 返回 `AUDITED`。G70--G77 形成 candidate
前，不运行算子 breadth 工作，也不把 mock provider 当作 Apple/CUDA 完整资格。
