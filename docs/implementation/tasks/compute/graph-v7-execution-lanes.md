# Graph v7：执行 lane 与资源预算

## Closure card

- component / owner：`graph` / Graph；round=`graph-v7-execution-lanes`；
- baseline：`e045411e6072f726c67cb25775a7f82b0b3d6ac4`；
  [v6](graph-v6-runtime-specialization-placement.md) candidate 为
  `db8d1d335c84fb89675f6be2c2350efc6e35b271`；
- trigger：v6 把 `ExecutionDomain` 同时用作 storage/placement domain 与唯一 Queue identity。真实 CUDA
  consumer 若要在同一 device/context/resource 内用 compute stream 与 H2D/prefetch stream 重叠，只能伪造
  两个 domain；这样会错误地产生物理 value、workspace 和跨域 copy。vLLM、SGLang 及 ds4 的真实路径均把
  device placement 与 compute/copy/communication stream 分开，这是可复现的产品表达缺口；另一个缺口是
  exact per-domain LifetimePlan 生成后没有 hard workspace budget，调用方不能在 compile 时阻止 overcommit；
- 主纵切：调用方已经明确 CPU/CUDA placement 的 prefill/decode fragment；同一 CUDA domain 使用 primary
  compute lane 与 transfer lane，把跨域 Copy 和后续 compute 用 device dependency 连接，一次冷准备后重复
  replay；
- 正交纵切：renderer/codec 的 upload -> compute -> download，证明 lane 不是 transformer 专名；
- 直接依赖 / 消费者：依赖既有 CLOSED Core/Error/Mem、Compute v5 candidate 与 CUDA backend；不改变
  Compute provider/backend SPI。直接消费者为 Graph 用户和未来推理 runtime；
- scope：把 storage/placement domain 与 execution lane 分离；为空的 lane 列表提供每 domain 一个 primary
  compute lane 的短路径；为 Copy 选择同 domain transfer lane（没有时使用 primary lane）；把 submit/
  prepare、dependency、capture、Completion 与 report 改为 lane identity；增加 per-domain workspace hard
  budget；保持 v6 的显式 placement、region、copy、liveness、alias 和 warm-route 合同；
- non-goals：逐节点 timing profile、HEFT/自动跨设备放置、TP/PP/DP、模型层/权重切分、KV/batch/admission、
  在线负载均衡、自动 benchmark、CPU worker/process/NUMA/page 策略、provider kernel tile/cache blocking、
  communication lane 或任意 DAG 全局最优。粗粒度设备/持久权重 placement 仍由推理 runtime 决定；CPU
  execution pool 由独立 Compute/Sched factor 审计；
- 性能 tier：P0 必须证明 lane 在 compile/prepare 固定，warm replay 不查策略、不分配、不搜索；P1/P2 只在
  immutable candidate 上比较相同 placement/copy/completion 的单-lane 抽象税，以及真实 pinned H2D +
  compute 能否产生 overlap。固定 5% 工程门；不以 synthetic scheduler score 冒充设备性能；
- 机器：W1-WIN 负责 CUDA 与 P1/P2；WSL GCC 14 负责 sanitizer/package；MacBook Air/Mac mini 负责
  portable correctness；
- 预算：五个本地推理来源和 oneDNN/oneTBB 各一次定向源码对账；API/算法两轮；每个 FIX 两轮；candidate
  后每个平台、三交付、P1/P2 各一次。达到预算未裁决则 BLOCKED，不扩建公共性能仪器；
- 预声明验证：implicit primary lane、显式 compute/transfer lane、重复/缺失/错误 context binding、同 domain
  不复制、跨 domain Copy 选 transfer lane、device/host dependency route、single/multi-lane capture、lane report、
  workspace exact/equal/exceed/overflow、OOM rollback、warm allocation/query、现有 Graph/Compute/CUDA 回归、
  逐头、ASan/UBSan/TSan、三平台、source/static/shared、P0、必要 P1/P2 与终局红队；
- stop：若 multi-stream capture 需要修改 Compute backend contract，另开 Compute/CUDA 卡并只精确阻塞该
  slice；若某种通信/辅助 lane 没有 typed operation/provider consumer，不为未来猜测扩 enum；若 overlap
  需要改变 placement 或业务 offload 策略，留给推理 runtime。

## 成熟实现对账与职责裁决

本轮锁定本地 clean revision：ggml `e91ded11bdcd`、llama.cpp `e71b80510c84`、vLLM
`51da0ca66c80`、SGLang `dcebe8c4733a`、ds4 `f62ca29a3087`、oneDNN `6b7f1fbbb18e`、
oneTBB `e00c68dcac0e`。

- llama.cpp/ggml 由 loader/runtime 决定 GPU layers、tensor split、CPU NUMA 与 decode/prefill threadpool；
  Graph backend scheduler 尊重已有 tensor/backend placement，再发出 split/copy/event。它没有逐 replay
  运行 timing oracle；
- vLLM 由 TP/PP/DP 决定进程和 device，profile model/activation/CUDA Graph 后用剩余显存确定 KV budget；
  offload 使用 pinned host backing、持久 copy stream、event fork/join 与提前若干 layer 的 prefetch；
- SGLang 把 worker 进程及内存绑定到 GPU 所在 NUMA node，kernel 再按 shape/cache block 裁剪线程；GPU
  runtime 独立维护 transfer/capture/communication side streams；
- ds4 使用持久 CPU pool 和显式 GPU stream；layer packing 在模型 runtime 按扣除 scratch/vendor workspace/
  safety margin 后的 device budget 进行，而不是交给通用 Graph 逐 node 猜测；
- oneTBB 的 arena 只规定 concurrency/NUMA/core type 上限，oneDNN kernel 再按 work amount 和 tile 缩小实际
  task 数。共同边界是“外层给资源预算，内层按具体工作裁剪”，不是让 Graph 读取 L1/L2/L3 后替 kernel
  选 tile。

因此资源权力固定为四层：

1. 推理 runtime/部署层拥有 device set、进程/TP/PP/DP、模型与持久权重 offload、显存/KV 总预算；
2. Compute/Sched runtime 拥有持久 CPU worker pool、CPU set/NUMA affinity 和同一 device 的 Queue 集；
3. provider/kernel 拥有每个 specialization 的 task/grain/tile、cache blocking、ISA 与 per-worker scratch；
4. Graph 拥有 DAG、region/boundary、workspace lifetime、execution lane、依赖、capture 与重复 replay。

L1/L2/L3 是 provider 的冷路径输入，不是 Graph 公共参数；OS page/NUMA/large-page/prefetch 由 Mem/loader
在知道 backing 生命周期时应用；进程数和模型分片需要模型及部署事实，不能由 HYLE Graph 单独推断。

## 能力与自然形状

| 能力族 | 冻结语义 | 当前 | 动作 |
|---|---|---|---|
| explicit domain placement | required/preferred 与 external resource 仍决定物理 storage domain | COMPLETE | KEEP |
| storage domain / execution lane separation | 同一 domain 可有 compute 与 transfer lane，不复制 value/workspace | MISSING | FIX |
| implicit simple path | 未给 lanes 时每个 active domain 自动产生一条 primary compute lane | MISSING | FIX |
| deterministic lane route | operation/region 走 compute；Copy 优先同执行 domain 的 transfer，否则 compute | MISSING | FIX |
| lane binding/dependency | replay 按 lane 绑定 exact Queue；跨 lane 使用已准备的 ordered/device/host route | PARTIAL | FIX |
| multi-lane capture truth | 能 capture 才报告；prefer 可显式回退 commands，require 明确失败 | PARTIAL | FIX |
| workspace budget | 每 domain 可选 hard limit；exact LifetimePlan 超限失败 | MISSING | FIX |
| report | domain、lane、role、segment、Queue/capture/workspace 要求分别可查 | PARTIAL | FIX |
| warm route | submit 不重新选 lane/placement/provider，不新增 heap/JIT/隐藏 copy | COMPLETE | KEEP/REPLAY |
| automatic heterogeneous placement | 模型 runtime 显式配置优先；Graph 保留 v6 确定 locality fallback | OUT_OF_SCOPE | REJECT |

冻结公开草图：

```cpp
graph::ExecutionDomain domains[]{
    {.context = &cpu, .workspace_resource = host},
    {.context = &cuda, .workspace_resource = device,
     .maximum_workspace_bytes = 512_MiB},
};
graph::ExecutionLane lanes[]{
    {.domain = graph::ExecutionDomainId{0},
     .role = graph::ExecutionLaneRole::kCompute},
    {.domain = graph::ExecutionDomainId{1},
     .role = graph::ExecutionLaneRole::kCompute},
    {.domain = graph::ExecutionDomainId{1},
     .role = graph::ExecutionLaneRole::kTransfer},
};
auto compiled = HYLE_TRY(graph::compile(
    graph, {.domains = domains, .lanes = lanes}, options));

graph::QueueBinding queues[]{
    {.lane = graph::ExecutionLaneId{0}, .queue = &cpu_queue},
    {.lane = graph::ExecutionLaneId{1}, .queue = &cuda_compute},
    {.lane = graph::ExecutionLaneId{2}, .queue = &cuda_copy},
};
HYLE_TRY(invocation.prepare_replay(queues));
```

`domains/lanes` 只在 compile 期间借用。Executable 保存 compact lane plan；Queue 仍由 caller/runtime 拥有并
在 prepare/submit 期间借用。每个 domain v7 至多一条 compute 和一条 transfer lane；重复 role 明确失败，
避免在没有 timing/typed semantics 时假装能决定多 compute streams。单 domain convenience 和空 lanes 均
路由 implicit lane，不迫使普通 CPU 用户理解 lane。

## 有限 FIX ledger

1. **G70**：增加 `ExecutionLaneId/Role/ExecutionLane`；`CompileEnvironment` 接 lanes；
   `ExecutionDomain` 增加 optional workspace hard limit；删除 domain=Queue 的错误注释与字段含义；
2. **G71**：plan/segment/report 同时保存 domain 与 lane；operation/region -> compute，Copy -> transfer-or-
   compute；same-domain lane 切换只产生 dependency，不产生 physical copy；
3. **G72**：`QueueBinding` 改为 lane identity；prepare/submit/capture/completion capacity 与 final aggregation
   按 physical Queue 去重，保持 exact owner 和 late-completion 生命周期；
4. **G73**：per-domain LifetimePlan 完成后检查 hard budget；report 加 lane count/role/queue/capture 和预算
   事实，不添加估计时间或“最优”措辞；
5. **G74**：补上述 correctness/P0/consumer-shaped tests，更新 architecture/user 文档；回放受影响的
   Graph/Compute/CUDA、三交付、sanitizer 与一次候选 P1/P2。

不在 G70--G74 内增加 operator、provider SPI、进程/线程池、模型配置、在线 profiler 或 cost database。

## P0 目标

- compile：保持 v6 的有界 placement/region search；lane route 为 `O(steps + domains + lanes)`，workspace
  budget 检查为 `O(domains)`；
- cold metadata：新增 `O(lanes)`，由 caller allocator 承担；OOM 全回滚；
- warm submit：保持 `O(bindings + steps + dependencies + segments)`，零 Graph heap、lane/placement/provider
  search、shape inference、JIT 和隐藏 copy；finalization 按 distinct physical Queue/terminal segment；
- 性能声明：Graph 只保证把已知结构映射到显式 lane 并消除可避免的 host serialization；是否获得 overlap
  必须由 pinned memory、真实 device dependency 和 consumer P2 证明，不能从 lane 数推断。
