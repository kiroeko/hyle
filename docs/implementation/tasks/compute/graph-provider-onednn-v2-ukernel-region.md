# Graph oneDNN provider v2：BRGEMM ukernel MLP region 恢复轮

## Closure card

- component / owner：`graph_provider_onednn` / Graph；round=`graph-provider-onednn-v2-ukernel-region`；
- baseline commit：`222d820e0b90492f5a349ab6382ac93090c09427`；previous blocker：
  [`graph-provider-onednn-v1`](graph-provider-onednn-v1.md)；
- trigger 与证据：v1 证明 oneDNN Graph 公开 execute 固定分配，但并未否定 oneDNN 作为 region kernel
  来源。已 `AUDITED` 的 `compute_provider_onednn` v2/v3 证明锁定 3.13.2 x64 BRGEMM ukernel 在 caller
  scratch 上 warm execute 零 allocation/JIT/copy；已 `CLOSED` 的 SIMD 提供 split-input
  `swiglu_f32_batch`。两者可在不修改 cached dependency source、不借 private Graph ABI 的前提下形成
  真实多节点 MLP region，满足 v1 registry 的第二条恢复条件；
- 正常 consumer 任务：用户为 Graph Context 显式组合 Graph-only oneDNN region provider 与 native
  fallback，冷编译固定 f32 gated MLP；每个 Invocation 取得独占 workspace，反复绑定 activation/output，
  一次 region submit 完成 gate/up/down projection 与 SwiGLU，最终返回覆盖 ordered Queue prefix 的
  Completion；
- 主产品纵切 / 具名来源：`RMSNorm -> {gate MatMul, up MatMul} ->
  GatedActivation{kSilu,kSeparate} -> down MatMul`。region 精确覆盖后四个连续拓扑 node，RMSNorm 保持
  provider-neutral 相邻 step；shape 取 llama.cpp/ggml、ds4、vLLM 与 SGLang 的 decode/短 prefill；
- 正交反偏置纵切：caller-owned dense f32 `M x K -> M x H -> M x K` branched MLP，含非 tile 整倍数的
  `M/K/H`，不使用模型、GGUF、token 或量化概念；
- 跨模块 public journey 与每段 owner：Compute 拥有 Context/Queue/Completion/Tensor 与 provider
  identity；Graph 拥有 DAG、region selection、constant binding、workspace/liveness、Executable/Invocation
  和 report；本 factor 仅拥有 gate/up/down BRGEMM artifacts、per-Invocation ukernel scratch 与两个
  `M x H` 中间 buffer。SIMD 只执行 gated activation；调用方/Graph 继续拥有输入、权重与输出 storage；
- scope：审计精确 pattern/boundary、BRGEMM artifact 和 workspace layout、SIMD alias、executor tiling、
  provider-only API、extension vtable、错误/OOM/并发/Completion、oneDNN 共用依赖配置、可选构建/安装与
  P1/P2 同合同；审计完成后只实现冻结的 gated f32 region；
- non-goals：恢复 oneDNN Graph API、增加通用 CPU 算子目录、单-op prepare、量化/bf16/f16/稀疏、
  Attention、RMSNorm fusion、动态 shape、GPU/SYCL、训练、修改共享 `.deps`、公开 oneDNN 类型、模型/KV/
  batching/sampling/serving、Native FFI；
- 直接依赖 / 直接消费者：SCOPED 时仍无依赖。候选 public dependency 为 `compute`；实际 private
  dependency 为 `compute_ops`、`error`、`graph`、`math`、`mem`、`simd` 和
  `compute_provider_onednn`，外部依赖为 locked oneDNN；不得让 Compute 或
  `compute_provider_onednn` 反向依赖 Graph。直接消费者是 Graph compile/Invocation、Graph package
  consumer 与完整 transformer MLP 纵切；
- 性能 tier 与原因：`P1` 证明 steady allocation、region boundary 与小形状开销，`P2` 比较完整 gated MLP
  admission-to-terminal-completion。比较对象为同一 Graph 的逐 op oneDNN/native sequence，以及同 shape、
  layout、threading、workspace owner 的 direct ukernel+SIMD sequence；固定 5% 非劣门；
- 适用机器 / 仪器 / material margin：Windows W1-WIN x64 资格化，WSL Linux x64 回放 correctness/
  sanitizer/package；两台 Apple 只验证 AUTO/OFF、显式 ON 诊断与 Graph native fallback，不签 x64 性能；
- 调查与纠正预算：pattern/Graph SPI、existing ukernel、SIMD、dependency/package 各一次源码审计；一个
  scratch region prototype；API/implementation、correctness/concurrency、delivery 各最多两轮；P1/P2 各
  一次 preflight 与一次根因纠正；若共享 ukernel owner 确需新 factor，必须先登记，不在实现中私造 target；
- 预声明验证：pattern 正反例、shape/stride/overflow、scalar differential、constant/boundary ordering、
  workspace size/alignment/OOM/rollback、两个并发 Invocation、abandoned Completion、unsupported/fallback、
  warm allocation=0、dispatch/copy/sync/report truth、nested executor、逐头、Windows/WSL/Apple 适用格、
  ASan/UBSan、source/static/shared、exact factor/product selection、offline lock/license、P0/P1/P2 和红队；
- stop / blocker 条件：若 region 必须复制既有 single-op provider 的可漂移实现、需要 Graph/Compute
  反向依赖、不能在 Graph v5 的显式结构策略或严格估值策略下诚实 admission、warm path 分配/JIT/隐藏 copy、内部 dispatch 无法如实报告，
  或任一正常/holdout shape 在公平 P2 material 固定输且无有限根因，则精确 BLOCKED；不得用测试 provider、
  苹果对橘子、强制 selection 或单个大 GEMM 胜利冒充 region 成立。

## SCOPED 待审计项

1. 四个 HYLE node 的拓扑、producer/input/constant 条件如何在 O(V) propose 中无误识别，哪些合法变体
   应拒绝而走 fallback；
2. gate/up 可否共享 shape artifact，down artifact 如何复用现有 x64 BRGEMM helper，而不复制一份会独立
   漂移的 provider implementation；若需要 common prime factor，其最小 owner/target/安装形状是什么；
3. 两个 `M x H` intermediate 是否足够，SwiGLU 原地覆盖 gate 后的 liveness、alignment、overflow 与
   ukernel per-task scratch 如何形成一个 caller workspace；
4. region-only provider 的普通 C++ 入口、identity/name/version、与 single-op oneDNN/native provider 的
   组合顺序和 guide 心智；未选 Graph 时不得给 Compute 增加 Graph 依赖；
5. `ParallelExecutorRef` 的 owner、nested progress、两次 sibling MatMul 是否串行或可安全并行，以及
   `uses_private_threads=false` / `blocks_host=true` / concurrent Invocation 的真实 report；
6. oneDNN dependency option、target、package dependency 和 license 如何由两个可选 factor 共用而不重复
   下载、重复打包或让 Graph-only selection失效；
7. P1/P2 的逐阶段 charging、direct comparator、双顺序、A/A 与 holdout shape；net-savings estimate 必须
   来自静态差异和合格证据，不能随意填正数来骗过 conservative selector。

本卡只授权审计。进入 `AUDITED` 并冻结有限 ledger 前，不添加产品源、依赖、公开头、feature、benchmark
或 Graph 修改；v1 的 oneDNN Graph public-API blocker 仍是历史事实，不因本次恢复被改写。

## AUDITED 结论（2026-09-13）

### 边界与自然形状

- Graph-only provider 必须是一个独立 `compute::Provider` 值：其 single-op `prepare` 返回
  `Unimplemented`，只通过 Graph v3 extension 提供 region。公开入口冻结为
  `hyle::graph::onednn::provider({.executor=...})`；它不要求用户构造 oneDNN 类型，也不让 Compute
  获取 Graph 依赖；
- 精确 pattern 为四个连续拓扑 node：两个共享 activation 的 dense row-major f32 MatMul、一个
  `GatedActivation{kSilu,kSeparate}`、一个 consuming gated result 的 dense row-major f32 MatMul。gate/up/
  gated 中间值必须各只有区间内唯一 consumer 且未 publish；其它 layout/dtype/attribute/额外 consumer
  均不提案并自然 fallback；
- provider context 保存 borrowed `ParallelExecutorRef` 与 Context metadata allocator。propose 用一次临时
  `value_count` 计数表和一次 O(V+E) 扫描验证 consumer/publish 约束，OOM 显式失败；不要求 Graph 增加
  benchmark-oriented reverse index；
- gate/up shape 相同，共用一个 immutable BRGEMM plan；down 使用第二个。每个 Invocation workspace 为
  两个对齐的 `M*H*f32` buffer 加 `max(gate_up_scratch, down_scratch)`，SwiGLU 原地覆盖 gate buffer；所有
  加法、乘法和对齐先检查 overflow；
- gate/up 在一个同步 executor episode 中按 N tile 执行，同一 task 在一次 oneDNN hardware context 内完成
  两次 BRGEMM，再逐 row 调用 `simd::swiglu_f32_batch`；barrier 后 down 在第二个 episode 执行。worker 数
  不计为 dispatch，因此 artifact `dispatch_count=2`、`blocks_host=true`、`uses_private_threads=false`；
- submit 只映射四个只读边界 tensor、一个输出和一次 Invocation workspace；non-coherent input 先
  invalidate、output 成功后 flush，所有 mapping 按 first-error 顺序 close。steady path不分配、不 JIT、不
  搜索、不 copy；同 Executable 的不同 Invocation 具有独占 workspace，可并发；
- oneDNN 不公开 JIT bytes，proposal 的 compile/persistent/workspace estimate 与 artifact persistent fact
  必须使用 Graph v5 的 unknown 语义，不能填零。短 preflight 已证明执行顺序足以使微小点值反转；本
  provider 不伪造可移植的正纳秒值，默认结构策略根据四节点覆盖、两个 execution episode 与零 copy/sync
  接纳，严格估值策略下自然不被选择。

### 能力与动作

| 能力族 | 层级 | 路线 | 覆盖 | 动作 |
|---|---|---|---|---|
| gated f32 MLP region | 复合机制 | 2 个 BRGEMM plan + HYLE SIMD + Graph workspace | `MISSING` | `BORROW/FIX` |
| region-only C++ provider | 原子配置 | 小型 Provider 值、borrowed executor、对象局部 extension | `MISSING` | `FIX` |
| 其它 dtype/layout/quant/attention | 其它能力族 | native/未来 provider 或 unsupported | `OUT_OF_SCOPE` | `DEFER` |

### 有限改动集

| ID | 问题与证据 | 允许改动 | 验收 |
|---|---|---|---|
| GP20 | 无真实生产 region provider | 新增公开 factory、Provider context 与精确 extension v3 | 普通 Context provider sequence 自然组合，single-op 不截流 |
| GP21 | pattern 可能吞掉外部 use | O(V+E) use/publish 扫描，只提案精确四节点形状 | 正反例、分支/published/extra-use fallback |
| GP22 | 不能复制 single-op BRGEMM | 私有依赖 OD40--OD44 seam，两个 plan、共享 gate/up artifact | 不含 dnnl 头/handle，无第二份 tiling/JIT owner |
| GP23 | 中间值与 scratch owner 缺失 | 一个 64-byte aligned caller workspace，两个 MxH buffer + 复用 scratch | overflow/OOM/alignment/lifetime/concurrent Invocation |
| GP24 | 多节点 submit 必须减少边界 | gate/up+SwiGLU 与 down 两个 executor episode；一次边界 mapping/Completion | warm allocation/JIT/copy=0，dispatch=2，scalar differential |
| GP25 | cost/report 不能伪造 | 采用 Graph v5 optional estimate 与结构 admission；P1 只检验抽象非劣和完整 MLP，不写回纳秒常数 | report 真值、双顺序、正常/holdout 5% NI |
| GP26 | 可选交付尚无 owner | public=`compute`；private=`graph,simd,compute_provider_onednn`；同一 oneDNN option/lock/package owner | OFF 不下载，ON 三交付，Apple AUTO/OFF + native fallback |

### API / usability freeze

```cpp
const std::array providers{
    hyle::graph::onednn::provider({.executor = executor}),
    hyle::compute::onednn::provider({.executor = executor}),
    hyle::compute::native::provider({.executor = executor}),
};
auto context = hyle::compute::Context::create(
    hyle::compute::cpu::device(), {.providers = providers});
```

Graph region、oneDNN single-op 与 native fallback 都由调用方显式排序；没有进程注册表或隐藏线程池。
executor backing 必须活过 Context 及其 artifacts/invocations。Graph-only provider 不可单独执行 operation；
不支持的图只返回 `Unimplemented` 给冷路径选择，不在 submit 时改变路线。

审计没有要求新的 semantic factor：BRGEMM 只有一个 mechanism owner 和一个直接私有消费者，单向内部 seam
足够；另造公共/common product 会增加身份而不改善生命周期。`graph_provider_onednn` 现可进入 `AUDITED`，
实现只限 GP20--GP26，并以前置 Graph v4 与 Compute oneDNN v4 ledger 为约束。

## Candidate

- immutable product commit：`49dcaa09139f79848af6ece1289e15739556ed9a`；
- public API：新增 `hyle/graph/providers/onednn.h` 与
  `hyle::graph::onednn::provider(ProviderOptions)`；公开面只暴露 borrowed
  `compute::ParallelExecutorRef`，不泄漏 oneDNN 类型；
- layout、算法和常态成本：只识别冻结的四节点 dense-f32 separate-input gated MLP；两个共享的私有
  BRGEMM plan、两个 `M*H` 中间区和复用 scratch 在冷路径固定。propose 为 `O(V+E)` 且一次临时计数
  allocation；每次 Invocation 拥有一个 64-byte aligned workspace；warm submit 两个同步 executor
  episode、一次边界 mapping、零 HYLE heap allocation/JIT/copy/provider search；
- correctness：Windows provider header 与 12 项行为测试通过，覆盖精确 region、decode/tail、single-op/
  published/extra-consumer 反例、严格 estimate 策略、相邻 RMSNorm、warm allocation、nested executor、
  并发 Invocation，以及 compile/Invocation OOM rollback/retry；
- 直接消费者与交付：`graph` 产品投影在 oneDNN ON 时包含本 factor，OFF 时 target 不存在且不下载；
  Graph package sample 冷编译并执行真实四节点 region。Windows Release/WERROR 的 source embedding、
  relocated installed shared 与 relocated installed static consumer 全部通过，安装包含精确 target、依赖头、
  package metadata 和 oneDNN license；
- candidate 后只允许本卡已声明的 P0/P1/P2、WSL sanitizer/package、两台 Apple AUTO/OFF/显式 ON
  诊断、红队和证据/文档变化。若公开 API、pattern、workspace、执行 episode 或产品依赖改变，必须先退回
  `AUDITED`。

## 红队与裁决

- 红队反证与处理：candidate 前 implementation review 已确认 ValueId 绑定不依赖常量传入顺序、所有
  size/alignment 算术复用 `math::try_*` 与 `mem::try_align_up`，不复制 BRGEMM mechanism；终局独立红队仍待
  candidate gates 后执行；
- 未决限制：qualified P1/P2、WSL ASan/UBSan 与交付、两台 Apple 可选能力诊断/native fallback、终局 P0
  和红队；
- reopen triggers：不适用，当前为 blocker 恢复审计；
- LOC：终局统计；
- candidate checkpoint：GP20--GP26 已在不可变产品提交中实现，Windows 行为与三交付通过；
  v1 的 oneDNN Graph API blocker 仍然有效且未被绕过。其余预声明 gate 未完成，不能提前写成
  `CLOSED`。

## 终局资格化与 BLOCKED 裁决（2026-09-13）

- 唯一 qualified W1-WIN 运行使用锁定 oneDNN 3.13.2、MSVC 19.50、Release static、portable ISA、
  LTO off，并完成预声明的 `n=15 -> n=31`。artifact/build identity、consumer provenance、A/A、5%/10%
  known-delta、timer-floor、placement 和 host/input stability 全部通过；bundle 与完整方向记录归档在
  [`benchmarks-compute-graph.md`](../../evidence/performance/benchmarks-compute-graph.md#graph-onednn-gated-mlp-region-p1p2)；
- decode natural-sequence、short-prefill fused-boundary 与 short-prefill natural-sequence 三个 workload
  都是八个 strata `PARITY`、non-inferiority=`PASS`。这证明完整自然 region 任务没有 material fixed loss，
  也没有把逐 op mapping/submission 成本偷偷从 Graph 一侧移走；
- 最小 decode fused-boundary 的八个 median ratio 为 `0.9622--0.9817`。五行是 `PARITY/PASS`，三行
  99.26% CI 下界 `0.9444--0.9495` 略穿过固定 5% 非劣界限，因此 combined exact=`INCONCLUSIVE`、
  engineering non-inferiority=`UNRESOLVED`。总 qualification gate 是 `PASS`，总 performance gate 是
  `INCONCLUSIVE`；不能把它改名为 parity 或继续追加样本寻找 PASS；
- 一次源码/P0 根因账单没有发现错误 kernel、allocation、JIT、copy 或 route search。相对 direct fused
  peer，Graph 必须额外完成 in-flight 原子 admission/terminal、external binding 完整性检查、五个
  `TensorView` staging、queue sequence、region vtable 与 Completion poll/settle。约 3% 中位方向来自当前
  通用 Graph 安全/组合语义；没有真实 consumer 依据允许增加 benchmark-only one-step fast path；
- 因工程性能门没有得到明确 `PASS`，本 factor 从 `CANDIDATE` 转为 `BLOCKED`，
  `blocked_from=CANDIDATE`。WSL sanitizer/package、两台 Apple 可选能力和终局独立红队在该终局 blocker
  后不再运行，也没有被写成 PASS；Windows 已取得的 correctness/三交付证据仍保留为事实；
- blocker owner：Graph runtime/plan 的通用 warmed-submit 成本合同，以及本 factor 的 5% P1 gate；
- resume condition：后续 Runtime/Graph 主干必须因动态 specialization、真实异步 backend、跨设备
  placement/copy/sync 或 capture/replay 等真实消费者需求形成新的产品 candidate；若该架构工作自然改变
  common submit path，才可重新从 `SCOPED` 审计本 factor、重建不可变 candidate 并运行一次新 P1。仅有
  更多样本、放宽 margin、删除安全检查或增加一条 oneDNN 专用 fast path 不构成恢复条件。

本 blocker 只属于可选 `graph_provider_onednn` 叶子，不阻塞 Compute/Graph 主干、CUDA/Metal/Vulkan
backend 或其它 provider 的合法施工；也不授权继续扩大 CPU 算子目录来摊薄固定成本。
