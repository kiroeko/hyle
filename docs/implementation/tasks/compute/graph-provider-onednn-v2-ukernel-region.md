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
- 直接依赖 / 直接消费者：SCOPED 时仍无依赖。候选 public dependency 为 `compute`，private dependency
  为 `graph`、`simd` 和 locked oneDNN；不得让 Compute 或 `compute_provider_onednn` 反向依赖 Graph。
  直接消费者是 Graph compile/Invocation 与完整 transformer MLP 纵切；
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
  反向依赖、不能给出正的可解释 net savings、warm path 分配/JIT/隐藏 copy、内部 dispatch 无法如实报告，
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

## Candidate

- commit：尚无；
- public API、layout、算法和常态成本变化：尚无；
- 直接消费者迁移：尚无；
- candidate 后仅证据/文档变化：尚未进入候选。

## 红队与裁决

- 红队反证与处理：待 AUDITED 后执行；
- 未决限制：上述七项；
- reopen triggers：不适用，当前为 blocker 恢复审计；
- LOC：终局统计；
- 最终状态与理由：`SCOPED`；恢复事实成立，但尚未批准实现或产品能力。
