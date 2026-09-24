# MR 真实模型画像路由系统重构

<!-- MODULE REVIEW: SCOPED -->

## Closure card

- component / owner：`compute`、`graph` 为共同合同 owner；实际迁移 owner 为
  `compute_provider_cuda`、`compute_provider_flashinfer`、`compute_provider_marlin`、
  `compute_provider_ggml_cuda`、`compute_provider_ggml_vulkan`、`compute_route_apple`、
  `compute_route_vulkan`、`graph_route_apple`、`graph_route_vulkan`。
- baseline commit：`af8a42e4c`（M3 错误候选已纠正，目标模型/权重/算子审计已固化）。
- trigger 与证据：
  [目标模型权重、量化与算子审计](llm-moe-model-weight-operator-audit.md)逐 header 证明：同一模型
  revision 有官方 checkpoint、动态 4-bit、Q8 等多个部署画像；单个画像又按 tensor role 混用
  BF16/F16/F32、Q8/Q6/Q5/Q4/Q3/IQ/MXFP4/FP8。现有 `RouteQuantizationRule` 只能描述单个
  operation 的一个量化 mode，固定的 input0/input1/output0 predicate 不能表达任意 arity；
  `WorkloadContext` 没有权重画像与 manifest 身份。这正好满足 Compute 的“真实 Provider 需要当前规则
  无法表达的模型/量化组合”以及 Graph 的“真实 consumer 需要新的冷资格维度或缓存合同”重开条件。
  这些旧类型只证明当前产品需要迁移，不构成新公开形状、字段或 matcher 算法的设计输入。
- 正常 consumer 任务：应用选择 CPU/CUDA/Apple/Vulkan 路线，提供模型、固定权重画像和真实
  Direct operation 或 Graph region；HYLE 在 prepare/compile 期按平台、硬件、驱动/toolkit、Provider、
  模型、画像、phase/region、实际 tensor/operation 条件绑定一个程序，重复 submit 不再搜索、计时或
  解释规则。
- 主产品纵切：本地 `Qwen/Qwen3.5-2B@15852e8` 的官方 BF16 与 Unsloth
  `UD-Q4_K_XL@f6d5376`，同一模型身份因画像、GDN/full-attention region 和实际 tensor 格式走不同
  合格规则；直接计算与 Graph 都重复执行两次。
- 正交反偏置纵切：Qwen3.8-2.4T `UD-IQ4_XS` 的 mixed IQ4/Q4/Q5/Q6/F32 MoE，以及 Kimi/DeepSeek
  的 MXFP4 expert + BF16/Q8 projection + F32 state/router。只使用容量缩小的真实结构和 header
  shape/format，不下载或冒充执行 2.4T checkpoint。
- 跨模块 public journey 与每段 owner：consumer 创建 owned `ModelIdentity` 和 owned weight/deployment
  profile identity（Compute）→ Direct `prepare` 或 Graph `compile` 携带 `WorkloadContext`
  （Compute/Graph）→ common matcher 合取环境、模型/profile、operation 与任意 tensor port
  （Compute）→ Provider 检查真实完整 DAG/region 并返回 `ProviderRouteKey`（各 Provider/Graph）→
  prepared artifact 重复 submit（四路线既有 owner）。
- scope：
  1. 从用户任务重新设计路由合同，不以现有 `RouteRule`、`WorkloadContext`、测试或 Provider 调用形状为
     兼容前提；先冻结事实 owner、身份、生命周期、选择和拒绝语义，再审计旧实现的删除/迁移清单；
  2. 增加独立、owned、可复制的模型制品 profile identity，含 publisher/repository、固定 model revision、
     configuration digest、variant、manifest digest 与 format family；它只标识部署制品，不冒充每个
     tensor 的真实 dtype/format；
  3. 路由 subject 支持 Direct operation 的全部实际端口，以及 Graph canonical boundary 的任意端口；
     predicate 支持方向、端口索引、dtype、`StorageFormatId`、rank 和有界/整除 shape 条件。量化 mode、
     group、轴、codes/scales 等来自实际 operation 参数和实际 tensor 端口，不再使用模型级 Q4/Q8 标签；
  4. Graph region 继续由 HYLE 管边界、Provider 检查完整真实 DAG；以 versioned region semantic、profile、
     canonical boundary 及真实 graph operation/tensor/state/effect facts 共同资格化，不复制第二套图 IR；
  5. 规则只表达 HYLE 开发时已经验证的条件与确定性优先级。准备期允许逐条资格化及合法 fallback，
     不运行候选、计时或学习；绑定后 submit 不携带、解释或查询路由数据；
  6. 全量替换所有旧路由产品调用者和 Direct/Graph 普通入口；不保留旧固定端口 API、旧全局量化字段或
     双轨兼容层；固定 plan/cache identity、结构化诊断、unknown-fact 和歧义规则语义；
  7. 完成 CPU、CUDA、Apple、Vulkan 四路线迁移、source/static/shared 和用户文档。
- non-goals：模型文件加载/tokenizer/采样/推理 engine；新增算子、kernel、shader、Provider 或 fallback；
  M3 模型链覆盖回填；M4 缺口实现；M5 冻结自实现比较；跨进程通用 plan cache；执行期 autotune；
  CPU+GPU 联合推理、HIP、分布式、视觉主干。
- 直接依赖 / 直接消费者：Compute tensor/operation/provider、Mem allocator；Graph compile/region provider；
  CUDA/Apple/Vulkan 路线及 CCCL/FlashInfer/Marlin/ggml CUDA/ggml Vulkan 的现有 route-rule 调用者。
- 性能 tier 与原因：`P0`。新字段和 matcher 只在 prepare/compile 冷路径；必须证明 submit artifact
  不增加字段解释、设备查询、分配、锁、计时或 Provider 搜索。本轮不以冷匹配纳秒差启动 P1。
- 适用机器 / 仪器 / material margin：Windows canonical 完成实现、MSVC 正确性与 source/static/shared；
  WSL 验证 GCC/Clang 公共头与 Linux 组合；MacBook Air 验证 Apple 普通消费和三交付，Mac mini 只作
  portability。CUDA SM89 与 Vulkan 当前设备用于真实规则，未知设备走成熟基线或明确拒绝。
- 调查与纠正预算：一次现有 public/consumer 自底向上审计；一次主纵切和一次正交纵切反证；有限 FIX
  冻结后最多两轮实现纠正。发现需要模型 loader、执行期搜索、跨路线万能 executor 或新增数学时停止并
  退回重新定界，不在本卡扩大。
- 预声明验证：identity 的 allocator/OOM/copy/move/lifetime/concurrent-read；profile exact/unknown/
  revision/manifest mismatch；任意 arity、重复/缺失端口、dtype/format/rank/shape/mode/group mismatch；
  Qwen mixed profile 与 MXFP4 mixed region；Direct/Graph prepare 一次、重复 submit 不重新匹配；
  Provider 迁移；Windows 定向 factors、Compute/Graph 三交付、WSL headers/组合、Apple 普通消费；
  `validation-select.py`、closure/docs/diff；P0 静态成本账单。
- stop / blocker 条件：有限 FIX、全部适用门和迁移完成即 `CLOSED` 并恢复 M3；任一公开模型事实无法
  与实际 DAG/tensor 共同校验、profile owner 不能安全保活、或适用平台无法裁决则 `BLOCKED`，不以
  私有字符串旁路，也不把未运行项写 PASS。

## SCOPED 决策

1. 路由分四层拥有事实：build envelope 裁剪可交付路线/Provider；route Context 固化平台、硬件、驱动、
   toolkit/runtime 和 Provider 身份；prepare/compile 提供模型制品画像和真实计算 subject；prepared
   artifact 独占 submit 所需的已绑定入口。事实不复制到不需要它的层级。
2. 模型制品画像是冷身份和复现/cache 维度；Provider eligibility 仍必须检查实际 operation/DAG 与
   tensor/state/resource facts，模型名、variant 或 profile 名不能单独授予资格。
3. Direct 的 subject 是实际 `Operation + TensorDesc ports`；Graph 的 subject 是完整 Graph、canonical
   proposal boundary、effects/state 和 region semantic。共同层只提供身份、端口和稀疏事实合取；复杂
   region 内部模式由 Provider 从真实 DAG 校验，不发明模型 DSL 或影子 Graph IR。
4. HYLE 交付版本化、确定性的默认 decision table。普通用户只选路线并提供真实请求；不提供 Provider
   顺序或 timing profile。相同事实必须产生相同候选次序；同优先级歧义是规则错误，不靠注册顺序碰运气。
5. build/Context 已知条件先裁剪；model/profile/region/shape 在 prepare 匹配。能力拒绝可走表中已声明
   fallback，已经接纳工作后的运行错误不得降级。submit 只使用已绑定 artifact。
6. 四路线、Provider 原生资源/程序/完成 owner 与 prepare/submit 分界不变；不建立万能 executor、Queue
   或逐 kernel vtable。
7. 本卡完成前 M3 暂停，已有 33 PASS 和所有拒绝/未运行格原样保留；本卡不改变覆盖分子。

## AUDITED 前必须回答

- 从真实模型表推导 owned model/artifact profile 的最小字段、完整/部分 selector 的合法层级、digest 表示、
  allocator 保活与并发读边界；
- 任意端口 predicate 和 operation semantic facts 是否足以覆盖主纵切与正交纵切，哪些复杂 region 内部
  事实必须由 Provider 直接检查 Graph；
- decision table 的显式优先级、歧义拒绝、能力 decline/fatal error、成熟 fallback 和结构化诊断合同；
- Graph proposal、compile、constant packing、plan identity 与 implementation key 分别由谁拥有，哪些
  身份进入重新准备/缓存失效，哪些只属于单次 compile；
- Direct 与 Graph 的自然 C++ 调用草图，以及 source/static/shared 的同一公开合同；
- 设计冻结后再列全部旧产品调用者和有限 `KEEP/FIX/BORROW/REJECT`，不能按旧实现反推新 API。

## 当前状态

本卡仅进入 `SCOPED`。目标模型调查已完成；产品实现、公开 API、测试和覆盖结论尚未因本卡改变。
