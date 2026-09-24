# MR 真实模型画像路由系统重构

<!-- MODULE REVIEW: CANDIDATE -->

## Closure card

- component / owner：`compute`、`graph` 为共同合同 owner；实际迁移 owner 为
  `compute_backend_cuda`、`compute_provider_native`、`compute_provider_onednn`、`compute_provider_ggml`、
  `graph_provider_onednn`、`compute_provider_cublas`、`compute_provider_cuda`、
  `compute_provider_flashinfer`、`compute_provider_marlin`、`compute_provider_ggml_cuda`、
  `graph_provider_ggml_cuda`、`compute_provider_ggml_vulkan`、`compute_route_cpu`、
  `graph_route_cpu`、`compute_route_cuda`、`graph_route_cuda`、`compute_route_apple`、
  `compute_route_vulkan`、`graph_route_apple`、`graph_route_vulkan`。
- baseline commit：`af8a42e4c`（M3 错误候选已纠正，目标模型/权重/算子审计已固化）。
- superseded records：旧 [B2 route identity](m2-b2-route-identity.md)、
  [共同 matcher](m2-route-qualification.md)与
  [旧调用者迁移](m2-route-qualification-complete-migration.md)只作历史证据；本轮不继承其 API 或完成结论。
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
- 跨模块 public journey 与每段 owner：consumer 创建 owned `ModelArtifactProfile`（Compute）→ Direct
  `prepare` 或 Graph `compile` 携带 `WorkloadContext`
  （Compute/Graph）→ common matcher 合取环境、模型/profile、operation 与任意 tensor port
  （Compute）→ Provider 检查真实完整 DAG/region 并返回 `ProviderImplementationId`、policy 与 preference
  （各 Provider/Graph）→
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

## 从真实需求冻结的公开形状

下列形状由[长期路由合同](../../../architecture/compute-routing-system.md)持有；本卡不以现有类名作兼容：

1. `ModelArtifactProfileDescription` 分开保存 model repository/family/name/fixed revision/configuration
   digest 与 artifact repository/fixed revision/variant/manifest digest/format family；
   `ModelArtifactProfile` 在一次 allocator allocation 中拥有这些字段，空 handle 表示模型无关调用。
2. `WorkloadContext` 持有 profile、phase 和 versioned region semantic。profile 不是 tensor 格式表，也不
   读取模型文件；旧 model-only identity 删除，不留 alias 或双轨。
3. `RouteEnvironment` 是 Context 捕获的 immutable native facts；`RouteSubject` 借用一次冷调用的真实
   operation、全部端口或 Graph region。规则不保存这些借用。
4. `RouteSelector` 分开保存 environment predicates、profile predicates、operation semantic predicates
   和任意 `input/output + index` 的 tensor predicate。端口支持 dtype、format、rank 与 axis 的
   inclusive range/multiple；量化 mode/group/accumulation 来自 operation schema 的 typed facts。
5. `RoutePolicyEntry` 有 versioned rule identity、显式 preference、目标 Provider 和稳定
   `ProviderImplementationId`。普通路线 façade 提供 HYLE 固定 policy；低层自定义 Context 属于专家
   入口。匹配相同 preference 的不同 Direct 目标返回 ambiguity，不用注册顺序选中。
6. Direct prepare 收集 hard-eligible proposals 后按 policy preference 绑定；只有 capability/resource
   decline 可试下一条。Graph 把同一 preference 纳入既有 whole-plan 有界规划，Provider 仍从真实 DAG
   校验 region。submit 不携带 selector/policy，也不调用 matcher。

## 审计覆盖图

| 真实纵切 | 设计如何表达 | 反证结论 |
|---|---|---|
| Qwen3.5-2B 官方 BF16 vs Unsloth UD-Q4_K_XL | model identity 相同、artifact identity 不同；实际 MatMul/GDN 端口分别检查 BF16 与 mixed Q/IQ formats | 不需要也不能有 profile-wide Q4 枚举 |
| Qwen3.8-2.4T UD-IQ4_XS MoE | exact artifact + region semantic；boundary 任意端口；Provider 从真实 DAG 检查 expert/router 内部的 IQ4/Q4/Q5/Q6/F32 | 共同层不能复制 512-expert 图 DSL |
| Kimi/DeepSeek official 与 Q8 GGUF | artifact 区分官方 packed FP4/FP8 与 GGUF；Provider 同时检查 MXFP4 expert、BF16/Q8 projection、F32 state/router | 模型名或 GGUF 文件标签均不足以资格化 |
| 空 profile 的渲染/工程 Direct 计算 | selector 不要求 profile，仍按环境、operation 和端口命中成熟通用路线 | 公共 API 没有过拟合 LLM loader |
| CUDA/Apple/Vulkan 原生路线 | typed environment fact 的 namespace 固定解释，policy 只在路线内选择 Provider | 不引入跨路线万能 executor/Queue |

## 产品调用者审计

| 边界 | 当前事实 | 本轮处置 |
|---|---|---|
| Compute workload owner 与 prepared artifact | model-only owner 已能保活字符串，但没有 artifact/manifest；artifact 已持有 workload | `FIX-MR1` 替换为单一 model-artifact profile owner并贯通 prepared 生命周期 |
| CUDA environment producer | backend Context 已捕获真实 device/SM/driver/runtime/toolkit，但使用旧 fact 类型 | `FIX-MR2` 只迁移 typed fact schema；backend 资源、stream、执行和完成不变 |
| common route matcher | 固定 input0/input1/output0 和一个量化结构；仅返回首 mismatch | `FIX-MR2` 全量替换为 selector、typed semantic facts 和任意端口/shape matcher |
| Provider implementation identity | `ProviderRouteKey/profile_key` 实际表示实现，不是 model profile | `FIX-MR3` 重命名为 `ProviderImplementationId/implementation`，删除歧义旧名 |
| Direct Provider 选择 | 默认 façade 隐式依赖 provider array 顺序，prepare 遇首个成功即返回 | `FIX-MR4` 冻结显式 HYLE preference，收集冷 proposal 后确定性绑定和诊断歧义 |
| Graph leaf/region 选择 | 已有真实 DAG、canonical boundary、effects、cost 与有界 plan search | `KEEP-MR1` 保留 owner/规划；`FIX-MR5` 加入 profile 与 explicit policy preference |
| CUDA/Apple/Vulkan 的八处旧 rule 调用 | 已有真实 hardware/operation/format facts，但形状受旧 matcher 限制 | `FIX-MR6` 迁移到新 selector；Provider 数学、资源、kernel/shader 保持不变 |
| CPU/Graph Provider 与路线 façade | 已有成熟 default roster 与 implementation key | `FIX-MR7` 把既有开发裁决写成显式 preference，不重做性能比较 |
| model/DAG loader 与跨进程 cache | 不属于 HYLE Compute/Graph | `REJECT-MR1` 不新增 loader、模型脚本、全局 registry 或持久化 cache |
| 新算子/kernel/Provider | MR 没有数学缺口 | `REJECT-MR2` 保持冻结，覆盖分子不变 |

## 有限 FIX ledger

1. **FIX-MR1 profile owner**：实现 description/owned handle/校验/OOM/copy/move/concurrent-read，替换
   model-only workload；manifest/config digest 接受规范 lowercase 64-hex，空 profile 零 allocation。
2. **FIX-MR2 selector**：实现 typed environment 与 operation facts、profile predicate、任意端口的
   dtype/format/rank/dimension range+multiple、完整 mismatch location 和非法规则检查；删除旧 rule API。
3. **FIX-MR3 implementation identity**：单轨重命名 Compute proposal、Graph proposal/report 和 custom
   Provider 字段；identity 仍只在同一 provider build/config 内解释。
4. **FIX-MR4 Direct binding**：默认 CPU/CUDA roster 使用显式 preference；prepare 冷收集、排序、
   ambiguity、unsupported/resource decline/fatal-stop；prepared submit 路径不变。
5. **FIX-MR5 Graph binding**：operation/region proposal 携带显式 preference，whole-plan search 先服从
   hard constraints 和用户 target，再使用已验证 preference/cost；compile request 持有 exact profile。
6. **FIX-MR6 rule migration**：迁移 collectives、FlashInfer、Marlin、ggml CUDA/Vulkan、Apple、Vulkan
   的旧 rule，覆盖高索引 ports、shape 与 mixed profile；不改计算实现。
7. **FIX-MR7 route integration**：CPU、CUDA、Apple、Vulkan 默认 façade 交付各自 immutable environment
   与 policy；Graph 四路线复用同一 Context/profile，不复制 policy。
8. **FIX-MR8 validation/docs**：定向 correctness、两组真实画像纵切、Direct/Graph 两次 submit 插桩、
   source/static/shared、适用 Windows/WSL/Apple/Vulkan/CUDA 门、用户文档和 P0 成本账单。

ledger 之外的算子、Provider、kernel、性能调优、模型加载和公共治理改动全部拒绝。任何新事实要求新增
数学、执行期选优或第二套 Graph 表达时，停止实现并退回 `AUDITED` 重新定界。

## CANDIDATE 验收

产品候选 `c9c11001419059068e3497b6eac10fcb70cb6e87` 完成 `FIX-MR1`--`FIX-MR8`；验收候选
`2ada22a89821d20b312407c9575df8f9a923df89` 只迁移一条 oneDNN OOM 测试的结构化诊断预期，产品实现
不变。完整收据见[终端证据](../../../../evidence/correctness/model-profile-routing-refactor/README.md)。

| Gate | Verdict |
|---|---|
| scope | PASS：只替换冷路由合同并迁移既有四路线调用者；未新增算子、Provider、kernel/shader，M3 覆盖不变 |
| architecture | PASS：owned exact profile、任意端口 selector、versioned route facts、显式 policy/preference 与 Graph 完整 DAG 资格化均按冻结设计落地；四路线执行 owner 保持独立 |
| correctness | PASS：Windows Compute/Graph/CPU/oneDNN/CUDA/Vulkan 定向及 selected factors 全过；WSL GCC/Clang 各 9/9；Air Apple 44/44+17/17；Mini portability 4/4 |
| delivery | PASS：Windows Compute/Graph source/static/shared relocation consumer 全过；Air 三交付 Compute/Graph consumer 全过 |
| usability_docs | PASS：普通用户仍只选路线并提供真实 workload/profile；公开架构、示例和诊断已迁移，selector plan READY |
| red_team | PASS：旧 API 产品扫描无命中；matcher 只在 cold construction/propose/prepare/compile；mismatch、歧义、合法 fallback 与 fatal-stop 已反证 |
| engineering_performance | PASS（P0）：submit 不解释规则、不查询设备、不搜索/计时/重选；新增成本限于 profile 创建、冷匹配和 prepared/plan metadata，P1 不适用 |

统一 LOC：Compute+Graph 实现 **60,422** 行 / **252** 文件；自实现不含测试 **155,607** 行 / **603**
文件；全库 **1,115,613** 行，固定候选统计自检通过。

限制与 reopen：本轮不提供模型加载/tokenizer/采样/推理 engine，不增加 operator/provider 数学覆盖，不做
CPU+GPU 联合推理、HIP、分布式或跨进程 plan cache。只有真实 consumer 证明现有 profile/selector/DAG
资格合同无法表达必要部署，或已绑定 submit 出现重新选路/生命周期回归时才重开本卡；M3 缺口继续由其
独立任务处理。

本卡进入 `CANDIDATE`。产品/API/算法已经冻结；下一步只执行候选后独立终审、closure/docs/diff 检查和
合法 `CANDIDATE -> CLOSED` 状态迁移，不再改产品实现或扩大验证。
