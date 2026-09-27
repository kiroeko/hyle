# M3 真实需求到成熟 Provider 的重新对账、补接与退场

<!-- MODULE REVIEW: AUDITED -->

**当前施工：D5。** D1–D4 已完成；22画像×四路线×Direct/Graph共176格已全部实测，
154通过、20冷拒绝、2数值失败。四路线三交付与CPU/Apple P1已完成，CUDA/Vulkan P1和D6仍待完成。
见[D5 当前卡](llm-moe-d5-real-fragment-acceptance.md)与
[本批证据](../../../../evidence/correctness/llm-moe-d5-real-fragments/qwen-nine-profile-complete-gate/README.md)。
[D6文档与覆盖对账](llm-moe-d6-coverage-review.md)已开始；A6及残余能力边界未解决前不签终审关闭。
后文各包的旧candidate和进展数字是历史，不替代当前ledger。

## Closure card

- component / owner：第一包由 `compute_ops` 持有需求矩阵；只有审计证明需要产品接线或删除时，才以
  具名事实重开对应 CPU/CUDA/Apple/Vulkan Provider、route 或 Graph owner。
- baseline commit：`4b74d2b860d5ee864f3e38931124fdd5b9db51cf`，真实模型画像冷路由重构已关闭，工作树干净。
- trigger 与证据：用户要求以新的真实算子阵列重新列入 Provider，并删除以前接入但实际无用的接线。
  [模型权重与算子审计](llm-moe-model-weight-operator-audit.md)已证明同一模型和单个制品内部按 tensor
  role 混用 BF16/F32、Q/IQ、FP8、MXFP4 等格式；[当前矩阵](llm-moe-coverage-matrix.json)的 256 个路线格
  只有 33 `PASS`，仍有 134 `VALIDATION_NOT_RUN`、21 `BACKEND_REJECTED`、64
  `NUMERIC_INCOMPATIBLE` 和 4 `UPSTREAM_MISSING`。旧 M3 又继承了“M2 已把全部成熟 Provider 接入”
  的前提，不能区分源码有符号、target 可构建、Provider 已注册、普通入口能命中和真实片段已运行。
- 正常 consumer 任务：普通 C++ 用户选择 CPU/CUDA/Apple/Vulkan 路线并提交真实 Direct operation 或
  Graph region 与 exact model/artifact profile；HYLE 在 prepare/compile 期命中一个已验证成熟实现，
  使用路线原生资源执行，重复 submit 不重新选路。
- 主产品纵切：本地 `Qwen/Qwen3.5-2B@15852e8` 官方 BF16 主干及其
  `UD-Q4_K_XL@f6d5376` 画像，覆盖 Gated DeltaNet、full attention、dense FFN、MTP、F32 recurrent state
  和按 tensor role 的实际格式。
- 正交反偏置纵切：Qwen3.8-2.4T mixed IQ MoE、Kimi K3 MXFP4/KDA、DeepSeek V4 FP4/FP8+
  DSA/mHC、GLM-5.3 FP8+KDA/稀疏 attention；使用真实宽度和缩小容量片段，不冒充整模型推理。
- 跨模块 public journey：模型审计事实（测试输入）→ owned profile/route subject（Compute）→ Direct
  prepare 或 Graph compile（Compute/Graph）→ Provider 原生 program/resource/completion（路线 owner）→
  两次 submit 与结果/失败/资源收据。模型 loader、tokenizer 和采样只属于测试消费者。
- scope：
  1. 从固定模型 revision、官方/主流量化 profile、tensor role 和 forward region 重建需求单元；
  2. 每个需求单元按四路线登记具体 Provider/上游入口/version、精确 dtype/format/shape/state 条件、
     当前 build/registration/policy/binding/execution 层级与证据；
  3. 把状态区分为已实际运行、上游可用但未绑定、已绑定但普通入口不可达、backend 拒绝、数值不相容、
     上游缺失和验证未运行；只有不可变候选上的真实 Direct/Graph 收据可写 PASS；
  4. 先按 Provider 原生 operation family/region 批量补接所有已证明可用但未接线的成熟能力；复杂融合由
     Provider 保持所有权，不拆成逐 kernel adapter；
  5. 删除没有模型或通用 Compute consumer、普通入口不可达、语义错误 fallback，或已有成熟路径完整
     替代且无独立用途的旧 Provider adapter/route；保留仍被非 LLM consumer 使用的通用能力；
  6. 用真实模型片段完成四路线 correctness、资源/失败、source/static/shared、默认规则与适用性能；
  7. 成熟覆盖完成后汇总残余缺口并停止；自研算法的最终裁决使用独立后续卡。
- non-goals：新增/优化 HYLE CPU/CUDA/Metal/Vulkan 数学、kernel、shader、量化器或 decoder；删除 SIMD
  基础库；模型文件加载/tokenizer/采样/推理 engine；完整 27B/2.4T 模型运行；HIP、分布式、CPU+GPU
  模型级联合推理；仅为“多一个库”引入后备 Provider。
- 直接依赖 / 消费者：`compute`、`graph`、四路线和现有 Provider factor；固定模型审计、provider map、
  package consumer 与本地 Qwen3.5-2B 测试输入。
- 性能 tier：矩阵/死代码清理为 `P0`；默认实际路线变化且 API/算法冻结后，对完整片段执行一次适用
  `P1`，计算必要转换、packing、上传下载、同步、融合损失和 workspace，不与冻结自研实现竞速。
- 机器 / 仪器：Windows 是产品真相；RTX 4070 验证 CUDA/Vulkan，WSL 验证 Linux/GCC/Clang 与合格
  CUDA-only 入口，两台 Mac 验证 Apple/Metal/MLX；低并行、重型路线串行。三交付只在批次稳定后执行。
- 调查与纠正预算：先用一次表驱动源码/产品对账覆盖全部需求单元，不逐算子开卡；每个 Provider family
  最多一次有限接线 ledger 和一次纠正。后备库只在主候选有具体反例后调查；性能失败先看源码成本。
- stop：成熟 Provider 可用但未接线数、错误可达数、适用验证未运行数均归零后关闭本卡；确实缺失项
  以具名证据进入后续最小缺口卡。出现必须新增自研数学、改变已关闭路由架构或模型事实无法确定时，
  停止并重新定界，不用相近格式或 CPU fallback 代签。

## 固定工作包

| 包 | 工作 | 完成定义 |
|---|---|---|
| D1 需求正规化 | 把八个模型的 operator/region、tensor role、官方与 GGUF profile 归并成去重需求单元 | 每格能回指固定模型/revision/header/config/source，不再使用全模型单一 Q4/Q8 标签 |
| D2 Provider 可达审计 | 对每格核对 upstream symbol、build、注册、route policy、prepare/compile、执行与收据 | 产出实际层级和唯一未覆盖原因；源码符号/target/枚举不计接入 |
| D3 批量补接 | 按 CPU、CUDA、Apple、Vulkan 的成熟 region 批量修 schema/lowering/resource/build/route | 所有 `AVAILABLE_UNBOUND` 变为实际 PASS 或具名拒绝；不新增自研数学 |
| D4 旧接线退场 | 审计 Provider adapter、route、CMake 和文档的真实需求/可达性 | 无需求、不可达、错误 fallback 和完全冗余接线删除；通用消费者与路线资源设施保留 |
| D5 真实片段验收 | Direct/Graph、两次 submit、状态/失败/资源、三交付及必要 P1 | 四路线每个适用需求格有真实收据；不以其它路线或 reference 代签 |
| D6 终审 | 红队、矩阵汇总、LOC、限制和状态关闭 | 成熟 Provider 阶段完成并停止，残余上游缺口与后续自研裁决分开 |

## Provider 首选与升级规则

| 路线 | 首选成熟来源 | 只在具名缺口时调查 |
|---|---|---|
| CPU | ggml CPU、oneDNN | XNNPACK、平台 BLAS |
| CUDA | ggml CUDA、cuBLASLt、Marlin、FlashInfer、SGLang 固定原版程序 | cuDNN Graph、FlashAttention；CUTLASS/CuTe 只作原版程序依赖 |
| Apple | MLX、ggml Metal | MPSGraph |
| Vulkan | ggml Vulkan、限定 IREE Vulkan | ncnn、MNN |

“首选”不等于全部保留。一个接线只有在至少一个需求单元或独立通用 consumer 实际可达，并满足精确
数值/资源/交付合同时才能存在；否则按 D4 删除或降为未接入候选。

## 删除与保留判据

1. 只被 benchmark、设备枚举、空 target 或注册表引用的 adapter 直接列为删除候选。
2. route policy 永远不可能命中、命中后必被 schema 拒绝，或依赖隐藏 CPU fallback 的接线必须修正或
   删除，不能作为 Provider 覆盖。
3. 成熟实现完整覆盖同一需求域且旧 adapter 没有不同 consumer、语义或有效默认规则时，删除冗余接线；
   不为其制造性能课题。
4. HYLE 自研算子原按[冻结合同](llm-moe-self-operator-freeze.md)留到后续统一裁决；用户现已明确授权在
   D4 按[批量退场卡](llm-moe-d4-self-operator-retirement.md)一次性清退错误自实现。不得新增、优化或为
   每个算子重开性能研究；完整成熟替代且无既有合格领先证据者直接退场，精确缺口才保留。
5. SIMD 基础库、四路线 backend/resource/completion 设施，以及有非 LLM 真实 consumer 的通用 Compute
   能力不因模型矩阵删除。

## SCOPED

旧 M3 已在 `b5843f1d9f3eca4edd86889b210da5116fc708b6` 停止，原 256 格和收据只保留为历史输入。本卡的
需求单元、Provider 层级、工作包、删除判据、预算与停止条件已冻结，`compute_ops` 依法恢复为
`SCOPED`。下一步只执行 D1/D2 表驱动审计，固定有限 `KEEP/BORROW/FIX/DEFER/REJECT`；在进入
`AUDITED` 前不修改产品、Provider、route、CMake 或自研算子。

## D1/D2 审计结果

[真实需求与 Provider 可达矩阵](llm-moe-provider-demand-matrix.json)已经替代旧 64×4 表作为本轮施工
输入。它固定了八个模型的 22 个可验收 profile、15 个去重语义 region 和 60 个路线语义格；
DeepSeek V4 Pro preview 的 0813 GGUF sibling 只保留为格式证据，不计验收 profile。旧表中的
`B16/Q4-S128/Q4-A64/Q8-B32/F8-B128` 只作历史 crosswalk，不能再产生 PASS、缺口或产品设计。

审计没有发现需要 HYLE 新写数学 kernel 的首轮理由，发现的是成熟能力没有穿过产品接缝：

1. 固定 ggml `456172ec` 已含 CPU/CUDA/Metal/Vulkan 的 Q3_K、Q4_K、Q5_K、Q6_K、Q8_0、
   IQ3/IQ4_XS、MXFP4 `MUL_MAT`/`MUL_MAT_ID` 和 Gated DeltaNet/KDA、Lightning Indexer 等程序；
   HYLE CPU adapter 的 `representable()` 仍要求 plain storage，CUDA adapter 也只从普通 `DataType`
   降低，Vulkan 只有 MXFP4 特例，Metal 私有程序只收 dense F32/I32。因此当前状态是
   `AVAILABLE_UNBOUND` 或 `BOUND_UNREACHABLE`，不是上游缺失。
2. oneDNN、cuBLASLt、FlashInfer、SGLang/CCCL、MLX 和限定 IREE 的现有合格域继续保留；它们分别
   承载 plain/fused、attention、indexing 或精确低精度补位，不把相近量化格式写成等价。
3. Marlin adapter 只实现人工 signed symmetric Q4/group128、M=1..64；真实目标画像没有该物理格式，
   仓库中除测试和文档外也没有 consumer。它进入 D4 退场候选：先从 CUDA 默认 roster 移除，若
   owner 审计仍找不到独立 consumer，再删除 factor/public alias/构建与文档。
4. 四路线基础资源、完成和失败设施及 SIMD 基础库继续保留。全部 HYLE 自研算子仍冻结；D3/D4
   不用性能比较借机改进或删除它们。

### 有限 FIX

| FIX | owner 批次 | 产品动作 | 完成边界 |
|---|---|---|---|
| `FIX-1` | `compute_ops`、`compute_provider_ggml` | 定义一个版本化 ggml-native storage catalog 和 checked descriptor builder | 只描述上游现有 K/IQ/Q8/MXFP4 物理格式，不写 codec |
| `FIX-2` | 四个 ggml Provider/route owner | CPU、CUDA、Metal、Vulkan 批量导入 native tensor，保留上游 `MUL_MAT`、`MUL_MAT_ID`、模型 region 与资源所有权 | 普通 Direct/Graph 能绑定；不按 kernel 建 vtable |
| `FIX-3` | 四 route + Graph owner | 用一个表绑定 15 个 region 的 exact model/profile/port signature 与硬件/驱动/toolkit 条件 | prepare 固定，submit 不选路、不计时 |
| `FIX-4` | 各路线 owner | 官方与 primary GGUF 的 reduced-capacity 真实片段、两次 submit、状态/资源/失败收据 | 不要求加载器或 27B/2.4T 整模常驻 |
| `FIX-5` | D4 十一个 route/provider owner | 按[批量 ledger](llm-moe-d4-retirement-ledger.json)整包或按域退场无目标、不可达、成熟实现已完整替代且无既有领先证据的旧实现 | 不删除 SIMD/路线设施/精确未覆盖残余，不新增或优化自研数学 |
| `FIX-6` | 对应交付 owner | 稳定后一次 source/static/shared 与适用 P1；汇总精确残余 | 未运行不写 PASS；具名上游缺失另开最小缺口卡 |

施工顺序固定为：公共 catalog → CPU/CUDA ggml → Apple Metal/MLX → Vulkan ggml/IREE → 旧接线退场 →
真实片段与三交付。相同 Provider 的 storage/lowering/receipt 按路线整批完成，不再逐算子开卡。

当前 D3 路线子批：

- [CPU/CUDA ggml native storage 接缝](llm-moe-d3-ggml-native-binding.md)
- [Apple ggml Metal native storage 接缝](llm-moe-d3-ggml-metal-native-binding.md)
- [Vulkan ggml native storage 接缝](llm-moe-d3-ggml-vulkan-native-binding.md)

CPU/CUDA 子批已形成 `08b8e2f1` candidate，Apple 子批已形成 `5a1b61fb` candidate，Vulkan 子批已形成
`2757ecc` candidate。三批均完成各自普通 Direct/Graph、原生资源/失败、代表数值与
source/static/shared 接线门，D3 至此完成；它们仍等待 D5 真实 profile 片段的完整成本与覆盖验收。
下一包进入 D4，只按既定删除判据审计旧接线，不借机研究、优化或扩张冻结的自研算子。

## AUDITED

D1/D2 已完成，以上六项是本轮全部产品 FIX。后续只有表中具名 owner 可以重开；先按合法状态迁移
绑定当前 card，再修改其产品代码。出现需要新自研数学、额外 Provider 或改变路由架构的新事实时，
退回重新定界，不扩大本 FIX 集。
