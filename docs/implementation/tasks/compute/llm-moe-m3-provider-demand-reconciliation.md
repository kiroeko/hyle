# M3 真实需求到成熟 Provider 的重新对账、补接与退场

<!-- MODULE REVIEW: SCOPED -->

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
4. HYLE 自研算子继续按[冻结合同](llm-moe-self-operator-freeze.md)留到后续统一裁决；本卡不得借清理
   Provider 接线提前研究、优化或删除这些算法。
5. SIMD 基础库、四路线 backend/resource/completion 设施，以及有非 LLM 真实 consumer 的通用 Compute
   能力不因模型矩阵删除。

## SCOPED

旧 M3 已在 `b5843f1d9f3eca4edd86889b210da5116fc708b6` 停止，原 256 格和收据只保留为历史输入。本卡的
需求单元、Provider 层级、工作包、删除判据、预算与停止条件已冻结，`compute_ops` 依法恢复为
`SCOPED`。下一步只执行 D1/D2 表驱动审计，固定有限 `KEEP/BORROW/FIX/DEFER/REJECT`；在进入
`AUDITED` 前不修改产品、Provider、route、CMake 或自研算子。
