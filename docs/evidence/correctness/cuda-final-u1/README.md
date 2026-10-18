# CUDA U1：继承共同需求的首轮静态接线对账

baseline `f5f137f1b3e48bdd56acd42f7471d3c412c7b6e3`，Windows canonical。
本轮直接继承 CPU 已完成的新模型／量化／MTP／DSpark 目录，**不重新推导算子需求**。
六份固定来源 SHA 与当前文件完全一致，104 制品身份／数值合同／参数类未改；
完整对账见 [demand-source-join.json](demand-source-join.json)。没有新 runtime PASS 或性能 WIN。

## 首轮结果

| 共同需求 | 当前 CUDA 源码事实 | 数量 | 后续动作 |
|---|---|---:|---|
| 559 GGUF 数值合同 | 上游 CUDA MatMul／Grouped 已有类型，HYLE 格式映射未接 | 119 | 批量接原格式；核对输出精度、activation codec、布局与完整消费者 |
| 同上 | 已有映射，但 Gather 只允许旧三个格式，原版 GET_ROWS 已有相应类型 | 5 | 接 Q3_K／Q5_K embedding，保留 bounds／失败／精度合同 |
| 同上 | 当前 stock core 无 IQ1_XS／XXS／XXXS 三类型 | 8 | 检出已固定 Unsloth 原扩展的 CUDA 程序，不能从 CPU 端口外推 GPU 或编自研替代 |
| 同上 | 格式 hook 存在，完整合同未在本审计运行 | 301 | 精确资格，包括真实模型内宽、普通两入口、资源／状态与重放 |
| 同上 | plain 候选，需逐原端口核对 admission | 126 | cuBLAS／ggml／其它合格成熟候选，plain 可用不等于原精度／Gather 已签 |
| non-GGUF packed | ArtifactWeightDecode／MlxQuantized 现有 CUDA Provider 无 hook | 34 | 接原 NF4／nested、AWQ、CT、MLX affine／MX4／MX8 的成熟 GPU 承载 |
| non-GGUF plain | 复用共同 45 项原 dtype／role／shape | 45 | 与 packed 准备和完整 consumer 一起验收，不能签 CPU fallback |
| 完整 semantic/state | 八族 49 个 family-region 关联、官方原版、新版与 sidecar 全保留 | 49 | 复用已定义程序，迁移执行环境；CPU 证据不代签 CUDA |

这些数量是已有计算合同／接缝，**不是待新写的数学算子数**。当前 124 项“上游已有未接”来自
八个标准类型的 119 项映射缺口与五项额外 Gather；十一种新增 GGUF 物理类型均已在共同目录中。
另 150 项原全 tensor geometry 会超过现适配层逻辑元素上界，需按已准入真实分片／expert 驻留规则
检查完整消费者；计数与上述分类重叠，不据它宣称 150 个新算子缺口。

## 接线／成本与构建的有限跟进

| ID | 具体源码事实与风险 | owner／有限动作 |
|---|---|---|
| W1 | tensor_program.cc 明确 decline 新类型，native_embedding_table 只 Q4_K／Q6_K／Q8_0 | compute_provider_ggml_cuda；原版格式／Gather 批量接线，Graph region 同源，不增数学 |
| W2 | IQ1 三扩展只借入 CPU；34 packed 的两个公共 op 在 CUDA 无 hook | runtime／实际成熟 Provider；检出固定原扩展／原 GPU 程序后冻结接缝，不以“同 4-bit”猜 packing |
| W3 | CUDA AOT 固定旧 DeepSeek preview 程序／四形状；新版 DSpark 与原 E8M0／FP8 精度不能凭旧名字继承 | compute_provider_cuda／实际成熟 carrier；按共同原版／新版分支合同逐程序核对 |
| W4 | ggml 整个 Provider 限 SM89，block128 AOT 限 Driver API13040；factory 固定顺序不是所有域 best-of | route/provider；资格、工程默认与已测特化分开；未知身份合法可用性审核，不无证移除数值 guard |
| W5 | ggml checked indices 每次 D2H＋stream sync；两个 stream/event 桥；LayerNorm centered-variance 序列；Hadamard 预置矩阵 | provider/Graph；计入完整成本，bounds／精度先等价再比较，保留原版融合与工作区，不能不做检查换性能 |
| W6 | cuBLAS heuristic 已在 prepare，workspace/handle 每 invocation；执行使用已绑定算法 | KEEP 冷热分界；验证普通重复调用无再次搜索，不发明逐 kernel 调度器 |
| B1 | 只有 compute-cpu preset，无标准 compute-cuda；普通 CUDA-only 尚需独立 CPU=OFF | U2；明确一键 CUDA 与高级 mixed 用法，实际 package/feature 由现有图解析 |
| B2 | HyleCudaSdk、runtime staging 已建立 CUDART/cuBLAS/cuBLASLt SHARED runtime 闭包与 Windows 邻 exe 复制 | U2；复用；验证当前 source/static/shared／搬移／reduced-PATH，未重跑不记新 PASS |
| B3 | backend 默认 80＋89-real，FlashInfer Linux-only；实际 driver_cuda_version 是 API 号，不是驱动发行版 | U2/U4；架构／软件身份与跨平台能力投影说明并核实，不能把 Provider 数量对称当模型能力对称 |

来源为 SHA 绑定源码与原版 ggml `456172ec733a` CUDA supports_op switch；该 switch
只证明成熟类型入口存在，不证明当前硬件、shape、precision、资源或 dispatch 资格。
CPU 当前 complete catalogue 只授予需求事实，CPU runtime PASS 不移植到 CUDA。

## 边界与尚未完成

U1 的共同需求／首轮 hook 对账已完成；W2／W3 的固定原成熟程序承载检出、region 的具体
Provider admission／资源映射仍须在本包有限预算内完成，再交 U2/U3。
本轮没有产品修改、registry 重开、构建或 P1。旧 CPU CLOSED、公共 Graph 的旧 Vulkan BLOCKED 保持。
先前 HF “0816” [身份查询](release-lookup.json)只作历史诊断，用户已明确复用 CPU 目录，
它不是另一个画像任务、待答问题或 blocking condition。

## 有限成熟来源检出结束

[原 GPU 来源身份](original-gpu-source-identities.json)保留 revision、archive SHA 和逐文件 SHA。
没有新模型需求、源码数学改写、产品构建或运行资格。

| 当前缺口 | 已检出的固定成熟 GPU 承载 | 接入／资格边界 |
|---|---|---|
| 8 个 narrow IQ1 合同 | Unsloth `46cbf0e9` 的 common／convert／MMVQ／MMQ 原 CUDA，18 文件匹配 C1 固定 SHA | 借入原数学／traits／codebook，保留当前 core ABI；CUDA admission／Group／原 A8／F32 accumulation 尚待运行 |
| 6 个 NF4/nested | bitsandbytes `83364904` 的 CUDA ops、dequantizeBlockwise 原 stream 入口及原 nested scale 序列 | 原码表与 group，二级 scale/offset；只准备一次，不每 token 全权重展开；非默认 codebook 不擅自当原 NF4 |
| 7 个 AWQ GEMM | AutoAWQ `88e4c76b` 原 unpack/reverse/dequant tensor 序列可在 CUDA tensor 执行；原原生 CUDA/Triton路径有调用点 | 实际 GEMM zero/scales／逆 permutation／F16 round，原程序承载须 U3 选定；CPU-only ATen 不算 GPU |
| 5 个 CT symmetric | compressed-tensors `ac8e2ba8` 的 unpack_from_int32 与原 dequant tensor 程序保留 device，从 ATen CUDA 执行 | 低位-first／signed offset8／原 group/scale dtype，不重解释为 AWQ 或要求新数学；GPU runtime/交付接缝待定 |
| 16 个 MLX affine/MX4/MX8 | MLX `2b2bb497` 原 CUDA QuantizedMatmul／GatherQMM、affine/fp dequant；SM80/naive 原 carrier及原类型映射 | 原 packed＋scale/bias／E8M0、selected rows/experts、F32 accumulate；直接 qmm fast path与共同精度必须先等价，不为性能改合同 |
| 新版 DSpark／三 sidecar | 共同 cpu_c4_model_branches／cpu_c4_qwen_mtp／cpu_c4_flash_sidecars 程序；所用原基础算子已存在于公共 schema | target hidden/main projection/Markov/confidence/三 stage、state/carry/head，按 CUDA 实际 admission／原 dtype 逐完整程序验证 |

U1 到此结束只读审计，有限交接 W1–W6／B1–B3；source 检出不自动授予新 Provider 身份或产品支持。
实际 GPU 准备程序／新 runtime 的 owner、软件／资源寿命、三交付方案必须在 U3 产品修改前冻结。
