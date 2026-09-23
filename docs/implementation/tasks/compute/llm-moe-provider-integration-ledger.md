# M0 交付：成熟 Provider 接入施工清单

## 当前裁决

本页是[总计划 M0–M6](llm-moe-coverage-completion-plan.md)的详细接入清单，owner `/root`，
产品基线 `ebe5af51`。模型、数值与验收需求不变；[64 行映射](llm-moe-provider-map.json)已逐行
指定四路线的候选入口、接法和资格限制。[源码身份](../../../../evidence/correctness/compute-llm-moe-m0/source-map.json)
保存 pin、SHA-256 和来源。**映射完成不代表256格产品验收通过**，原覆盖矩阵仍全部 UNPROVEN。

用户再次明确：CPU、CUDA、Metal、Vulkan 的全部 HYLE 自实现算子冻结；目录、语言、Provider
包装、嵌入字符串或上游补丁不构成例外。接入阶段允许资源/错误/编译/调用适配，不允许新增或
改写数学 kernel、排序/归约/scan 算法、手写 shader 或以 MLIR/CUTLASS 模板改名自写。
成熟库的已有原子组合与原版融合程序属于接入；其来源、转换和完整成本必须可追溯。

本页的拟定 FIX 必须由相应 owner 的有限卡审计到 AUDITED 才能实施，当前状态以 registry 为准。
M1-G 已按[独立资源卡](llm-moe-m1-ggml-runtime.md)完成，M1-R 已按[路由卡](llm-moe-m1-route-binding.md)
关闭冷模型条件/原生环境接缝；四环境原生、三交付及适用插桩通过，新增模型覆盖0。
[M1-C](llm-moe-m1-cuda-admission.md)已CLOSED：Windows/WSL普通入口与三交付、原生失败/资源、
host ASan/UBSan及两台Mac适用门通过；只关闭有限接入范围，新增模型覆盖仍为0。[M1-A](llm-moe-m1-apple-admission.md)也已CLOSED，原版Metal/MLX普通组合、两台Mac三交付、资源/host插桩与最终selected门通过；[M1-V](llm-moe-m1-vulkan-admission.md)也已CLOSED：两机普通原版recurrent/MXFP4/IREE Scan/FP8、资源失败/插桩/六格三交付和120个selected门完成；下一步M2；
其余未审范围不授权产品改动。
旧 BLOCKED 不因M1完成自动恢复；oneDNN Q2现已按用户确认的
[工程裁决](llm-moe-onednn-aa-cause-followup.md#当前工程裁决旧样本无效恢复-m2)记为无效测量，
撤销产品FAIL含义和接入阻塞，保留原始日志且不改成性能PASS。

## M2 批量接入执行口径（2026-09-23）

本清单从现在起按Provider原生region/能力族消费，不按单个算子重复施工。每个确定接入单元维护
一张表驱动账本：原版入口和版本、适用需求行、dtype/layout/shape、普通Direct/Graph绑定、
资源/失败、正确性、交付与成本。完整region已真实承载需求片段时，只补边界翻译和逐行验收，
不把region拆成逐kernel adapter。

同一路线的一组能力在一个AUDITED有限表内实现，开发期只跑受影响门；全部稳定后一次完成该
路线批次的跨环境、插桩、source/static/shared组合消费、最终selected和必要P1。未变接缝按身份
复用，枚举项不得新开卡、续预算或重跑整套交付。原版不接受的格记录具名反例并转到本表已经
准入的成熟后备；仍不得转成自研授权。进度按“接入单元的准入范围全部绑定并验收”计数，局部
算子、文档或探针不增加分子。整个M2完成后暂停汇报，等待用户再进入M3。

## 1. Provider 准入处置

“必需”是其下述准入范围必须完成接入与适用验收；不要求接入库中与模型需求无关的 API。
“已有复用”也必须补齐新范围，不继承旧 CLOSED。“后备”不计未接入债务，但主方案出现具体反例
时须先核对对应后备，不能直接转成自写。条件资格失败必须重定界，不能偷删需求或宣称已覆盖。

| Provider | 裁决与需求 | C++ 接法 / 版本与许可 | 明确边界 |
|---|---|---|---|
| ggml CPU | 必需；F01–F17 广覆盖与模型组合 | retained backend graph、persistent pool；`456172ec`，MIT | M1-G资源补丁已验收；M2-B1已接首批普通Direct/Graph与CPU-I标准Scan/TopK，Windows开发验证通过；完整批量域/四环境/三交付/P1仍未完成 |
| ggml CUDA | 必需；F01–F17 的 CUDA 原版后端 | native backend region；`456172ec`，MIT | M1-C已接原生stream/device-buffer与有限F32/attention片段，完整批量覆盖仍待M2；不以既有 HYLE CUDA 核代签 |
| ggml Metal | 必需；F09–F17 成熟递归/稀疏/MoE 片段 | native backend region；`456172ec`，MIT | M1-A 已接 F32 whole recurrent 与 MLX 普通 Direct/Graph 组合、原生资源/完成及三交付；批量覆盖待 M2 |
| ggml Vulkan | 已有复用并扩接；F01–F17 | 现有 managed region 批量映射；`456172ec`，MIT | M2-B1高rank广播/norm、任意轴Sum/Mean、半精度Copy/Cast/既有算术族及I32 Copy/浮点和I32 Concat已接普通入口；任意轴Softmax/广播additive mask同一候选也通过Windows/Air开发门；完整21行域/新增三交付/P1仍未完成 |
| oneDNN | 已有复用并扩接；CPU F01/F04–F06/F18 | primitive/Graph partition、reorder + MatMul；3.13.2 `71094df4`，Apache-2.0 | locked pin 已有 FP8/E2M1/UE8 reorder 源码；格式组合与设备资格待验证；Q2现裁决为INVALID_MEASUREMENT，不再作为接入阻塞，也不宣称性能PASS |
| MLX | 已有复用并扩接；Apple F01–F08/F18 | C++ typed/compiled array region；0.32.2 `1f8e74e3`，MIT | locked pin 的 `from_fp8/to_fp8/quantized_matmul/gather_qmm/qqmm` 已定位；默认 nvfp4 不等于所有 F18 档位 |
| cuBLASLt | 已有复用；CUDA F05 与量化解码后的合法浮点 GEMM | CUDA Toolkit 的公开 C API，精确 toolkit/driver 由已有环境与 package receipt 绑定 | 不增加第二份 toolkit 下载器；属于 NVIDIA SDK 分发合同 |
| Marlin | 已有复用；CUDA F06 合格 W4A16 域 | 已锁 vLLM `51918e25` 原版窄源码，Apache-2.0 | group/zero/packing 与数值档位逐一匹配，不能只匹配4bit |
| SGLang | 已有复用并扩接；CUDA activation/norm、F18 量化/专家程序 | `06992c92` 已有 C++/CUDA 源码片段，Apache-2.0 | AB 只有 W1 static 局部收据；不把 Python/Torch 安装当 SDK 接入 |
| FlashInfer | 必需；合格Linux CUDA的F07/F15 attention、KV 与成熟融合候选 | 0.6.18 `69ff11fc` 的 CUDA headers/固定入口，Apache-2.0 | M1-C原版WSL head256/SM89 FP16与BF16普通入口、资源与三交付通过；Windows本pin编译失格，已接ggml合格attention；完整覆盖仍待M2 |
| CUTLASS / CuTe | 作为成熟程序的内部依赖准入；F05/F18 | 优先 SGLang/FlashInfer 已采用的固定版本；候选源 `f614dc40`，BSD-3-Clause | 不创建空 Provider，也不从模板另写 GEMM；SM89 FP8 示例不等于 block128 已满足 |
| IREE Vulkan | 限定分工资格及M1普通接入完成；F01.3/F02.4/F18精确数值补位 | `6ef606b6`；原版标准算子离线参数化region与私有C runtime，Apache-2.0 WITH LLVM-exception | M1 checked I32 Scan和两种scale的FP8 MatMul含三交付；M2新增原版F32/F16→BF16转换，Windows/Air普通混合入口已通过，新增域完整门仍待完成；FP4由ggml MXFP4承载，不以compiler名义写算法 |
| cuDNN Graph | 后备；F07 的合格 attention 域 | frontend `91dbf3e9`，Apache-2.0；backend 是 NVIDIA SDK，版本在实际准入时锁定 | 已选 ggml/FlashInfer 承载；公开支持表中 Ada decode head≤128，不能代签 Qwen head256 |
| FlashAttention | 后备；F07 的融合 attention | `1bda8f92`，BSD-3-Clause；固定原版 CUDA 程序 | 与已选 ggml/FlashInfer 重叠；Torch 包装不进入 SDK，Windows 原生接法若启用须单独资格化 |
| MPSGraph | 后备；Apple 具名 MLX/ggml Metal 不满足的域 | Apple 系统 framework 的 graph executable；版本随 macOS/SDK receipt | 当前无必须另引入的独有域，不宣称它不支持模型；失败时按具名域启用调查 |
| ncnn | 后备；Vulkan 量化/布局补位 | `70e7bfd6`，BSD-3-Clause；`Layer::forward(VkMat, VkCompute, ...)` | C++ Vulkan seam 已定位；没有取得本合同 F18 全域证据，不能凭名称宣称支持或不支持 |
| MNN | 后备；Vulkan 量化/布局补位 | `397f8533`，Apache-2.0；Expr/后端 execution | Vulkan MatMul 类存在；须先证明精确格式、原生资源与失败合同再提升为必需 |
| XNNPACK / 平台 BLAS | 后备；CPU 具名 primitive/GEMM 缺口 | XNNPACK `81a0804a`，BSD-3-Clause，C operator/subgraph；平台 BLAS 按 OS/SDK 锁定 | 广覆盖和低精度已分给 ggml/oneDNN；没有证据要求重复接一套普通 GEMM |

上述候选的 Git 全长身份和已读入口见源码记录；系统 SDK 的版本来自执行环境而非虚构 Git pin。
后备不意味着已经完成全面支持审计；具体主方案失格时按下一节有限资格项升级调查。

进度分母固定为**10个确定接入/复用单元 + 1个条件资格单元**：ggml四后端分别计4，oneDNN、
MLX、cuBLASLt、Marlin、SGLang、FlashInfer计6；IREE Vulkan是条件单元。CUTLASS/CuTe作为成熟
程序依赖不重复计Provider，runtime patch和后备库不计已接入。IREE若失格，须记录替代方案和
分母变化；不能把该单元删除后声称对应需求已完成。当前完整新范围通过数为0，不否认已有子集实现。

### 有证据的版本与数值差异

- llama.cpp `4c9233c0` 的 bundled ggml 与独立 `456172ec` 不能整体视为相同：本次核对的 CUDA
  文件差异是 `GGML_OP_DUP` 的 I32/I16 接受条件。M1 统一 vendor 源身份并测该域，不能静默换 pin。
- CPU ggml Q4/Q8 dot 路径会使用其量化激活类型。F06 的 W8A16 明确禁止这一步；采用合格成熟
  reorder/有界解码加浮点 GEMM，或真正保持该合同的成熟量化程序。转换成本和压缩驻留一起验收。
- locked ggml 没有本合同 E4M3 FP8 tensor 类型。不能用它的 MXFP4 权重支持代签 W4A8、UE8 scale
  或 FP8 index。CPU/Apple/CUDA 的成熟格式入口已定位；Vulkan 加入具名 Q-VK，而不是自写 shader。
- 目标模型的 DeltaNet/KDA 已有 ggml fused 与原版图组合；DeepSeek mHC/稀疏索引与 GLM pooled
  tail 也有现成图。按类型/head 选择完整合法片段，不因融合入口不接受某域就重写数学。
- GLM Flash 使用已登记 `de253435` 支持分支；MTP 另由 C0 的 SGLang NextN 来源约束，不能用
  base 模型分支存在代签 MTP。模型加载和推测采样仍归测试 consumer。

## 2. 先关闭的有限资格项

这些是**接入验证工作**，不是自实现授权；可与不依赖该项的 Provider 接入分别推进。
每项只出一次可复现结论和下一动作。调查预算包含在对应 M1/M2 卡内，不层层叠加。

| ID | 精确问题与实验 | 限额 / 失败后的动作 |
|---|---|---|
| Q-RUNTIME | ggml context、threadpool 与 plan 可恢复失败；持久 pool；半初始化回滚 | 源码/设计2小时；只审资源 seam。原 CPU BLOCKED 仅在独立 patch factor 按原恢复合同验收后恢复 |
| Q-SHAPE | ggml rank4 与 HYLE rank6/stride；动态索引、重复 scatter、稳定 I32 scan | 每路线一组代表组合；优先合法折叠/分段与 MLX/oneDNN/CUB/IREE 原版操作；不改需求 |
| Q-NUMERIC | L2 epsilon、gated norm 舍入、all-masked softmax、tie/NaN 策略 | 用 C0 oracle 在一份参数表验证；不相容时换成熟组合，不能以更宽容差隐藏 |
| Q-WXA16 | 三种 Q4/Q8 编码的无损 packing 与 W*A16/F32 累加 | 一个 Q4/Q8 混合线性片段，预填逻辑 codes/scales；禁 A8 重量化、禁全模型浮点常驻 |
| Q-FP8 | F18 的权重/激活/index 格式与 F32/UE8 scales、SiTU 再量化 | 各路线一个覆盖五行的受控格式 bundle；同量化值和独立格式 oracle，不下载大模型 |
| Q-VK | IREE 参数化 I32 scan 与 E4M3/E2M1/UE8→标准 MatMul region，在 RTX4070/MoltenVK 同设备运行 | 首次资格含准备最多4小时、每设备一个串行 bundle；检查 Vulkan1.3 基线、压缩驻留与三交付接法；失败只登记具体缺项，下一轮先核对 ncnn/MNN/具名上游实现，不默认自写 |
| Q-CUDA-API | M1-C原版WSL C++ head256 prefill/decode数值通过；Windows本pin编译失格转ggml原版attention；两请求同时在途/资源完成仍由产品卡验收 | 接法调查计入M1-C总2小时；Windows根因纠正预算已停止；保留原始失败和具名成熟替代，不降低计算需求、不移植整个引擎 |
| Q-STATE / Q-SPARSE | recurrent state、索引 carry、压缩窗口与原生 shape 条件 | 从已登记模型图借用完整组合，在边界维度差分；不拆融合来追求逐节点统一 |

Q-VK 是一个窄候选的可行性门，不把通用编译器重构纳入 HYLE。拟采用标准 cast/quantization/
scan/matmul 操作及上游类型合法化，编译器只作构建工具；source/static/shared 交付同一组参数化
程序与原生 runtime，普通消费者不安装 Python、不提供 MLIR、不运行编译器。若该交付形状不可行，
必须明确裁决该候选，不能偷偷增加 SDK 使用负担。首次 Q-VK 的 IREE packed-FP4 路径已失格，日志和部分通过项见 M1-V 卡；原版 LLVM UE8 预转换已消除观察到的 NaN。限定组合 IREE scan/FP8 + 原版 ggml MXFP4 已取得两台机器的数值与 C++ 原生资源证据，见[进度收据](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/native-progress.json)。后续[限定资格](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/checked-queue-qualification.json)已补齐两台机器的 checked-overflow 和共享单队列自动续执行；IREE独立owner已完成SCOPED/AUDITED；原生runtime/queue/quota/异步产品门在Windows/Air通过，Air host ASan也通过，见[原生检查点](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/iree-native-checkpoint.json)。普通HYLE Scan/FP8混合入口、资源失败与产品三交付现已完成，见[最终收据](../../../../evidence/correctness/compute-llm-moe-m1-vulkan-admission/final-acceptance.json)；M1限定接入关闭，完整模型域不计覆盖，原调查预算不重置。

## 3. M1 有限接入工作包

| 包 | owner / 先决条件 | 固定的拟定 FIX | 验收与停止 |
|---|---|---|---|
| M1-G | `compute_ggml_runtime`与`compute_provider_ggml_vulkan`已按[资源卡](llm-moe-m1-ggml-runtime.md)CLOSED；原CPU owner仍BLOCKED | 一个共同 ggml vendor 源与按路线启用的原生后端；复用现有 context 失败 patch 的已审资源部分；CPU persistent pool/plan、OOM/线程创建失败回滚；解决 ggml target 重名/旧新 pin 混用 | 真实 norm→MatMul→状态片段，重复执行不重建 pool；allocation/thread failure、两个 invocation、close；对应源码/静态/动态和 route-OFF；不改原版数学 |
| M1-R | compute / graph 与四路线具名owner；以registry实际状态绑定，CPU/CUDA既有CLOSED须按模型条件新需求重定界 | 批量 lowering 的共用语义检查、capability/诊断、模型/配置/phase 与软件/硬件规则 identity；常量、状态、多输出、effect/lease；每路线独立资源程序 | Direct 与 Graph 一组正交片段；未知模型/版本不匹配、缓存失效；热执行无探测、搜索或计时；不造万能执行器 |
| M1-C | 六owner已按[接入卡](llm-moe-m1-cuda-admission.md)CLOSED，GQA两owner按独立卡CLOSED | 原版ggml CUDA/Linux FlashInfer、外来buffer与两event边界、每invocation资源/失败/完成；managed成本明确，不冒称prepared零查询 | Windows/WSL普通Direct/Graph、同GPU cuBLAS组合、三交付及host插桩，Air/Mini OFF与公共语义；源码/设计调查已停止 |
| M1-A | 五 owner 按[接入卡](llm-moe-m1-apple-admission.md) CLOSED，有限接入域/P0 | 原版 ggml Metal whole recurrent 已接普通 Direct/Graph；MLX native buffer/region 边界、独立 invocation、共同 GatedDeltaNet 语义已实现；自研冻结 | Air/Mini 普通入口、原生及三交付、Air host ASan/UBSan、Windows/WSL 共享资源与 OFF 门通过；最终 119 个 selected 门及独立候选复核通过；P0，新增模型覆盖0 |
| M1-V | 五owner按[父卡](llm-moe-m1-vulkan-admission.md)及[IREE卡](llm-moe-m1-vulkan-iree-product.md)CLOSED，有限接入/P0 | 原版recurrent/MXFP4和IREE checked Scan/FP8接普通Direct/Graph，复用实际feature/queue/quota与原版异步推进；默认冷绑定，无自研数学 | Windows/Air六格普通三交付、资源/失败/host插桩、三环境OFF及120 selected门通过；完整数值/shape域仍待M2/M3，新增覆盖0/256 |
| M1-B | 以上每包所属 build owner 的有限范围 | 复用 lock/features/vendor targets/private patch/resources/license/install；同份 CMake 声明贯通三交付；短 build 根与共享只读缓存 | 开关隔离、单/组合 Provider、源码父工程、静态/动态干净 consumer；关闭路线零下载/链接；不重写构建系统或治理工具 |

新增 ID 在实现前依法 QUEUED→SCOPED→AUDITED 登记；此表中的拟定名称不是空 target，不能算接入。
M1-G 的独立 patch factor 完成维护/升级/OOM/并发/三交付审计之前，不改变旧 CPU owner 的恢复条件。
新 ggml 公共 target 只是一份 vendor 依赖，CPU/CUDA/Metal/Vulkan 程序和执行状态仍各归本路线。

每个 seam 源码/设计调查默认2小时；出现改变 owner、资源语义或交付形状的反例立即重新定界。
这是调查上限，不把正常编译/适用测试伪装成研究时间，也不通过反复增开同名任务重置预算。

## 4. M2 按模型片段批量接入

每批使用上面的四路线承载表，一次把对应 family 映射到成熟 operation schema/region，
按 CPU→CUDA→Apple→Vulkan 串行完成重型验证。源码工作可逐 seam 推进；不用等自研比较才能接库。

| 批 | 需求行 | 实际消费与可复用映射 | 到点产物 |
|---|---|---|---|
| B1 张量与状态基础 | F01–F04/F19 | embedding→norm/gate→residual/index/state；一份参数/类型表驱动四 Provider lowering | 四路线普通 Direct/Graph、动态错误、两请求/旧结果、适用三交付；满足全部门后记入M3-01，按用户最新指示直接继续 |
| B2 线性与注意力 | F05–F08 | 原版 dense/quantized FFN、GQA prefill/decode、KV/RoPE/YaRN/Hadamard；用整片段复用 packing/workspace | 相同 codes/scales 的 Q4/Q8 消费，head256、变 position/length、压缩驻留和必要转换账单 |
| B3 递归混合 | F09–F11 | 原版 conv/DeltaNet/KDA fused 或现成图组合；state 输入输出整体接入 | prompt1/17/128、257-token chunk 与单步等价、非零状态；本地2B B16/Q4/Q8 的24层、8步 teacher-forced decode |
| B4 MoE 与模型片段 | F12–F14 | 上游 routing/dispatch/grouped expert/combine、latent/SiTU、mHC/AttnRes/MTP 组合 | 空/偏斜专家、重复 combine、真实宽度、共享状态；不把引擎调度/采样带入 SDK |
| B5 稀疏与压缩 | F15–F17 | 原版 MLA/DSA/CSA/HCA、shared/pooled index/tail 与增量状态 | 同一选择索引的层间复用、压缩边界、Pro/Flash 两份配置、MTP index carry |
| B6 新数值档位 | F18 | 成熟量化/类型转换、dense/grouped MatMul 与 index 程序；每种 scale/激活单独资格 | 五行精确格式片段 + 本地2B FP8 四路线完整旅程；无原生指令时仍在同 GPU 合格执行 |

批次记录区分 SOURCE_IDENTIFIED、BACKEND_DOMAIN、BOUND、正确性、资源、三交付和性能证据。
M2 的批量映射不按256格各建一套 adapter；不同 Provider 仍有自己的一次性 schema 与原生程序。
必要性能门只验证成熟默认路径及接入成本，**不与冻结 HYLE 算子竞速**；API 固定的不可变候选才跑
适用 P1。同一树/合同/机器证据可复用，文档修改不重跑。oneDNN Q2 不追加原预算外的采样。

## 5. 统一停止与最后裁决

M3 汇总64行四路线的实际证据；准入第三方未接通数不为0、资格项未裁决或适用验证未运行时，
M4/M5 均不得启动。只有成熟组合、版本和后备路径都存在具名反例的残余缺口，才允许 M4 单独
审计最小补缺；预计新增自研0，接口缺陷不能算算法缺失。

按2026-09-20用户指示，**进入M5前必须暂停并汇报成熟Provider覆盖、模型验证和剩余缺口，
等待用户明确继续**。原M3-01专门通知/暂停已取消。

M5 才核对所有 CPU/CUDA/Metal/Vulkan 冻结算法。有效 LOSS 即删除可完整替代域；算法优化预算0，
总分析2小时/采样2小时，不按旧算子数量续杯。无需求旧域直接清理，不新造性能课题。SIMD 基础库
保留；上个 goal 专用 SIMD 增量来源核对为0，Compute SIMD 算子封装仍参加最终裁决。

完成全部模型机制、四路线、普通入口、资源/失败、三交付和适用性能后，按总计划 M6 通知完成并停止。
计划完成与实现完成分别报告；当前没有新的产品覆盖或性能 PASS。
