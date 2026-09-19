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

本次交付施工计划后先向用户报告。模块保持 SCOPED，产品 FIX 为空；本页的拟定 FIX 必须由相应
owner 的后续有限卡审计到 AUDITED 才能实施。旧 ggml CPU、oneDNN 的 BLOCKED 不自动恢复。

## 1. Provider 准入处置

“必需”是其下述准入范围必须完成接入与适用验收；不要求接入库中与模型需求无关的 API。
“已有复用”也必须补齐新范围，不继承旧 CLOSED。“后备”不计未接入债务，但主方案出现具体反例
时须先核对对应后备，不能直接转成自写。条件资格失败必须重定界，不能偷删需求或宣称已覆盖。

| Provider | 裁决与需求 | C++ 接法 / 版本与许可 | 明确边界 |
|---|---|---|---|
| ggml CPU | 必需；F01–F17 广覆盖与模型组合 | retained backend graph、persistent pool；拟统一到 `456172ec`，MIT | 当前 CPU lock 是 `8599e0ea`，迁移与 failure patch 单独审计；不是已完成 |
| ggml CUDA | 必需；F01–F17 的 CUDA 原版后端 | native backend region；`456172ec`，MIT | stream/device-buffer 桥未接；不以既有 HYLE CUDA 核代签 |
| ggml Metal | 必需；F09–F17 成熟递归/稀疏/MoE 片段 | native backend region；`456172ec`，MIT | 独立于 MLX runtime；保留上游融合，原生资源/完成需接通 |
| ggml Vulkan | 已有复用并扩接；F01–F17 | 现有 managed region 批量映射；`456172ec`，MIT | 既有少量原子不代替完整片段；原生类型/head/subgroup 条件逐域验证 |
| oneDNN | 已有复用并扩接；CPU F01/F04–F06/F18 | primitive/Graph partition、reorder + MatMul；3.13.2 `71094df4`，Apache-2.0 | locked pin 已有 FP8/E2M1/UE8 reorder 源码；格式组合与设备资格待验证；Q2 原阻塞仍在 |
| MLX | 已有复用并扩接；Apple F01–F08/F18 | C++ typed/compiled array region；0.32.2 `1f8e74e3`，MIT | locked pin 的 `from_fp8/to_fp8/quantized_matmul/gather_qmm/qqmm` 已定位；默认 nvfp4 不等于所有 F18 档位 |
| cuBLASLt | 已有复用；CUDA F05 与量化解码后的合法浮点 GEMM | CUDA Toolkit 的公开 C API，精确 toolkit/driver 由已有环境与 package receipt 绑定 | 不增加第二份 toolkit 下载器；属于 NVIDIA SDK 分发合同 |
| Marlin | 已有复用；CUDA F06 合格 W4A16 域 | 已锁 vLLM `51918e25` 原版窄源码，Apache-2.0 | group/zero/packing 与数值档位逐一匹配，不能只匹配4bit |
| SGLang | 已有复用并扩接；CUDA activation/norm、F18 量化/专家程序 | `06992c92` 已有 C++/CUDA 源码片段，Apache-2.0 | AB 只有 W1 static 局部收据；不把 Python/Torch 安装当 SDK 接入 |
| FlashInfer | 必需；CUDA F07/F15 attention、KV 与成熟融合候选 | 0.6.18 `69ff11fc` 的 CUDA headers/固定入口，Apache-2.0 | C++ prefill/decode 入口接收 stream；Windows、head256、SM89 与资源资格先验证 |
| CUTLASS / CuTe | 作为成熟程序的内部依赖准入；F05/F18 | 优先 SGLang/FlashInfer 已采用的固定版本；候选源 `f614dc40`，BSD-3-Clause | 不创建空 Provider，也不从模板另写 GEMM；SM89 FP8 示例不等于 block128 已满足 |
| IREE Vulkan | **条件准入资格先行**；F02.4、F06/F18 精确数值补位 | `6ef606b6`；标准算子离线编译成参数化 region，C runtime，Apache-2.0 WITH LLVM-exception | 源码有 SPIR-V FP8 legalization、scan/sort/scatter、VkDevice 包装；Q-VK 必须先过，禁止以 compiler 为名手写算法 |
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
| Q-CUDA-API | FlashInfer 原版 headers/依赖可由 Windows/WSL C++ 使用；head256 prefill/decode、独立请求、stream/workspace | 接法调查最多2小时；若入口失格保留 ggml 合法 baseline、登记具体改用的成熟入口；禁止无边界移植整个引擎 |
| Q-STATE / Q-SPARSE | recurrent state、索引 carry、压缩窗口与原生 shape 条件 | 从已登记模型图借用完整组合，在边界维度差分；不拆融合来追求逐节点统一 |

Q-VK 是一个窄候选的可行性门，不把通用编译器重构纳入 HYLE。拟采用标准 cast/quantization/
scan/matmul 操作及上游类型合法化，编译器只作构建工具；source/static/shared 交付同一组参数化
程序与原生 runtime，普通消费者不安装 Python、不提供 MLIR、不运行编译器。若该交付形状不可行，
必须明确裁决该候选，不能偷偷增加 SDK 使用负担。当前未执行 Q-VK，不能写 PASS。

## 3. M1 有限接入工作包

| 包 | owner / 先决条件 | 固定的拟定 FIX | 验收与停止 |
|---|---|---|---|
| M1-G | 拟新增 `compute_ggml_runtime`；原 CPU owner 仍 BLOCKED；必要 build owner 只定向开卡 | 一个共同 ggml vendor 源与按路线启用的原生后端；复用现有 context 失败 patch 的已审资源部分；CPU persistent pool/plan、OOM/线程创建失败回滚；解决 ggml target 重名/旧新 pin 混用 | 真实 norm→MatMul→状态片段，重复执行不重建 pool；allocation/thread failure、两个 invocation、close；对应源码/静态/动态和 route-OFF；不改原版数学 |
| M1-R | compute / graph 与四路线 owner，现有 SCOPED | 批量 lowering 的共用语义检查、capability/诊断、模型/配置/phase 与软件/硬件规则 identity；常量、状态、多输出、effect/lease；每路线独立资源程序 | Direct 与 Graph 一组正交片段；未知模型/版本不匹配、缓存失效；热执行无探测、搜索或计时；不造万能执行器 |
| M1-C | CUDA backend/route + 拟新增 ggml CUDA / FlashInfer factor | 复用本路线 native stream/buffer/event；固定原版后端 import/export 或小资源桥；prepared workspace、capture/失败/完成 | 同 GPU ggml→成熟 GEMM/attention→结果的组合；错误后资源仍可回收、旧结果保活；Windows/WSL；DLL 邻置 |
| M1-A | Apple route + 拟新增 ggml Metal factor | MLX 与 ggml Metal 各自持有原生 program；建立必要 buffer/完成边界；whole recurrent region，不分裂为逐 kernel 接口 | Air/Mini 正确性；状态交错、old output、原生同步成本；三交付；不能把共享地址当作无需同步 |
| M1-V | 现有 ggml Vulkan/两 route owner；Q-VK 后才决定新增 IREE factor | 扩现有 managed region schema；需要时接独立 IREE 原生 Vulkan region/parameter buffer/完成；共享设备互操作受限时明确有界 GPU copy | RTX4070 和 Air/MoltenVK；所需 feature 诊断、同设备链路、失效/teardown 与三交付；无 CPU 模型 fallback |
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
| B1 张量与状态基础 | F01–F04/F19 | embedding→norm/gate→residual/index/state；一份参数/类型表驱动四 Provider lowering | 四路线普通 Direct/Graph、动态错误、两请求/旧结果、适用三交付；满足全部门后触发 M3-01 通知并暂停供用户提问 |
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

M5 才核对所有 CPU/CUDA/Metal/Vulkan 冻结算法。有效 LOSS 即删除可完整替代域；算法优化预算0，
总分析2小时/采样2小时，不按旧算子数量续杯。无需求旧域直接清理，不新造性能课题。SIMD 基础库
保留；上个 goal 专用 SIMD 增量来源核对为0，Compute SIMD 算子封装仍参加最终裁决。

完成全部模型机制、四路线、普通入口、资源/失败、三交付和适用性能后，按总计划 M6 通知完成并停止。
计划完成与实现完成分别报告；当前没有新的产品覆盖或性能 PASS。
