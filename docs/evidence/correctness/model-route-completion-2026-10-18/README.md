# 名单模型的融合与量化接线补全

2026-10-18，在 `d0359b2ce` 基础上检查当前产品源码。范围为既定 8 个文本模型、
22 个官方/GGUF 制品配置，覆盖 CPU、CUDA、Apple、Vulkan 的普通路由。
[模型来源](../../../implementation/tasks/compute/llm-moe-model-sources.json)及
[制品目录](../../../implementation/tasks/compute/llm-moe-provider-demand-matrix.json)只提供固定模型事实；
旧表中的通过、拒绝和流程状态不作为当前结论。接线阶段未运行完整模型，也不声称性能胜出；
后续Qwen完整24层三phase执行与尚未接受的精度差异见文末及后续底座报告。
本报告保留接线阶段的定向结果；后续紧凑MLA、压缩页/评分、权重热成本及架构迁移的当前证据见
[非多模态底座审查](../inference-foundation-review-2026-10-18/README.md)。新架构仍须继续补查这些模型需求，
不能用局部修复条数、历史覆盖百分比或删除重复入口代替功能与整链证据。

## 范围与接线依据

| 模型 | 官方主要格式 | GGUF 主要格式 | 配置数 |
|---|---|---|---:|
| Qwen3.5-2B | BF16、F32 recurrent | K/IQ4/Q8、F16/F32 | 3 |
| Qwen3.8-27B | BF16、F32 recurrent | K/IQ3/IQ4/Q8、BF16/F32 | 3 |
| Qwen3.8-2.4T-A95B | BF16、F32 recurrent/router | IQ4/K/Q8、F32 | 3 |
| Kimi-K3 | MXFP4 expert、UE8M0、BF16/F32 | MXFP4、Q8/BF16/F32 | 3 |
| DeepSeek-V4-Pro | FP4 expert、FP8、UE8M0、BF16/F32 | 无已选 GGUF 配置 | 1 |
| DeepSeek-V4-Flash | FP4 expert、FP8、UE8M0、BF16/F32 | MXFP4、Q8/BF16/F32 | 3 |
| GLM-5.3 / GLM-5.3-Flash | FP8 block128、BF16/F32 | K/Q8、F32 | 6 |

检查 embedding/LM head、普通/批量投影、norm/gate、attention、GDN/KDA/conv、
MoE dispatch/expert/combine、indexer/压缩、MTP 和状态边界。路由必须先满足真实格式、
布局、形状、资源与数值合同；有成熟实现却未接通才修接线，不能以改 dtype 或放宽误差冒充支持。

当前 BF16/F32、GGUF 投影与 expert、官方 FP8 decode/linear、GroupedFloat4/8 都已有实现承载。
CPU 主要使用 ggml/ATen，CUDA 使用 ggml/cuBLAS/DeepSeek/SGLang/Torch，Apple 使用 MLX/ggml Metal，
Vulkan 使用 ggml/typed operations/IREE。官方 FP4/FP8 的分组和 scale 合同与 GGUF codec 分开核对。
旧 R11 的 Apple/Vulkan grouped FP4/FP8 拒绝已过时；它们当前已经接通。
按实际 GET_ROWS 而非 MatMul 计算，静态核查 CPU 当前为21种、Apple为18种；
CUDA/Vulkan 本轮各18种运行通过。CPU/Apple的这个目录核查没有新增运行结论。

## 已修复

- **CPU/CUDA Graph region 的多输出接线**：旧 adapter 只收一个输出，recipe 和 CUDA
  边界资源也只处理 output 0。现按实际输出数量交给已有 native support 判断并完整接线，
  不增加按算子名字放行的第二份白名单。
- **CUDA/Vulkan GGUF embedding**：分别由 8/8 种扩到 18 种，接入锁定 ggml 已有的
  IQ2_XXS/XS/S、IQ1_S/M、IQ3_XXS/S、IQ4_NL/XS、MXFP4 等 GET_ROWS。
  CUDA 保留 IQ4_NL/MXFP4 行宽 256 对齐要求；没有用完整解码权重代替选行。
- **CUDA F32 批量 MatMul**：普通 cuBLAS adapter 只收二维，ggml adapter 又拒绝 F32
  三／四维，默认路由无人承载。现接通已有批量实现，显式保留 F32 source precision，
  避免只声明 F32 accumulation 却仍允许 TF32 源运算；二维分工、dense 与广播方向限制不变。
- **Windows CUDA 原生批次**：移植锁定 FlashInfer 的 host 编译接缝，并统一平台、依赖、
  公开能力宏、安装与测试装配。保留原 attention/cache 内核与有限硬件域，不以 WSL 或 CPU 参考代替 Windows 原生调用。
- **Qwen3.8-27B 的 GQA 比例 6**：实际 QH24/KVH4，旧 batch qualifier 只允许1/2/4/8。
  上游 scheduler/kernel 已使用运行时分组除法；将6接入原路径，不增加新内核或任意比例放行。
- **Materialized Graph 的独立 QK/V 深度**：控制器曾拿 Q 描述绑定输出，Dqk≠Dv 时在准备期失败。
  改为 recipe 的真实输出描述，64→128 与 MLA 形状 576→512 均可沿原 Direct/Graph 参考路径执行。
- **高维原生分页／ragged**：CPU 接通已有动态深度 flash 的 plain 512→512、576→512；
  CUDA 接通已有 VO-split 原生 512→512；Vulkan 接通原 FullF32 shader 的 plain F16
  512→512、576→512；Apple 接通原 compact full tile 的 plain F16/BF16 两类高维。
  Apple/Vulkan统一route的准备／控制器几何判断。直接读取驻留KV，不展开整套历史。

## 定向结果

| 检查 | 结果与边界 |
|---|---|
| CPU/CUDA 多输出融合 | 各一次小闭包通过，两个 canonical test TU 也编译通过；scalar GDN、D128 逐通道 KDA、融合 recurrent 三种图，各三个节点合成一个 region、同时发布两原输出及后继输出、重复执行两次 |
| KDA 接受前缀 | CPU/CUDA/Apple/Vulkan 从 K=3 快照的中间设备子区间恢复并继续最后一个 token，两个结果与独立 FP64 公式一致；不以 oracle 状态替换实际反馈。Apple/Vulkan 最大绝对误差分别约 5.47e-9 / 4.84e-9 |
| CUDA/Vulkan embedding | 两条既有回归各 1/1 通过、各 18 格式，对照原 CPU codec 解码；两次选择/重绑、非法索引拒绝、旧结果保留，Vulkan 另检查失败后新 invocation；各只重编 adapter/test 两个主机 TU |
| CUDA F32 批量 MatMul | 2/2 通过：三／四维 Direct，双 queue/slot、重绑与旧输出保留；Graph MatMul→Neg 单 region 并发布两值。保留低位尾数的 F32 输入对照独立 FP64 公式，实际运行使用组件 Context，未完整链接默认 roster |
| Windows CUDA 原生批次 | RTX 4070 / SM89 / CUDA 13.3 / MSVC 19.50：四个 native CUDA TU 编译通过；普通 `cuda::create_context()` → `BatchProgram::prepare()` 发现 H24/KV4/D256 承载。8 组 paged/ragged×F16/BF16×H4/KV2或H24/KV4 的组件 Direct/Graph，16 次独立 FP64 oracle 全通过，最大绝对误差0.0022713，小于原有0.007，不放宽误差 |
| 构建配套 | Windows/Linux 产品公开依赖与宏的同一关联测试通过；FlashInfer、Vulkan、MLX三套原生补丁 tree/NOTICE 与当前依赖锁的同次 CMake 检查通过。只做配置 fixture 和指纹检查，无 SDK 重建 |
| Materialized 非对称深度 | 同一小探针旧控制器在 `prepare_batch` 失败，新控制器 1/1 通过：paged/ragged×F16/BF16×64→128或576→512，两请求混合 Q2/Q1、两次 payload 重绑，Direct/Graph 共32次数值执行，对照独立 FP64 公式；canonical CPU Graph 测试 TU 另以 /W4 /WX 编译通过 |
| CPU 高维原生批次 | canonical 定向测试 1/1、8 case通过：paged/ragged×F16/BF16×512→512(H64/KV1)或576→512(H96/KV1)，两请求 Q2/Q1、乱序物理页、KV31/33，最大绝对误差约3.49e-5，小于原.0025；只更新相关主机 TU，复用 native cache。未新增高维 Graph、Q8 或整模型运行结论 |
| CUDA D512 原生批次 | 普通 CUDA Context 发现 H64/KV1/D512 承载；8个新增 case 的组件 Direct/Graph 共16次 FP64 oracle通过，另复核原D256的16次，最大绝对误差0.0029732<原.007。新增一个小 CUDA TU，保留原四个 native 对象；最终主机 guard另编译并通过6个无payload跨度越界负例。高维运行覆盖 causal/plain/正scale，full/window/split路径本轮仅编译与源码复核 |
| Vulkan 高维原生批次 | 2/2定向测试通过：H64/KV1的512→512、H96/KV1的576→512，paged/ragged×两组payload，Q1/Q17，完整FP64 oracle检查约590万活跃F16输出，原.0025容差不变；另复核6个低维子域。只更新1个native host及相关主机TU，复用同一SPIR-V；最终删除共享判断后的冗余条件另编译通过，属于已测路径的等价消重 |
| Apple 高维原生批次 | MacBook Air Mac16,12，H64/KV1的512→512、H96/KV1的576→512，各paged/ragged×F16/BF16，共8组合、Q1/Q17通过完整FP64 oracle，最大误差6.104e-5，小于原F16 .0025/BF16 .008。实际compact PSO分别为16/18 KiB，低于设备32 KiB；Q1也走该tile。复用既有Steel/kernel与SDK/metallib，只更新相关小主机TU；最终资格消重另编译通过，未重复数值 |
| 日常测试装配 | 非对称深度和 CUDA 公开批次消费者均注册进现有 Graph batch 测试；CPU+CUDA、CPU、无对应路线三种轻量 CMake fixture均配置成功并只登记实际可用目标，无依赖构建 |

复核源码、原始命令和结果见 [Vulkan embedding](vulkan-embedding.zip)、
[融合与 KDA](fusion-kda.zip)、[CUDA F32 批量矩阵](cuda-f32-batched.zip)、
[Windows CUDA 批次](windows-cuda-batch.zip)、[CUDA 18 格式 embedding](cuda-embedding18.zip)、
[Vulkan 18 格式 embedding](vulkan-embedding18.zip)、[非对称深度控制器](materialized-asymmetric.zip)、
[CPU 高维原生批次](cpu-highdim.zip)、[CUDA D512 与高维边界](cuda-highdim.zip)、
[Vulkan 高维原生批次](vulkan-highdim.zip)、[Apple 高维原生批次](apple-highdim.zip)。
13 格式的旧 Vulkan 包保留为先前结果，不代表当前仍只支持 13 种。
CPU/CUDA 的运行复用组件 Context 和 native 缓存；两个 canonical Graph test TU 另做编译检查，
未运行完整 canonical 测试二进制。Apple KDA 使用原 NativeProgram 小探针，Vulkan 使用公开入口。
CUDA 旧验证 native 缓存缺少三个与本轮无关的 IQ1 codec，
验证对象明确拒绝它们，不能用本轮结果代签这些格式。没有复制 SDK 或执行全库构建。
Windows 首次链接/运行发现旧 concat、sampling 与 batch extension 缓存，更新相应小单元后通过，
原失败记录保留；没有把旧缓存不匹配算成新的产品缺陷。Windows 本轮实际运行覆盖 plain K/V，
FP8 cache、滑窗与外部 capture 沿原源码承载，未在本轮新增对应的 Windows 运行结论。
旧 Windows batch absence 负向消费者已移出活动测试，原源码保留在 Windows 包中，
由当前正向消费者替代。相关清理按用户约定先写 `test.ps1` 再执行，未运行旧清理脚本。

## 不应误判的缺口与剩余边界

- GDN/causal conv 的 F32 recurrent 合同是实际模型要求，BF16 权重不意味着这些状态也应为 BF16。
- LightningIndexer及后续独立packed indexer已有多query和逐query可见性；TopK产生device IDs，
  不要求逐token读回索引。SinkAttention现支持正batch及共享`[Q,S]`/独立`[B,Q,S]` mask，
  各query不同选中KV可以批量接一次Sink。CPU/CUDA typed分页评分→TopK→padded选页→Sink已有ready证据；
  Apple/Vulkan有native padded ready读取及选中输出接普通Graph的证据。它们不等于native分页评分、page Graph/pending授权，
  通用原生paged batch也仍无逐querymask/sink端口。完整DeepSeek从空历史、原始QAT、压缩及接受前缀后继续的整链仍须补查。
- 通用原生分页批次仍有自己的格式/形状域，不能从普通 SDPA/MLA 成功推导原生分页支持。
  CUDA 新 512→512 不代表 576→512：普通 FlashInfer 576→512 最小 CTA32 的静态 shared
  为109,072 bytes，超过实测 SM89 的101,376 bytes；不能只改谓词。专用 MLA 还要求
  V=CKV 及拆分的 latent/rotary 表示，不能偷换当前三个独立 Q/K/V 端口的含义。
  三个独立Q/K/V的576→512仍不能原样据此放开；materialized参考的gather、GQA展开及score成本保持可见。
  后续已有独立typed紧凑MLA表示：CPU/CUDA paged与ragged、Apple/Vulkan paged直接借用共享CKV的前512项作V，
  保真实512/576行跨度，不复制整份V。对应Direct/Graph及CUDA pending证据见后续报告；
  Apple/Vulkan compact ragged、GPU窗口及Vulkan native BF16 attention仍须按真实资格和证据处理，不能从公共schema推导支持。
- 共同 BatchSchema 的 Q/output 与 plain cache 当前为 F16/BF16；普通 F32 SDPA/MLA
  是另一条可组合路径，不能用这次窄类型分页结果代签 F32 cache 或整模型精度。
- 公开 GGUF 的三个 IQ1 narrow 变体仍未接入 CUDA/Apple/Vulkan GET_ROWS；不能由 18 种成功
  推导全部 21 种的 embedding 均可用，也不能把 MatMul/expert 承载当作 GET_ROWS 承载。
  这属于公共catalog边界，不能虚构为22制品已选embedding的缺口；完整expert bank的已修过度门与设备单buffer限制也分别记录。
- 模型加载、tokenizer、生成/接受策略及完整引擎留在引擎层。早期T23/1932节点图的四输入additive-mask SDPA默认无承载，
  RouteSearch修复只使其从65,536状态穷举变为直接报node300缺实现；该普通F32 additive-mask功能洞仍在，
  不因Qwen改用语义匹配的typed RaggedAttention就称已修。[早期诊断](../../../../.scratch/architecture-refactor-2026-10-18/route-search/08-model-diagnostic.stderr.log)
  后续完整Qwen3.5-2B官方BF16 CUDA消费者已执行24层、每phase1938节点/998 step，Q/KV为23/23→4/27→1/28，
  next token 20→248046→198与独立reference一致；150份导出张量均finite、未使用KV尾部全0。
  logits相对L2依次1.874%/1.549%/2.067%，hidden为1.990%/1.323%/2.460%；无既定容差不能写数值PASS。
  首层首次实质偏离已定位到post-attention RMSNorm一个BF16 ULP；同输入的CPU F64及F64→F32→BF16参考
  与Hyle全部47,104个输出逐位相同，Torch仅该点不同；这不证明24层全部偏差来源已经核清。
  单次trace不是benchmark，也不代签其他制品、平台或完整引擎。
  [执行日志](../../../../.scratch/compute-execution-refactor-2026-10-18/qwen35-ragged/05-full-text.stderr.log)、[完整比较](../../../../.scratch/compute-execution-refactor-2026-10-18/qwen35-ragged/comparison.json)
  [首层高精度对照](../../../../.scratch/compute-execution-refactor-2026-10-18/qwen35-layer0-probe/ffn-slice/f64-reference.json)

继承范围是实际制品/phase的默认Direct与Graph路线、融合和量化接缝、CPU＋GPU权重/状态驻留、动态长度与请求布局，
以及prefill→续算→decode→接受前缀后继续的共同状态与失败寿命；现有组合已能表达的需求不重复造API。
carrier共享、冷Plan共享和selected-prepared权限迁移只替换重复组织方式；generated-output、统一执行和旧入口删除仍须完成，
不能据此声称8模型/22制品或全部非多模态Compute/Graph功能已经核完。
新增native同槽准备已有CUDA16组合Direct与两节点共享metadata完整oracle，exact Context与实际workspace冷query/grow-only另有CPU九项、canonical CUDA及mock2/2；
core adapter初始17项与共享provider上限修正后三项均通过。这些是局部架构证据，不能替代尚未迁移的row-cache/page/pending合同或整链精度接受。
[native同槽包](../../../../.scratch/compute-execution-refactor-2026-10-18/graph-batch/source-results.zip)、[Context/workspace结果](../../../../.scratch/compute-execution-refactor-2026-10-18/batch-retained-workspace/summary.json)、[adapter日志](../../../../.scratch/architecture-refactor-2026-10-18/region-adapter/tests/04-capability-tests.stdout.log)
