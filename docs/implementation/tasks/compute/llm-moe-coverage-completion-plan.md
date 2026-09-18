# Compute / Graph：LLM 与 MoE 能力补齐计划

## 本轮边界与基线

执行已于 2026-09-17 获用户授权并创建新 goal，当前从
[C0 输入与共同合同](llm-moe-c0-local-contracts.md)推进；以下“本次修订”描述此前计划交付。

2026-09-17 用户要求制定实施和验证计划：以 Qwen3.8-27B / 2.4T、Kimi K3、DeepSeek V4
Pro / Flash、GLM 5.3 / Flash 的语言主干与 MoE 为覆盖目标，避免同一能力在四路线缺席。
用户已明确：大模型定义**算子覆盖需求**，不要求现有机器完整加载；本机另选小模型做完整旅程。

- 原计划 baseline：`39f22b29fd88d6eb57ce6ab3ed9c619933d30778`。本次本地模型修订 baseline：
  `c292b2cedf4b5d38bfe596657547e2b186832a74`，`main`，核对时工作树干净。
- 本页是用户下一轮新 goal 的唯一执行计划。模型数据、同步、量化与验证安排均合并在本页；
  旁边的 JSON 只保存来源/文件清单，不形成另一份施工队列。本次修订没有启动产品实施。
- [四路线基础架构](four-route-foundation-acceptance.md)已经完成。本计划是下一阶段的有限能力扩充，
  不是恢复 R1–R7、旧大 goal 或旧百分比。后续用户授权执行时，依 C0–C7 和 owner 协议实际实施；
  这次计划交付的“仅文档”事实不限制下一轮授权。
- 长期合同仍见 [Compute](../../../architecture/compute-design.md)、
  [Graph](../../../architecture/graph-design.md)、[四路线](../../../architecture/compute-routes.md)
  与[能力准入](../../../architecture/capability-completeness.md)。
- 完成的含义：下列冻结模型需要的计算语义、准入数值档位、状态与图组合，在 CPU、CUDA、
  Apple MLX/Metal、Vulkan 的普通 C++ Direct / Graph 中均有实际可执行路径及适用证据。
  不承诺所有未来模型、所有格式、无限 shape 或任意设备上的绝对 SOTA。
- 本阶段覆盖文本输入、prefill、decode、必要 MTP 计算和本地 MoE；视觉塔、图像/视频预处理、
  通用 CNN、扩散、语音、训练不属于这次 LLM/MoE 范围。具名模型具有视觉能力不等于本轮验收视觉。
- 模型文件读取、tokenizer、请求/KV 分配策略、批处理策略、采样/投机接受策略与服务属于 consumer。
  HYLE 负责其需要的张量、索引、缓存访问/更新、依赖、资源寿命和计算；不以引擎边界为由漏掉机制。
  不实现 CPU+GPU 模型联合计算、跨卡模型切分、分布式、HIP 或推理引擎。

## 1. 冻结需求来源

完整模型 ID、不可变 revision、配置摘要与 SHA-256、实现源码 pins 见
[模型来源快照](llm-moe-model-sources.json)。它记录需求来源，不记录产品 PASS，也不是第二份 registry。
七个大模型只作需求来源，不下载它们的完整权重；本地 2B 权重已由用户提供，见第 2 节。
以下是源码/配置调查结果，详细数学合同在 C0 固定；本次没有运行模型。

| 需求模型 | 已核对的区别 | 必须落到能力与验证中的内容 |
|---|---|---|
| [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 64 层；48 linear + 16 full attention；hidden 5120；Q/KV 24/4，head 256 | Gated DeltaNet、因果短卷积、门控 GQA、部分 RoPE、FP32 递归状态；dense FFN 17408 |
| [Qwen3.8-2.4T-A95B](https://huggingface.co/Qwen/Qwen3.8-2.4T-A95B) | 92 层；69 linear + 23 full；512 experts，top-10 | 前者的混合计算加 shared/routed experts、不同 Q/K/V head 比例；不能用 dense FFN 代替 MoE |
| [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | KDA + gated MLA；LatentMoE；896 experts，top-16；AttnRes | 逐通道衰减 KDA、latent 投影/归一化、SiTU-GLU、跨层残差注意力；MXFP4 权重与对应激活数值规则 |
| [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) | hidden 7168；384 experts，top-6；CSA/HCA + sliding window | 压缩池化/增量状态、稀疏索引、Hadamard、mHC/Sinkhorn、hash/score routing、受限 SwiGLU、FP4/FP8 |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | hidden 4096；256 experts，top-6；压缩层安排与 Pro 不同 | 同族机制但独立 shape / layer 配置；不能只用 Pro 的结果签 Flash |
| [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) | MLA/DSA；78 层；256 experts，top-8；indexer full/shared | 稀疏选择结果跨层复用、RoPE 交错、FP32 router、FP8 block scaling；共享索引的真实依赖与寿命 |
| [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | 45 层；34 KDA + 11 DSA；288 experts，top-8；mHC | KDA 与 pooled sparse indexer、tail token 处理、mHC；不是 GLM-5.3 的单纯缩小版 |

版本选择按上表正式 repo 的本次 snapshot；不静默替换成 `0731`、`0813`、Base 或第三方量化版。
上游后续版本只有出现新的计算需求或可复现错误时才另行更新本计划的需求版本。

### 首包必须解决的三个有限来源问题

1. **Qwen3.8 输出门**：两个配置声明 `output_gate_type=swish`；本次固定的 Transformers
   `Qwen3_5Attention` 却直接调用 `sigmoid`，没有读取该字段。先对照官方支持的 SGLang/vLLM
   路径、精确权重/配置与独立公式确定 oracle；本次未裁决哪一个是 Qwen3.8 的合格参考。
   Qwen3.5 的正确输出不能替 Qwen3.8 解除这一项。
2. **Kimi 量化激活**：模型说明为 MXFP4 weights / MXFP8 activations，但配置中的
   `input_activations` 是 null；配置解码、BF16 参考执行与部署激活量化要分别绑定实际代码。
   不从权重四位推导激活也四位，不把解码到 BF16 的参考当原生 MXFP8 性能对照。
3. **MTP 与可选分支**：配置中的 head/layer 必须对应可用权重与实际 forward；一些通用
   Transformers 路径忽略附加层。按具名消费绑定独立 MTP 计算片段，不把普通 next-token
   logits 当 MTP 已测，也不在 HYLE 实现 speculative acceptance 策略。

这些是参考资格问题，不是推翻四路线架构的理由。每项只调查具名分支和公式，预算见 C0。

## 2. 唯一本地验收模型、同步与量化数据

### 2.1 固定用户已下载的 2B

用户指定 `C:\Users\kiro\Downloads\Qwen3.5-2B` 为本轮唯一原始模型数据来源。
移除原计划的 0.8B 要求；快速回归和完整验收都使用这份 2B，不再为增加另一个小模型下载权重。
各机器只接受从 Windows 本地数据同步的验证副本，缺文件时先报缺失，不能自动访问 HF 下载替代品。

本次已只读核对本地实物，完整文件身份见 [本地模型清单](qwen35-2b-local-input.json)：

- 13 个文件，共 4,571,274,023 bytes；唯一 Safetensors 权重文件为
  `model.safetensors-00001-of-00001.safetensors`，4,548,221,488 bytes。
- 权重头有 632 个张量：596 个 BF16、36 个 F32；分别为 2,274,067,232 和 2,592 个数值元素。
  index 与 header 的名称/分片映射一致，shape/dtype/字节长度与 payload offsets 完整且无重叠。
- 24 个语言层、18 DeltaNet + 6 full attention；本地还含 `mtp.*` 权重。MTP 的实际 forward
  资格仍须 C0 绑定，存在文件不代替运算验证。
- 所有文件已计算 SHA-256；整个清单的身份为
  `4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155`。
  本地 config 与既有 `15852e8c16360a2fea060d615a32b45270f8a8fc` 配置哈希一致；未重新联网
  比较全部权重字节，不声称已经验证整个本地目录等同于该远程 revision。实际验收以本地清单为准。

本地原目录只读消费。权重、量化派生物和完整参考张量放仓库外，Git 只保存来源、参数、哈希与结果。
4.55 GB 是整个权重文件大小；语言主干实际常驻量、状态、workspace、runtime 和加载峰值分别记录。

### 2.2 跨机器只同步本地资产

| 环境 | 输入方式 / 预定缓存 | 核对与运行责任 |
|---|---|---|
| Windows | 直接读取用户原目录；派生物放 `C:\hyle-cache\models\qwen35-2b\<source-id>\<profile>` | 原始模型和量化制品的 canonical 来源；不在 Downloads 中原地改格式 |
| WSL | 从 `/mnt/c/Users/kiro/Downloads/Qwen3.5-2B` 本地复制到 `~/.cache/hyle/models/qwen35-2b/<source-id>/original` | 首次按清单校验；正式运行使用 Linux 本地文件系统，不从互联网重新取得权重 |
| MacBook Air | 从 Windows 经 `ssh kiro@172.22.10.198` 对应的 OpenSSH 传输连接同步到 `/Users/kiro/.cache/hyle/models/qwen35-2b/<source-id>/original` | 默认身份；承担已声明的 Apple / MoltenVK 正确性与 Air 性能资格 |
| Mac mini | 从 Windows 经 `ssh -i C:\Users\kiro\.ssh\codex_mac_172_22_10_107_win kiro@172.22.10.107` 对应连接同步到同样的用户缓存结构 | 只按既定 correctness/portability 范围验证；不替 Air 签性能 |

`source-id` 使用完整清单身份，报告可显示前 12 位；profile 为 original 或具名派生格式。
C0 首次准备时串行同步，先写新的 `.partial` 目录，核对每个文件大小/SHA-256 后发布为可用缓存，
记录接收 receipt。已有身份一致的副本直接复用；缺失或损坏只重传相应文件，不每轮重传 4.55 GB。
失败的副本不得参与验证，也不反向覆盖 Windows。产品源码仍按既有 Git / 完整 archive 规则同步，
模型数据清单不替代源码候选身份。

验证进程只使用显式本地路径并关闭模型自动下载。软件依赖另按具名版本与已有缓存准备；缺少依赖
要报出具体项，不能顺带抓取模型。参考框架和 HYLE 串行运行、释放前一套 GPU 权重后再开始下一套；
传输/加载峰值单独计费，稳态计算比较从双方已完成约定准备的边界开始。本次尚未分发或加载模型。

### 2.3 四路线共用的 BF16 / 4bit / 8bit 验收档位

从同一份本地 BF16 权重离线生成有限派生物；使用具名成熟工具/公式，冻结版本、目标 tensor 名称、
参数、校准输入（如适用）、输出 codes/scales 与哈希。只生成一次，在四路线同步同一套逻辑量化值。
各路线可另做无损 packing，不能各自重新量化后把不同权重当成相同输入。

| 档位 | 数值合同与完整模型要求 | 首次验收 |
|---|---|---|
| B16 | 本地原始 BF16 语言主干与规定的 F32 参数/状态 | C3：四路完整 prefill/decode；作为原始模型基线 |
| Q4-S128 | signed symmetric INT4、group128、FP16 scale；矩阵 activation/output 为 FP16，F32 累加，所需转换显式入图 | C2 片段，C3 四路量化 2B 完整语言主干 |
| Q8-B32 | Q8_0 数值编码：每 32 个 signed INT8 权重一个 FP16 scale；W8A16/F32 累加路径，不能静默把 activation 再量化成 INT8 | C2 片段，C3 四路量化 2B 完整语言主干 |
| F8-B128 | E4M3 FP8 权重 block128×128、动态 activation group128 与具名 scale/rounding；以 W8A8 计算合同验证 | C6：四路 FP8 2B 完整语言主干及目标模型片段 |
| 其余已准入域 | affine Q4/group64；目标模型的 UE8M0 scale、MXFP4/MXFP8、索引量化等分别列行 | C2/C6 四路完整计算片段；本地 2B 不冒充具名大模型的所有量化组合 |

Q8_0 是这里准入的权重数值编码，不代表 HYLE 加入 GGUF 模型加载器或自动支持所有 Q8/INT8 方案。
来源为已登记 ggml/llama.cpp/ds4 的 block storage 消费，C0 绑定精确 codec 和同合同执行对照。
Q4/Q8/FP8 档位只量化冻结名单中的语言线性权重；embedding/tied head、Norm、卷积、状态、门控参数等
如何保留高精度在 C0 按名称明确，不从“4bit 模型”推导所有张量都四位。每档报告实际量化比例和字节数。

FP8 等格式在缺少原生指令的设备上仍需同设备的合格执行路径，转换/分块解码成本计入；实现为空、
CPU fallback 或整个模型浮点展开常驻都不满足四路线共同覆盖门。BF16 跑通不能代替 Q4/Q8/FP8 验收。

### 2.4 独立测试消费者与分层验证

模型知识由仓库测试目录的独立 consumer 持有，不进入安装 SDK，不向 HYLE 增加模型加载/采样 API。
验证工具用成熟解析器读取本地 Safetensors/config/tokenizer，准备命名 tensor 和固定 token IDs；
小型 C++ consumer 用公开 Direct/Graph 接口构造计算并绑定资源，所有被测计算实际经过相应 HYLE 路线。
参考工具另行执行相同输入/量化值并保存输出。数据读取器、reference、consumer 各有独立位置与身份。

| 验证层 | 固定范围 | 能证明什么 |
|---|---|---|
| 快速回归 | 取这份 2B 的真实权重片段；一个 DeltaNet + 一个 full-attention/FFN，变化输入与状态 | 短周期查接口、数值、量化与状态问题；不标为整模型 |
| 完整语言主干 | B16/Q4-S128/Q8-B32/F8-B128 各在四路执行 24 层；batch=1，prompt 1/17/128，8 步 teacher-forced decode | 完整图、持续状态、logits 与量化组合；reference 和 HYLE 顺序执行 |
| 状态边界 | 257-token 分块与逐 token 等价、非零初始状态/reset；batch=2/更长边界放有限片段 | 防止短模型 smoke 漏掉 chunk、序列隔离和缓存偏移 |
| 七个大模型的计算需求 | 已冻结配置/源码下的真实宽度、head 比例、专家数、格式与状态规则；有界输入和缩小容量的组合图 | 覆盖大模型独有 MoE/MLA/CSA/HCA/mHC 等机制；不称为运行过大模型 checkpoint |

大模型测试无需下载完整权重；确定性输入/权重用例必须保留被测机制和真实维度，并由独立公式/
合格参考执行核对。若个别问题确需额外真实权重片段，单独记录最小需求和获取条件，不能自动改为
全模型下载。原尺寸整层超容量时，按独立专家、真实宽度块和状态边界拆验证单元，明确组合限制。

实现正确性使用同一份量化 codes/scales/激活规则作对照；与 B16 比较量化损失单列，不能混成一个
宽松容差。固定 token、分层抽样、最终 logits/状态、误差门与量化基线在 candidate 前冻结；
不把生成了自然语言当成正确性通过，也不把模型默认 262144 context 全部预分配。

## 3. 四路线覆盖合同

验收单位是 **能力 × 数值档位 × shape/layout 域 × Direct/Graph × 路线**，不是一个 op 名称。
能力可以由原生 kernel、成熟库的单算子、完整融合片段或 region 承载；普通用户不为补齐一行
自行写 custom kernel / provider adapter。只存在 MLX 原生函数、SPIR-V 专家入口或 schema 不算完成。

### 数值与格式

- 通用档位：FP32；模型需要的 FP16 / BF16 存储、计算输出及规定的 FP32 累加/归一化/状态。
  精确的中间提升、舍入、epsilon、广播、NaN/Inf、饱和与越界合同逐能力固定。
- 现有 signed symmetric Q4 group128 与 affine Q4 group64 分别登记；二者不是同一种 Q4。
  先统一逻辑 codes / scales / zero 或 bias 语义，再允许路线自己的 packing。
- 本地量化 2B 增加 Q8_0 / W8A16 的有限 INT8 权重档位；block32、FP16 scale 与实际激活精度
  单独验收。它与 FP8/W8A8 是不同的八位计算合同，不能互相代签。
- 新准入：官方实现要求的 E4M3 FP8 block scaling（含 FP32 与 UE8M0 scale 的区别）、
  E2M1 FP4 group32 / UE8M0，以及经来源资格确认的激活量化。权重、激活、KV/index quantization
  是不同能力；精确 block/scale/rounding 独立列行。不会顺带承诺所有 GGUF / AWQ / GPTQ 格式。
- 缺原生 BF16/FP8/FP4 指令的设备必须有同设备的合格实现或有界分块解码/计算，保持所选逻辑
  量化值和误差合同。解码 workspace 与成本计费，不展开并常驻整个模型的浮点副本；不得回落到
  CPU 计算模型内容。高于参考精度的中间运算也需验证，不能自动当数值等价。
- 功能共同支持不要求相同速度、物理布局或 kernel。每路线保留设备特定的优选实现，按 GPU
  型号/架构/SM、硬件能力、driver / toolkit / OS / vendor 版本和数值域在研发期固定规则。
  准备只匹配绑定，执行不计时、搜索或裁决。没有资格的硬件不宣称性能已验证。
- C0 把下表展开成**实际需要的组合**，不生成所有 op/dtype/shape 的无意义笛卡尔积。
  普通发行的共同能力必须全覆盖；专家裁剪需显式暴露 capability 缺席，不能继续自称完整 profile。
  C1 同步落实普通入口的能力查询/准备期诊断；未支持的 dtype、shape 或版本域要能明确说明，
  不等执行到中途才发现缺算子。公开支持表由实际验证单元回映，新增行要同时声明四条路线。

### 有限能力族与当前缺口

`部分` 指当前确有子集实现，但未满足这里完整的模型合同；`缺` 指未取得普通 Direct/Graph
内建路径；都不是本阶段 PASS。更细路线审计在 C0 绑定源码，不能从历史 CLOSED 继承新能力。

| ID | 能力族 / 必需片段 | CPU | CUDA | Apple | Vulkan | 首次交付包 |
|---|---|---|---|---|---|---|
| F01 | view/reshape/transpose/split/concat/broadcast、必要 materialize、cast | 部分 | 部分 | 部分 | 部分 | C1 |
| F02 | embedding/gather/scatter-add、mask/select、index/offset、必要 scan/permute | 部分 | 缺 | 缺 | 缺 | C1 |
| F03 | residual、逐元素运算、SiLU/SwiGLU、clamp、softplus/sqrt、SiTU、输出门 | 部分 | 部分 | 部分 | 部分 | C1 |
| F04 | RMSNorm/LayerNorm/L2Norm、gated norm、稳定 reduce/softmax/top-k | 部分 | 缺 | 缺 | 缺 | C1 |
| F05 | dense/batched/broadcast MatMul/GEMV、模型所需 transpose/stride、FP16/BF16 | 部分 | 部分 | 部分 | 部分 | C2 |
| F06 | 两类 Q4 与 Q8_0 的逻辑 carrier、压缩驻留和 MatMul | 缺 | 部分 | 部分 | 缺 | C2 |
| F07 | causal GQA/MQA、mask/ragged、KV 读写/追加/间接访问、门控 attention | 部分 | 缺 | 缺 | 缺 | C2 |
| F08 | partial/interleaved RoPE、位置偏移、YaRN、需要的 Hadamard rotation | 部分 | 缺 | 缺 | 缺 | C2 |
| F09 | causal depthwise conv1d，prefill 与带历史状态的 update | 缺 | 缺 | 缺 | 缺 | C3 |
| F10 | Gated DeltaNet：chunk / recurrent、QK L2 norm、FP32 状态 | 缺 | 缺 | 缺 | 缺 | C3 |
| F11 | KDA：逐通道衰减、gate 下界、状态布局、ragged/chunk / decode | 缺 | 缺 | 缺 | 缺 | C3 |
| F12 | MoE score/hash/grouped top-k、bias 只用于选择、权重归一化 | 部分 | 缺 | 缺 | 缺 | C4 |
| F13 | token dispatch/permute、grouped expert GEMM、combine、shared/latent experts | 缺 | 缺 | 缺 | 缺 | C4 |
| F14 | mHC/Sinkhorn pre/post/head、AttnRes、必要 MTP 计算组合 | 缺 | 缺 | 缺 | 缺 | C4 |
| F15 | MLA 的投影/吸收形式、不同 QK/V head dim、可选 NoPE/门控 | 缺 | 缺 | 缺 | 缺 | C5 |
| F16 | DSA/indexer、跨层选择复用、pooled index/tail、稀疏 gather-attention | 缺 | 缺 | 缺 | 缺 | C5 |
| F17 | CSA/HCA、sliding window、learned/overlap compression、增量压缩状态 | 缺 | 缺 | 缺 | 缺 | C5 |
| F18 | FP8 / MXFP4 存储与动态激活量化、dense/grouped GEMM、索引量化 | 缺 | 缺 | 缺 | 缺 | C6 |
| F19 | 以上片段的动态参数、状态 effects、资源复用、固定规则与 native 重复执行 | 部分 | 部分 | 部分 | 部分 | C1 起，每包同步 |

基础 FP8 storage codec 不等于 F18 的模型计算已经存在。某个复合能力可由完整 Provider region
实现，但必须把其内部数值/状态合同回映到这些行，不强迫拆掉成熟融合，也不把 wrapper 算多份覆盖。

## 4. 横向工作包

每包的四路线缺一不可；允许在包内逐路提交，实现和适用证据齐全后才签该包完成。
先达到本地 2B 的 B16/Q4/Q8 自然完整用法，再补目标模型独有的 MoE/Attention/FP8/FP4 机制。

| 包 | 有限产物 | 当包必须完成的真实消费与停止点 |
|---|---|---|
| C0 本地输入与共同合同 | 按清单同步 2B，取得本地 B16 参考输出；绑定独立测试 consumer、离线量化工具/张量名单；解决三个来源问题；F01–F19 展开为有限 profile/domain/case 表；分配 owner 与卡 | 数据同源可复现；MUST 有来源、四路承载、自然调用草图、数值 oracle、预算与直接消费者；各 owner AUDITED 前不写产品 |
| C1 基本张量与状态边界 | F01–F04；四路 shape/parameter/资源读写的必要补齐；先固定 state 输入/输出、alias 和拒绝合同 | embedding → norm/gate → residual/索引片段；两独立运行状态、错误/别名/关闭；Direct + Graph + 三交付 |
| C2 线性与常规 Attention | F05–F08；两类 Q4 与 Q8_0 的跨路线落实；有界 cache 访问机制 | dense FFN、门控 GQA 的 prefill→decode；本地 Q4/Q8 真实片段；head 比例、offset、padding、压缩驻留与显式精度转换完整 |
| C3 递归混合计算 | F09–F11；卷积、DeltaNet、KDA 的 chunk/update 与 state effects | 四路本地 2B 的 B16/Q4-S128/Q8-B32 完整语言主干和多步 decode；Qwen3.8 真实宽度块；Kimi/GLM KDA 状态用例 |
| C4 MoE 与层间组合 | F12–F14；shared/latent expert、SiTU 组合、mHC/AttnRes、具名 MTP 计算 | 四个 MoE 家族各自规则的完整本地 dispatch→expert→combine；空专家/偏斜/ties；跨层残差与附加 head 的组合/寿命 |
| C5 稀疏与压缩 Attention | F15–F17；压缩状态、index 复用、窗口与位置边界 | Kimi gated MLA、DeepSeek Pro/Flash CSA/HCA、GLM full/shared index 与 Flash pooled-tail；真实维度片段 + 缩小容量的模型组合图 |
| C6 现代低精度完整链 | F18；FP8/MXFP4 carrier/scale/rounding、同设备执行、冷 packing 共享 | 四路本地 2B 的 F8-B128 完整旅程；C4/C5 的量化 dense/MoE/Attention 链；独立解码 oracle、峰值驻留/带宽/同步/完整性能 |
| C7 覆盖终审 | 汇总每包已完成的矩阵/receipts、普通 C++ recipes、默认能力查询与限制 | MUST 无缺席/未运行；七个需求模型计算回映闭合，本地 2B 的 B16/Q4/Q8/FP8 四路旅程通过；不首次接线算子 |

C3 的小模型验收随此前能力新增/修改按 selector 重放受影响路径；其余不改的性能证据复用。
后续每个需求家族的组合用例在 C4–C6 当包引入，不把模型图接线全部推迟到 C7。
来源未裁决只阻塞相应需求行；可独立固定的本地 2B 能力由各自卡继续推进，不能让无关分支造成全线
停工。该安排不减少 MUST，也不把有未决项的包或 C7 写成已完成。

### 下一轮 goal 的启动动作

1. 核对实际 HEAD、工作树、AGENTS 与 registry，保留计划提交之后的改动，不回退到本页 baseline。
2. 核对本地模型清单；在 C0 记录验证副本、参考依赖与数据身份，执行首次必要同步和有限 B16 参考
   forward，保存固定输入、分层抽样、logits 与状态。外部参考通过只签数据/reference 资格。
3. 为受影响 owner 建立当前卡，先 SCOPED，再按有限 FIX 到 AUDITED；C0 的参考/夹具只作准备，
   C1 开始真正补产品。首次可运行里程碑是 C3 的四路线 B16/Q4/Q8 2B 完整文本计算。
4. 继续 C4–C6 补齐大模型独有机制和 FP8/FP4，C7 汇总所有适用证据后结束。任何预算到点或实质
   外部 blocker 都给具名问题与恢复条件，不虚报通过、不无限调查，不自动启动下一代模型目录。

### owner 与实施协议

- 2026-09-18 用户进一步固定执行顺序：**登记/检出主流 Provider → 接入真实计算片段 → 验证并列出
  剩余覆盖缺口 → HYLE 补缺或实现有证据的优势算法**。C1–C6 均遵守此顺序；不能先排四份自写
  算子队列，再把 Provider 接入留到最后。主流库清单、现有接入事实与待接入范围统一维护在
  [Provider 承载复核](llm-moe-c1-provider-reuse-audit.md#四路线-provider-候选与接入优先级)。
  对候选库按真实片段分配接入责任；重叠能力通过资格和等责比较择优。检出固定版本只是来源准备，
  编译成功只是依赖准备，普通 Direct/Graph 实际可用且完成适用验证才计入产品覆盖。
- 成熟实现优先是实施顺序：先从参考模型的实际计算链定位可用的第三方算子、融合片段或完整
  region，再处理 HYLE 的描述、资源和状态接入。只有查明所需域确实缺失，或完整成本比较证明
  HYLE 更好，才新增自有算法；类型/布局/API 不同及旧 pin 未覆盖不能直接作为自写理由。
  四路线覆盖格是验收责任，不是四份自写 kernel 清单。保留成熟库内部融合和资源管理。
- 审计必须分别记录直接库调用、编译固定上游源码、修改后的上游片段，以及参考算法后自写。
  使用 SDK 原语、复用 HYLE SIMD 或引用第三方公式，不等于已经接入第三方算子 Provider。
  已实现的自有路径也不自动取得默认优选资格；按有限范围核对后复用、迁移或仅保留必要缺口域。
  当前 C1 的具体核对与下一步边界见 [Provider 承载复核](llm-moe-c1-provider-reuse-audit.md)。
- **已有自有算子同样接受替换和删除。** 成熟 Provider 完成同一数值/状态合同后，比较完整片段，
  包括必要转换、上传下载、同步、融合得失、驻留与结果寿命；既有 HYLE 实现不因已经投入开发而
  获得保留资格。若没有可证明的完整收益，优先采用合格成熟实现。确有具体优化机制时沿当包有限
  纠正预算尝试；没有可信机制，或预算内仍不能取得优势，就迁移 Direct/Graph 消费者并删除被
  完整覆盖的重复实现、构建分支和过时文档，停止维护双轨。删除前验证替代路径及原支持域，
  Provider 尚未覆盖的必要域和有硬件/版本/shape 条件的 HYLE 优势域分别保留，不静默丢能力。
  不合格或不同责任的性能对照先纠正比较，不能据此删实现，也不能以反复换测法拖延裁决。
- **SIMD 明确保留。** 用户指出 SIMD 还服务于推理之外的通用消费，本轮 Provider 替换不得
  删除或裁剪 SIMD 模块、公共能力或其非推理消费者；不能用某个推理片段输赢推导 SIMD 无用。
  退场规则只针对 Compute/Graph 中被合格 Provider 完整替代且无独立收益/需求的重复算子路径。
- 用户同时要求 CMake 主导的 build system 能简明描述第三方接入。每个 Provider 的 route/feature、
  锁定源码/补丁、vendor options、targets、运行资源和许可证在 owner 处集中描述；复用现有 lock/
  cache/package registry 投影三交付与关闭路线隔离。带补丁依赖使用可重复的 build-local 源码，
  不修改共享原始缓存。随真实接入验证并定向简化，不能另造独立下载/打包脚本作为正常产品路径。
- C0 给 `compute_ops` 分配共同语义，`compute` 分配必要 Direct/result/格式合同，`graph` 分配
  图语义/资源 effects；不是先扩一个万能执行器。旧 API 无兼容要求，按需求迁移真实消费者。
- CPU：`compute_provider_native`、必要的 `compute_provider_onednn` /
  `graph_provider_onednn` 与 `compute_route_cpu` / `graph_route_cpu`。
- CUDA：`compute_provider_cublas`、`compute_provider_marlin`、`compute_provider_cuda` 与
  `compute_route_cuda` / `graph_route_cuda`。Apple/Vulkan 分别归现有同名 compute/graph route。
- backend / Mem / Sched / Build 只因有证据的基础能力缺陷或新的必要依赖/交付边界定向入卡；
  不全库重审，不把上层模型策略搬入基础模块。
- 成熟实现候选沿[已登记研究 pins](compute-graph-route-unification-audit.md#本轮架构分析统一合同与独立执行)
  的 ggml/llama.cpp、vLLM、SGLang、ds4、MLX/oneDNN 定向检查。新的模型 reference pin 不自动
  成为产品依赖升级；Python/Torch wrapper 不是可交付的 C++ Provider。
- 当前 `compute_provider_ggml` 的 BLOCKED 不解除，`compute_sched` 的 QUEUED 不自动开工。
  可借用合格源码/片段，但不得借改名 adapter 绕过已知 failure/lifetime blocker。
- 实施开始时核对当时 HEAD/registry；每个包绑定实际 owner 的新卡及合法能力缺口 trigger，
  `SCOPED → AUDITED`，固定有限 FIX 后才改产品。多 owner 共享一张卡时明确各自 FIX 与 gate。
  后继包不沿用前包 AUDITED 授权；新增独立 factor 先登记 QUEUED。
- 研究预算：C0 额外调查最多 90 分钟，三个来源问题各最多两条具名实现路径；各后续包在卡中
  分别声明调查与纠正预算，默认每个未知机制最多 60 分钟/两条候选来源。预算到点留下明确结论、
  实施方案或具体 blocker/恢复条件；不能凭“还需研究”无限续期，也不把写计划算能力完成。

## 5. 验证方法

### 当前执行顺序补充：Provider 覆盖与实现资格（2026-09-18）

用户指出 C1 的实际任务切分仍偏向四路线分别自写，要求改正施工顺序。C0–C7 的需求、包边界
和验收范围保持；各包的实施顺序统一为：**成熟 Provider 实际承载 → 正确性/资源/交付资格 →
同合同完整成本比较 → 固定条件路由与迁移 → 明确缺口的有限补缺**。已有实现的具体缺陷在资格
阶段定向修复，不能等到性能胜出后才检查，也不能以“性能更好”为缺检查的实现签收。

- “Provider 已接入”必须指明普通 Direct/Graph 实际可用的能力、dtype/shape、硬件/版本条件；
  只有源码、target、Context 或一个示例 region 时，按该局部事实报告。上游已有但 HYLE 尚未绑定
  是接入缺口；转换、两输出、checked 状态或资源边界不同是适配事项，不能直接改列自写算法。
- CPU、CUDA、Apple、Vulkan 都先从已登记成熟实现选择覆盖一组需求的算子/融合片段/region。
  每个具名缺口至多检查当前卡预算内的两条候选；不要求把候选清单里所有库都引入，也不再按
  每个算子逐路线扩充自有 kernel。具体当前顺序以 [C1 活动卡](llm-moe-c1-tensor-state.md)为准。
- 性能候选先满足同一输入、输出、精度、alias、错误、完成与资源寿命合同。形状/类型/静态容量
  等检查在准备期完成，真正变化的绑定/索引/数值范围在必要的运行边界检查；已验证且不可变的
  事实可复用。普通 checked 路径不得静默改成 unchecked，也不得拿省略验证/同步/结果责任的
  裸实现作为赢家。上游自身的检查不足也由接入责任方处理，不能转嫁给普通用户。
- 发现已有实现越界、溢出、错误丢失、未完成即释放、OOM 回滚不完整或数值违反合同，先登记
  具体反例、影响域和 owner 的有限 FIX。未修复域不得参加优选；必要时从该域的默认规则撤下，
  使用已有合格实现或明确拒绝，不能静默算错。CLOSED/BLOCKED 依原协议定向恢复，不全库重审。
- 比较后的规则允许按硬件、SM/架构、驱动/toolkit、dtype/shape 和完整片段条件分别选优。
  用户选择路线，准备只匹配绑定，执行不计时或换赢家。无明确完整收益时采用合格成熟实现；
  被完整替代且无独立价值的自有路径在迁移验证后清理，SIMD 按用户约定保留。

该补充是当前实施约束，不增加模型、算子或公共治理范围，也不增加任何产品完成度。

### 数值、状态与完整组合

1. **独立公式**：小尺寸用标量 FP64/规范定义的舍入参考；量化从同一 codes/scales 独立解码。
   大维度用已资格化官方/成熟实现。HYLE 与 oracle 不能只是同一 vendor kernel 的两个 wrapper。
2. **真实形状**：保留模型 hidden/head/专家维度、激活/scale 布局及边界，覆盖 prefill、decode、
   非 tile 整除、空/单行、mask 与不等长请求；容量变化必须与数值语义变化分开标记。
3. **逐步状态**：完整序列与多种 chunk 划分/逐 token 输出、最终状态对比；非零初始状态、reset、
   batch 内不同长度、请求交错、保留旧结果。覆盖卷积 K=4、压缩 ratio=4/128、窗口 128 的前后边界。
4. **MoE**：top-k 边界/ties、bias 只影响选择而不改变原始 routing weights、空专家、全选同组、
   极端 token 偏斜、共享专家、latent 投影、hash token IDs、scatter 累加顺序与精度。
5. **图与 Direct**：同逻辑输入/参数/状态得到同域结果；参数变化不误用旧 capture/packing；
   effects/alias 真实，两个 Invocation 状态独立，constants 可共享；失败后接受的工作及 owner 不丢。
   冷编译与稳态重放成本分开，有限 shape 变体/运行参数满足模型片段，禁止 token 循环重建整张图。
6. **模型**：本地 2B 的 B16/Q4/Q8/FP8 固定 token 输入、逐层抽样、logits 与多步 teacher-forced decode 对比；
   检查 greedy 结果时记录 top-1 margin，浮点近 tie 用预声明数值规则裁决，不要求随机采样文本相同。
   每个大模型的完整计算路径还需缩小容量的结构用例；它与原尺寸片段证据共同证明需求回映。
7. **误差**：在 candidate 前为各 profile 固定 atol/rtol、范数误差、state 漂移/步数及量化误差标准。
   离散索引语义单独判定；不得看到结果后放宽，不用统一的宽松百分比掩盖 gate/scale/状态错误。

### 平台、交付与资源

- CPU：Windows、WSL Linux、Apple CPU 的适用编译/正确性；CUDA：Windows/WSL RTX 4070；
  Apple：MacBook Air 原生 MLX/Metal，Mini 仅既定 correctness/portability；Vulkan：RTX 4070 与
  Air/MoltenVK。新未测厂商/架构可保留合法实现路径，单独标记验证缺席，不外推性能资格。
- 每包每路线实际新增能力必须在 source / installed static / installed shared 普通 consumer
  执行；一个完整组合可覆盖多行，不要求每个数学原子复制三套 fixture。GPU OFF 只签隔离，不能代 ON。
- 覆盖申请失败/非法参数、异步接受后失败、保留结果、关闭/drain、并发运行状态、quota/峰值驻留；
  host sanitizer 与适用 device 工具各写实际范围。不故意制造设备挂死，不把 vendor 内部未观测内存记零。
- 真相仅 Windows；WSL/macOS 是验证副本。低并行构建 2、CTest 1，重型任务串行。
  Windows 启动前把所需 DLL 放 EXE 同目录并检查依赖；验证 launcher 复用 `SetErrorMode(0x8003)`。
  Tracy 验证用 `TRACY_ONLY_LOCALHOST=1`，不触发提权/防火墙调整或影响用户桌面。
- 现有容量：Windows RAM 约 64 GiB、RTX 4070 显存 12282 MiB，WSL profile 32 GiB、Air 16 GiB。
  先做内存账单，再运行；不给 T 级模型安排本地全权重加载，也不把 offload 伪装成单路线验收。

### 性能与公平比较

- P0 先检查常驻量/临时量、拷贝、解码、同步、launch/遍历次数、融合边界与渐近复杂度。
  Prefill 不采用仅对单 token 合理的逐 token Python 式执行来取得“功能齐全”；性能异常先查结构。
- 每包每路线只对 API/算法固定的不可变 candidate 启动一个预声明 P1 bundle；正常/holdout shape、
  同 workload A/A、双顺序与有效样本预算在卡中冻结，沿现有合格 placement/仪器，不重写统计系统。
  多能力可共用一个真实片段；未变路径按 tree/contract/toolchain/machine 身份规则复用证据。
- 同一比较必须有相同输入数值、量化档位、状态、输出责任、执行拓扑与完成点。packing/转换/
  上传下载/同步/融合损失/资源回收逐项 CHARGED 或明确两侧一致地分账。裸 kernel 与拥有完整
  生命周期的 Graph 只能作分账诊断，不据此宣布算法或 Sched 输赢。
- 正确性 oracle、实现竞争 P1、完整模型 consumer P2 各有目的。外部引擎含调度/tokenizer 时，
  不拿其 tokens/s 和 HYLE 单 region 时间排名；端到端性能比较须冻结相同 consumer 工作。
- 先选合格成熟完整实现，再验证 HYLE 替换是否有完整收益。失败后只在固定预算内做源码/成本驱动
  的纠正；候选改变重新回 AUDITED。无法达到门时如实 LOSS/INCONCLUSIVE 与 blocker，不能无限
  换顺序重测、补样本或继承 CPU 历史例外来解除新问题。
- 某格式/设备没有合格同合同 peer 时，在候选前固定独立原生组合对照或可解释的服务成本预算，
  仍验证适用性能风险；结果限于该预算/对照，不制造 SOTA 或免测的 N/A。

## 6. 完成判据与进度

实施 C0 产出的明细矩阵必须具备：requirement ID、模型/分支来源、dtype/layout/shape 域、
普通入口、四路实现与规则、Direct/Graph case ID、oracle、资源/交付/P1 证据及未覆盖域。
后续仅因明确需求变化改版本；验收使用实际 case/result 回映检查，不能手填四个 PASS。

- 功能进度分别报告已验证的共同能力行/总行，以及四路线详细单元格；同一公共数学能力不因多个
  wrapper 或模型重复计数。文档、研究、枚举、schema 和旧 CLOSED 都不算本阶段实现进度。
- C7 完成要求所有 MUST 明细都有四路自然入口、正确性/状态/资源、三交付与适用性能裁决，
  本地 2B 的四种规定数值档位完整文本旅程通过，七个需求模型回映无遗漏；没有“以后再补另一路”的收尾。
- 容量不足的原始大模型整层/整模型运行不在本地 MUST；报告清楚实际执行的验证单元和组合限制。
  能力本身因缺实现/来源/适用验证无法裁决时必须 BLOCKED，不能用容量说明掩盖。
- 模块状态仍唯一来自 registry。每包完成/重大范围变化/用户询问时报告实现、验证、缺口、阻塞，
  使用 `scripts/loc-report.py` 按完整 Compute+Graph 实现边界计数，不把 LOC 当完成度。
- 所有约定 gate 通过后通知“本轮 LLM/MoE Compute + Graph 能力补齐完成”，列明模型版本、
  数值/shape/硬件资格与 consumer 边界后停止；不自动扩视觉、其它架构或算子大全。

## 本次计划交付

本次修订将本地模型、跨机器同步、离线 Q4/Q8/FP8 与测试消费者统一合并到本页；移除 0.8B 依赖，
保存实际文件清单与结构检查结果。没有创建新 goal、修改产品或改变当前模块状态。
统一 LOC 脚本 `--self-check`：Compute+Graph 实现 37,035 行 / 130 个文件，不含测试；
边界复用 [R7 完整路径清单](../../../../evidence/correctness/compute-graph-r7/terminal-loc.json)。
维护验证：selector 的 docs/closure 两门 PASS，其余 116 门按影响面跳过；来源/清单的元数据
一致性检查与 `git diff --check` PASS。不构建、不运行模型或 P1。
本地 header/index/大小/文件哈希已核对，尚未同步到其它机器、生成量化派生物或取得计算正确性证据。
源码中的未决来源问题仍留给具名 C0 核对，不计为产品 blocker 已解决。
