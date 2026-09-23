# Compute / Graph：模型驱动的成熟 Provider 接入与完成计划

## 当前合同与重启基线（2026-09-19）

用户要求重写全部活动计划并启动新 goal，纠正逐算子自实现的执行方向。本页替代原 C0–C7
施工队列，是新 goal 的唯一总计划；[M0 当前卡](llm-moe-m0-provider-scope.md)、
[Provider 目录](llm-moe-c1-provider-reuse-audit.md)和需求矩阵是本页的明细，不形成独立队列。

- 产品基线：`ebe5af5109afb01b029c1ba85484d03be61d6fe6`，Windows `main`，重写前工作树干净。
  保留所有已有实现、提交及合格证据，不回退仓库。旧 C1 尚未完成，不沿用旧百分比。
- 顺序固定：**模型能力 → 必需算子/融合片段/状态与数值域 → Provider 与接入方式 →
  四路线成批接入 → 覆盖验收 → 具名剩余缺口 → 最后存量淘汰**。
- 预计新增自研算子为 **0**。先将 ggml/llama.cpp 的现成计算链接入；其它成熟库按必需覆盖
  或具名性能价值准入。不得预排 Native/CUDA/Metal/Vulkan 四份自写 kernel 清单。
- 所有已有 HYLE 自实现算子冻结至 M5；包含 CPU SIMD 算子 Provider、CUDA、Apple Metal、
  Vulkan 及修改上游算术后的自有部分。不能扩域、调研、优化、续用旧 FIX 或提前开性能对比。
  已有实现继续保留当前有效能力；SIMD 基础库及非推理消费者不删除。
- “库里有、HYLE 未接通”属于接入缺口。layout、dtype、两输出、状态、owner 或失败接口不同，
  先解决图/资源适配与受审的窄运行时桥，不能据此自行重写算法。
- 所有 **必需接入** 的 Provider 及其需求域完成实际接入和覆盖核对前，禁止新增自实现，也禁止调研或性能比较自己的算子。
  候选未审、后端未核对、接入未完成、版本过旧、适用验证未运行，都不等于算法缺失。
- 本计划完成后明确通知并停止；不扩成算子大全、模型加载器、服务引擎、CPU+GPU 联合推理、
  HIP 或分布式模型执行。文本 LLM/MoE 为范围，视觉/语音/训练不在本轮。
- 2026-09-20用户新增停止点：**进入M5之前暂停并汇报**成熟Provider覆盖、模型验证与剩余
  缺口，等待用户明确继续；不得先启动旧自研的性能比较、裁决或删除。M1–M4继续执行。
- 2026-09-23用户新增更早的当前停止点：**M2全部成熟Provider接入完成后、进入M3前暂停并
  汇报**。用户恢复后，原M5前停止点仍有效；这不缩小M2分母，也不把部分B1签成M2完成。
- 长期边界仍由 [Compute](../../../architecture/compute-design.md)、
  [Graph](../../../architecture/graph-design.md)、[四路线](../../../architecture/compute-routes.md)
  与[能力准入](../../../architecture/capability-completeness.md)决定。各路线保持原生资源与执行，
  HYLE 管理图和片段边界，允许 Provider 一次执行完整 region。

## 1. 冻结需求来源

完整模型 ID、不可变 revision、配置摘要与 SHA-256、实现源码 pins 见
[模型来源快照](llm-moe-model-sources.json)。它记录需求来源，不记录产品 PASS，也不是第二份 registry。
七个大模型只作需求来源，不下载它们的完整权重；本地 2B 权重已由用户提供，见第 2 节。
以下是已固定的源码/配置需求；详细合同与已有参考资格见 C0，本页不把参考运行当产品验收。

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

### 已完成的来源资格，按身份复用

[C0 来源资格](llm-moe-c0-source-qualification.json)已区分 Qwen 输出门、Kimi 激活量化与
各模型 MTP 分支；[C0 记录](llm-moe-c0-local-contracts.md)已保存本地/验证机输入、B16 参考、
量化派生物与限制。新 goal 不重做这些已完成准备，也不把它们计为 HYLE 产品通过。
M0 补充 llama.cpp 图节点 → ggml 后端 → Provider 接入方式的全量映射，沿用模型和数值合同。

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
  来源资格已由 C0 绑定，存在文件不代替 HYLE 运算验证。
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
首次准备已在 C0 串行同步；后续修复副本仍串行进行，先写新的 `.partial` 目录，核对大小/SHA-256 后发布为可用缓存，
记录接收 receipt。已有身份一致的副本直接复用；缺失或损坏只重传相应文件，不每轮重传 4.55 GB。
失败的副本不得参与验证，也不反向覆盖 Windows。产品源码仍按既有 Git / 完整 archive 规则同步，
模型数据清单不替代源码候选身份。

验证进程只使用显式本地路径并关闭模型自动下载。软件依赖另按具名版本与已有缓存准备；缺少依赖
要报出具体项，不能顺带抓取模型。参考框架和 HYLE 串行运行、释放前一套 GPU 权重后再开始下一套；
传输/加载峰值单独计费，稳态计算比较从双方已完成约定准备的边界开始。已分发输入与 B16 参考的结果见 C0；HYLE 完整模型执行仍待验收。

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
如何保留高精度沿用 C0 按名称冻结的名单，不从“4bit 模型”推导所有张量都四位。每档报告实际量化比例和字节数。

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
- 已完成的 C0 将下表展开成**实际需要的组合**，不生成所有 op/dtype/shape 的无意义笛卡尔积。
  普通发行的共同能力必须全覆盖；专家裁剪需显式暴露 capability 缺席，不能继续自称完整 profile。
  M1–M3 同步落实普通入口的能力查询/准备期诊断；未支持的 dtype、shape 或版本域要能明确说明，
  不等执行到中途才发现缺算子。公开支持表由实际验证单元回映，新增行要同时声明四条路线。

### 有限能力族与原始缺口快照

`部分` 指原计划 baseline 确有子集实现，但未满足这里完整的模型合同；`缺` 指未取得普通 Direct/Graph
内建路径；都不是本阶段 PASS。更细路线审计在 M0 绑定 Provider 和后端源码，不能从历史 CLOSED 继承新能力。

| ID | 能力族 / 必需片段 | CPU | CUDA | Apple | Vulkan | 验收分组（历史 ID） |
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

C1–C6 在上述表格与 JSON 的 `first_package` / case ID 中仅保留为验收分组身份，
分别对应 M3 的六组消费，**不再是施工顺序或自实现授权**。M0–M6 是唯一新队列。
原始缺口快照不代表当前已接入情况；后者见 Provider 目录及实际 receipt。

## 4. 新施工清单：M0–M6

### M0：先确定模型需求、Provider 与接入方式

当前 owner `/root`；[M0](llm-moe-m0-provider-scope.md)的计划交付已完成：
[64行四路线映射](llm-moe-provider-map.json)、
[Provider准入与详细施工清单](llm-moe-provider-integration-ledger.md)。以下勾选只表示计划产物，
不表示产品覆盖。实施已进入M1，当前状态以registry及其绑定的有限卡为准；
[M1-G资源卡](llm-moe-m1-ggml-runtime.md)与[M1-R路由卡](llm-moe-m1-route-binding.md)
已关闭各自有限资源/冷条件范围；[M1-C CUDA接入](llm-moe-m1-cuda-admission.md)及其GQA语义卡也已
CLOSED，Windows/WSL普通入口/三交付/适用插桩与Mac适用门通过。
[M1-A Apple接入](llm-moe-m1-apple-admission.md)也已CLOSED：原版Metal与MLX组合、普通入口、
两台Mac三交付、host插桩及最终selected回归通过。[M1-V](llm-moe-m1-vulkan-admission.md)也已CLOSED：原版recurrent/MXFP4/IREE Scan/FP8普通入口、资源/失败、两机六格三交付和最终120个selected门完成。新增256格覆盖仍为0；进入M2批量成熟路径接入。
尚未审计的包仍须逐owner
固定有限FIX，不继承旧卡的产品授权。

- [x] **M0-01 模型计算映射**：对 Kimi K3、DeepSeek V4 Pro/Flash、GLM 5.3/Flash、
  Qwen3.8 2.4T/27B、Qwen3.5-2B，按已固定配置/源码把64需求域回映到实际图节点、融合片段、
  状态与量化。每个节点记录源文件/符号、pin、输入输出及必需 shape；不从 HYLE 旧 API 推导需求。
- [x] **M0-02 四后端承载表**：以 llama.cpp 使用的 ggml 图与 CPU/CUDA/Metal/Vulkan 分派
  为第一候选，区分源码存在、后端接受、HYLE 已绑定、正确性和三交付通过。已有研究直接复用。
  一次遍历模型图，按 Provider 的操作族/region 批量映射，不逐原子重新调查。
- [x] **M0-03 有限 Provider 准入表**：逐个将目录候选裁决为必需接入、已有可复用、后备或
  不准入；每项有需求 ID、具体理由、pin/license/C++ 入口及接入方式。ggml 四后端为首选广覆盖
  基线，oneDNN/MLX/已接 CUDA 库保留各自成熟能力；新增专用库必须有具名价值。
  不要求为数量接入所有重叠库，也不能把“尚未评估”写成不需要。未裁决必需域禁止进入 M4。
- [x] **M0-04 接入与冻结边界**：每条路线固定一个资源/执行 seam 和批量 lowering 方案，
  给 M1/M2 分配 owner、有限 FIX、适用验证与预算；归档自实现冻结清单。零产品算法改动。

M0 的产物必须能回答每个需求“由哪个库、哪个入口、以何种方式进入 HYLE”。现有来源支持
“主要是接入工作”的判断；四后端的完整数值域仍需实际核对，不从 llama.cpp 可加载某模型
外推每个后端、每种量化合同均通过。GLM-5.3-Flash 的已登记支持分支单独绑定，不默默换 pin。

### 模型与软硬件条件路由（M0/M1 必须固定）

规则可按模型家族、模型版本/配置身份、prefill/decode、片段语义、shape/数值/量化档位，结合
GPU 型号/SM/架构、driver/toolkit/OS/Provider 版本，选择专用计算方法、融合片段或完整 region。
consumer 显式提供模型/工作负载事实，HYLE 不做模型加载；未提供模型身份时仍有自然通用入口。
模型名称不能跳过语义检查，未知配置走合格通用基线或明确拒绝；模型事实、规则和软件身份纳入
受影响计划/编译缓存资格，变化时重新准备。准备只匹配绑定，热执行不查询、搜索或测速。
这是可交付的架构要求，M1 验收必须包含专用匹配、未知模型、版本不匹配和缓存失效，不能只写愿景。

### M1：先打通能承载一批算子的共同接入边界

- [x] **M1-01 ggml CPU 运行时**：优先解决已登记的 context/threadpool 可恢复失败与持久 pool
  阻塞。按原恢复条件核对精确版本或独立登记窄 source-patch factor；维护/升级、OOM、并发、
  三交付审计齐备后才恢复 owner。不能绕过 abort，也不能因这项接口工作改写 CPU 算子。
- [x] **M1-02 每路线原生桥**：固定 tensor/storage、device/context、执行与完成、workspace、
  constants、状态输入输出和多输出的所有权。CPU persistent executor、CUDA stream/capture、
  MLX array/compiled region、ggml Metal/Vulkan backend 分别承载；不造万能 Queue/vtable。
- [x] **M1-03 批量映射结构**：按 provider operation schema/region lowerings 共用 dtype/layout/
  参数与能力查询；同类算子的枚举和测试由表驱动。按真实消费保留融合，不能为逐节点排程拆坏库。
  Direct 可复用单节点/片段程序，Graph 可接完整 region；用户只选路线，无需写 provider glue。
- [x] **M1-04 CMake 接入**：以现有 route/feature、lock、vendor targets、私有补丁、资源、许可证
  和 package 合同集中描述依赖，source/static/shared 共用同一声明。关闭路线不下载或链接其库。
  缓存原件只读，补丁在 build-local 源码；不另造正常用户下载/编译脚本或重写公共治理。

可独立的路线 seam 顺序推进；重型构建/验证串行。每个 seam 用一组有真实计算、复用状态和失败
边界的片段证明，不以空 target、设备枚举或 copy 签收。seam 验收一次，后续批量映射复用证据。

M1勾选表示共同资源/批量映射结构和各路线有限真实纵切验收完成，不表示所有第三方需求域已经接通。
CPU ggml的普通Provider/lowering仍归M2：M1-G已完成其独立资源patch恢复前提，原owner须依法另建
SCOPED卡恢复，不能直接清除BLOCKED；oneDNN原性能阻塞也不由M1解除。各域的四路线覆盖继续由M2/M3逐项回填。

### M2：四路线接入成熟库的实际计算覆盖

B2--B6 的需求、四路线成熟承载、有限 FIX、批量验收和停止条件统一见
[M2 B2--B6 批次合同](llm-moe-m2-batch-contracts.md)。总账中的短表只用于显示顺序，不代替
启动批次时必须绑定的当前 owner 卡。

当前执行顺序：2026-09-21用户已接受oneDNN原因调查的机制证据与历史限制，批准恢复M2，
并授权按工程证据翻案。[当前裁决](llm-moe-onednn-aa-cause-followup.md#当前工程裁决旧样本无效恢复-m2)
将旧Q2视为环境干扰最可能的无效测量，撤销产品FAIL含义与接入阻塞；原始日志保留，
不改成性能PASS，不重跑Q2。继续成熟库新能力资格与批量接入，M5前停止点保持。

#### 2026-09-23 批量执行补充合同

用户要求提高接入速度，后续M2以“**路线 × Provider原生region/能力族**”作为最小施工单元，
不再以单个算子作为重复施工单元。本补充适用于现有B1卡和后续B2–B6：

1. 一个能力族共用同一份schema/lowering、资源程序、Direct/Graph入口和表驱动资格；优先接
   Provider完整region。完整region已经真实执行模型片段且逐需求语义通过时，不再为内部每个
   原子另写adapter；覆盖矩阵仍逐行记录实际承载、dtype/layout/shape和验证结论。
2. 同一能力族只做一次AUDITED有限表和一次稳定候选。不得按枚举项新开子卡、重置源码预算，
   或重复调查同一Provider源码。原版不接受的域记录精确反例和已准入后备责任。
3. 开发期只运行受影响编译和定向正确性；路线批次稳定后，集中执行一次跨环境、host插桩、
   source/static/shared组合消费、最终selected门和必要P1。不变的资源、失败、package接缝按
   tree/contract/toolchain/machine身份复用，不逐算子重跑。
4. 每条路线维护一张待接/已绑定/已验证/明确不相容的能力族账本。只有产品路径或共享接缝发生
   变化才增加施工；文档行、历史探针和重复通过不计进度。一次构建尽量覆盖同路线全部待验族，
   重型验证仍串行、并行度2。
5. 当前先把已稳定的Apple B1作为第一个集中收口批次，再用同一方法完成CPU、CUDA、Vulkan
   的B1剩余族；随后B2–B6按Provider region批量推进。完成整个M2后暂停并向用户报告，不进入M3。
6. 仪器只保留能裁决本批实际风险的最小集合。路线validator核对已准入Provider、补丁、资源和
   合同的精确身份，不要求合法patched tree等同unpatched upstream，也不把其它路线的控制密度
   机械复制到本批。已有共同仪器只有在可复现缺陷妨碍当前裁决时才定向修正，不扩建公共治理系统。
7. 路线/Provider批次的退出门是：所有已准入成熟能力均已接通并完成适用验证，未接通项为0；
   原版不具备或与公共语义不相容的域必须留下精确源码/调用反例和后续残差owner。批次关闭不把
   这些残差写成PASS，也不要求在M2偷渡自研来凑满21行；M3按256格汇总真实PASS与残差，M4才在
   “所有成熟接入完成”之后裁决最小补缺。整个M2仍须四路线全部成熟批次和B2–B6完成才算结束。
8. 每条路线从B1到B6复用一个Windows活动验证树和一个环境profile；SDK、Provider pin、DLL邻置
   与低并行参数由驱动一次解析，不能每个能力重新拼参数或新建树。开发期一次增量构建覆盖本批
   全部改动，只跑表驱动正确性；卡片只在能力族边界和批次收口更新。首次依赖编译、环境配置失败
   和重复文档门不计能力进度，也不得成为逐算子施工步骤。

2026-09-23，[M2-B1 成熟 Provider 基础批次](llm-moe-m2-b1-terminal.md)已按上述第7项退出门
结束：四路线已准入的原版能力均已接普通 Direct/Graph，适用环境与 source/static/shared 聚合门
通过；具名上游/语义残差保持 UNPROVEN 并交 M3/M4。只有 Apple 的21个路线格为PASS，跨四路线
完整行仍为0/64。B2可以启动；共享M2 owner保持AUDITED，不伪造组件CLOSED后立即重开。

Vulkan B1的具体实现、失败和源码边界保存在[路线卡](llm-moe-m2-b1-vulkan.md)；最终聚合验证与
未准入残差以[B1终局](llm-moe-m2-b1-terminal.md)为准。

Apple B1已经按同一集中合同CLOSED：MLX成熟路径的F01–F04/F19共21格完成普通Direct/Graph、
资源/失败、两台Mac适用门、三交付和完整region P1；矩阵为21/256路线格，跨四路线完整行仍0/64。
产品候选为`fd03ba10`，未启动冻结自研比较；旧不可达自定义Metal链继续留到M5裁决。

2026-09-24，[M2 B1--B6 已汇总关闭](llm-moe-m2-terminal.md)：所有已准入成熟入口完成批量接线与
适用验收，未接通的已准入入口为0；具名上游、设备、精确数值和完整模型链残差保持
`UNPROVEN`，由 M3 按真实消费逐格裁决。M2 没有新增自研数学。当前按用户停止点暂停，不自动进入
M3。

| 批次 | Provider 与接法 | 一批要接的需求 | owner 与停止点 |
|---|---|---|---|
| M2-CPU | ggml CPU backend/完整 region；oneDNN primitive/partition 按准入表组合 | F01–F19 的 CPU 必需域，优先现成模型图中的 tensor、线性、attention、递归与 MoE 路径 | 现有 ggml/oneDNN/CPU route owner，新增 factor 先 QUEUED；普通 Direct/Graph 与持久资源可用 |
| M2-CUDA | ggml CUDA backend/region；cuBLASLt、Marlin、SGLang 复用；cuDNN/FlashInfer 等仅按 M0 准入 | 同一模型需求全集的 CUDA 域；保留成熟融合与量化/attention/专家程序 | CUDA provider/route/backend 的具名 owner；同 GPU 多 Provider 所有权/同步清楚 |
| M2-Apple | MLX typed/compiled region；ggml Metal 按已冻结需求接入，避免重复数学实现 | 同一需求全集的 MLX/Metal 承载与必要互操作 | Apple Compute/Graph route，ggml Metal 如需新 factor 先登记；原生状态/资源复用 |
| M2-Vulkan | 扩展已接 ggml Vulkan managed region 的批量映射；其他库仅填具名成熟覆盖 | 同一需求全集的 Vulkan 承载，保留上游 shader/融合/调度 | ggml Vulkan 与两个 Vulkan route owner；包括 Air/MoltenVK 实际适用域 |

每批的有限清单由 M0 的需求映射产生，不预写算子算法。原 pin 缺能力先查模型实际使用的版本、
后端与完整组合；升级或源码摘取必须保留 provenance，不把“参考实现重写”标为 BORROW。
四批均落实普通路由、正确性、资源/失败、适用三交付及 P0。API/算法稳定后对成熟默认路径按卡
作必要 P1，**不插入旧自实现的竞速/优化回合**。已完成的受影响门不因下一枚举项重复全库构建；
同一稳定路线批次只执行一次集中交付/跨环境/P1闭包。

Scatter浮点add按[M2-S公共数值裁决](llm-moe-m2-scatter-numerical-semantics.md)执行：所有更新参与，
允许重结合与存储精度或更宽的中间值；固定模型的F32端口/误差门保持。旧逐次舍入资格失败
保留历史结论，成熟接法在新合同下仍需独立资格；不把数值档修正当成覆盖或自研授权。

### M3：按模型计算链验收，核对真实剩余覆盖

当前施工边界见 [M3 四路线真实模型计算链验收卡](llm-moe-m3-model-chain-acceptance.md)。该卡先审计
六组共享测试消费者与有限接入缺陷，转为 `AUDITED` 后才允许修改产品或回填覆盖结论。

下列是消费分组，可随 M2 对应能力接入顺序执行；不是新的自实现队列：

| ID | 需求域 | 真实验收与产物 |
|---|---|---|
| M3-01（原 C1） | F01–F04/F19 | embedding→norm/gate→residual/index/state；两个 invocation、旧结果、alias/错误；四路线 Direct/Graph 和三交付 |
| M3-02（原 C2） | F05–F08 | dense/quantized FFN、GQA prefill→decode、RoPE/KV；相同 Q4/Q8 codes/scales，压缩驻留和转换成本 |
| M3-03（原 C3） | F09–F11 | 卷积/DeltaNet/KDA chunk/recurrent；本地2B B16/Q4/Q8 四路线24层和8步 decode；非零初态及 chunk 边界 |
| M3-04（原 C4） | F12–F14 | 四个 MoE 家族的 dispatch→expert→combine、shared/latent、SiTU、mHC/AttnRes/MTP；空专家、偏斜和 ties |
| M3-05（原 C5） | F15–F17 | MLA/DSA/CSA/HCA、full/shared/pooled index 与 tail、压缩状态；真实宽度片段和缩小容量结构图 |
| M3-06（原 C6） | F18 | 四路线 FP8 本地2B、MXFP4/FP8 专家与索引合同，独立解码 oracle、量化质量/实现误差分别判定 |

- [x] **M3-07 缺口审计**：64行×4路线逐域回填原矩阵，分开“上游缺失 / 未接通 / 后端不接受 /
  数值或状态不相容 / 适用验证未运行”。前四类有精确调用或源码证据，最后一类绝不能写 PASS。
- [x] **M3-08 Provider 接入完成门**：M0 所有必需 Provider 已接通其准入范围并完成适用验证，
  所有候选处置有理由，未完成接入数量为0。此门未过，M4 新自实现数量必须保持0。

M3 于 2026-09-24 完成[中间审计](llm-moe-m3-terminal.md)：256 格已有初次分类，33 PASS、21
`BACKEND_REJECTED`、64 `NUMERIC_INCOMPATIBLE`、4 `UPSTREAM_MISSING`、134
`VALIDATION_NOT_RUN`。随后审查确认 134 个未运行格不能作为 M3 终局残差；M3 恢复为活动验收，按
M3-01/02/04/05/06 的五个共享批次执行并回填。已有明确准备期拒绝的 89 格不做无效重测，本地 2B
完整 HYLE 旅程仍为 0/16；M3 完成前不进入 M4。

2026-09-20用户取消M3-01完成后的专门通知与暂停，验收完成后直接继续后续阶段；
正常工作包进展仍按实际实现与证据报告。不把旧C1文档、测试数量或局部交付算作该里程碑完成。

### M4：剩余缺口，默认空集

- [ ] 对 M3 仍缺的每一域，先核对其它已准入成熟库、合法算子组合、所需上游版本/窄移植。
  接口或后端桥未完成仍回 M1/M2，不能改叫算法缺失。
- [ ] 只有全部必需接入完成、存在可复现必需计算缺口，且成熟实现/组合确不能满足，才写具名
  gap record：模型/需求 ID、四后端证据、排除理由、最小补缺、owner、有限 FIX 和验证预算。
  依当前 owner SCOPED→AUDITED 后才能写代码，不自动授权一整族算子。
- [ ] 若已有冻结实现恰好覆盖该缺口，先以其固定版本验证必需域，不以“有证据优势”为由提前
  优化或扩张。纯追求 SOTA 不授权本轮新增算法；SOTA 主要通过成熟实现和条件规则达成。
- [ ] 若没有真正缺口，M4 以 **0 个新增自研算子**结束；不为了填阶段而制造代码。

### M5：最后一次存量速裁与删除

**用户停止点：本阶段开始前必须暂停并汇报，收到用户明确继续后才可启动以下工作。**
此停止点取代自动从M4进入M5；不恢复已取消的M3-01专门通知/暂停。

这是受预算约束的收尾，**不是优化项目**。详见[冻结与退场合同](llm-moe-self-operator-freeze.md)。

- [ ] 成熟替代已满足同域数值、检查、状态、资源和交付，存量只按真实片段成组比较。
  复用身份一致的合格证据；没有被消费的域直接进入删除，不为每个旧算子新造 benchmark。
- [ ] 每路线最多一个固定候选 bundle，沿现有合格仪器、同 workload A/A、双顺序及有限样本。
  比较前一次性核对责任与成本，避免 apples-to-oranges；失格结果不能裁决输赢。
- [ ] **HYLE 有效 LOSS：立即迁移并删除对应已覆盖域，无算法优化、补样本或换姿势重测。**
  有可信条件 WIN 只保留该域；持平或到预算仍无优势时采用合格成熟实现，删除重复路径，
  并如实区分“未证明优势”与 LOSS。替代未覆盖域回 M4，不静默删除必要能力。
- [ ] 删除实现、注册/默认规则、专属构建分支和过时说明；保留必要独立 oracle/回归覆盖。
  SIMD 基础库不删除；它的 Compute/Graph 算子 Provider 按相同规则淘汰。
  上个 goal 专为算子新增到 SIMD 内、且 Highway/xsimd 均无对应实现的部分，也只在本阶段按
  来源/用途证据裁决；不能连带删通用 SIMD。当前变更来源核对未发现上个 goal 修改 SIMD 产品，
  见[精确边界与记录](llm-moe-self-operator-freeze.md#simd-的基础能力与算子专用增量分开)。

### M6：终审与停止

- [x] 完成并报告用户2026-09-21要求的[oneDNN同实现A/A原因调查](llm-moe-onednn-aa-cause-followup.md)；
  用户接受机制与历史限制，授权旧Q2工程改判为无效测量并恢复M2；原始FAIL日志保留，
  当前不再视为产品失败/接入阻塞，也不虚构性能PASS。
- [ ] 四路线64需求域、普通 Direct/Graph、状态/失败/资源、三交付及适用性能证据闭合。
- [ ] 本地2B B16/Q4/Q8/FP8 四路线完整语言主干通过；七个大模型的必需机制映射无遗漏。
- [ ] 条件路由、能力查询、当前用户文档、默认支持表与实际实现一致，旧自实现无未裁决尾项。
- [ ] 按统一脚本报告全 Compute+Graph 实现 LOC，清理本 goal 可丢弃验证产物，保留必要证据。
- [ ] 通知“本轮 LLM/MoE Compute + Graph 能力补齐完成”，列出资格与限制，停止等用户讨论。

## 5. 验证、预算与授权规则

### 验证要证明实际消费，避免重复劳动

- 正确性：同一输入与量化 codes/scales、独立 oracle、模型真实宽度/头比例/专家规则；状态比较
  覆盖 chunk/逐 token、非零初态/reset、请求交错和保留旧结果。量化损失与实现误差分开判。
- 接入资格：准备检查静态 shape/dtype/容量/alias，运行保留动态索引/错误/完成责任；上游缺少
  的必要检查由接入边界处理。不能拿省检查或同步的裸 kernel 与完整 Graph 比输赢。
- 平台：CPU Windows/WSL/Apple CPU；CUDA Windows/WSL RTX4070；Apple Air/Mini；Vulkan
  RTX4070 与 Air/MoltenVK。Mini 只签正确性/可移植性，不替 Air 签 Formal 性能。
- 三交付：一个真实组合消费覆盖一批需求，source、installed static/shared 都实际执行。
  共享 Provider seam、交付依赖和未变输入按身份复用；新接入域做相应增量，不逐算子重复全套。
- 性能：成熟默认路径在 M2/M3 自己完成适用门；M5 只裁决旧自研是否值得保留。P0 先看转换、
  同步、融合、launch、workspace/驻留、资源回收和常态复杂度。P1 仅不可变候选且 API/算法固定。
  规则按 GPU/SM/架构/driver/toolkit/vendor 版本和 dtype/shape 绑定，准备只匹配，执行不搜索。
- 先运行 change-impact selector，仅 docs/state 改动不构建、不 P1。适用未运行写 UNPROVEN/
  BLOCKED，聚合不可裁决写 INCONCLUSIVE；不能因计划改写把旧测量失败改成通过。

### 有限预算与停止

- M0 是一次表格化源码核对：复用已有八模型/64域证据，全局新增调查最多两个工作小时；每个
  未决 seam 最多45分钟。到点提交具名结论或具体阻塞，不转向自实现，也不无限追加来源。
- M1/M2 每 Provider 先审一张有限卡，再成批绑定；每个具名产品缺陷最多两次源码驱动纠正。
  批次完成后只做选中且适用验证，不另开“再全面看看”任务。
- M5 准备/公平性核对最多每路线30分钟；一个原仪器预声明 bundle，最多原预算且不超过31个
  有效配对样本/比较，不新增专用性能工具。每路线最多一次有具名仪器缺陷的修正，纠正调查
  最多30分钟；产品算法纠正预算 **0**。旧 CPU Q2 等 blocker 不因本预算重新获得采样机会。
  全阶段新增分析合计最多2小时、性能采样 wall-clock 合计最多2小时；额度耗尽时直接按
  已有证据裁决，未证明优势且可被合格成熟实现完整覆盖的路径退场，不逐核追加研究预算。
  删除后的必要构建/正确性检查由 selector 决定，不借其名义重启性能调查。
- 测量失格不判 HYLE 输；成熟替代已合格且完整覆盖时默认采用它，不为保存旧代码无限追测。
  成熟路径自身的必要资格仍未过则如实阻塞该域，不能假报 goal 完成。

### owner 与工作区

- 当前授权只来自新计划、新活动卡和 registry。旧 C1 的 AUDITED 子集撤销；经记录的合法
  状态迁移回到新 scope。所有 CLOSED/BLOCKED 的原恢复条件保留，不借改名绕过。
- 通用 Compute/Graph 只因必要图/资源 seam 定向改动；Mem/Sched/Build 不全库重审。
  新独立 factor 先注册 QUEUED，再 SCOPED/AUDITED；不把注册空 target 算接入。
- Windows 是产品真相，WSL/macOS 是验证副本。构建 parallel2，重型任务跨机器串行；
  Windows DLL 邻置 EXE、PATH 隔离及 SetErrorMode，禁止 UAC/修改电源策略/防火墙弹窗。
- WSL/macOS 的源码任务目录均为可丢弃副本。每次验证先核对并删除该批具名任务目录，再从
  Windows 不可变提交生成完整归档、校验 SHA-256 后解包；禁止在旧副本上增量覆盖，也禁止
  从验证机向 Windows 反向同步源码、文档或状态。
- 模型沿用户本地清单同步，不重复下载；依赖与构建只保留活动缓存和必要可复现证据，阶段末
  删除本 goal 的失败/过期副本前验证绝对路径。不触碰用户模型、源码或无关临时文件。
- 提交推送使用 `[skip ci]`。每轮维护检查、registry 前序 revision 和 skipped reasons 都保存。

## 6. 进度与本次重启交付

进度按实际产物报告，不继承旧“60%”：需求映射完成数/64、必需 Provider 接入数/冻结总数、
四路线已验收格/256、共同能力行/64、本地2B完整旅程/16、存量待裁决/删除域分别列出。
Provider 总数在 M0 冻结，后续变化说明具体需求；不能用库名数、文档或 LOC 代替完成度。

重启时矩阵仍为0/64行、0/256格完整验收；这是严格全门口径，已有实现与局部有效证据继续复用。
本轮重写不新增产品通过。统一 `loc-report.py --self-check` 在产品基线得到 Compute+Graph
**49,270 实现行 / 200 文件**，不含测试及第三方源码；重写未改产品，计数身份不变。
旧 goal 停止，不宣称完成。新 goal 以本页的 M0 起步，前轮四路线基础重构成果不重做。
