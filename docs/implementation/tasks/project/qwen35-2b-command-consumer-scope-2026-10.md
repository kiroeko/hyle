# Qwen3.5 2B 命令行推理范围与工作量

这个本地模型适合成为 HYLE 的首个完整文本消费者。已有真实权重片段覆盖可以复用，主要
剩余工作是加载/tokenizer、24 层模型接线、驻留状态回送、生成循环及整链数值检查。
首个单请求 BF16 命令不需要先实现所有量化/多请求/多机能力；但它也不能替代已有十项
基础库缺口的逐项纠正与验收。

2026-10-13 用户提出读取 `C:\Users\kiro\Downloads\Qwen3.5-2B` 做命令行推理，并要求前次
发现的缺口全部处理。当前输入 baseline 为 `1a15794c0`；本页为 **RESEARCH_COMPLETE 与
实施范围提案**，不是产品 owner closure card，不改变 registry，也不宣称 HYLE 已整模型运行。
承接[一期收口与引擎启动方案](inference-consumer-adaptation-plan-2026-10.md)及
[单机基础能力审计](inference-single-machine-foundation-audit-2026-10.md)。

**2026-10-16 当前交付决定**：用户选择先完成全部G01–G10，再开发本页的独立命令行引擎；
首个性能交付限定为Qwen3.5-2B纯文本、单机单卡、full CUDA，关闭MTP/speculative与多模态，
同机与llama.cpp比较prefill/decode tokens/s。旧goal已清除，本页登记后续任务，不自动创建
或恢复goal，也不宣称命令已经实现。下面原C1–C5可复用，执行顺序以本决定和末节为准。

## 1 本地实物与已跑参考

原模型 13 个文件、4,571,274,023 bytes 本轮再次逐文件大小/SHA256 校验，匹配原
[输入清单](../compute/qwen35-2b-local-input.json)。原目录只读，未下载或替换权重。
纯文本实际消费 320 个权重，独立参数 storage 共 3,763,655,360 bytes，tied head 不重复分配。
完整 safetensors shard 为 4,548,221,488 bytes，其中还包含视觉和 MTP；文本首版不加载它们。

| 本地配置 | 实际值 | 对实施的影响 |
|---|---|---|
| 文本层 | 24：18 linear attention，6 full attention | FFN 为 dense；attention 是 hybrid，不能用纯 Transformer 模板省略 recurrent 状态 |
| hidden / FFN / vocab | 2048 / 6144 / 248320 | 完整词表 projection 与 embedding 占用明显；不能用七行 embedding fixture 当全模型 |
| full attention | Q8/KV2，D256，output gate | Q projection 实际 4096；每 head 拆 Q 与 gate，再做 gated attention 输出 |
| RoPE | partial factor 0.25，rotary64，theta10000000，MRoPE 配置 | 纯文本位置按实际公式生成；不把既有随机 frequency/RoPE fixture 当整模型位置实现 |
| linear attention | K/V heads16，D128，conv kernel4 | 保 18 份 conv 与 FP32 recurrent 状态，区别初始 prefill 与 decode |
| checkpoint dtype | 大部分 BF16，A_log/部分参数 F32 | 每个 tensor 按 header 读取；不能整模型强制同一种 dtype |
| output | tied embedding | final norm→全部 logits→选择 token；结束 ID 来自 tokenizer/config |

本机实际为 i9-13900K、64 GiB 级 RAM、RTX4070 12 GiB，驱动617.14。参考加载前查询实际
free memory，并要求文本权重加 1 GiB 余量可用；这不是任意上下文/并发的显存保证。
[官方模型说明](https://huggingface.co/Qwen/Qwen3.5-2B)也描述了 3:1 GDN/full attention 与上述
关键结构；本地 config/header 是本轮精确参数依据。

实际离线执行记录见[命令预检证据](../../../../evidence/correctness/qwen35-cli-preflight/README.md)：

- 使用已安装 Torch2.8.0+cu129、Transformers5.3.0、Safetensors0.8.0、Tokenizers0.22.2。
- 纯文本 24 层，greedy，chat template 的 non-thinking 选项，vision/MTP 不执行。
- 算术提示生成 `5` 与结束 token；另一提示实际完成 32 个生成 token。
- 使用 Torch eager attention 和已有 fallback GDN/conv 公式，不宣称优化引擎性能。

该执行只证明原输入、tokenizer、模型组合及本机资源能支持短文本生成。它是外部 reference，
不计 HYLE 运行通过，不给正式 token/s，不推导最大上下文或三个部署形态已完成。

## 2 复用到哪里

[模型片段覆盖](../../../user/model-fragment-coverage.md)中 Qwen3.5-2B official BF16 的 CPU、
CUDA、Apple、Vulkan 均有已验收片段；不能从片段资格推导整模型正确或已高性能运行。
[真实片段卡](../compute/llm-moe-d5-real-fragment-acceptance.md)保存原 oracle、数值纠正与
实际交付。既有[完整文本参考脚本](../../../../tests/compute_models/qwen35_reference.py)已生成
prefill1/17/128与8步 teacher-forced decode 的中间层、logits、conv/recurrent/KV 状态。

| 已有基础能力 | 首个消费者的剩余任务 |
|---|---|
| MatMul、RMSNorm、SiLU/gating、views/casts、RoPE、GQA/SDPA | 按模型逐层组织物理描述、常量、残差与正确 dtype 边界 |
| CausalConv1d、GatedDeltaNet 与 raw-gate recurrent 片段 | 反馈 conv history 和 recurrent state，补 Q/K L2、gated norm、完整 projection 组合 |
| 完整词表 projection、embedding 片段和 checked IDs | 读全 248320 行，重用 tied backing，greedy/EOS、tokenizer 解码 |
| F1/F2/F3 的租约、范围搬运与文件 stage | 将全部实际权重映射/放置，保资源 through actual retirement，分开冷加载与 warm 执行 |
| Graph 与 finite replay，F4 版本 | 准备阶段绑定稳定数学；明确状态更新与输出失效；checkpoint 不必每 token 复制大目录 |

初版可以用 Direct/有限 region 组合，随后按实际热点收敛 Graph；不能每个 token 重建权重/
所有 prepared objects，也不能未经裁决直接调第三方模型 forward 来称为 HYLE 实现。
模型数学由公开 HYLE Compute/Graph 消费，成熟 tokenizer/格式库可由引擎直接借用。
如首个开发命令用 Python 做 tokenizer 前后处理，记录其边界与费用，不能偷偷引入 Native FFI。

## 3 首个命令的有限合同

拟议完成形状如下，**二进制尚未实现，下面不是当前可运行命令**：

```powershell
.\hyle-infer.exe --model "C:\Users\kiro\Downloads\Qwen3.5-2B" `
  --route cuda --prompt "请计算 2+3，简短回答。" `
  --max-new-tokens 64 --context-limit 1024 --sampling greedy
```

首版为独立 C++ 消费工程，不放入 HYLE factor DAG，不将模型 loader/tokenizer/生成循环
下沉为库产品。首个执行档选本机 CUDA BF16 单请求、上下文有界1024；CPU 和 CPU＋CUDA
沿同数学分别加入，其后在实际 Air/Vulkan 资格下验证其他路线。先实现一个档不取消一期
三部署形态，也不把各 GPU 的未运行格填成通过。

首版不要求视觉、视频、MTP/speculative、MoE 模型、quant KV、随机 sampling、continuous
batching 或多机控制。它们在十项基础能力修正或后续引擎合同中分别验收，不消失。
长期262144上下文声明也不是本命令在12GiB上必须满足的容量。

### 五个消费者工作段

| 段 | 具体实现 | 完成证据 |
|---|---|---|
| C1 输入与 token | safetensors/config 读取、全部名字/extent/dtype 校验；chat template、tokenizer、EOS；仅加载语言权重 | 同 input source ID；token IDs 与成熟 tokenizer 对拍；missing/overflow/截断拒绝 |
| C2 冷准备与模型图 | 320 tensors 与 tied head；两类 layer 的全部 projections、norm/gate、残差、FFN、最终词表 head | 可发现 public HYLE 调用；冷/热费用分开；无需运行私有 HYLE 类型 |
| C3 状态与执行 | 18 conv/recurrent +6 K/V 的位置、写范围和生命周期；首次 prefill 与单 token decode；有界 scratch/slots | 原 producer/lease/native owner 保活，失败不发布新状态；旧结果/取消/关闭 |
| C4 生成命令 | 全链 forward→greedy→EOS/上限，连续输出，明确错误码与取消；诊断时间/资源 | 实际至少64步用例及自然 EOS；token/文本可回查，不只算算术两 token |
| C5 整链对拍 | teacher-forced prompt1/17/128与固定8步；每层 hidden、logits、全部状态，随后自由生成 | 独立 reference，不将实际 HYLE hidden 喂回 oracle；near-tie 单列，不临时放宽精度 |

首个自然短输出是诊断里程碑；C1–C5 完成才是首个 HYLE 命令资格。prefill 可先逐 token
验证状态数学，但这种路径必须标明 reference/诊断档；高性能 prefill 需要真实分块/批量数学
与非零初态 continuation 域，不能把 decode 循环冒充成熟 bulk-prefill 档位。

### 必须认真对拍的模型接缝

- Qwen 普通 RMS 的 scale 是 `1+weight` 且有明确 F32 运算/最终 dtype 边界；GDN gated RMS
  的 scale/SiLU gate 是另一个公式。简单把同名 norm 全部接一种 recipe 会错。
- linear Q/K 的 L2 使用 `sum(x*x)+eps`，不是直接替成普通 mean RMS；HYLE GDN 自身另有
  `1/sqrt(Dk)`，不能在 caller 再重复缩放。raw a/b、负 decay 与 dt_bias 按已签片段接。
- conv history 的值、顺序、dtype 与 recurrent 的 FP32 初/终态逐层反馈；新请求全部重置。
- full-attention Q/gate 是 per-head chunk，RoPE64/剩余维度/绝对位置正确；KV active 长度递增，
  capacity 无效区不得参与。旧 BF16 RoPE 失败已被原 adapter 纠正，不再当当前未修缺陷。
- prefill chunk 与 cached continuation 不能机械借 Transformers5.3.0 的多 token cached 分支；
  [原参考卡](../compute/llm-moe-c0-local-contracts.md)已有重置 initial-state 的反例。对拍应
  使用独立正确公式；首次 prefill + 单 token decode 参考仍可用。

这些是模型消费者的语义检查，不是先验地宣布五个新的 HYLE bug。实际最小 consumer 暴露
公开承载缺口后，才带其数学/资源反例进入对应 owner 的活动卡。

## 4 十项缺口与本模型的关系

用户现在要求前次缺口全部处理，不能继续仅按“本模型不需要”把它们从总交付中删去。
先跑本模型是验证与需求入口；十项总清单独立保留。候选经反证已存在完整路线时，结案要
给 public recipe、成本/域和测试依据；不能为凑十次改动强造 API，也不能仅口头称上层负责。

| 项目 | 首个 BF16 单请求能验证什么 | 仍须单独纠正/验收的范围 |
|---|---|---|
| G01 驻留读取 | 6 个 full-attention layer 的 active 长度、实际 copy/scratch、dense/view 路线 | 多请求/paged 等价消费，长 Q/reference KV 展开及 CPU/Apple/Vulkan 高效档 |
| G02 联合分页 | 首个纯 CUDA 命令不覆盖；CPU＋CUDA 真数学加入后检查 bindings/搬运 | 七路线中真实 hybrid 的准确 page frame、metadata 与原 native slot；各三交付 |
| G03 增量更新 | 每步 K/V append、conv/recurrent 回送、old readers、producer 与取消 | F5 pending→ready 组合、必要 idle input release、跨页/shared页与各 native route；out-of-place GDN 不代签全部 mutable 路径 |
| G04 quant KV | 原 BF16 cache 无压缩编码 | 冻结一个真实 cache 编码的 append/storage/scales/数学完整旅程及路线资格，不能用 roundtrip 代签 |
| G05 attention 域 | D256/GQA4、partial RoPE、output gate、真实位置/有效区 | window/sink/不同D/其他被准入域各锁有限目标；不能将所有未来特例自动并入本轮 |
| G06 metadata/decode | warm plan/metadata/copy/waits、跨 step reuse | 多请求 tail/跨页/重排、active/delta、decode 专用档与实际 capture 域 |
| G07 packed weights | 原 BF16 不覆盖低精度；实际模型校验冷/热 weight residency | 一个选定 packed 路径、selected row/expert checked IDs、原生 owner 和 warm 展开/同步成本 |
| G08 输出选择 | 全词表 greedy 与必要小结果返回 | GPU 随机数值域、seed/counter、重排、分布与 exact top-p 等受选域；不造模型 Sampler 政策 |
| G09 联合流水 | 单请求逐层链只有必要依赖；不能证明无依赖支路 overlap | 两 slots/microbatch、独立 CPU/GPU 支路、实际 trace，原 Apple/Vulkan 片段推进与关闭 |
| G10 CPU预算 | CPU 路线实际线程/Provider，prefill/decode 阶段 | private pool、并发 invocation/嵌套预算与共享进程边界，不能只读一个 threads 选项 |

当前十项的调查已完成，新增接缝的产品修正与逐项新旅程验收尚未完成。每项要结束为
有证据的 `FIX/BORROW 完成` 或 `已有完整 ROUTED 反证`；APPLICATION 政策剥离要留下
其机制覆盖依据。用户要求的通用缺口不能仅以未优化/延后宣称总任务完成。特殊数学域仍
必须先冻结有限模型/编码，全部 peer feature 求并集不是“十项全部处理”的含义。

## 5 工作量粗估

以下是基于已有片段与参考的工程估计，**尚未由完整 HYLE prototype 校准**，不是已有实现
行数，也不是本次会话的耗时承诺。开发量以熟悉代码的一位工程师、可复用现有依赖/产物为
条件；构建、数值定位和三平台验收可使日历时间更长。

| 范围 | 引擎自有产品代码粗估 | 工程工作量粗估 | 主要不确定性 |
|---|---|---|---|
| 首个 CUDA BF16、文本/greedy 单请求 CLI，C1–C5 | 2500–6000 行；另1500–3500测试/对拍/准备工具 | 1–3 人周 | 已有数学组合能否直接接全层、精度累计、resident cache/输出寿命与依赖重用 |
| 同模型 CPU、单GPU、CPU＋GPU及四计算路线的单请求/正确关闭 | 合计5000–10000行；不是在上行后再加同样数目 | 累计3–6 人周 | native layout/资源不同、联合图缺口、平台验证与 CPU private pool |
| 实用引擎：单/多请求、有限量化、流输出及三部署，高性能闭环 | 仍参考原条件14–30k产品行数；首版后实测修订 | 原12–24人周条件估计 | batch/admission/prefix、编码、整链质量与性能回流，不能由2B短运行直接减少估计 |

十项基础库纠正另计，不把其新增 LOC 当成引擎减少 LOC。狭窄组合问题和高效 native 数学
承载差异很大；沿[前次条件评估](foundation-post-goal-engine-effort-2026-10.md)，窄接缝约
500–3000产品行/2–8人周；新增两个高效 native/等价 adapter 的情形约3000–8000行/8–16人周。
这两个情形不是十项全量报价，也不能直接相加为已签总工期。G04/G05/G08 的具体域尚未冻结，
十项全量总数现在给确定值会失真；先用上表和逐项有限卡确定实际分解。

“命令能生成文本”比“全部三路线高性能与十项缺口完成”小一个范围。也不是只补50行主函数：
目前的片段 fixture 没有模型 loader、tokenizer、自回归循环或完整24层正确性证据。

## 6 实施顺序与退出条件

1. **本轮完成的准备**：固定原输入，实查结构/硬件，离线 reference 短生成，列出模型接缝与
   十项回映。源目录、product/registry 不改，reference 不冒充 HYLE。
2. **首个 HYLE 命令**：独立消费者完成 C1–C5；必要库缺口带真实反例到 owner 卡，按
   SCOPED→AUDITED 冻结有限 FIX。复用原缓存/构建树，按 selector 只选实际改动范围。
3. **两条验收支线**：一条扩同模型到一期三部署/真实单多请求；另一条按既有十项完成有限
   机制修正及留出用例。首个单请求 CLI 不作为第二条已完成证据。
4. **统一交接**：全部当期库项与实际 engine 域分别裁决；同模型/精度/硬件/请求与完成条件下
   比较 prefill/decode token/s、TTFT、ITL、资源账。诊断参考的两次短运行不能提供这个 verdict。

停止条件不写“再也发现不了问题”，而是原输入/C1–C5/有限十项及所选域全部有真实裁决。
新增真实缺陷保留最小反例与 owner，只有活动卡重新纳入才施工；未做二期多机/多GPU不计
入本轮失败，仍保[二期边界](inference-consumer-adaptation-plan-2026-10.md#7-二期多-gpu-与多机边界)。

## 7 G01–G10之后的full CUDA命令与性能交付

用户确认剩余G07 packed权重、G08输出数值选择、G09联合流水、G10 CPU预算全部保留，
完成有限范围后再启动引擎。不能因首个BF16单请求不依赖某项而跳过它，也不能把四项
扩成新的无限缺口调研。引擎单独作为HYLE的C++消费者；不把模型、tokenizer、请求政策
放进HYLE组件图，不把未来引擎性能门前移为基础库的循环前置条件。

### 选择与完成边界

G04当前选择的是库的**原生FP8驻留/F32 partial路线**；引擎第一组比较选择双方可匹配的
**普通BF16权重/KV基线**。两者是不同选择，不要求Qwen引擎强制使用FP8。若锁定的
双方CUDA实现不能支持共同BF16 KV，则在采样前选定共同支持的F16 KV并显式记录，
不允许一侧悄悄fallback或扩展为临时修库。原checkpoint中的F32 tensor仍按原dtype保留，
GGUF转换须核实数值等价和必要布局变换，不能拿Q4权重与BF16权重作实现优劣对照。

第一条命令仍沿上文`hyle-infer --model ... --route cuda`形状，须真正完成C1–C5和连续
输出；确切参数在引擎scope冻结，当前示例不冒充可执行程序。full CUDA要求24层模型
计算、权重、六层KV和十八层conv/recurrent状态实际由CUDA承载，禁止CPU数学fallback；
CPU可负责文件加载、tokenizer、host提交与文本显示。CUDA选择不等于不需要CPU线程预算。
原本地checkpoint只读；视觉、音频/视频、MTP分支不加载/执行，不使用draft模型。

初始实际HYLE资格以已有WSL CUDA路线为准，双方必须在同一OS/WSL环境、同一物理GPU
运行，不能拿HYLE WSL与llama.cpp Windows或另一张卡的数字直接排名。运行前读实际
硬件/驱动/工具链身份与空闲显存；不把既有RTX4070身份或旧测量当作新执行事实。

### 公平性能合同

| 项 | 冻结与输出要求 |
|---|---|
| 模型与输入 | 同源文本权重、同token IDs/chat template、相同thinking模式、prompt、context与生成长度；greedy作为首个可复现生成档 |
| 执行 | 单请求、所有层和状态CUDA；双方允许正常batch/chunk/capture优化，记录各自实际配置、KV dtype和中间精度，不强迫实现相同 |
| 阶段 | 分别输出prefill和decode tokens/s、实际token/forward次数与耗时；写清首token归属、最终logits/必要选择及checked完成边界 |
| 费用 | 必需KV追加、conv/recurrent更新、格式转换、同步与状态维护计入对应阶段；模型冷加载、一次性repack/JIT、warmup与文本打印另列，不漏掉反复发生的准备 |
| 资源 | 记录峰值/稳态显存、权重/KV/recurrent/workspace；同一资源预算下比较，容量收益单列，不用多付副本的warm路径冒充同约束peer |
| 测试长度 | 首个短命令可用原1024上限；性能首批预定prompt128/512/2048、生成128、context4096，资源可行性在采样前核定，失败不单边缩形状 |
| 重复 | 一批匹配warmup、独立重复和双执行顺序，串行占用GPU，保留全部样本；先查同实现控制与噪声，不边调参边挑有利重复 |
| 结论 | 各prompt/context分别给出HYLE与llama.cpp的pp/tg、比值和波动/区间；允许明确WIN/LOSS/未决，不要求用单一平均数隐藏阶段差异 |

真正CLI生成的阶段计时为消费者结果。`llama-bench`可作独立pp/tg/pg诊断，但它的synthetic
token/forward路径不等于完整greedy采样和文本输出；两组结果分列、不得混算。
本地参考当前可核身份为`23b0202a189c44a54625aadcb37a946dd1d6278d`，实际实施前再固定
可运行构建/命令及源码SHA，不无条件更新依赖或把本页reference pin当HYLE产品依赖。
[官方llama-bench合同](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/tools/llama-bench/README.md)
将pp/tg/pg分开；本地源码的实际decode/synchronize与时间边界也须在预检核实。

### 精度策略与终止

采用“任务质量合格范围内追求速度”，不要求逐位一致或所有中间结果F32。先对原模型和
匹配BF16基线完成teacher-forced层/logits/状态、自由生成和固定文本质量语料检查；误差、
near-tie及必要质量阈值在性能采样前冻结，不能运行后为快路径临时放宽。格式/shape、
越界、错误传播与生命周期仍严格验证，数值误差容忍不豁免这些语义。

实用FP8/packed等优化档可以在质量门通过后另列“质量/显存/速度”取舍；同精度基线和
各自最佳合格配置分开回答，不能用HYLE强精度与对方低精度的数字直接归因实现优劣。
llama.cpp也显式在attention使用F32累加，见
[固定源码](https://github.com/ggml-org/llama.cpp/blob/23b0202a189c44a54625aadcb37a946dd1d6278d/src/llama-graph.cpp#L2850)。
先前不计warm冷展开的BF16参考程序属于HYLE构造的FlashInfer comparator，**不是llama.cpp**；
其约4–5%warm差距既不证明本引擎更慢，也不构成相对llama.cpp的独有精度/内存优势。

C1–C5、full CUDA身份、分阶段费用和上述有限对照全部有真实结果后结束首轮交付；
结果落后则按实际trace和material消费者影响另定有限修正，不靠继续扩基础库能力清单
推迟引擎。Qwen hybrid/GDN数学和prefill状态不得删减成纯Transformer或逐token伪高效
prefill。单CPU/CPU+GPU的既定一期目标和二期多GPU/多机仍分别保留。
