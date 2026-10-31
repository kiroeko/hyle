# HYLE 推理引擎

入口在 `engine/`，当前完整文本模型为 Qwen3.5-2B，支持普通生成、MTP、多轮、多会话、共享前缀和有界缓存恢复。
项目当前主要用真实模型检验底层推理支持，基础模块仍可独立使用。

运行范围与限制：

- 单机 CPU、单 GPU 或 CPU＋一张 GPU；具体制品和平台以[验证范围](#制品与验证范围)为准。
- 使用本地权重清单和 token IDs，输出包含生成 IDs 与计时的 JSON；分词需要另行准备 Python 环境。
- 尚无通用模型目录加载、C++ tokenizer、引擎应用／HTTP 服务 API 或图片输入；其它模型家族仍是组件级实现与检查。

首次运行按“构建→准备本地输入并运行”两节完成；需要采样、MTP或多会话时再查[常用选项](#常用选项)。
开发范围与任务见[看板](../developer/inference-progress.md)。

## 构建

先完成[环境准备](../developer/setup/README.md)，在仓库根目录运行默认 CPU 构建：

```text
python scripts/init.py
python scripts/build.py --target hyle_engine --jobs 2
```

新配置默认 CPU、Release、共享库、engine，测试和 trace 关闭；已有配置保留选择。
CUDA、混合或基础库配置见[常见用法](../developer/setup/README.md#常见用法)，例如具名 CUDA 构建：

```text
python scripts/init.py --name qwen-cuda --components engine --backends cuda
python scripts/build.py --name qwen-cuda --target hyle_engine --jobs 2
```

`engine` 是顶层新配置的默认组件。后端可选 `cpu`、`cuda`、`metal`、`vulkan` 或平台支持的组合，
它们表示编译能力。运行时用 `--backend cpu|cuda|apple|vulkan` 选择实际主后端，
其中构建选项 `metal` 对应运行时名称 `apple`。只有一个 GPU 路线时默认选它；
只有 CPU 时默认选 CPU；同时编入多个 GPU 路线时必须显式选择。
选择失败会报错，不自动切换后端；编入 CPU 路线后可在同一程序中显式选 CPU。
构建脚本打印实际目录，程序在 `<build-dir>/bin/hyle`，Windows 为 `hyle.exe`；
多配置生成器在 `bin/Release/` 等配置子目录。调试配置使用 `init.py --config Debug`。
`--tests` 准备测试环境，测试仍分为[构建与运行](../developer/setup/README.md)两步。

只要基础模块时选择 `--components util`、`--components mem,container` 或 `--components compute,graph`，
不构建引擎。`util` 是具体工具模块，不代表全部基础库。制作 SDK 见[交付与集成](distribution.md)。

## 准备本地输入并运行

默认准备器读取本地模型的 `config.json`、safetensors 索引与张量头，校验所需角色、类型、形状和文件范围，
对实际使用的原权重范围计算 hash，生成权重偏移与 token IDs；不下载模型、不复制整份权重。
指定 `--role-files` 时则核对固定 repository／revision 及原角色 hash，GGUF 使用 `--gguf-role-files`。
本地原权重与 tokenizer 文件须保持可访问。Python tokenizer 需要兼容的 Transformers 环境，
初始化 C++ 构建不会自动安装它或获取模型。

```text
python engine/models/qwen35/prepare.py --model LOCAL_MODEL --output .scratch/qwen-input --prompt "Please calculate 2 + 3. Answer briefly."
<build-dir>/bin/hyle --inputs .scratch/qwen-input --mtp off --steps 64 --capacity 1024
```

输出目录必须是新目录，只保存偏移、token IDs 与输入记录，共用原权重文件。
需要 MTP 时，在准备命令加 `--mtp`，运行时用 `--mtp on`。准备器会加入额外 15 份原 MTP 参数，
仅准备不执行模型；同一份完整清单可切换普通／MTP。
只需普通模式时可省略准备器的`--mtp`；物理角色计数与输入身份见[模型输入说明](../../engine/models/qwen35/README.md#输入与数值政策)。

## 常用选项

`--steps` 是最多输出数，包含 prefill 首 token，遇 EOS 可提前结束。
命令行使用默认模型EOS；逐请求停止政策等C++扩展见[模型请求说明](../../engine/models/qwen35/README.md#自然生成批次与状态)。
Windows PowerShell 将运行路径改为实际 `hyle.exe`；上面的 `<build-dir>` 是脚本报告的目录。
输入不足、模型身份不符、上下文容量不足或不支持的参数组合会明确失败。

### 输入分块与冷准备

`--prefill-chunk-tokens 256`限制一次prefill实际消费的token行数，让调度可在块间服务其它请求，
默认不限制块长。中间块只更新状态，最后一块才生成首token；末尾pending只消费一次。
`--capacity`仍限制完整上下文，分块不扩大它，也不支持固定`prompt/decode`重放。
所选路线可用容量载体合批，但只提交各成员真实输入行。形状与舍入变化可能改变生成结果，分块不保证更快。
实际行数、载体容量及状态发布语义查[模型运行说明](../../engine/models/qwen35/README.md#自然生成批次与状态)。

`--warmup-shapes planned`在ready前准备已有计划的执行载体；默认`lazy`按需准备。
它保留真实阶段、形状与采样配置，不枚举全部长度、不接纳会话、不推进RNG；未计划形状仍需有界冷接纳。
它可能增加启动时间与持久内存，`--cache-bytes`不是总显存上限。
这也不是设备执行或CUDA capture预热，首次真实绑定仍可能准备replay。

### 连续接力

普通 CUDA greedy 自然生成自动使用连续接力，无需额外参数；已有限检查单活动请求及双活动请求。
CUDA greedy 的K1 MTP也已接入单活动请求的自然接力；其它MTP宽度和采样配置仍走原单步执行。
当前仅为最多输出至少 10 个 token 的自然生成准备两层候选；prefill 保留原路径，后继只使用已有实际捕获。
完整同形、同序的活动请求组可持续接力；异形分组保留原同步执行。输出额度不足时先完成已接纳工作，再暂停等待额度。
其它后端、上述范围外的MTP、混合执行、随机采样、scheduler 或显式 commands 自动使用原单步行为；显式缓存预算容不下额外候选时也自动使用单步。
遇到当前物理绑定缺少捕获时，先完成已提交工作，再沿原路径准备捕获并恢复接力；设备及清理错误继续报告。
接力会增加载体和状态内存；不满足执行条件或额外资源不足时自动单步推进。具体速度与内存边界见[最新性能对照](../developer/performance/decode.md#目标与当前基线)。


### 采样与 MTP

默认 greedy；也可用 `--sampling random --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42`。
普通与 MTP 共用温度→top-k→top-p 分布，`--top-k 0` 关闭该过滤。
每个会话有独立随机流；失败或撤回的成员不推进已发布随机位置。相同seed不保证不同后端或普通／MTP逐token相同。
逐请求bias／mask、历史罚分、动态条件及停止政策属于`engine/`内的C++请求扩展，
没有对应CLI参数或已安装的引擎SDK／server入口；其合同查[输入与请求政策](../../engine/models/qwen35/README.md#输入与数值政策)。

`--mtp-max-drafts` 控制每轮最多提出的草稿数，默认 1。`--mtp-min-probability` 设置
草稿分布最大概率的最低门槛，范围为 0 到 1，默认 0；达到门槛才继续草稿，实际长度可以更短。
这不是额外置信度模型。制品、后端与草稿数的实际运行范围见[模型覆盖](../implementation/model-operators.md)。

### 可选草稿词表

可用 `--mtp-draft-vocabulary FILE` 显式缩小草稿head，完整target head与embedding继续保留。
当前支持原BF16／F16／F32稠密head、无logits modifier的greedy及零概率阈值；可用于K1与递归多草稿。
显式选项遇到随机、modifier、非零阈值或不支持的权重会报错；未指定该选项仍使用原完整词表。
文件绑定准备清单的原source identity及实际词表大小，格式为：

```text
hyle-draft-vocabulary-v1
<weights.tsv第二行的64位source identity>
<模型实际词表大小>
<以空白分隔的原token IDs>
```

IDs必须非空、互异且在原词表范围内；冷期按原ID排序，保持greedy平局顺序，图内输出仍还原为原token ID。
选中行只导入一次，额外保留在原资源作用域；例如2B BF16模型的32K行需要128 MiB权重及128 KiB映射。
JSON的 `original_*_weight_bytes` 仍只统计原角色，额外payload见 `draft_vocabulary_weight_bytes`／`draft_vocabulary_map_bytes`；
这些权重都不属于 `--cache-bytes`，payload字节数也不是物理显存峰。
子集会改变提案和接受率，需按实际输入检查速度；代码语料先验不代表通用语言覆盖或质量保证。

### CPU 与 GPU 分置

需要明确的 CPU＋GPU 分置时，用 `scripts/init.py` 的 `--backends cpu,cuda`、`cpu,metal` 或
`cpu,vulkan` 分别准备对应构建，运行时选择 GPU 主后端并添加 `--cpu-ffn-layers 2`，将前两层 target FFN 及其
原始权重放到 CPU；注意力、缓存和 MTP 留在所选 GPU。默认值 0 保持 GPU 放置。
混合执行的适用制品和后端见[模型覆盖](../implementation/model-operators.md)；功能可用不代表比纯GPU更快。

命令行保持原 CPU 线程默认值，不增加线程参数。程序集成可复用公开的
`compute::cpu::DefaultContextOptions`；allocator／executor 支撑对象须活到全部派生制品和原生工作退休，
具体组合见 [CPU executor 示例](../../samples/package/graph/cpu_executor/README.md)和[架构说明](../architecture/inference-design.md)。

### 多轮与多会话

多轮 schedule、会话数量、同时活动数、缓存预算、trace 与数值限制见[模型运行说明](../../engine/models/qwen35/README.md)。
当前多会话以共享模型、计划和有界执行资源组织，不按用户复制模型；缓存数量和实际批次宽度是独立配置。
普通与 MTP 共用 `prompt/generate` 自然请求文件和多轮／多会话生命周期，用 `--mtp off|on` 切换。
每轮 `prompt` 文件只提供新增用户 token；后续轮可使用空文件继续生成，消费者会恰好消费一次上一轮的末尾 pending。首轮 prompt 必须非空。
普通模式也保留显式 `prompt/decode` 固定输入重放；同一会话清单不能混用两种语法。
自然生成统一输出 `sessions[].result`，单请求为 `sessions[0].result`；其中 `generated_ids` 是实际生成 IDs。
自然续文可能随合法计算形状发生差异，不能把 ID 一致当完整任务质量保证。

## 制品与验证范围

| 范围 | 当前边界 |
|---|---|
| Qwen3.5-2B完整入口 | 官方BF16及固定GPTQ、NVFP4、GGUF Q4等制品有列明范围的完整短旅程；不代表任意量化目录可直接加载。 |
| 较大Qwen制品 | Qwen3.8-27B有特定CPU／混合配置的完整短旅程；不能推广到全部制品或纯GPU常驻。 |
| NF4、CT asymmetric INT4、MLX affine4／8 | 有压缩权重及有限状态链／原组件接入；完整对应大模型、质量和吞吐未交付。 |
| GLM、DeepSeek、Kimi | 组件级实现与检查，没有完整引擎入口。 |

模型、格式、后端与原始检查统一查[模型与算子覆盖](../implementation/model-operators.md)。
加载器不会预先展开整库，但实际计划可能分解计算并使用workspace；4bit文件大小不是运行内存上限。

原 CT channel/token FP8 与 local NVFP4 混合清单使用
`--fp8-reader compressed-tensors-0.17.2a20260716` 固定参考算法；存在原 KV 校准参数时，
还须显式选择缓存政策：`--kv-cache-dtype bf16` 保留原 BF16 缓存并省略校准角色加载；
也可用 `--kv-cache-dtype e4m3fn`，读取每层原 BF16 K/V scale，采用固定校准的 target E4 缓存。
它可与 `--mtp` 合用，draft KV 保留原 BF16／plain 存储，不推造 draft 校准参数；
缺失、非正或非有限 scale 会在冷准备时拒绝，不合成默认 scale。
两者都将校准参数的名字、形状及 hash 保留在输入身份和记录中，不把它们装成模型权重张量。
E4 的缓存行载荷为 BF16 的一半，实际总 backing 仍包括页、元数据和其它状态。
缓存政策不改变 activation、recurrent 或投影计算精度。channel/token 保留 BF16 QDQ 与逐权重舍入，
local NVFP4 保留原独立全局乘数和 F32 输入预舍入；它们不套用 block128 或 ModelOpt 的算法。
这类清单的头部／投影与小规模加载不表示对应27B制品已完成整模型生成。
借入分离target／MTP文件时，使用完整的`--role-files`原角色记录。

## 查看性能

普通与 MTP 分开对照，报告 prefill、decode、内存／显存以及真实多轮综合速度，加载时间另列。
最新正式窗口、计数边界和已知热点统一保留在[decode 性能资料](../developer/performance/decode.md)，
不以模型改名、组件测试通过或减少 kernel 数量推定端到端提速。
