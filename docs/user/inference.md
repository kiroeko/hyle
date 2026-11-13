# HYLE 推理引擎

入口在 `engine/`，当前完整文本模型为 Qwen3.5-2B，支持普通生成、MTP、多轮、多会话、共享前缀和有界缓存恢复。
项目当前主要用真实模型检验底层推理支持，基础模块仍可独立使用。

运行范围与限制：

- 单机 CPU、单 GPU 或 CPU＋一张 GPU；具体制品和平台以[验证范围](#制品与验证范围)为准。
- token CLI 与内部工作进程通过同一模型提供者装载。CLI 从文件或标准输入读取 token IDs，标准输出为生成 ID；诊断信息写入标准错误。文本 CLI 和 Responses 文本服务使用本地 tokenizer 与模板，详见[文本接口](text-api.md)。Qwen 的 JSON 报告与固定重放使用下方独立诊断目标。
- 当前没有通用模型目录加载、C++ tokenizer 或图片输入；其它模型家族仍是组件级实现与检查。文本服务的实际平台与协议范围以文本接口指南为准。

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
python scripts/prepare-qwen35.py --model LOCAL_MODEL --output .scratch/qwen-input --prompt "Please calculate 2 + 3. Answer briefly."
<build-dir>/bin/hyle generate --inputs .scratch/qwen-input --tokens .scratch/qwen-input/prompt.ids --max-output-tokens 64 --capacity 1024
```

输出目录必须是新目录，只保存偏移、token IDs 与输入记录，共用原权重文件。
需要 MTP 时，在原权重准备命令加 `--mtp`，准备器加入额外15份原MTP参数，仅准备不执行模型；同一完整清单可切换普通／MTP。
量化 GGUF 头会同时从本地 tokenizer 的固定准备提示和特殊 token 生成来源绑定的半词表制品，不运行模型。
符合自动选择条件的 CUDA 目录可通过下方来源匹配的草稿制品启用 MTP；具体选择见[采样与 MTP](#采样与-mtp)。
只需普通模式时可省略准备器的`--mtp`；物理角色计数与输入身份见[模型输入说明](../../engine/models/qwen35/README.md#输入与数值政策)。

### 一次准备并复用

独立工具可从本地 Qwen 模型目录或单文件 GGUF 生成 Hyle 输入。GGUF 仍需要同模型的本地配置与 tokenizer：

```powershell
python scripts/prepare-qwen35.py --model <本地配置和tokenizer目录> --gguf <本地模型.gguf> --output .scratch/qwen-ready --mtp
hyle generate --inputs .scratch/qwen-ready --tokens .scratch/qwen-ready/prompt.ids --max-output-tokens 64 --capacity 1024
```

工具按本地内容计算来源身份、校验张量范围并生成 `weights.tsv`、输入 ID；原始输出头为量化编码的 GGUF 默认同时生成来源绑定的草稿子词表。本地文件的内容身份不等于已核验发布方的原始文件。工具不下载或复制原模型；`weights.tsv` 引用的原权重文件必须保持不变且可访问。重复加载同一准备目录不需要重新执行工具。直接读取相同 GGUF 内容时，来源身份不依赖其文件名；不同来源记录生成的身份可能不同，草稿制品必须匹配实际输入清单。

HF 原权重可省略 `--gguf`。`--draft-vocabulary tokenizer` 显式从本地 tokenizer 生成子词表，`--draft-rows N` 调整行数；`none` 不生成。默认 tokenizer 方法不运行模型，也不保证对所有提示或设备都最快。需要按模型输出离线校准时，先构建下方独立诊断目标，再指定 `--draft-vocabulary calibrated --hyle <本地hyle_qwen35_diagnostic.exe>`；它运行固定提示的普通生成。已有同来源校准结果可用重复的 `--calibration-run <目录>` 复用，此时不需要再次运行模型。

已有 Hyle 输入时，也可以只生成可复用资产，不重建权重清单：

```powershell
python scripts/prepare-qwen35.py --inputs <已有输入目录> --tokenizer-dir <同模型的本地tokenizer目录> --output .scratch/qwen-draft --mtp --draft-rows 114688
hyle generate --inputs <已有输入目录> --tokens <prompt.ids> --draft-vocabulary .scratch/qwen-draft/mtp-draft-vocabulary.ids --max-output-tokens 64 --capacity 1024
```

需要独立 Q4_K 草稿头时，可以在一次准备中同时生成子词表和头：

```powershell
python scripts/prepare-qwen35.py --model <本地HF模型目录> --output .scratch/qwen-q4-head --mtp --draft-vocabulary tokenizer --draft-head-q4 <本地ggml-base.dll> --quantizer-version llama.cpp@<40位实际提交号>
hyle generate --inputs .scratch/qwen-q4-head --tokens .scratch/qwen-q4-head/prompt.ids --max-output-tokens 64 --capacity 1024
```

该工具只支持其可读取的原始稠密 HF 头；有子词表时生成与其绑定的紧凑头，不自动推荐 K3。对已有输入生成独立资产后，运行时还需显式指定 `--draft-head .scratch/qwen-draft/mtp-draft-head.tsv`；直接准备的新输入目录在同时有子词表时可自动发现这些制品，仅准备全头时可用 `--draft-count 1 --draft-head FILE`。完整词表仍来自模型本身，无需生成另一份权重或词表文件。引擎加载模型后自动准备并持久化这些资产尚未实现。

## 常用选项

正常 `hyle generate` 和 `hyle --engine-worker` 共用 [application.h](../../engine/application.h) 的解析和提供者装载；模型登记位于 `model_catalog.h`，公共解析器不解释模型私有几何或状态。

| 用途 | 参数与合同 |
| --- | --- |
| token 输入与额度 | CLI 要求 `--tokens FILE` 或 `--tokens -`，`--max-output-tokens` 默认 64，包含首 token；EOS 可提前停止。worker 的请求、输入与额度来自管道帧，不能混用这些 CLI 请求参数。 |
| 端点与执行容量 | `--backend auto\|cpu\|cuda\|apple\|vulkan`；`--capacity` 默认 4096。CLI 默认一个常驻／活动会话，worker 默认两个，分别用 `--max-sessions` 与 `--active-sequences` 配置。它们不是草稿数或流水深度。 |
| prefill | `--prefill-chunk-tokens N` 控制块长；当前 Qwen 冷准备还按模型容量与 512 行上限裁剪，不查看未来轮次。 |
| 采样 | 默认 greedy；`--sampling random --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42`。额外采样程序可用 `--additional-sampling random,0.8,40,0.95`，CLI 用 `--sampling-index N` 选择，请求 RNG 独立。 |
| MTP | 不传 `--draft-count` 时由 Qwen 冷策略按来源制品和能力选择。`--draft-count 0` 准备普通生成，正数显式准备对应 MTP 上限；未指定草稿文件的显式正数使用原头与全词表。 |
| 不可变辅助资产 | `--asset ROLE=FILE` 传递提供者命名的文件；`--draft-vocabulary FILE` 和 `--draft-head FILE` 是两个草稿角色的简写。这里没有路径哨兵，`auto`／`full` 是真实文件名；自动发现通过省略参数表达。未使用或不支持的显式资产报错。 |
| 执行方案 | `--profile NAME` 声明额外执行准备，后续 `--draft-count`、`--overlap`、`--profile-asset ROLE=FILE` 属于该方案。CLI 的 `--execution-profile N` 与采样索引分开。当前 Qwen 每个所有者只准备方案零，明确拒绝额外方案；公共机制不伪装未实现的组合。 |

正常执行自动选择可用快路；`--overlap on\|off` 可作明确对照。权重、模型状态和真实退休仍由原实现持有，CLI 不构造第二份缓存。

## Qwen 专用诊断

以下选项、自然日程、固定重放和 JSON 字段属于显式诊断消费者，不由正常 `hyle` 入口解析。需要时单独构建：

```text
python scripts/build.py --target hyle_qwen35_diagnostic --jobs 2
<build-dir>/bin/hyle_qwen35_diagnostic --help
```

该目标不在默认产品构建中；它复用同一模型运行实现与冷准备，额外接入有限日程、trace 和报告消费。校准工具的 `--hyle` 参数也应指向它。正常多会话文本使用公共 worker／Responses，见[文本接口](text-api.md)。

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
它可以与自动 MTP 一起使用，可能增加启动时间与持久内存；`--cache-bytes`不是总显存上限。
这也不是设备执行或CUDA capture预热，首次真实绑定仍可能准备replay。

### 连续接力

普通 CUDA 自然生成在符合执行条件时自动使用连续接力，无需额外参数；greedy 已有限检查单活动请求及双活动请求，随机采样已有限检查单活动请求。
CUDA greedy单会话K1—K3已接入自然接力。固定K2／K3把起草、验证及draft KV修复放入同一张轮图，
中间token和下一轮接受位置留在设备。冷期只按配置容量和已到达首轮准备，后续长度与配额不参与规划。
符合条件的 MTP 根据来源绑定的草稿制品读取推荐草稿数；没有推荐时默认为 K1。K3 的收益与资源边界见性能表。
单会话有界生成按配置容量准备接力；其它有限日程在有多于一个输出额度且执行能力允许时准备两层候选。
需要强制只使用不依赖未来请求形状的路径时，可设置 `--require-bounded-generation on`；不符合该条件的配置会报错。
prefill 保留原路径，后继只使用已有实际捕获。
完整同形、同序的活动请求组可持续接力；异形分组保留原同步执行。输出额度不足时先完成已接纳工作，再暂停等待额度。
其它后端、上述范围外的MTP、混合执行、随机采样 MTP、scheduler 或显式 commands 自动使用原单步行为；显式缓存预算容不下额外候选时也自动使用单步。
遇到当前物理绑定缺少捕获时，先完成已提交工作，再沿原路径准备捕获并恢复接力；设备及清理错误继续报告。
接力会增加载体和状态内存；不满足执行条件或额外资源不足时自动单步推进。具体速度与内存边界见[最新性能对照](../developer/performance/decode.md#目标与当前基线)。


### 采样与 MTP

默认 greedy；也可用 `--sampling random --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42`。
普通与 MTP 共用温度→top-k→top-p 分布，`--top-k 0` 关闭该过滤。
每个会话有独立随机流；失败或撤回的成员不推进已发布随机位置。相同seed不保证不同后端或普通／MTP逐token相同。
逐请求bias／mask、历史罚分、动态条件及停止政策属于`engine/`内的C++请求扩展，
没有对应CLI参数或已安装的引擎SDK／server入口；其合同查[输入与请求政策](../../engine/models/qwen35/README.md#输入与数值政策)。

`--mtp auto|off|on` 默认 `auto`。在符合条件的 CUDA 自然 greedy 生成中，目录里的来源匹配的
`mtp-draft-vocabulary.ids` 可以选择 MTP；制品可推荐草稿数，没有推荐时选择 K1。
没有词表制品时，具备完整 MTP 角色及 F32 量化路径的 GGUF 来源也可自动选择全词表 K1。
其他情况默认关闭；`--mtp off` 强制普通生成。启用 MTP 会增加内存，收益取决于模型、词表和输入。
在自动策略允许的执行条件下，显式选择草稿头或词表也可启用 MTP；`mtp_selection_reason` 会报告 `explicit_draft_asset`。
关闭 MTP、草稿数为 0，或自动策略回落时，不读取未使用的草稿制品；若显式指定了制品文件，冲突会报错并说明原因。
实际选中的无效或来源不符制品也会报错。
JSON 的 `mtp_selection_reason` 报告冷期选择或回落原因，`mtp_depth_source` 说明草稿数来自显式参数、制品推荐还是配置默认值。
`draft_vocabulary_selection_reason` 说明词表是否使用、缺失或因来源／后端能力而跳过。
`bounded_generation_plan_reason`、`decode_overlap_plan_reason`、`device_verification_plan_reason` 和 `overlap_depth_plan_reason` 解释有界生成、接力、设备验证及连续后继的计划选择；`cache_budget` 表示配置的缓存预算触发回落。显式计划的工作负载没有 decode 时，设备验证报告 `no_decode`；默认有界会话仍为可能到来的后续轮次保留能力。运行时是否实际可接力仍以 `decode_overlap_available` 为准。
`insufficient_capacity` 表示容量不足以同时容纳输入和草稿位置；显式启用仍受实际容量和能力限制。
`--mtp-max-drafts 1` 可覆盖已启用的 K3，按 K1 执行。
`--mtp-max-drafts` 控制每轮最多提出的草稿数，默认采用制品推荐值，没有推荐时为 1。
草稿数设为 0 时，`auto` 和 `off` 使用普通路径；显式 `on` 保持 MTP 路径但不提出草稿。
制品、后端与草稿数的实际运行范围见[模型覆盖](../implementation/model-operators.md)。

### 草稿词表

默认 `--mtp-draft-vocabulary auto`：MTP开启、无logits modifier的greedy时，
读取 `mtp-draft-vocabulary.ids`。制品的 `head prepared` 偏好选择目录里来源匹配的 `mtp-draft-head.tsv`；
该头不存在时从支持的原始头提取所选行。`head original` 选择原始头；显式 `--mtp-draft-head` 优先。
文件不存在或采样政策不适用时，已启用的 MTP 使用完整词表；
自动词表不能由所选后端或原始头提取时会跳过，并按原始头能力选择全词表 MTP 或普通模式；显式指定的不支持词表会报错。
被选中的制品若source identity不匹配会拒绝加载。`--mtp-draft-vocabulary full` 强制完整词表作对照，
也可用 `--mtp-draft-vocabulary FILE` 显式选择另一份准备文件。
子集头的接受率取决于实际生成内容；多语言或长尾词较多时，全词表头可能更快。
选择完整词表时，自动发现的compact头不适用，引擎改用原始头；显式指定不匹配的compact头会报错。
支持原BF16／F16／F32稠密head；CUDA还支持GGUF Q8_0／Q4_K／Q5_K／Q6_K编码行的子集，保持原量化代码与scale字节。
子集会增加所选头的行副本；自动选择不按剩余显存回落，显存紧张时可选 `--mtp-draft-vocabulary full` 或 `--mtp off`。
Q5_K 的行复制、1～4行投影与原ID映射已用合成数据验证；本地2B模型已验证Q5_K完整头及子集头的CUDA单请求greedy生成。
可用于K1与多草稿；完整target head与embedding继续保留。编码行副本目前只支持CUDA，其它后端跳过自动词表并拒绝显式编码词表。
显式文件遇到随机、modifier或不支持的权重会报错。

准备阶段可指定语言偏好，例如 `prepare.py ... --mtp --draft-languages zh,en`，使用本地 tokenizer 的中英和共享代码提示选择自动词表；默认行数仍为原词表的一半，可用 `--draft-rows` 调整。未指定语言时沿用原有提示。此方法只统计提示中的 token，不执行模型。

也可用本地模型生成独立校准语料，按生成内容选行；例如为中英文和共享代码生成词表，默认选32768行：

```powershell
python scripts/prepare-mtp-draft-vocabulary.py --hyle <hyle_qwen35_diagnostic.exe> --inputs <准备目录> --tokenizer-dir <本地tokenizer目录> --languages zh,en --work-dir .scratch/draft-calibration --output .scratch/draft-vocabulary-zh-en.ids
hyle_qwen35_diagnostic --inputs <准备目录> --backend cuda --mtp on --mtp-draft-vocabulary .scratch/draft-vocabulary-zh-en.ids --schedule <自然请求文件>
```

该工具只用本地文件；不指定语言时使用24个独立提示，指定语言时使用对应提示和共享代码提示。内置 `ar,de,en,es,fr,hu,ja,ko,ru,th,vi,zh`，可用逗号选择多个语言，其中 `hu` 是匈牙利语。普通greedy每个提示默认最多生成256个token，可用 `--tokens 1024` 扩大校准；只统计生成的ID。指定语言时优先保留本地 tokenizer 的特殊 token，行数不足时按原 ID 选取。校准内容多也不能保证覆盖全部语言或优于全词表，应按实际负载比较。
按频次选行，未出现的ID按原ID升序补齐；不使用八轮验收负载。校准在启动之前单独运行，
原始输出保留在新建的work目录，不覆盖已有结果。没有准备文件的模型不会在启动时自行校准。
也可用重复的 `--calibration-run DIR` 复用同一source的已有普通greedy校准；该模式不启动GPU，也不需要tokenizer。
支持本工具的校准目录，以及含train／validation划分的已归档生成目录；只导入train，核对源身份、原始ID和会话退休。
对带分组的已归档语料指定 `--languages zh,en` 时，只统计对应语言和 `code` 组；每种指定语言都需要有训练样本。自备语料也可使用其他语言标签，并用 `--languages` 选择。无分组的旧校准目录不能按语言过滤。离线复用没有 tokenizer，因此不额外推断特殊 token。
`--rows N` 可选任意非零且不超过原词表的行数，选择次序保持不变。
如果已在目标负载上确认 K3 合适，可以在生成时指定 `--recommended-drafts 3`；
该工具不会自动评测或推断最佳草稿数。没有指定时制品不带推荐，默认采用 K1。
文件绑定准备清单的原source identity及实际词表大小，格式为：

```text
hyle-draft-vocabulary-v2
<weights.tsv第二行的64位source identity>
<模型实际词表大小>
recommended_drafts none|<正整数>
head original|prepared
<以空白分隔的原token IDs>
```

IDs必须非空、互异且在原词表范围内；冷期按原ID排序，保持greedy平局顺序，图内输出仍还原为原token ID。
CUDA单活动请求、通过冷期设备能力检查的BF16 head可直接读取原矩阵的选中行，只额外保留ID映射；
其它设备、dtype或批次能力在冷期导入一次行副本。例如2B BF16模型的32K行，直接读取时额外权重为0、映射128 KiB，
行副本路径则另占128 MiB权重。v1 词表需要重新生成或转换为上述 v2 格式。不可变投影和映射供原并行编译流程读取，热路径不新增分配。
JSON的 `original_*_weight_bytes` 仍只统计原角色，额外payload见 `draft_vocabulary_weight_bytes`／`draft_vocabulary_map_bytes`；
`draft_vocabulary_indexed_rows` 表示是否直接读取原矩阵。额外权重不属于 `--cache-bytes`，payload字节数也不是物理显存峰。
子集会改变提案和接受率，需按实际输入检查速度；有限校准语料不代表通用语言覆盖。

### 独立全词表草稿头

BF16目标也可使用离线准备的全词表Q4_K草稿头。目标验证头与embedding继续使用原权重。
准备工具只转换这一张矩阵，不下载模型或ggml库；输入是已有的BF16／F16／F32 HF准备目录，hidden须为256的倍数。
本次验证使用llama.cpp `2cbca389f515df455412dafa07e2b1359cc72a6d` 的本地 `ggml-base.dll`，Q4_K参考量化、无importance matrix：

```powershell
python scripts/prepare-mtp-draft-head.py --inputs <准备目录> --ggml-library <本地ggml-base.dll> --quantizer-version llama.cpp@2cbca389f515df455412dafa07e2b1359cc72a6d --output .scratch/full-draft-head
hyle_qwen35_diagnostic --inputs <准备目录> --backend cuda --mtp on --mtp-max-drafts 3 --mtp-draft-head .scratch/full-draft-head/mtp-draft-head.tsv --schedule <自然请求文件>
```

输出目录必须是新目录。工具完整写入并验证编码行后才发布 `mtp-draft-head.tsv`；运行时校验元数据、原source identity、形状及payload范围，
通过原权重流加载额外投影。2B全词表头的payload为286064640字节（272.8125MiB）；这不是完整进程的显存峰值。

默认 `--mtp-draft-head auto` 根据词表制品的头偏好选择原始头或输入目录里的准备头；
没有词表制品时，允许自动准备头的路径会查找该头，自动全词表 GGUF 路径使用原始头。
`--mtp-draft-head original` 禁用独立头；同时加 `--mtp-draft-vocabulary full` 才是原始全词表头对照。
独立Q4_K头可与显式子集映射组合。CUDA在冷期按原ID复制完整编码行，保持代码与scale字节；
复制和ID上传退休后才替换完整草稿头，不更换目标验证头或embedding。头与映射分别绑定原source identity。
例如从已有独立校准生成114688行映射，再用于K3：

```powershell
python scripts/prepare-mtp-draft-vocabulary.py --inputs <准备目录> --calibration-run <旧校准目录> --calibration-run <新增train校准目录> --rows 114688 --work-dir .scratch/draft-subset-calibration --output .scratch/draft-subset.ids
hyle_qwen35_diagnostic --inputs <准备目录> --backend cuda --mtp on --mtp-max-drafts 3 --mtp-draft-head .scratch/full-draft-head/mtp-draft-head.tsv --mtp-draft-vocabulary .scratch/draft-subset.ids --schedule <自然请求文件>
```

显式映射仍可与独立头组合。默认K3可使用下一段准备的成套制品。
2B的114688行Q4_K头payload为126MiB、映射448KiB。组合路径目前只验证CUDA，要求启用草稿和无modifier的greedy。
更换草稿头会改变提案、接受率与轮次划分，速度与数值边界见[性能表](../developer/performance/decode.md#固定-k3本轮复测)。

### 自动K3的准备与运行

先按上文用独立校准生成114688行映射，再离线生成可直接运行的默认目录：

```powershell
python scripts/prepare-mtp-draft-head.py --inputs <含MTP的BF16准备目录> --ggml-library <本地ggml-base.dll> --quantizer-version llama.cpp@2cbca389f515df455412dafa07e2b1359cc72a6d --default-k3-vocabulary <114688行映射.ids> --output .scratch/mtp-ready
hyle_qwen35_diagnostic --inputs .scratch/mtp-ready --backend cuda --capacity 32768 --schedule <自然请求文件>
```

运行命令无需额外MTP、K或草稿头参数。输出目录必须是新目录；工具离线编码并只保存所选Q4_K行，
借用原权重的绝对文件范围，并发布原`weights.tsv`格式、输入IDs和带 K3 推荐的`mtp-draft-vocabulary.ids`。
完整子集payload、映射与原权重引用写完后才发布头元数据；头身份同时绑定原模型与规范I32映射。启动时只校验和加载，不下载、生成词表或量化权重。

已验证系统为Windows11 x64、RTX4070／SM8.9、驱动617.14、CUDA13.3.73。制品保存GGUF Q4_K行数据与原ID映射，
其读取依赖相同Q4_K存储约定及兼容后端；其它系统未作本次性能验收。
自动准入要求CUDA、容量至少2、单会话自然greedy；实际草稿深度仍受可用容量和执行能力限制。
CPU分置、prefix、多会话、scheduler和随机采样不自动启用。`--replay commands`、计划形状准备及关闭decode overlap均可使用自动MTP。
准备头缺失时使用原始头；选中的制品身份过期或行数不符时拒绝加载。
制品元数据表达K3偏好，本身不证明该模型或负载上K3最快；可用`--mtp-max-drafts 1`显式选择K1。
这些条件只选择当前模型制品及执行政策，不读取未来prompt、接受结果或输出配额，不做动态K。
离线转译只运行一次，不计入推理性能；冷启动加载与ready单列。主要比较实际输出除以prefill＋decode墙时，
再分别列两个阶段；启动不摊入生成速度。完整进程值只作另列诊断。速度验收范围与启动、显存成本见[性能表](../developer/performance/decode.md#固定-k3本轮复测)。

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
普通与 MTP 共用 `prompt/generate` 自然请求文件和多轮／多会话生命周期，用 `--mtp auto|off|on` 选择。
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

普通 decode 的数值回归可用 `scripts/inference-quality.py` 逐行计算完整词表 KL，
不保存完整 logits。它同时驻留 BF16 oracle 和一个候选模型，串行提交 GPU 工作；
这不是速度测试。需要 Python NumPy、准备好的三格式本地输入和 GGUF，以及 CUDA 测试构建：

```powershell
cmake --build <build> --config Release --target hyle_qwen_quality_worker
python scripts/inference-quality.py run --hyle-worker <build>/bin/hyle_qwen_quality_worker.exe --hyle-runtime <build>/bin --peer-worker <tools>/llama_quality_worker.exe --official-bin <b12063-bin> --bf16-model <BF16.gguf> --q8-model <Q8_0.gguf> --q4-model <Q4_K_M.gguf> --bf16-inputs <BF16-inputs> --q8-inputs <Q8-inputs> --q4-inputs <Q4-inputs> --corpus tests/data/inference-quality-prompts.tsv --output <fresh-output> --hyle-version <source-commit> --peer-version <peer-commit>
```

上游 worker 源码为 `tests/tools/llama_quality_worker.cc`，使用固定 b12063 的公开
`include/`、`ggml/include/` 头文件编译，并链接其 `llama`、`ggml`、`ggml-base` 导入库；
Windows 可从对应官方 DLL 的导出表生成导入库。需要 C++20；不链接 Hyle 的 ggml。
worker 会核对实际载入的 DLL 位于 `--official-bin` 指定目录。

默认取新提示集的 8192 个 BF16 greedy teacher 位置，每提示最多 64 个，遇 EOS 停止；
每个候选强制使用同一前缀。提示通过 GGUF 自带 chat template 和 tokenizer 转换，不下载模型。
每个提示后清空上游状态，Hyle 走原会话退休并核对页、版本和借用资源。
输出仅保留 token、逐位置指标、命令和 stderr；可以用
`python scripts/inference-quality.py analyze <output>` 重算提示分组 bootstrap。
分组按提示重采样，不能把相邻位置当成独立样本。此合成短提示集用于有限数值回归，
不代表通用质量、长上下文或 MTP 资格；护栏与实际结果见下方性能资料。

普通与 MTP 分开对照，报告 prefill、decode、内存／显存以及真实多轮综合速度，加载时间另列。
最新正式窗口、计数边界和已知热点统一保留在[decode 性能资料](../developer/performance/decode.md)，
不以模型改名、组件测试通过或减少 kernel 数量推定端到端提速。
