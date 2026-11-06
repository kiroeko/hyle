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
需要 MTP 时，在原权重准备命令加 `--mtp`，准备器加入额外15份原MTP参数，仅准备不执行模型；同一完整清单可切换普通／MTP。
符合自动选择条件的 CUDA 目录可通过下方来源匹配的草稿制品启用 MTP；具体选择见[采样与 MTP](#采样与-mtp)。
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
它可以与自动 MTP 一起使用，可能增加启动时间与持久内存；`--cache-bytes`不是总显存上限。
这也不是设备执行或CUDA capture预热，首次真实绑定仍可能准备replay。

### 连续接力

普通 CUDA greedy 自然生成自动使用连续接力，无需额外参数；已有限检查单活动请求及双活动请求。
无草稿概率门槛的CUDA greedy单会话K1—K3已接入自然接力。固定K2／K3把起草、验证及draft KV修复放入同一张轮图，
中间token和下一轮接受位置留在设备。冷期只按配置容量和已到达首轮准备，后续长度与配额不参与规划。
符合条件的 MTP 会根据准备目录中的草稿制品选择 K1 或 K3；没有 K3 制品时显式开启默认为 K1。K3 的收益与资源边界见性能表。
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

`--mtp auto|off|on` 默认 `auto`。在符合条件的 CUDA 自然 greedy 生成中，来源匹配的 K3 命名制品优先选择 K3；
没有该制品但目录有来源匹配的 `mtp-draft-vocabulary.ids` 时，
自动选择原始草稿头的 K1。
其他情况默认关闭；`--mtp off` 强制普通生成。启用 MTP 会增加内存，收益取决于模型、词表和输入。
`--mtp-max-drafts 1` 可覆盖已启用的 K3，按 K1 执行。
`--mtp-max-drafts` 控制每轮最多提出的草稿数，自动 K3 制品为 3，其余默认为 1。`--mtp-min-probability` 设置
草稿分布最大概率的最低门槛，范围为 0 到 1，默认 0；达到门槛才继续草稿，实际长度可以更短。
这不是额外置信度模型。制品、后端与草稿数的实际运行范围见[模型覆盖](../implementation/model-operators.md)。

### 草稿词表

默认 `--mtp-draft-vocabulary auto`：MTP开启、无logits modifier的greedy、零概率阈值时，
默认 K3 目录组合 `mtp-draft-head.tsv` 与 `mtp-draft-vocabulary-k3.ids`；
其他 MTP 目录读取 `mtp-draft-vocabulary.ids`；该文件也可让符合条件的自动模式选择 K1。
文件不存在或采样政策不适用时使用完整词表；
存在但source identity不匹配时拒绝加载。`--mtp-draft-vocabulary full` 强制完整词表作对照，
也可用 `--mtp-draft-vocabulary FILE` 显式选择另一份准备文件。
支持原BF16／F16／F32稠密head；CUDA还支持GGUF Q8_0／Q4_K／Q6_K编码行的子集，保持原量化代码与scale字节。
可用于K1与多草稿；完整target head与embedding继续保留。编码行副本目前只验证CUDA，其它后端明确拒绝。
显式文件遇到随机、modifier、非零阈值或不支持的权重会报错。

准备阶段可用本地模型生成独立校准语料，默认选32768行：

```powershell
python scripts/prepare-mtp-draft-vocabulary.py --hyle <hyle.exe> --inputs <准备目录> --tokenizer-dir <本地tokenizer目录> --work-dir .scratch/draft-calibration --output <准备目录>/mtp-draft-vocabulary.ids
```

该工具只用本地文件，固定24个独立提示、普通greedy各生成至多256个token，只统计生成的ID。
按频次选行，未出现的ID按原ID升序补齐；不使用八轮验收负载。校准在启动之前单独运行，
原始输出保留在新建的work目录，不覆盖已有结果。没有准备文件的模型不会在启动时自行校准。
也可用重复的 `--calibration-run DIR` 复用同一source的已有普通greedy校准；该模式不启动GPU，也不需要tokenizer。
支持本工具的校准目录，以及含train／validation划分的已归档生成目录；只导入train，核对源身份、原始ID和会话退休。
`--rows N` 可选任意非零且不超过原词表的行数，选择次序保持不变。
文件绑定准备清单的原source identity及实际词表大小，格式为：

```text
hyle-draft-vocabulary-v1
<weights.tsv第二行的64位source identity>
<模型实际词表大小>
<以空白分隔的原token IDs>
```

IDs必须非空、互异且在原词表范围内；冷期按原ID排序，保持greedy平局顺序，图内输出仍还原为原token ID。
CUDA单活动请求、通过冷期设备能力检查的BF16 head可直接读取原矩阵的选中行，只额外保留ID映射；
其它设备、dtype或批次能力在冷期导入一次行副本。例如2B BF16模型的32K行，直接读取时额外权重为0、映射128 KiB，
行副本路径则另占128 MiB权重。已有准备文件不需要重做。不可变投影和映射供原并行编译流程读取，热路径不新增分配。
JSON的 `original_*_weight_bytes` 仍只统计原角色，额外payload见 `draft_vocabulary_weight_bytes`／`draft_vocabulary_map_bytes`；
`draft_vocabulary_indexed_rows` 表示是否直接读取原矩阵。额外权重不属于 `--cache-bytes`，payload字节数也不是物理显存峰。
子集会改变提案和接受率，需按实际输入检查速度；有限校准语料不代表通用语言覆盖。

### 独立全词表草稿头

BF16目标也可使用离线准备的全词表Q4_K草稿头。目标验证头与embedding继续使用原权重。
准备工具只转换这一张矩阵，不下载模型或ggml库；输入是已有的BF16／F16／F32 HF准备目录，hidden须为256的倍数。
本次验证使用llama.cpp `2cbca389f515df455412dafa07e2b1359cc72a6d` 的本地 `ggml-base.dll`，Q4_K参考量化、无importance matrix：

```powershell
python scripts/prepare-mtp-draft-head.py --inputs <准备目录> --ggml-library <本地ggml-base.dll> --quantizer-version llama.cpp@2cbca389f515df455412dafa07e2b1359cc72a6d --output .scratch/full-draft-head
hyle --inputs <准备目录> --backend cuda --mtp on --mtp-max-drafts 3 --mtp-draft-head .scratch/full-draft-head/mtp-draft-head.tsv --schedule <自然请求文件>
```

输出目录必须是新目录。工具完整写入并验证编码行后才发布 `mtp-draft-head.tsv`；运行时校验元数据、原source identity、形状及payload范围，
通过原权重流加载额外投影。2B全词表头的payload为286064640字节（272.8125MiB）；这不是完整进程的显存峰值。

默认 `--mtp-draft-head auto` 在输入目录发现此文件时优先使用它，随后才考虑自动32K映射。
`--mtp-draft-head original` 禁用独立头；同时加 `--mtp-draft-vocabulary full` 才是原始全词表头对照。
独立Q4_K头可与显式子集映射组合。CUDA在冷期按原ID复制完整编码行，保持代码与scale字节；
复制和ID上传退休后才替换完整草稿头，不更换目标验证头或embedding。头与映射分别绑定原source identity。
例如从已有独立校准生成114688行映射，再用于K3：

```powershell
python scripts/prepare-mtp-draft-vocabulary.py --inputs <准备目录> --calibration-run <旧校准目录> --calibration-run <新增train校准目录> --rows 114688 --work-dir .scratch/draft-subset-calibration --output .scratch/draft-subset.ids
hyle --inputs <准备目录> --backend cuda --mtp on --mtp-max-drafts 3 --mtp-draft-head .scratch/full-draft-head/mtp-draft-head.tsv --mtp-draft-vocabulary .scratch/draft-subset.ids --schedule <自然请求文件>
```

普通独立头仍优先于旧自动映射；显式映射仍可组合。默认K3使用下一段专门准备的成套制品，不把旧32K目录自动升为K3。
2B的114688行Q4_K头payload为126MiB、映射448KiB。组合路径目前只验证CUDA，要求启用草稿、无modifier的greedy和零概率阈值。
更换草稿头会改变提案、接受率与轮次划分，速度与数值边界见[性能表](../developer/performance/decode.md#固定-k3本轮复测)。

### 自动K3的准备与运行

先按上文用独立校准生成114688行映射，再离线生成可直接运行的默认目录：

```powershell
python scripts/prepare-mtp-draft-head.py --inputs <含MTP的BF16准备目录> --ggml-library <本地ggml-base.dll> --quantizer-version llama.cpp@2cbca389f515df455412dafa07e2b1359cc72a6d --default-k3-vocabulary <114688行映射.ids> --output .scratch/mtp-ready
hyle --inputs .scratch/mtp-ready --backend cuda --capacity 32768 --schedule <自然请求文件>
```

运行命令无需额外MTP、K或草稿头参数。输出目录必须是新目录；工具离线编码并只保存所选Q4_K行，
借用原权重的绝对文件范围，并发布原`weights.tsv`格式、输入IDs和`mtp-draft-vocabulary-k3.ids`。
完整子集payload、映射与原权重引用写完后才发布头元数据；头身份同时绑定原模型与规范I32映射。启动时只校验和加载，不下载、生成词表或量化权重。

已验证系统为Windows11 x64、RTX4070／SM8.9、驱动617.14、CUDA13.3.73。制品保存GGUF Q4_K行数据与原ID映射，
其读取依赖相同Q4_K存储约定及兼容后端；其它系统未作本次性能验收。
自动K3目前限原HF BF16 Qwen3.5-2B几何、BF16 KV、CUDA、容量4至32768、单会话自然greedy、零草稿概率阈值、
lazy准备及默认接力／replay；CPU分置、prefix、多会话、scheduler和随机不自动启用。
专用映射存在但头缺失、身份过期或行数不符时拒绝加载。普通独立头、旧32K目录和显式开关保留原策略。
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
