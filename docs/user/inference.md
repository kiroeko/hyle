# HYLE 推理引擎

HYLE 正在由真实模型路径形成完整推理引擎，入口位于 `engine/`；基础模块仍可独立使用。
当前主要用这条路径检验底层支持，应用入口随测试推进，server 与应用 API 留待后续。
当前完整文本模型为 Qwen3.5-2B，包含普通生成、有界多草稿 MTP、多轮、多会话、
共享前缀和有界缓存恢复。CUDA、Apple 和 Vulkan 已有列明范围的真实模型执行；
CPU 入口的普通／MTP 短旅程已完整通过，缓存元数据计费已修复，见[检查记录](../evidence/correctness/qwen-mtp-cpu-layout-budget-2026-10-24.json)；已检查范围及各项边界见[推理工程范围](../implementation/inference-roadmap.md)。
官方 BF16 及固定 GPTQ、NVFP4、Unsloth MTP GGUF Q4 制品已有完整模型短旅程；
量化输入通过显式原角色清单借入文件范围，具体边界见[实现范围](../implementation/inference-roadmap.md)。这不代表可以直接加载任意量化模型目录。
NF4 已接原量化状态校验、F16 激活／缓存及六输入压缩权重驻留，[CPU／CUDA 有限状态链](../evidence/correctness/nf4-source-resident-ordinary-consumer-2026-10-25.json)通过；完整 27B 模型未运行。加载器不预先展开整库，实际计划会报告压缩融合、分解及 workspace；CPU 多行仍可能选择分解，不能仅按 4bit 文件大小估算运行空间。
CT asymmetric INT4 已接原零点角色和同一加载／投影流程，[原 CPU 组件](../evidence/correctness/ct-asymmetric-qwen-original-projections-2026-10-25.json)的七组投影通过；完整 27B 模型及该原权重路径的其它后端尚未验证。
选定 MLX affine4／8 group64 制品已接原压缩投影、embedding 与共享 head，保留源 norm／conv 语义；[原 CPU／CUDA 组件及小型 CPU 状态链](../evidence/correctness/mlx-source-resident-ordinary-consumer-2026-10-25.json)通过。完整 2B／27B、质量及吞吐未测，不代表任意 MLX 目录可直接加载。

目前使用准备后的本地权重清单和 token IDs，输出 JSON 中的生成 IDs 与计时。
尚无通用模型目录加载、C++ tokenizer、聊天服务或图片输入；其它模型的有限组件检查也不表示已有完整引擎。
当前范围和后续顺序见[推理工程计划](../implementation/inference-roadmap.md)。

## 构建

先完成[环境准备](../developer/setup/README.md)，在仓库根目录运行，例如 CUDA：

```text
python scripts/init.py --name qwen-cuda --components engine --backends cuda
python scripts/build.py --target hyle_engine --jobs 2
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

准备器读取[固定模型输入清单](../implementation/tasks/compute/qwen35-2b-local-input.json)，
核对本地文件身份，生成权重偏移与 token IDs；不下载模型、不复制整份权重。
本地模型目录须与清单匹配，并保持可访问。Python tokenizer 需要准备器所用的 Transformers 环境，
初始化 C++ 构建不会自动安装它或获取模型。

```text
python engine/models/qwen35/prepare.py --model LOCAL_MODEL --output .scratch/qwen-mtp-input --prompt "Please calculate 2 + 3. Answer briefly." --mtp
<build-dir>/bin/hyle --inputs .scratch/qwen-mtp-input --mtp off --steps 64 --capacity 1024
<build-dir>/bin/hyle --inputs .scratch/qwen-mtp-input --mtp on --steps 64 --capacity 1024
```

输出目录必须是新目录，只保存偏移、token IDs 与输入记录，共用原权重文件。
`prepare.py --mtp` 加入额外 15 份原 MTP 参数，仅准备不执行模型；同一份完整清单可切换普通／MTP。
普通模式校验完整清单后只加载文本权重；官方 BF16 为 320 个文本张量加 15 个 MTP 张量，量化制品的物理张量数随格式而异。
JSON 的 `text_tensors`／`mtp_tensors` 统计实际加载的物理角色，`weight_count_scope` 标明口径。只需普通模式时可省略准备器的 `--mtp`。
原 CT channel/token FP8 与 local NVFP4 混合清单使用
`--fp8-reader compressed-tensors-0.17.2a20260716` 固定参考算法；存在原 KV 校准参数时，
还须显式选择缓存政策：`--kv-cache-dtype bf16` 保留原 BF16 缓存并省略校准角色加载；
普通推理也可用 `--kv-cache-dtype e4m3fn`，读取每层原 BF16 K/V scale，采用固定校准的 E4 缓存。
后者不能与 `--mtp` 合用，缺失、非正或非有限 scale 会在冷准备时拒绝；不合成默认 scale。
两者都将校准参数的名字、形状及 hash 保留在输入身份和记录中，不把它们装成模型权重张量。
E4 的缓存行载荷为 BF16 的一半，实际总 backing 仍包括页、元数据和其它状态。
缓存政策不改变 activation、recurrent 或投影计算精度。channel/token 保留 BF16 QDQ 与逐权重舍入，
local NVFP4 保留原独立全局乘数和 F32 输入预舍入；它们不套用 block128 或 ModelOpt 的算法。
目前已检查完整原头部、小原权重投影和小清单真实加载，见[接线记录](../evidence/correctness/mtp-qwen-ct-mixed-prepared-binding-2026-10-24.json)；
这不代表该 27B 制品已完成整模型生成。借入这类分离 target／MTP 文件时使用完整的 `--role-files` 原角色记录。
`--steps` 是最多输出数，包含 prefill 首 token，遇 EOS 可提前结束。
底层普通请求还可独立设置额外停止 token、忽略模型 EOS，并在缓存命中与恢复时保留请求政策；
当前命令行继续使用默认模型 EOS，该非默认请求配置尚未扩展到 MTP。
Windows PowerShell 将运行路径改为实际 `hyle.exe`；上面的 `<build-dir>` 是脚本报告的目录。
输入不足、模型身份不符、上下文容量不足或不支持的参数组合会明确失败。

普通自然生成可加 `--prefill-chunk-tokens 256`，限制每次 prefill 实际消费的 token 行数，
让请求调度在块间继续服务其它请求；默认不限制块长。中间块只推进真实 KV／递归状态，
最后一块才生成首 token、推进采样游标。后续轮的 pending 输入仍只消费一次，
`--capacity` 继续限制完整上下文，分块不会扩大它。当前该选项不支持 MTP 或固定 `prompt/decode` 重放，
不支持的组合会直接报错；它不代表设备执行预热或全模型 varlen 合批。
有限量子直接消费真实尾块，例如量子 8 的 23 行分为 `8+8+7`，P154／量子64为 `64+64+26`。
普通单请求在所选路线支持时复用容量图，短尾的私有计算载体可大于实际输入；
缓存、递归状态、位置和采样只提交真实输入行。其它路线按真实长度接纳 exact 图，设备错误仍直接报告。
合法分块及算法选择会影响计算形状和舍入，不能保证与整段生成逐位相同，也不保证分块更快。

`--warmup-shapes planned` 在 ready 前准备已有计划中会使用的执行载体，默认 `lazy` 保持按需准备。
该选项保留实际批宽、行数、阶段和采样配置，不枚举所有上下文长度，不接纳会话或推进 RNG。
它可增加启动时间与持久载体内存；`--cache-bytes` 不是这些资源的总显存上限。
启用前缀缓存和有限块长时，容量及中间／末块完成方式进入同一冷计划；
默认无界输入、MTP 或其它未计划形状仍需有界冷接纳。额外形状容量由初始冷键数冻结，独立于会话缓存数量。
这不是设备执行／CUDA capture 预热，首次真实绑定仍可能准备 replay。

普通 CUDA greedy 自然生成自动使用连续接力，无需额外参数；已有限检查单活动请求及双活动请求。
当前仅为最多输出至少 10 个 token 的自然生成准备两层候选；prefill 保留原路径，后继只使用已有实际捕获。
完整同形、同序的活动请求组可持续接力；异形分组保留原同步执行。输出额度不足时先完成已接纳工作，再暂停等待额度。
其它后端、MTP、混合执行、随机采样、scheduler 或显式 commands 自动使用原单步行为；显式缓存预算容不下额外候选时也自动使用单步。
遇到当前物理绑定缺少捕获时，先完成已提交工作，再沿原路径准备捕获并恢复接力；设备及清理错误继续报告。
接力会增加载体和状态内存；不满足执行条件或额外资源不足时自动单步推进。具体速度与内存边界见[最新性能对照](../developer/performance/decode.md#目标与当前基线)。


默认 greedy；也可用 `--sampling random --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42`。
普通与 MTP 共用温度→top-k→top-p 分布，`--top-k 0` 关闭该过滤。
每个会话有独立随机流，提交失败或取消不推进已发布的随机数位置；前缀命中保留当前会话的随机流。
固定随机地址的恢复、重试及真实取消已检查，见[功能记录](../evidence/correctness/qwen-mtp-random-integration-2026-10-24.json)。
相同 seed 不保证不同后端或普通／MTP 路径逐 token 相同。

`--mtp-max-drafts` 控制每轮最多提出的草稿数，默认 1。`--mtp-min-probability` 设置
草稿分布最大概率的最低门槛，范围为 0 到 1，默认 0；达到门槛才继续草稿，实际长度可以更短。
这不是额外置信度模型。CPU／CUDA 已检查多草稿、提前截断和普通回退，见
[多草稿与检查点记录](../evidence/correctness/qwen35-mtp-multidraft-checkpoint-controls-2026-10-24.json)。
Apple／Vulkan 另已运行 K4 随机和双用户 K2 随机完整短旅程；具体制品和新增配置的完成范围以看板为准。

需要明确的 CPU＋GPU 分置时，用 `scripts/init.py` 的 `--backends cpu,cuda`、`cpu,metal` 或
`cpu,vulkan` 分别准备对应构建，运行时选择 GPU 主后端并添加 `--cpu-ffn-layers 2`，将前两层 target FFN 及其
原始权重放到 CPU；注意力、缓存和 MTP 留在所选 GPU。默认值 0 保持 GPU 放置。
固定 BF16 模型的 CUDA 普通、K1／K4、随机、多会话及恢复见[CUDA 分置记录](../evidence/correctness/qwen35-cpu-ffn-cuda-placement-2026-10-24.json)；
Apple／Vulkan 的完整随机 K1 短旅程见[原生混合记录](../evidence/correctness/mtp-qwen35-cpu-native-placement-2026-10-24.json)。这些功能检查尚未刷新正式性能对照。

命令行保持原 CPU 线程默认值，不增加线程参数。程序集成可复用公开的
`compute::cpu::DefaultContextOptions`；allocator／executor 支撑对象须活到全部派生制品和原生工作退休，
具体组合见 [CPU executor 示例](../../samples/package/graph/cpu_executor/README.md)和[架构说明](../architecture/inference-design.md)。

多轮 schedule、会话数量、同时活动数、缓存预算、trace 与数值限制见[模型运行说明](../../engine/models/qwen35/README.md)。
当前多会话以共享模型、计划和有界执行资源组织，不按用户复制模型；缓存数量和实际批次宽度是独立配置。
普通与 MTP 共用 `prompt/generate` 自然请求文件和多轮／多会话生命周期，用 `--mtp off|on` 切换。
每轮 `prompt` 文件只提供新增用户 token；后续轮可使用空文件继续生成，消费者会恰好消费一次上一轮的末尾 pending。首轮 prompt 必须非空。
普通模式也保留显式 `prompt/decode` 固定输入重放；同一会话清单不能混用两种语法。
自然生成统一输出 `sessions[].result`，单请求为 `sessions[0].result`；其中 `generated_ids` 是实际生成 IDs。
自然续文可能随合法计算形状发生差异，不能把 ID 一致当完整任务质量保证。

## 查看性能

普通与 MTP 分开对照，报告 prefill、decode、内存／显存以及真实多轮综合速度，加载时间另列。
最新正式窗口、计数边界和已知热点统一保留在[decode 性能资料](../developer/performance/decode.md)，
不以模型改名、组件测试通过或减少 kernel 数量推定端到端提速。
