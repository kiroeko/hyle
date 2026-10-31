# Qwen3.5-2B 模型实现与诊断

这是 `hyle` 程序当前使用的真实文本模型实现，默认关闭 MTP，尚未接视觉。
当前仍以真实模型检验底层支持，使用本地权重清单和 token IDs；通用目录 loader、C++ tokenizer、
聊天服务及稳定引擎应用 API 留待后续。第一次构建、准备输入与运行见[引擎指南](../../../docs/user/inference.md)。
模块用户仍可独立使用 Compute／Graph；模型、生成政策和请求编排归 `engine/`。

官方 BF16 与选定量化制品的整模型／组件检查范围见[推理工程范围](../../../docs/implementation/inference-roadmap.md)，
不能从某个有限旅程外推全部模型、量化、平台或任务质量。最新速度与内存对照统一见
[decode 性能资料](../../../docs/developer/performance/decode.md)，本页不重复旧性能窗口数字。

## 输入与数值政策

`prepare.py --model LOCAL_MODEL --output NEW_DIR` 默认扫描本地 `config.json`、safetensors 索引及
张量头，校验实际角色、类型、形状和文件范围，对使用的原权重范围计算 hash，输出偏移与 token IDs。
`--role-files` 使用固定 repository／revision 与角色 hash 的显式原文件记录，GGUF 使用
`--gguf-role-files`。准备器不下载或复制整模型；原权重文件必须持续可访问。
本地 tokenizer 由已安装的兼容 Transformers 调用，C++ 初始化脚本不安装该 Python 环境。

官方 BF16 输入包含 320 个文本张量；`prepare.py --mtp` 另含 15 个原 MTP 张量。
完整清单可用 `--mtp off` 只加载文本权重，或用 `--mtp on` 执行草拟与验证。
量化制品按实际物理角色计数，不能套用 BF16 张量数。FP8／量化 reader 与 KV 政策说明见
[输入准备](../../../docs/user/inference.md#准备本地输入并运行)。target E4 KV 可与原 plain／BF16 draft KV 组合，
二者独立描述、计费与追加；不推造 draft 校准参数，有限检查见[E4 MTP](../../../docs/evidence/correctness/inference-mtp-e4-2026-10-28.json)。

官方 BF16 的 target 与 draft 末分数使用 `dense-linear-f32`：原 BF16 权重与激活得到 F32 scores，
中间线性输出仍为 BF16；实际输入的 `score_dtype`／`score_policy` 写入 JSON。
形状、批宽、分块及合法归约可引起数值变化，同 seed 不保证普通／MTP或不同后端逐 token 相同。
完整状态、质量与性能分别检查；历史逐字节结果只覆盖其记录中的精度与分块。

普通与 MTP 请求都支持固定的逐请求 bias／mask、历史罚分与逐假设前缀条件提供者。
这些是 `engine/` 内的 C++ 请求扩展，目前没有对应 CLI 参数或安装后的引擎 SDK／server API。
提供者取得当前 turn 已提交输出与未来草稿前缀；pending 已在前者，后者不重复它。
q 与 p 复用同一不可变条件行；被取消或失败的候选不发布新的 head、账簿或 RNG。
使用范围与有限检查见[采样说明](../../../docs/user/inference.md)，严格决策回放仍不凭空恢复外部 bias／mask 载荷。

## 自然生成、批次与状态

普通与 MTP 共用 `--schedule FILE`／`--session-schedules FILE` 自然请求循环；
每轮交替写 `prompt IDSFILE` 和 `generate N`，N 包含 prefill 首 token。
只有一行 prompt 时可用 `--steps N`，显式 generate 与显式 steps 互斥。
下一轮 prompt 只含新增用户输入，可以为空；首轮必须非空。上一轮的末尾 pending 恰好补入一次，
不重复返回，也不以恢复时的新预测替换历史输出。MTP 不接受固定 `decode IDSFILE` 输入带。

```text
prompt first.ids
generate 64
prompt next-user.ids
generate 64
```

```text
hyle --inputs MTP_DIR --mtp on --session-schedules sessions.txt --capacity 1024
```

Model 共享权重、冷图族及执行资源，每个 Session 保留自己的完整 KV、conv／GDN、位置、pending 与 RNG；
MTP 另保 draft KV、frontier／lag／catchup 和 carry／prior hidden。N 是可保留完整版本数，R 是活动序列上限，
二者不表示复制 N 份模型或同时发射 R 个独立图。初始 prompt 和轮次检查点可进入有界 Radix 目录，
键只覆盖实际已消费 token 与模型／数学身份；树的分叉不凭空生成状态。
完整命中跳过主模型计算；无 modifier 的默认 greedy 可直接返回已保存 ID，其余需重新选择的政策走 frontier 采样。
恢复同一请求保留历史 pending／RNG，decode 不逐 token 查树。

普通 Phase 使用一个槽，已接通的设备后继 Phase 使用三个固定槽轮换，最多保留两个有序在途批次。
CUDA greedy 普通自然生成以及单活动请求的 K1 MTP 可自动接力；其它采样、MTP宽度、后端、混合执行、
资源不足或不满足当前绑定条件时沿原单步路径推进。每槽有自己的 IO／状态事务，旧读者和原生工作退休前不能覆盖。
后继仍由原 Model／Graph 的提交、完成、状态发布与退休机制管理，没有额外执行器；控制由调用方串行化。
实际接力条件与冷准备选项见[用户指南](../../../docs/user/inference.md#准备本地输入并运行)。

有限 prefill 按真实有效行推进，例如 P154／量子64是 `64+64+26`。
兼容的 prefill／Verify 可在有相应 Provider 能力时使用容量载体合批；私有载体可以大于有效行，
KV、递归状态、位置和采样只提交各成员实际行。其它组合按 exact 形状分组。
`batches.input_rows` 是各成员真实目标行数，`active_rows` 是总和，`physical_rows` 是载体容量；
Verify 提案截短不按规划上界计数，接受部分前缀也不抹掉已执行行。

已知形状与后端选择在冷路径规划，首 prefill 和普通已知 decode 的 T1 载体在 ready 前准备。
`--warmup-shapes planned` 可准备已有计划的其它载体；默认按需准备，未列 exact 形状仍可进入有界冷接纳。
首次编译／载体准备／capture 都按实际发生的启动或请求边界计时，不把按需准备写成稳态热路径路由。
主 Phase、NextN、frontier 和接受采样沿共同程序载体持有原 Invocation／Completion／资源租约。

成功后先收齐结果和页事务、核 expected base，再发布完整状态；失败保留旧 head，按真实退休归还资源。
普通／MTP已提交多成员批次有幸存者时，只丢弃停止成员的发布；幸存者使用原批次结果，不重算。
停止成员保留旧head／ledger／RNG；单成员、全部停止、提交前停止或明确abort仍中止整episode。
取消不是GPU抢占，原Store／Append及Completion仍须整束真实退休，丢弃Completion不表示资源可写。
空闲 Session 可导出完整 HostImage 后退休／恢复；缺少可用快照时按账簿中已消费输入重建，
保持历史已返回 token 和 RNG，不重新采样过去输出。带外部 modifier 的严格决策回放仍需其原载荷。
完整条件控制的有限范围见[控制检查](../../../docs/evidence/correctness/inference-natural-logits-control-2026-10-29.json)。

自然生成结果在 `sessions[].result`，单请求也读取 `sessions[0].result.generated_ids`。
prefill／decode 按实际 episode 的不重叠 host 端点区间计时，和为 `session_seconds`；
末端日志与退休尾成本另列 `post_endpoint_seconds`。流水 episode 的 `step_seconds` 可以重叠，不能相加替代真实 wall。
`logical_user_prompt_tokens` 只计新增用户输入，`pending_flushed_tokens` 计多轮 pending，二者合为逻辑输入；
`reused_tokens` 与 `restored_*` 分别记录复用及恢复，`actual_target_rows`／`actual_draft_rows` 记录真实计算。
每请求延迟和参与次数可以重叠，不能相加当作物理调用或总耗时；含缓存的有效输入率也不等于相同计算量的算子速度。

## 构建与参数

普通源码构建使用[三脚本入口](../../../docs/developer/setup/README.md)，默认组件为 `engine`，
只要基础库时显式选择所需产品。也可对已安装 SDK 单独构建此引擎：

```text
cmake -S engine -B .scratch/qwen35-build -DCMAKE_BUILD_TYPE=Release -DHyle_DIR="<SDK>/lib/cmake/Hyle"
cmake --build .scratch/qwen35-build --config Release --target hyle_engine
```

该 SDK 须有 `compute;graph;graph_progress;compute_stream;mem` 及所需后端，应用只查包、链接与暂存运行资源。
后端选择、默认普通生成、采样、MTP 和 CPU／GPU 分置的常用参数统一见
[引擎指南](../../../docs/user/inference.md#常用选项)；构建后端与运行时名称的对应见
[构建说明](../../../docs/user/inference.md#构建)。Model／Session 属于引擎私有实现；
基础库 SDK 不包含 Model／Session 等引擎应用接口。

普通与 MTP 都在执行前校验完整旅程：prompt、可选 continuation 和后续最多 `steps−1` 行输入
须容纳于 `--capacity`；`--steps` 包含 prefill 首 token，自然生成遇 EOS 提前停止。

### 输入、会话与诊断参数

常用参数可运行 `hyle --help` 查看；以下补充模型特有的文件、资源与诊断合同。

- `--continuation IDS`：普通非schedule模式在prefill后追加给定token IDs；不适用于MTP。`--steps N`是最多输出数，包含prefill首token。
- `--schedule FILE`：普通／MTP均支持prompt／generate自然多轮；一行prompt可配合`--steps`，显式generate与显式steps互斥。普通模式另支持prompt／decode固定重放，与显式steps互斥；MTP拒绝固定重放。schedule与continuation／decode-ids互斥。先检查完整语法，再读取所有ID并校验会话容量，最后创建Model。
- `--session-schedules FILE`：执行多会话manifest，与`--schedule`互斥；普通／MTP的prompt-only自然请求可配合`--steps`。同一manifest的所有会话必须使用一致的自然生成或固定重放语法，在读取token文件前检查。每行`SESSION_ID SCHEDULE_FILE`，路径相对manifest，支持双引号和整行`#`注释；Windows清单中的路径使用`/`，例如`user1 "C:/inputs/user one.txt"`，避免双引号中的反斜杠被当成转义。ID为ASCII字母／数字开头，后续可含`_`／`-`且不能重复。`--capacity`是每会话上下文上限，所有会话和ID均先校验再加载模型。
- `--max-sessions auto|N`：保留的设备缓存会话上限；默认auto取本次输入会话数。普通文本和MTP都可淘汰空闲head；MTP生成中的head不淘汰，轮次间可淘汰并在下轮恢复。共享backing按解析后的容量预留，不为各会话复制权重或执行载体。
- `--active-sequences auto|R`：活跃请求和一次真实Graph批次的序列上限，批宽不超过R；不是同时提交R个独立Invocation。自然请求按角色、完成方式、采样和可用容量组合，支持有效行的载体可合批不同长度，其余按exact形状分组。默认auto随输入、缓存数量和预算解析，满足`1≤R≤N`；显式值受输入数量和缓存容量约束。
- `--cache-bytes auto|N`：KV页池、状态池及保留原生结果backing的请求字节额度。CUDA的auto按实际N／R、草稿候选边界及在途深度规划；显式额度不足时缩小可驻留数量，不能容纳活动请求则在加载前报错。CPU／Apple／Vulkan的auto不设字节额度，仍受N／R和版本数约束。显式额度包含各自实际资源计费，原生结果在完成后、发布前准入；失败保留旧快照。淘汰归还页／版本，仍有读者和Model池的材料继续计费。该参数不是全进程或物理显存上限。
- `--prefix-cache off|auto`：多会话manifest默认auto，单会话schedule及普通生成默认off；显式auto须使用schedule。在prompt／轮次恢复边界借用同Model和数学政策下真实保存的完整Snapshot，再计算后缀。完整prompt命中跳过主模型；无modifier的默认greedy可直接使用已保存ID，其它须重新选择的政策走frontier采样。恢复同一请求保留历史pending／RNG。目录最多N项，与会话head共用N份完整版本预算，在途候选材料另按实际深度准备；decode直接使用Session。
- `--progress direct|scheduler`：默认direct；scheduler在原槽上准备进度驱动，复用同一提交机制。
- `--replay auto|commands|prefer-capture`：默认auto，按冷计划中各批宽／child／槽的使用上界选择；一次使用走Commands，预计复用才PreferCapture。辅助NextN没有可靠使用次数提示，Auto仍按一次处理。当前主Phase有随机采样或history罚分时使用Commands，NextN带head的history分支亦使用Commands；接受采样和frontier程序当前显式使用Commands。其它满足条件的分支可把PreferCapture交给原Graph，捕获范围仍由原生能力和实际绑定决定。`planned_replay`记录冷选择，逐步日志记录实际录制／复用；用户不逐步选择Provider或另设流水开关。
- `--decode-ids FILE`：固定单token输入带，允许跨过预测EOS；若同时给steps，须等于输入数加一。正常生成遇到endoftext或im_end停止。
- `--trace DIR --trace-level decisions`：仅导出logits与最后hidden；默认full还导出conv/GDN状态与`kv-layout.json`。分页帧不公开原始bank载荷，后者记录实际页布局，不再导出dense KV文件。路由写入DIR/routes，不增加图输出。
- `--layer0-probe DIR`：诊断时另行发布layer0边界并比较实际路由；诊断图会影响合法融合，因此与普通图分别冷准备，均计入启动时间。多会话的边界输出按`session<序号>_<ID>`子目录隔离，不覆盖其他会话；共享图的路由说明仍在DIR。
- `--preparation-diagnostics 1`：仅普通greedy的固定 `prompt/decode` schedule，不支持自然生成或MTP；分开记录layout、batch和replay准备时间，注明实际槽及其首用，并读取六份公开BatchPlanReport。记录冷预分配，报告在原host输出端点后读取、最终端点后打印；标记为diagnostic，不用于吞吐对比。默认关闭，不增加这些报告查询。

各执行模式的JSON均报告同一`cache_budget_scope`、固定池`cache_shared_backing_bytes`、当前额外保留结果`cache_retained_output_backing_bytes`及额度内live／peak；未设字节额度时`cache_requested_byte_limit`为null。它们是同一个Model的全局统计，多会话子结果中的值不能相加。`native_execution_private_bytes_unknown`单列原生执行私有成本未知，`physical_vram_limit_known=false`；这些预算统计不能替代实际内存／显存旁采。

Apple／Vulkan提供按需构建的`hyle_qwen35_retained_budget_probe`目标，使用`--inputs <已准备的MTP输入> --out <尚不存在的结果目录>`。它在真实计算完成后施加纯额度压力，核旧完整状态、失败回收及原槽重试；不分配对应压力的设备内存，不用于测速。实际命令与结果见[保留状态预算记录](../../../docs/evidence/correctness/qwen-retained-budget-2026-10-22.json)。

普通文本和MTP均已接真实多请求批次。[本批MTP证据](../../../docs/evidence/correctness/qwen-mtp-batch-2026-10-22.json)包含B2的不同接受边界、lag1新输入、整批取消／原槽重试、有界重建及0Graph完整命中。R1／R2续文确有分歧，跨批宽质量未收口；本批为诊断运行，不作吞吐或显存收益结论。

## 固定输入重放（诊断）

本节描述普通 `prompt/decode` 固定输入带及其日志；自然生成的有效行合批与后继执行见上文。

普通模式的固定重放manifest每行是`prompt IDSFILE`或`decode IDSFILE`，须从prompt开始并交替；Qwen消费者允许最后一轮仅有prompt，表示只取首token。共享parser的默认完整prompt／decode合同不变。
ID文件路径相对manifest所在目录；含空格的路径用双引号，ID文件使用空白分隔的整数。例如：

```text
prompt prompt1.ids
decode decode1.ids
prompt prompt2.ids
decode decode2.ids
```

```text
hyle --inputs DIR --schedule session/manifest.txt --capacity 1024
```

每个prompt块产生一个逻辑结果，每个decode ID产生一次单token逻辑调用，固定输入允许跨过预测EOS；同波次的兼容调用可以合成一次真实Graph批次。
后续prompt沿同一个Session及原KV、conv/GDN状态追加，不重新加载或重置缓存。
图族按所有会话的shape并集冷编译，Model按有序调用冷统计每个形状的串行槽使用上界，预选Commands或PreferCapture。完整prompt命中跳过主模型计算和预热；需要重新选择输出的政策仍执行frontier采样Graph。切换会话不复制载体。准备诊断的prior-use计数属于共享执行槽，跨会话不清零，失败后重建该槽才清零。

普通随机生成中，同策略、无logits修改且不执行top-k／top-p过滤的多请求批次，直接沿原Categorical算子批量采样；每行保留原请求的随机身份和draw位置。过滤策略继续按原逐行路径执行，用户无需另设优化开关。对应有限对照保留在[采样记录](../../../docs/evidence/performance/inference-random-four-request-screen-2026-10-26.json)，当前性能以[最新对照](../../../docs/developer/performance/decode.md)为准。
日志标签为`round1_prefill`、`round1_decode1`等；每个阶段另有
`memory_phase=roundX_prefill_begin/end`或`roundX_decode_begin/end`标记，供独立显存旁采定位，不代表消费者已测得显存数值。

仅schedule模式的stdout追加`phase_count`、`turn_count`、`processed_tokens`、`generated_tokens`和`phases`摘要。
`processed_tokens`是逻辑输入总数；`generated_tokens`是每个prompt末尾返回的一枚首token，加上每次decode返回的一枚token，包括完整缓存命中的返回。顶层`actual_graph_calls`／`actual_graph_tokens`只计一次真实Graph工作，包含淘汰后的恢复；一次R请求批次计一次调用、R×T个输入token。每Session的`graph_participations`／`graph_input_tokens`记录它参与的批次与输入，不能相加参与次数当成Graph调用数。`reused_tokens`只计覆盖的新增逻辑prompt输入，恢复原边界的等长命中为0。`prefix_hits`／`prefix_misses`／`prefix_full_hits`和目录淘汰数另列。
每phase记录上下文范围、完整阶段`seconds`及原逐调用`step_sum_seconds`。
schedule模式的逐调用记录和标签在冷端预分配，执行时只填原计时、capture和token字段；最后一个host端点之后，才按原两行格式顺序打印日志。普通生成及MTP仍保留原逐调用stderr，完整decode计时包含调用间日志和调度。
`session_seconds`从首阶段输入准备前至最后token在host可见，保留实时阶段标记和控制间隙；
`startup_total_seconds`还包括输入解析、权重加载和冷图准备。最后调用端点之后的日志不计入这两个时间。
首用layout准备、capture和实例化均留在原完整调用计时内，不能只报重放段或平均各阶段TPS；
单会话综合有效输出率使用`generated_tokens / session_seconds`；另列`processed_tokens / session_seconds`为总输入处理率。批次decode率按decode波次的实际输出数除以对应波次wall另报，不能平均各阶段TPS。
trace或layer0 probe会标记`diagnostic=true`，其中间导出可能拖长会话时间，不作为速度窗口。
未给schedule时保留原CLI结果和日志格式。

多会话manifest例如：

```text
alice alice/manifest.txt
bob bob/manifest.txt
```

```text
hyle --inputs DIR --session-schedules sessions.txt --capacity 1024
```

一个prompt／decode完整turn是一份活跃请求；按活跃上限组成cohort，每个波次最多取各会话的一个下一调用。先完成该波次全部前缀查找／恢复，再按实际T和prompt／decode种类分桶，不padding、不将下一turn越过当前依赖。每桶使用一次原Invocation；所有桶完成后才发布本波次检查点，同波次调用不消费彼此的新检查点。turn完成后会话变空闲。这是一个Queue上顺序执行的真实批次，不是连续batch服务。
stdout的`execution`为`equal_T_batched`。`logical_dispatch_session_ids`记录逻辑调度，`batches`记录实际批宽、T、参与Session及`restore`标志，`waves`记录每波逻辑／实际token、输出数、Graph调用和wall；全局调用数只在顶层出现一次。`sessions`各自记录状态和阶段。`trace`按`session0_alice`等子目录隔离，路由仍写入`DIR/routes`。
各会话／phase的wall及逐调用累计包含等待同批／其它桶的时间，彼此会重叠；不能相加Session的wall、`step_sum_seconds`或TPS作为批次速度。总输出率用全部输出数除以`interleaved_seconds`，prefill／decode分项使用对应波次wall，含启动比较使用`startup_total_seconds`。
off模式已知shape／slot及首组状态的准备计入`shape_prepare_seconds`，后续接纳／恢复计入请求wall；已有Model载体不重复创建，模型加载桶不包含状态分配。auto模式的lookup／接纳／首次准备／恢复均留在请求wall。`cache_prepare_seconds`在off包含接纳及必要恢复，在auto只计目录准入，不能当成全部准备。跨模式比较须保冷准备边界一致，或使用含全部启动工作的端到端时间。
缓存淘汰后按原token tape和实际chunk记录重算，不读取参考状态、不重新采样或改写已经返回的输出，也不把`128＋1＋suffix`合成一次大prefill。`restored_calls`／`restored_tokens`和`restore:true`批次单列额外工作，不计作逻辑新输出。
`restore_contract`为`recompute_original_inputs`：合法批宽／归约变化可能改变重算的hidden、logits和greedy ID，恢复不是精确导入旧状态。`recomputed_decision_changes`保留旧／新决策；当前head保留真实重算的next ID。前缀模式另建ReplayHistory记录这次实际next，不修改旧Snapshot或共享历史，后续检查点的head与重算记录一致。分块记录是消费者的重算依据，不是通用前缀key，也不代表跨批宽数值质量已验收。诊断恢复向量写入`trace/restores`，应与原逻辑调用的向量数分开统计。
前缀匹配和接纳／恢复／suffix槽准备计入请求wall。完整命中的trace仅写`cache-hit.json`并标记`tensor_payload_available=false`，不导出未经执行的hidden／logits；诊断应使用新目录，不把旧文件当成本次输出。所需suffix形状从本次有限输入中的真实checkpoint匹配推导，在加载后的冷图族中一次准备。图族的算子候选和制品准备预算按请求变体数有界计算，日志列实际额度和使用数；不需要用户填写内部搜索上限。
`cache_hits`／`cache_misses`、`evicted_session_ids`和live／peak请求字节描述实际缓存行为。目录仅在请求边界使用，decode直达已pin的Session。
共享Model与2／4会话隔离的[原回归](../../../docs/evidence/correctness/qwen-multi-session-isolation-2026-10-21.json)和dense状态容量限制的[三组检查](../../../docs/evidence/correctness/qwen-session-capacity-2026-10-21.json)已通过。完整分页快照后的[单会话与四会话淘汰重建](../../../docs/evidence/correctness/qwen-paged-snapshot-2026-10-21.json)保留18／72份决策向量对照。[实际前缀／分支与有界淘汰恢复](../../../docs/evidence/correctness/qwen-prefix-reuse-2026-10-21.json)另核完整命中0Graph、T1／T16两条分支和之后的原会话续写。[CUDA在途回收](../../../docs/evidence/correctness/qwen-native-retirement-2026-10-21.json)使用真实队列阻塞，核取消／丢弃Completion不提前返还资源，实际退休后才能重新接纳。独占尾页与Radix的本批结果见下；其它原生后端的完整消费者及并发吞吐仍待补。

当前前缀目录使用有界压缩Radix树，按实际token与模型／数学身份查询完整快照，结构分叉不自动生成检查点。
lookup只发生在请求边界；无modifier的默认greedy完整命中可为0Graph，其它需要重新选择输出的政策另走frontier采样，普通decode不逐token查树。目录及LRU在初始化时预留容量，活跃命中可同时只读持有。
Session独占自己的尾权，同页续写只写未提交后缀；前缀命中不会复制尾权，分支沿原COW路径，旧快照范围始终不变。
[本批实现、回归与性能](../../../docs/evidence/performance/qwen-exclusive-radix-2026-10-21.json)不代表其它原生后端或GPU并发吞吐已交付。

## 专项诊断与历史依据

下列日期记录保留当时冻结代码、命令与结果；其数值、状态布局及未运行项不自动更新为当前版本的结论。

`hyle_qwen35_retirement_probe`与`hyle_qwen35_mtp_session_probe`已登记CUDA／Apple／Vulkan构建；原生会话探针当前仅开放`--scenario prefix`，退休探针的三平台有限T2检查已通过列明范围。`hyle_qwen35_batch_probe`仍仅在CUDA SDK下提供。普通模型数学与状态通过公共Compute／Graph接口执行；这些按需诊断可借用原生私有状态来检查在途寿命，结果按各证据实际后端分别说明。
CUDA 测试目标 `hyle_qwen_request_lifecycle_test INPUT_DIR` 使用原 P154 BF16 输入检查普通随机 R2 的取消、存活请求续跑、有界输出及增量续轮；不用于测速。也可设置 `HYLE_QWEN_REQUEST_INPUTS` 后运行同名 CTest；未提供输入时跳过，显式不存在的目录按失败处理。[原生结果](../../../docs/evidence/correctness/qwen-live-request-lifecycle-2026-10-25.json)来自归档消费者。正式入口严格编译、链接及主机错误合并检查已通过；新增 `--retirement-status-only` 仅检查主机状态分类，未注入设备错误，正式入口未重跑 GPU。

同一目标的 `--chunked-prefill [INPUT_DIR]` 检查普通 P154／Q64 的无输出中间块、decode 穿插、部分恢复及取消重试；对应 CTest 为 `hyle_qwen_chunked_prefill_test`，输入规则相同。[已测结果](../../../docs/evidence/correctness/qwen-ordinary-chunked-prefill-2026-10-25.json)来自冻结消费者；正式入口仅完成严格语法检查，未再运行 GPU，也不用于吞吐对比。
`--dual-model [INPUT_DIR]`／CTest `hyle_qwen_dual_model_test` 检查同卡两实例的独立退休与销毁，[原 CUDA 过程已通过](../../../docs/evidence/correctness/qwen-dual-model-lifetime-2026-10-25.json)；正式入口随后通过新精度／暂存组合的严格语法检查，未重跑 GPU。它会加载两份原文本权重，运行前须预留约两倍权重空间，不能解释为物理 context 隔离或多请求吞吐测试。
退休检查独立于默认构建，可显式构建`hyle_qwen35_retirement_probe`，运行时传`--inputs DIR --decode-ids IDSFILE`；使用原P128输入及至少两个decode ID，不复制权重。它检验原执行链的寿命，不用于测速。
探针当前覆盖同页独占后缀取消：实际退休前保留资源，退休后旧快照仍持页，所有旧读者释放后才可重新取得整页。
该目标另可用`--inputs MTP_DIR --mtp on [--verify-rows 1|2]`检查真实MTP取消，和`--decode-ids`互斥。默认2行验证；1行分支实际覆盖没有attention读者的草拟KV写effect。两种分支均已实跑：使用自然接受后的lag1前缀，取消并丢弃Completion后保留全部14份KV写事务、状态、候选和执行借用；真实退休后核旧字节及同槽重建。另核成功terminal后的接受策略异常不会发布head，回收后原完整前缀仍可续算。这是有限取消／host异常检查，没有注入GPU设备错误，也不用于测速。

请求backing按实际geometry、target／draft各自的K/V行字节、上下文C、驻留版本N、活动序列R、
验证边界K和在途深度D计算，见[当前容量实现](cache_plan.h)。页容量包括每个在途序列的COW尾页，
各候选前缀保留独立递归／hidden材料；N份完整版本额度不因候选空间增加而扩大。
真正能释放的旧base才可抵扣，目录别名或旧读者仍pin住的版本继续计费，无法容纳时在提交前拒绝。
CPU元数据、Apple原生生成backing及后端私有执行占用另有实际计费／未知边界，不能把旧CUDA公式当作总显存上限。

独立目标`hyle_qwen35_mtp_session_probe`运行时传`--inputs MTP_DIR --output NEW_DIR`；默认使用原23-token输入的前15／23枚，在同一N2／R1 Model上先独立执行、退休，再交错自然生成。[原检查](../../../docs/evidence/correctness/qwen-mtp-sessions-2026-10-22.json)的378份向量逐字节一致，只对应当时的R1范围。追加`--scenario batch`检查真实B2、接受边界[2,1]、取消／原槽重试、已提交输入重建及独立的非argmax pending政策；本批14次Graph中，同分块重建54项逐字节一致，跨B的108项有4项draft K/V数值不同。完整质量、其它原生后端及吞吐仍须分别检查。

追加`--scenario numerics`可检查[同旧状态的批宽控制](../../../docs/evidence/correctness/qwen-mtp-batch-numerics-2026-10-22.json)：该历史BF16末分数检查中，R1和真实B2分别完成P15，再各用自然R1轮次到L18；控制Session借用同一个完整旧Snapshot，分别执行R1／R2，不重建或复制历史。实际10次Graph、13组54项向量中，两份来源各自的目标状态／输出跨B逐字节一致，draft K/V另保真实差异；首异A随prefill来源走。若自然生成没到指定边界，报告跳过控制，不强制token。该历史结果不签当前F32末分数政策的质量或吞吐；本批未重新编译／运行这个独立探针。

追加`--scenario prefix`可检查[三平台MTP前缀与准入](../../../docs/evidence/correctness/qwen-native-protocol-2026-10-22.json)：完整命中比较52份状态，不伪造hidden／logits；真实续算比较54份向量。T1／T8均与相同输入分块的独立基线对照，534项逐字节一致，不能用分块结果签与整段prefill数值相同。误把真实teacher-forced T1交给生成finish时，在原Completion退休后、状态发布前拒绝；14份事务回收，旧head不变，同槽可正确重试。CLI另实跑N1／N2排队、字节缩容、0Graph完整命中和两种后缀；本批未测速。

独立`hyle_qwen35_batch_probe`用与CLI相同的Model批次接口执行真实B2／T1：普通线性层使用packed行，Conv／GDN用独立B维，历史通过普通Graph的Concat收集、结果Slice／Copy回各自输出租约，KV用公共`PagedArena::reframe`组合与投影。一次submit对应一个原Invocation／Completion，不是两次evaluate合并统计；不同T由CLI分桶，不能用会改变recurrent状态的虚假padding。运行传`--inputs DIR --output NEW_DIR`，使用原P128输入，不复制权重。探针比较独立B1同分块、旧状态、结果分流和继续推理，并检查实际队列阻塞后整批取消／丢弃Completion、两候选保留到真实退休、原R2槽重建及重试；B变化可改变合法矩阵归约，原始误差指标与greedy ID分别记录，未定义跨B完整模型质量阈值。状态打包／拆分成本须从实际批次的完整性能窗口读取。

CLI批次及同源R1对照的prefill、decode、总会话与采样显存见[本轮实测](../../../docs/evidence/performance/qwen-cli-batch-2026-10-22.json)。批次合计吞吐不代表单会话提速，跨批宽数值质量未验收。

加载沿现有FileResourcePipeline，用两个8MiB暂存读取原权重范围并上传：CUDA用原pinned／device资源，Apple接原生shared资源和文件管线。官方 BF16 文本选择320个原始张量，MTP另加15个，均保留原分配和持久目标租约；量化清单按实际物理角色加载。
完整Model／ready／首token计时包含文件关闭、管线退休和临时运行时销毁。新进程未清OS页缓存，不能称冷盘加载。
`step_seconds`包含该episode的输入上传、layout准备、绑定、replay准备、提交、等待及token读回；首次exact形状可能在启动或有界请求冷接纳时准备。trace导出不在该子段内，流水episode可重叠。
普通生成和MTP的`prefill_seconds`从实际prepare前计至head发布且token在host可见；`decode_seconds`从同一prefill端点计至最后成功finish端点，`session_seconds`是两段连续wall之和。`startup_total_seconds`（同`startup_to_endpoint_seconds`）含加载与冷准备；`step_sum_seconds`单列，不能替代完整elapsed。四个`generation_*`内存标记仅供旁采近似分段，不替代计时端点；trace运行标为diagnostic，不混入速度表。stdout将实际Graph调用、目标验证行、草拟行、已提交行和真实输出分别计账，接受token数不冒充计算量。
`seconds`和`submit_seconds`是更窄子段，不可直接与另一引擎的完整decode相比；完整启动至首token看`first_token_elapsed_seconds`。

当前完整窗口、计数边界和已知热点统一见[decode性能记录](../../../docs/developer/performance/decode.md)。
报告中的compiled storage保留unknown计数，不等于完整显存峰值；GPU旁采峰值是采样下界，
驻留集、private commit与GPU共享内存分开解释，mmap与文件缓存也影响进程归属。

`reference.py`是独立Torch参考，候选不会读取参考状态或结果。固定输入与向量相同属于回归检查，独立质量验收仍未完成；
BF16舍入、归约和RoPE诊断见[底座审查证据](../../../docs/evidence/correctness/inference-foundation-review-2026-10-18/README.md)。
历史输入与运行命令以所属证据为准；新使用者按引擎指南准备本地输入，源码构建不要求某个既有`.scratch`目录。
