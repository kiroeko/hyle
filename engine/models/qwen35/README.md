# Qwen3.5-2B 模型实现与诊断

这是Hyle正式推理入口当前使用的Qwen模型实现，使用本地官方权重执行完整24层纯文本计算，默认关闭MTP，尚未接视觉。Qwen BF16 greedy K1、多会话／多轮／B2及有界重建已在CUDA、Apple和Vulkan实际运行；[三平台单槽结果](../../../docs/evidence/correctness/qwen-serial-carrier-2026-10-22.json)覆盖同一N2／R2三会话两轮，[完整前缀检查](../../../docs/evidence/correctness/qwen-native-protocol-2026-10-22.json)每平台各534项向量与自身相同分块参考逐字节一致。既有Vulkan大权重展开、结果复用和冗余双槽引起的OOM已修复。[本轮有限取消／退休与Vulkan额度恢复](../../../docs/evidence/correctness/qwen-native-protocol-2026-10-22.json)已通过；Apple仅证明原生FIFO接纳Model Job，未证明该Model的GPU命令已提交。此前列明制品的有限MTP流程与四臂性能窗口已完成；它们不代表全部模型、量化、平台或完整任务质量通过。当前功能边界与用法见[引擎指南](../../../docs/user/inference.md)。
模型与生成策略属于引擎层，Compute／Graph保持模型无关。当前仍需本地输入准备，通用loader、C++ tokenizer和服务入口属于后续工作。
CPU入口的缓存元数据计费已修，普通／MTP P154＋4 短旅程通过，见[库存与有限预算记录](../../../docs/evidence/correctness/qwen-mtp-cpu-layout-budget-2026-10-24.json)。固定 GPTQ 制品也已完成 CPU／CUDA 普通与 MTP 四条短旅程，见[原量化完整模型记录](../../../docs/evidence/correctness/qwen-mtp-original-gptq-2026-10-24.json)；更多量化、采样与状态的已检查范围见[推理工程范围](../../../docs/implementation/inference-roadmap.md)，不据此声称完整质量或性能通过。

以下精度说明对应官方BF16权重；量化制品以JSON中的实际`score_policy`为准。target与MTP draft末分数显式使用公共`dense-linear-f32`合同：原BF16权重和BF16激活直接得到F32 scores，中间linear仍为BF16。`score_dtype`／`score_policy`写入JSON，区别于旧BF16投影后再Cast(F32)的舍入；同一Model的缓存只属于其固定数学政策。CUDA默认路由已接cuBLASLt混合输出实现，权重只改描述符方向，消费者无需判断Provider。[本轮数值与完整窗口](../../../docs/evidence/performance/qwen-cublas-scores-2026-10-22.json)保留同一F32合同：普通固定输入的hidden／分数字节相同，P154／64自然输出窗prefill中位＋15.48%、综合＋1.82%，decode范围重叠、近持平；dedicated单进程旁采峰少108MiB。11项相关测试、真实双会话追赶和N1重建通过；这些不是完整模型质量或其它系统通过。下文历史逐字节结果仅对应原证据中的精度和分块。

`prepare.py` 校验原输入清单的13个文件，输出权重偏移和token IDs；Python只做输入准备与独立参考。
官方 BF16 输入默认准备320个文本张量。`prepare.py --mtp`另含15个原始MTP张量；同一完整清单可用`--mtp off`只加载文本权重，或用`--mtp on`执行草拟与验证，默认 greedy、K1。量化制品的物理角色数不同，JSON 按实际加载清单计数；权重沿同一管线加载，见[原始GPU字节检查](../../../docs/evidence/correctness/qwen-mtp-weights-2026-10-22.json)。
CPU／CUDA 已补有界多草稿、随机采样、置信阈值、完整及部分生成后检查点恢复，见[新增控制与检查点记录](../../../docs/evidence/correctness/qwen35-mtp-multidraft-checkpoint-controls-2026-10-24.json)。Apple／Vulkan 已完成 K1 普通／MTP、greedy／random，以及 [K4 随机和双用户 K2 随机](../../../docs/evidence/correctness/mtp-qwen35-native-multik-random-complete-input-2026-10-24.json)有限旅程。
MTP复用Recipe和原Graph执行链：prefill保存独立草拟KV；每轮设备I32边连接草拟ID与目标T2，收齐原Completion后选择接受边界，一次提交target／draft／carry。接受后落后一行时，下轮用真实token和prior hidden追赶。输出额度或剩余上下文只有一行时走目标T1，同时维护草拟KV；遇EOS停止。CUDA自然接受／拒绝、追赶、旧快照分支及T1的[实际结果](../../../docs/evidence/correctness/qwen-mtp-round-2026-10-22.json)已记录。[取消、独立CPU参考和MTP peer](../../../docs/evidence/correctness/qwen-mtp-retirement-2026-10-22.json)另核原生退休、完整前缀恢复和有限协议输出。[P154／64自然输出四臂窗口](../../../docs/evidence/performance/qwen-mtp-window-2026-10-22.json)已完成；peer MTP从第19枚输出起分歧。后续固定同state对照已证明T1／T2形状数值差足以翻转本次相近top2，所测pending恢复未见额外误差；不等于全部rollback或完整质量通过，也不作同token速度胜负。

Model串行提交一个Graph batch，每个实际准备的形状只保留一个执行槽；批宽和MTP接受边界不增加在途Graph数量。T1 family／计划保持冷编；自然普通生成已知要decode时，在ready前准备T1载体，其余未选中的载体按需创建。旧快照通过独立readonly owner保活，完整结果读取和原生退休前不归还执行槽；捕获仍按实际绑定身份失效。[三平台多会话与CUDA取消检查](../../../docs/evidence/correctness/qwen-serial-carrier-2026-10-22.json)已记录，本次改动尚未做完整性能对比。
普通与MTP共用自然生成循环：`--session-schedules FILE`运行多个会话，`--schedule FILE`运行单个会话，用`--mtp off|on`切换。普通路径只加载文本权重并使用Text图，不准备draft资源。自然多轮schedule交替写`prompt IDSFILE`和`generate N`，N为本轮最多输出数，包含prefill首token。仅一行prompt仍可配合`--steps N`，显式generate与显式steps互斥。MTP拒绝旧`decode IDSFILE`固定输入带，不将其解释为自然生成。`--active-sequences auto|R`按输入、缓存数量及预算解析；每波按实际T、角色及draft lag分桶，每桶用同一Model／Graph执行一个真实批次，不padding。各成员独立选择接受边界，全部expected base核对后一次提交所有head；取消任一成员则取消整批。
驻留数可小于会话数：生成中的head受到保护，轮次间的空闲head按LRU淘汰，实际退休后才准入下一条。CPU账本保存实际已消费输入块及生成观察；正常恢复沿这些输入用原Graph重算，将draft补到target前缀并归一为lag0，不重新采样历史。接受两行时消费的是旧pending和当时接受的draft，bonus仍未消费。完整恢复成功才公开新Session；真实新预测、`recomputed_decision_changes`与`normalized_lag_rows`另列，允许合法数值变化。严格复现原R1角色／分块的旧恢复入口仅保留作诊断，不用于正常缓存恢复。
下一轮prompt只提供新增用户输入，也可指向空IDS文件表示继续生成；首轮仍须非空。消费者把上一轮已返回但尚未写入KV的末尾token补在前面，恰好消费一次，不重复返回；该token由引擎输出记录保存，不能用恢复后的新greedy预测替换。draft落后一行时在同一Graph里追赶。因此新输入U行对应target的U＋1行，lag1时draft再多一行。EOS只结束当前生成轮次。
多会话manifest默认`--prefix-cache auto`，单schedule默认off。初始prompt与完整生成轮次均可发布检查点，键只包含实际已消费的输入；快照保留target／draft KV、conv/GDN、carry、位置、pending和draft lag。只有真实保存的状态边界可以命中，radix分支不凭空制造中间状态；部分命中只计算实际后缀。完整prompt命中跳过target／draft计算：greedy执行0次Graph，random按当前请求RNG执行1次frontier采样Graph。恢复同一请求的完整生成检查点时保留它自己的RNG和pending，不重新抽样；未命中则沿已提交账本重建。目录与会话head共用N份版本预算，压力下先撤掉可淘汰的目录引用，N1续算不额外保留旧检查点。多轮自然生成使用schedule；`--continuation`、固定decode带、准备诊断和layer0 probe不适用于MTP入口。
`--trace`按`prefill`、`mtp1`等轮次导出实际目标候选hidden／logits和已选状态，T2保留两行；多会话按`session<序号>_<ID>`子目录隔离。接受决策另列在stdout的`rounds`，不能把被拒绝行当成已提交状态。

```text
hyle --inputs MTP_DIR --mtp on --session-schedules sessions.txt --steps 64 --capacity 1024
```

自然多轮的schedule示例（此时命令行省略显式`--steps`）：

```text
prompt first.ids
generate 64
prompt next-user.ids
generate 64
```

普通／MTP自然生成统一在`session_count`、`dispatch`及`sessions[].result`中报告；单请求也读取`sessions[0].result.generated_ids`，不保留旧扁平结果。排队后prefill与decode会交错；顶层`phase_timing=disjoint_continuous_episode_intervals`按每个实际episode分类累计，从上次host端点至本次token可见，包含lookup、准入及协调间隙，P＋D等于一次`session_seconds`。最后端点后的日志、目录及退休尾成本另列`post_endpoint_seconds`。每会话延迟包含等待其它会话的时间，不能相加作为总耗时。单请求也沿同一循环和计时协议；首prefill及普通已知decode的T1载体在ready前准备，后续会话准入及其它载体首用计入真实请求时间。
`logical_user_prompt_tokens`计新用户输入，`pending_flushed_tokens`计多轮补入的末尾token；二者合为`logical_prompt_tokens`。`reused_tokens`另计前缀复用，`prefill_target_rows`／`actual_target_rows`／`actual_draft_rows`只计真实计算。`restored_*`单列恢复工作，并计入actual工作量，不增加逻辑输入／输出；恢复位于下一轮prefill wall内。全局`actual_graph_calls`每桶只计一次，不包含0Graph命中；每会话`graph_participations`计参与次数，不能相加当作物理调用数。每会话P／D是各轮延迟之和，轮间等待另列`between_turn_seconds`，三者合为该会话wall；不同会话延迟不可相加。`prefill_tokens_per_second`是含复用的有效输入率，不能拿它冒充相同计算量的算子速度。完整命中trace只有`cache-hit.json`，明确无本次hidden／logits载荷。
`main.cc` 用一份模型语义图表达prefill、带状态的多token continuation和decode，在启动时预列本次调用序列，
以有限shape图族一次冷编译。Model共享权重、图族、不可变child及按实际执行并发使用的Phase／IO／workspace；每个驻留Session保存完整KV／conv／GDN与位置，不各自复制执行载体。每个实际准备的child最多两个Invocation槽，共用IO所以整体独占借用；空闲会话不占槽。
自然请求冷编有限shape，提前准备首prefill及普通已知decode的T1载体；其余实际批宽／前缀suffix首次选中才在请求内创建载体。固定输入重放仍沿已知调用计划提前准备。只实例化已有编译child，不临时建图、搜索Provider、编译或padding；首次准备仍计入对应冷准备或请求及含启动总时间。
conv/GDN状态真实反馈，KV使用有界共享页池，只写新增行及COW尾页。普通文本Snapshot同时保留12份KV帧、36份conv/GDN读租约与位置；MTP另保2份草拟KV、frontier／lag／catchup token及目标carry／prior。收齐成功结果、取回append事务并撤回已退休状态绑定后，对全部参与会话检查expected base后，以无抛交换提交完整head。失败保留旧head并按真实退休回收候选，不发布部分状态。

六个BF16 GQA attention共用一份BatchSchema，每次执行共享同一owned layout与metadata输入。
schema按最大请求数R和每请求最大预列chunk设界，具体Q/output行数是当前批次的合计；各请求的KV可见长度由layout和positions分别表达。
例如23-token prefill→4-token continuation→单token decode使用同图族的23/4/1三个child；head深度等模型常数保持固定。
每槽冷准备IO与状态端口；每步更新token、position与layout，绑定精确页写事务和状态租约，沿原Invocation执行`graph::batch::prepare`→`prepare_replay`→submit→Completion，再取回结果与状态。
单槽借用不授权并发覆盖tokens、KV或状态；CUDA共享原Queue，Apple沿原生Runtime FIFO，控制须外部串行化。借用在写输入前取得，实际退休、状态提交及最后report／trace读取后才归还；失败只在真实退休后销毁本次Invocation，下次显式prepare重建。`evaluate`组合原有prepare／submit／finish；拆开调用可在真实工作在途时停止会话。Model拥有一个批次episode及参与Session的强引用，Session只借用pending链接。`stop`停止任一参与者时采用整批取消政策，关闭本次借用而不关闭Model池；其它会话保原head，退休后可以重试。`try_retire`只在原Completion及页权限退休门通过后清理本会话状态。目录retiring项不可查询或重新接纳，实际收集前继续计费。取消不表示GPU抢占，丢弃Completion也不表示页已可写。显式shutdown仍drain，所有Session须先于Model销毁；普通CLI与独立探针均使用同一批次episode，没有第二套执行器。

构建消费者的已安装SDK须包含`compute;graph;graph_progress;compute_stream;mem`产品及所需路由。SDK导出的能力给出可选集合，运行时用`--backend cpu|cuda|apple|vulkan`选择主后端；构建的`metal`路线使用运行时名称`apple`。默认选择唯一的GPU路线，没有GPU时选择CPU，同时编入多个GPU路线则要求显式选择；失败不自动回退。CUDA保留原Context／StorageOwner／设备内存／pinned暂存／Queue；Apple使用原生Context／共享内存／ExecutionSource，Vulkan使用原生Context／device与staging资源／ExecutionSource；原生路线不制造通用Queue。Backend只装配冷环境与外部I/O，Recipe和MTP数学共用。
CMake只查包和链接，不构建原生依赖，也不增加用户后端开关。示例：

```text
cmake -S engine -B .scratch/qwen35-build -DHyle_DIR="<SDK>/lib/cmake/Hyle"
cmake --build .scratch/qwen35-build --config Release --target hyle_engine
```

Apple生成的conv／GDN与MTP carry／prior沿原Graph结果租约保活，在选择接受边界后、取回任何页事务及发布整批快照前，按完整实际backing进入缓存预算。同一预算／资源内可证明共享的backing只计一次；失败沿原退休／回滚流程保留旧快照。结果槽、旧读者或在途工作仍持有租约时继续计费。MLX中间值／私有分配、权重、IO及驱动占用另计；原生计算可能先于保留准入发生，缓存额度不约束其瞬时分配峰值。下文CUDA性能和取消／批次结果不自动扩展到Apple。
`--inputs DIR`指向准备后的输入；可用选项：

- `--continuation IDS`：prefill后追加多token输入；`--steps N`指定生成阶段数。
- `--mtp on|off`：默认off；on需要`prepare.py --mtp`的清单。`--steps`仍为最多输出token数，包含prefill首token；每轮可输出接受的草稿和一枚目标token，不重复输出已返回的pending token。普通版和MTP均按完整旅程校验：prompt、continuation与后续最多`steps−1`行输入须放得进`--capacity`，不另设128枚输出上限；自然生成遇EOS提前停止。stdout分列实际验证行数、草拟写入行数、提交行数及自然agreement／accept数；短样例不代表普遍接受率或提速。
- `--mtp-max-drafts N`：须配合 `--mtp on`，默认1，0禁用草稿。实际草稿数还受上下文、输出额度和置信阈值约束；冷图按需要的验证宽度准备。
- `--mtp-min-probability P`：须配合 `--mtp on`，范围[0,1]、默认0。用实际草稿分布的最大概率决定是否继续，不另造置信度头；截短或无草稿时保留正确的目标续算和 RNG 位置。
- `--cpu-ffn-layers N`：分别选择 `--backends cpu,cuda`、`cpu,metal` 或 `cpu,vulkan` 构建，将前 N 层 target FFN 放在 CPU，其原权重直接加载到 host，不在 GPU 保留副本；注意力、缓存与 MTP 留在所选 GPU。默认0。CUDA 普通／MTP、多 K、随机、多会话和恢复见[CUDA 分置记录](../../../docs/evidence/correctness/qwen35-cpu-ffn-cuda-placement-2026-10-24.json)，Apple／Vulkan 随机 K1 见[原生混合记录](../../../docs/evidence/correctness/mtp-qwen35-cpu-native-placement-2026-10-24.json)。JSON 分列真实权重域、共享执行预算范围与每次执行的已知 workspace；原生私有开销未知，不能据此推定全进程或物理显存上限。
- `--schedule FILE`：普通／MTP均支持prompt／generate自然多轮；一行prompt可配合`--steps`，显式generate与显式steps互斥。普通模式另支持prompt／decode固定重放，与显式steps互斥；MTP拒绝固定重放。schedule与continuation／decode-ids互斥。先检查完整语法，再读取所有ID并校验会话容量，最后创建Model。
- `--session-schedules FILE`：执行多会话manifest，与`--schedule`互斥；普通／MTP的prompt-only自然请求可配合`--steps`。同一manifest的所有会话必须使用一致的自然生成或固定重放语法，在读取token文件前检查。每行`SESSION_ID SCHEDULE_FILE`，路径相对manifest，支持双引号和整行`#`注释；ID为ASCII字母／数字开头，后续可含`_`／`-`且不能重复。`--capacity`是每会话上下文上限，所有会话和ID均先校验再加载模型。
- `--max-sessions auto|N`：保留的设备缓存会话上限；默认auto取本次输入会话数。普通文本和MTP都可淘汰空闲head；MTP生成中的head不淘汰，轮次间可淘汰并在下轮恢复。共享backing按解析后的容量预留，不为各会话复制权重或执行载体。
- `--active-sequences auto|R`：活跃请求和单次真实Graph批次的序列上限，实际按等T分桶，批宽不超过R；不是同时提交多个独立Invocation。默认auto随输入、缓存数量和字节预算解析，满足`1≤R≤N`；显式值受输入数量和缓存容量约束。
- `--cache-bytes auto|N`：共享KV页池、初始／caller状态池及选中保留的原生结果backing的请求字节额度。CUDA默认auto按N份可提交版本及最多R份在途候选解析；显式额度不足时缩小可驻留数量，不能容纳活跃请求则在加载前报错。Apple／Vulkan的auto不设字节额度，仍执行N／R及版本数边界；显式额度按实际池及完成后的结果准入，不用CUDA公式预估可驻留会话数，空间不足会报错并保留旧快照，可提高额度或减少会话／上下文后重试。淘汰归还页／版本，Model池及仍有读者的结果继续计费。这不是全进程或物理显存上限。
- `--prefix-cache off|auto`：多会话manifest默认auto，单会话schedule及普通生成默认off；显式auto须使用schedule。在prompt或轮次恢复边界查找同Model／数学图族的精确已保存前缀，借用完整Snapshot并计算剩余输入。完整prompt命中的greedy返回已有next ID、执行0次Graph；random使用当前请求RNG执行1次frontier采样Graph。恢复同一请求的完整生成检查点保留其pending与RNG，不重采样。decode直接使用Session。目录最多N项，旧检查点与会话head共用N份完整版本额度，额外R份只供候选。
- `--progress direct|scheduler`：默认direct；scheduler在原槽上准备进度驱动，复用同一提交机制。
- `--replay auto|commands|prefer-capture`：默认auto，在Model范围冷聚合每个批宽／child／槽的使用上界，只用一次走commands，预计复用才prefer-capture。固定输入重放保留有序调用的精确slot计数；自然请求、动态批次、前缀suffix和恢复按有限波次保守计数。也可显式固定其中一种模式。生成可能提前结束，不保证每份capture都会复用；换会话仍由实际资源／布局身份决定是否重录。`planned_replay`记录冷选择，逐步日志记录实际录制与复用；不做虚假预热或把首次执行移出计时。
- `--decode-ids FILE`：固定单token输入带，允许跨过预测EOS；若同时给steps，须等于输入数加一。正常生成遇到endoftext或im_end停止。
- `--trace DIR --trace-level decisions`：仅导出logits与最后hidden；默认full还导出conv/GDN状态与`kv-layout.json`。分页帧不公开原始bank载荷，后者记录实际页布局，不再导出dense KV文件。路由写入DIR/routes，不增加图输出。
- `--layer0-probe DIR`：诊断时另行发布layer0边界并比较实际路由；诊断图会影响合法融合，因此与普通图分别冷准备，均计入启动时间。多会话的边界输出按`session<序号>_<ID>`子目录隔离，不覆盖其他会话；共享图的路由说明仍在DIR。
- `--preparation-diagnostics 1`：仅schedule模式；分开记录layout、batch和replay准备时间，注明实际槽及其首用，并读取六份公开BatchPlanReport。记录冷预分配，报告在原host输出端点后读取、最终端点后打印；标记为diagnostic，不用于吞吐对比。默认关闭，不增加这些报告查询。

各执行模式的JSON均报告同一`cache_budget_scope`、固定池`cache_shared_backing_bytes`、当前额外保留结果`cache_retained_output_backing_bytes`及额度内live／peak；未设字节额度时`cache_requested_byte_limit`为null。它们是同一个Model的全局统计，多会话子结果中的值不能相加。`native_execution_private_bytes_unknown`单列原生执行私有成本未知，`physical_vram_limit_known=false`；这些预算统计不能替代实际内存／显存旁采。

Apple／Vulkan提供按需构建的`hyle_qwen35_retained_budget_probe`目标，使用`--inputs <已准备的MTP输入> --out <尚不存在的结果目录>`。它在真实计算完成后施加纯额度压力，核旧完整状态、失败回收及原槽重试；不分配对应压力的设备内存，不用于测速。实际命令与结果见[保留状态预算记录](../../../docs/evidence/correctness/qwen-retained-budget-2026-10-22.json)。

普通文本和MTP均已接真实多请求批次。[本批MTP证据](../../../docs/evidence/correctness/qwen-mtp-batch-2026-10-22.json)包含B2的不同接受边界、lag1新输入、整批取消／原槽重试、有界重建及0Graph完整命中。R1／R2续文确有分歧，跨批宽质量未收口；本批为诊断运行，不作吞吐或显存收益结论。

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
图族按所有会话的shape并集冷编译，Model按有序调用冷统计每个形状的串行槽使用上界，预选Commands或PreferCapture。完整prompt命中跳过prompt计算和预热；random仍有上述frontier采样Graph。切换会话不复制载体。准备诊断的prior-use计数属于共享执行槽，跨会话不清零，失败后重建该槽才清零。
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
lookup只发生在请求边界；greedy完整命中为0Graph，random完整prompt命中另执行1次frontier采样Graph，普通decode不逐token查树。目录及LRU在初始化时预留容量，活跃命中可同时只读持有。
Session独占自己的尾权，同页续写只写未提交后缀；前缀命中不会复制尾权，分支沿原COW路径，旧快照范围始终不变。
[本批实现、回归与性能](../../../docs/evidence/performance/qwen-exclusive-radix-2026-10-21.json)不代表其它原生后端或GPU并发吞吐已交付。

`hyle_qwen35_retirement_probe`与`hyle_qwen35_mtp_session_probe`已登记CUDA／Apple／Vulkan构建；原生会话探针当前仅开放`--scenario prefix`，退休探针的三平台有限T2检查已通过列明范围。`hyle_qwen35_batch_probe`仍仅在CUDA SDK下提供。普通模型数学与状态通过公共Compute／Graph接口执行；这些按需诊断可借用原生私有状态来检查在途寿命，结果按各证据实际后端分别说明。
CUDA 测试目标 `hyle_qwen_request_lifecycle_test INPUT_DIR` 使用原 P154 BF16 输入检查普通随机 R2 的取消、存活请求续跑、有界输出及增量续轮；不用于测速。也可设置 `HYLE_QWEN_REQUEST_INPUTS` 后运行同名 CTest；未提供输入时跳过，显式不存在的目录按失败处理。[原生结果](../../../docs/evidence/correctness/qwen-live-request-lifecycle-2026-10-25.json)来自归档消费者。正式入口严格编译、链接及主机错误合并检查已通过；新增 `--retirement-status-only` 仅检查主机状态分类，未注入设备错误，正式入口未重跑 GPU。
退休检查独立于默认构建，可显式构建`hyle_qwen35_retirement_probe`，运行时传`--inputs DIR --decode-ids IDSFILE`；使用原P128输入及至少两个decode ID，不复制权重。它检验原执行链的寿命，不用于测速。
探针当前覆盖同页独占后缀取消：实际退休前保留资源，退休后旧快照仍持页，所有旧读者释放后才可重新取得整页。
该目标另可用`--inputs MTP_DIR --mtp on [--verify-rows 1|2]`检查真实MTP取消，和`--decode-ids`互斥。默认2行验证；1行分支实际覆盖没有attention读者的草拟KV写effect。两种分支均已实跑：使用自然接受后的lag1前缀，取消并丢弃Completion后保留全部14份KV写事务、状态、候选和执行借用；真实退休后核旧字节及同槽重建。另核成功terminal后的接受策略异常不会发布head，回收后原完整前缀仍可续算。这是有限取消／host异常检查，没有注入GPU设备错误，也不用于测速。

下述请求backing公式对应CUDA预留池；Apple的native生成backing未计全，不能将公式作为其完整字节上限。
普通无MTP页大小16。设每序列容量C、可保留完整快照数N、一次批次最多R条序列（1≤R≤N）：12份KV arena各容纳`N*ceil(C/16)+R`页，36份conv/GDN池各有N＋R个块。单版本有效载荷合计19,537,920字节，各backing另有1字节尾部；总额为`12*((N*ceil(C/16)+R)*16*1024+1)+19,537,920*(N+R)+36`。query／KV行和page refs的schema上限是批内各请求之和，物理页池按不同物理页计数。CLI按实际解析的R准备该额度；R1、C256、N1／N2示例仍为42,418,224／65,101,872字节。额外R份不是历史检查点额度：若已有L份完整版本，本次r名成员只会真正释放D份不同旧base，须满足`L+r-D≤N`。目录或未选会话保留的旧base不能抵扣，淘汰目录别名后按真实版本数重算；无法容纳时拒绝，不提交超过N份的完整版本。新建初始head也须先满足N份额度，同一Snapshot别名不重复计版本。

MTP同样按实际R准备：同样页几何下增加2份独立草拟arena，conv/GDN预留`N+2R`份材料，并保留`2*(N+R)`个4096B carry/prior槽。总请求backing为`14*((N*ceil(C/16)+R)*16*1024+1)+19,537,920*(N+2R)+36+4096*2*(N+R)+1`，不是物理显存上限。

独立目标`hyle_qwen35_mtp_session_probe`运行时传`--inputs MTP_DIR --output NEW_DIR`；默认使用原23-token输入的前15／23枚，在同一N2／R1 Model上先独立执行、退休，再交错自然生成。[原检查](../../../docs/evidence/correctness/qwen-mtp-sessions-2026-10-22.json)的378份向量逐字节一致，只对应当时的R1范围。追加`--scenario batch`检查真实B2、接受边界[2,1]、取消／原槽重试、已提交输入重建及独立的非argmax pending政策；本批14次Graph中，同分块重建54项逐字节一致，跨B的108项有4项draft K/V数值不同。完整质量、其它原生后端及吞吐仍须分别检查。

追加`--scenario numerics`可检查[同旧状态的批宽控制](../../../docs/evidence/correctness/qwen-mtp-batch-numerics-2026-10-22.json)：该历史BF16末分数检查中，R1和真实B2分别完成P15，再各用自然R1轮次到L18；控制Session借用同一个完整旧Snapshot，分别执行R1／R2，不重建或复制历史。实际10次Graph、13组54项向量中，两份来源各自的目标状态／输出跨B逐字节一致，draft K/V另保真实差异；首异A随prefill来源走。若自然生成没到指定边界，报告跳过控制，不强制token。该历史结果不签当前F32末分数政策的质量或吞吐；本批未重新编译／运行这个独立探针。

追加`--scenario prefix`可检查[三平台MTP前缀与准入](../../../docs/evidence/correctness/qwen-native-protocol-2026-10-22.json)：完整命中比较52份状态，不伪造hidden／logits；真实续算比较54份向量。T1／T8均与相同输入分块的独立基线对照，534项逐字节一致，不能用分块结果签与整段prefill数值相同。误把真实teacher-forced T1交给生成finish时，在原Completion退休后、状态发布前拒绝；14份事务回收，旧head不变，同槽可正确重试。CLI另实跑N1／N2排队、字节缩容、0Graph完整命中和两种后缀；本批未测速。

独立`hyle_qwen35_batch_probe`用与CLI相同的Model批次接口执行真实B2／T1：普通线性层使用packed行，Conv／GDN用独立B维，历史通过普通Graph的Concat收集、结果Slice／Copy回各自输出租约，KV用公共`PagedArena::reframe`组合与投影。一次submit对应一个原Invocation／Completion，不是两次evaluate合并统计；不同T由CLI分桶，不能用会改变recurrent状态的虚假padding。运行传`--inputs DIR --output NEW_DIR`，使用原P128输入，不复制权重。探针比较独立B1同分块、旧状态、结果分流和继续推理，并检查实际队列阻塞后整批取消／丢弃Completion、两候选保留到真实退休、原R2槽重建及重试；B变化可改变合法矩阵归约，原始误差指标与greedy ID分别记录，未定义跨B完整模型质量阈值。状态打包／拆分成本须从实际批次的完整性能窗口读取。

CLI批次及同源R1对照的prefill、decode、总会话与采样显存见[本轮实测](../../../docs/evidence/performance/qwen-cli-batch-2026-10-22.json)。批次合计吞吐不代表单会话提速，跨批宽数值质量未验收。

加载沿现有FileResourcePipeline，用两个8MiB暂存读取原权重范围并上传：CUDA用原pinned／device资源，Apple接原生shared资源和文件管线。官方 BF16 文本选择320个原始张量，MTP另加15个，均保留原分配和持久目标租约；量化清单按实际物理角色加载。
完整Model／ready／首token计时包含文件关闭、管线退休和临时运行时销毁。新进程未清OS页缓存，不能称冷盘加载。
`step_seconds`包含输入上传、layout准备、绑定、replay准备、提交、等待及token读回；首次shape编译在启动时完成，trace导出不在step内。
普通生成和MTP的`prefill_seconds`从实际prepare前计至head发布且token在host可见；`decode_seconds`从同一prefill端点计至最后成功finish端点，`session_seconds`是两段连续wall之和。`startup_total_seconds`（同`startup_to_endpoint_seconds`）含加载与冷准备；`step_sum_seconds`单列，不能替代完整elapsed。四个`generation_*`内存标记仅供旁采近似分段，不替代计时端点；trace运行标为diagnostic，不混入速度表。stdout将实际Graph调用、目标验证行、草拟行、已提交行和真实输出分别计账，接受token数不冒充计算量。
`seconds`和`submit_seconds`是更窄子段，不可直接与另一引擎的完整decode相比；完整启动至首token看`first_token_elapsed_seconds`。

本次有限图族已完成原128+31旅程的64份hidden/logits逐byte对照，以及23/4/1 scheduler＋capture旅程的6份对照。
两组独立性能窗口确认准备开销减少；decode小差异反向，未证明稳定提速。详见[有限图族实测](../../../docs/evidence/performance/qwen-finite-family-2026-10-20.json)，
最新同窗指标见[decode性能记录](../../../docs/developer/performance/decode.md)。报告中的compiled storage保留unknown计数，不等于完整显存峰值。
默认auto及其一次性chunk／重复槽选择见[最新实测与完整证据](../../../docs/evidence/performance/qwen-auto-replay-policy-2026-10-21.json)；该策略修复了首次缓存续算损项，完整短decode的首次准备成本仍在排查。
另有独立的[进程内存／显存采样](../../../docs/evidence/performance/qwen-memory-current-2026-10-20.json)：GPU峰值为采样下界，
驻留集、private commit和GPU共享内存分别记录；mmap与文件缓存会影响归属，不能将进程差直接当作全机RAM节省。

`reference.py`是独立Torch参考，候选不会读取参考状态或结果。固定输入与向量相同属于回归检查，独立质量验收仍未完成；
BF16舍入、归约和RoPE诊断见[底座审查证据](../../../docs/evidence/correctness/inference-foundation-review-2026-10-18/README.md)。
原输入位于仓库`.scratch/inference-execution-refactor-2026-10-18/qwen35-text/`，不复制权重或SDK。
