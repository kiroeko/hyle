# 推理看板

更新：2026-11-05。看板由用户维护；agent 按顺序做任务卡，完成后只把该卡标为完成并附提交号和关键数字。规则见 [AGENTS.md](../../AGENTS.md)。

**目标：** 在一两个有人在乎的场景做出可复现、明显领先 llama.cpp 的推理结果，并公开讲清原因。长期方向见[两项核心期许](../architecture/inference-principles.md)。

## 里程碑

| 里程碑 | 完成标准 | 状态 |
|---|---|---|
| M1 首轮冷开销 | 完成 M1-5 收尾，整理一篇讲清根因的技术文章。性能以同窗实测和 M1-5 验收标准为准。 | 进行中 |
| M2 主场景 | 按 M2-1 的实测选定一个场景，在该场景领先 llama.cpp 至少 1.3 倍，或做到它做不到的事 | 未开始 |
| M3 发布 | README 第一屏是结果表和一条复现命令；文章公开 | 未开始 |
| 上游 PR | 1–3 个性能 PR 合入 vLLM、SGLang、llama.cpp 或 FlashInfer，方向取自 M1、M2 查清的问题 | 未开始 |

**当前执行：** 下一张为M1-11本机2B速度收尾，待做；本卡结束后停止，不自动接旧卡或M2。M1-10e已完成，114688行Q4子集头及合格准备目录下BF16自动K3已交付。M1-10d的底座、完整轮图和后继重叠已合入main `d03321af6`；原32K速度验收未通过，32K重排方向停止。M1-9b、M1-7a与3.B保持结项；M1-8和M2未启动。

**暂停：** GLM、Kimi、DeepSeek 组件，主场景用不到的后端，基础库新功能，多卡，通用 loader、tokenizer、server。

## 任务卡（按顺序做）

### M1-11 本机2B速度收尾｜待做
- **目标：** 收敛用户本轮“显存无关的机会”review中的八项线索。在本机、现有2B制品和固定负载内，合入有完整消费者收益的改动，或给出未采用的证据与边界；不承诺所有模型、格式和负载都已没有优化空间。review中的预计收益、历史局部带宽与耗时只作线索，不作本卡验收数字。
- **范围／基线：** 从`6c7b2bd78`冻结基线exe及其实际DLL／运行环境；候选复用现有构建和依赖。先读实际硬件、工具链与驱动。只用本地Qwen3.5-2B BF16 HF、Q8_0、Q4_K_M；固定C32768、现有八个不同prompt、每轮最多64输出，自然EOS。普通、显式K1／K3与实际auto默认分开报告；BF16合格准备目录的auto为114688行Q4头K3，量化auto资格不因计时通过而改变。沿用`.scratch/inference-full-matrix-20261105/`的本地制品与负载，不重跑九格式或容量／长度的笛卡尔积；不下载权重，保留PAUSED。
- **先做：** 复用`scripts/inference-upstream-compare.py`、既有配对profile与teacher工具，在`.scratch/inference-speed-closeout-20261105/run.py`补齐本卡薄封装；这是假定由本卡实现的入口，建卡时尚未实现。显式传入冻结基线、候选exe和实际版本，新输出目录不得混入旧样本；禁止直接沿用旧脚本硬编码的exe／版本。先核对输出数、上下文、工作量、计时端点与默认政策，再做一次K3提交归因及Q4／Q8 K1配对profile。profile与正式计时分开；正常桌面和后台保持，GPU推理与profiler串行。
- **机会清单：** 每项先确认当前main是否仍有该成本；只有可归因的完整进程／合并decode潜在收益达到1%才进入性能实现。先处理K3重复执行的真实性与K1精度，再评估K1子集头，最后处理单次配对profile筛出的剩余大项；不要求把八项都改一遍。

| review项 | 本卡处理与边界 |
|---|---|
| 1. 量化K1改走F32 | 查实际转换、数学边界与设备后继资格；验证后再实现，不以删除精度门代替适配，不沿用预计的5%～10%收益。 |
| 2. 量化K1子集草稿头及词表 | 在第1项结论后单独对照；只用本地原头和既有train校准，保持原ID映射，验证／holdout不参与选行。最终组合收益须重测，不把两项百分比相加。 |
| 3. BF16 K3放大Q4子集头 | M1-10e与自动K3已交付（`2bf864ba8`、`0b86c7d42`）；保留为本卡基线，不重做选行、不再扩行数。 |
| 4. K3重复提交 | 统计`overlap_successor_discards`、`refusals`、真实提交、丢弃后重做及对应GPU区间；区分正常投机丢弃、失败回落、重复执行和重复发布。只把实际关键路径浪费算作可省时间，不要求discards为零；若有重复发布，按正确性缺陷先修。 |
| 5. Q4 K1第3／4轮偏慢 | 与Q8 K1在相同负载下配对profile，先复现再归因；历史每轮约0.55ms不直接当当前可省时间。不把显示／后台周期尖峰预先归因为引擎，不通过关后台或改变显示设置取得正式收益。 |
| 6. 普通路径小kernel | 一次配对profile核对共享输入量化、MMVQ＋add、COW append、norm／GDN／conv及空隙；只实现达到1%完整收益预算的项，不重新扫全部kernel参数。 |
| 7. Prefill | 只补BF16普通模式在占用深度4K／16K后的pp512配对profile；先算其完整进程收益预算。M1-9b已完成的捕获复用不重做，不扩其它深度／格式。 |
| 8. 启动 | 核对冷family串行与加载／staging的当前成本，只处理有实测预算的瓶颈；ready费用保留在完整进程内，不挪ready标记，不靠预知未来请求准备形状。 |

- **验收命令：** 本卡先实现并用`--help`核对上述入口。`python -B .scratch/inference-speed-closeout-20261105/run.py prepare --baseline <冻结基线exe> --baseline-version 6c7b2bd78 --candidate <候选exe> --candidate-version <实际版本> --output <新目录>`冻结本轮参数与本地路径；随后分别运行`run.py diagnostic --output <同目录>`、`run.py timing --output <同目录>`、`run.py memory --output <同目录>`、`run.py correctness --output <同目录>`与`run.py check --output <同目录>`。这些phase均沿用prepare参数；diagnostic只作归因，timing／memory每臂固定n3，不自动补样到通过；check缺数据或未过门槛须非零退出，不能仅打印报告即视为通过。
- **正式性能：** 冻结基线、候选、`b12063 / 2cbca389f`同窗交错，每臂至少3个独立进程，记录实际版本与上游政策，不重新扫上游K；服务端口2048、`--metrics`。每个候选先做受影响配置的筛选，过筛后验收该配置及共享改动影响的回归；结项覆盖三格式的普通／K1／K3和实际默认，不扩其它工作负载。4K／16K pp512只用于本项归因及受影响prefill改动的验收。
- **通过标准：** 完整进程tok/s为主，候选相对同窗基线的中位数不得降低；decode改动还须合并decode≥基线101%、候选与基线三进程范围不重叠；prefill／启动改动须在对应完整负载的完整进程tok/s提升≥1%且范围不重叠。所有受影响配置每轮decode≥基线98%；ready增量≤100ms。按进程累计输出／墙时后再取中位数，不平均逐轮速度比；首轮录图／预热不能扣除。分别列出ready、首轮冷费用、逐轮prefill／decode、生成综合速度与完整进程tok/s。prepare须统一跨引擎外部起止端点，禁止用“Hyle退出／上游最后HTTP响应”的不同比值宣称完整进程优势；无法统一时停止汇报。独立资源进程记录GPU dedicated／shared、私有提交及working set，披露增量；不得缩小容量、减少输出或常驻后台采样混入计时来取得收益。
- **正确性：** 同数学实现改动，八轮最多512个输出ID与冻结基线逐项一致，EOS／实际产出一致。BF16换草稿头沿用主八轮及既有24条holdout的同前缀b12063 BF16 teacher检查，分叉只允许teacher top2且top1−top2<0.1，逐项报告非top1，不写成全部一致。Q8／Q4目标精度改变须在受影响普通／MTP路径实际重验8192位置的配对teacher与既有KL／top1护栏：Hyle KL≤上游同格式KL×1.10＋同组Hyle BF16 KL，top1一致率不低于上游同格式减1个百分点；普通worker结果不能代签MTP路径。相关构建与测试通过，核对状态版本、RNG、有序发布、权限、并发及真实退休；所需检查做不到就停下报告，不放宽质量门。
- **停止／结项：** 同一方向连续两次实现或调查没有可测量进展，保存命名实验分支并停下汇报，本卡保持进行中；“范围重叠／未证明收益”不写成“确定无收益”，未测不写成“估计<1%”。只有八项均有“已交付／本卡合入／试过未证明收益／有预算依据的估计<1%不做”结论、相关验收全过且无未解释的正确性问题，才标完成；完成不以必须新增优化提交为条件。结项只留一张简表和三格式当前数字，原日志放本卡`.scratch/`，不追加实验过程或证据包。完成、受阻或超范围时都停止，不自动启动显存、旧卡或M2。
- **不做：** 显存归因／优化、开关与能力门整理、引擎抽取、第二个模型、跨平台扩展、GPTQ／AWQ／NVFP4等其它格式、更多上下文／输出长度、K>3、自适应K、概率阈值、词表重排／概率加权／上下文词表、精确head剪枝、ForgeKernel、放宽Q8 KL门槛；不在main增加未采用方案的实验开关。

### M1-10d K步草稿录成一张图｜已实现，32K速度未通过
- **目标：** 固定K3的合并decode ≥ K1的105%。把K步起草（MTP层 → 32K草稿头 → 设备argmax → 取嵌入 → 下一步）录进同一张图，草稿token不回主机，并保住K1已有的轮间后继重叠。参考SGLang EAGLEDraftCudaGraphRunner和TRT-LLM MTPWorker，记录版本。新卡从HEAD开始；Source接力补丁存档，不提交。
- **先做：** 查清K1的轮图怎样把草稿和验证放进同一张图，判断K步能否沿用；草稿KV的位置和页表在提交前按K行预留。如果现有Graph/Compute接口做不到设备端可变位置，只能每轮做一次主机同步，就停下汇报。
- **验收：** 正式八轮同窗，K1、新K3、b12063各跑3个独立进程。K3的`actual_graph_calls/decode_graph_calls ≤ 2`，`overlap_successor_submissions > 0`。合并decode ≥ K1的105%，每轮 ≥ 97%。ready增量 ≤ 100ms；K≥2现在多约1.0s，next-notes记录的原因是`begin_cold_families`调用处排除了K≥2。512项输出与HEAD的K3逐项一致。
- **不做：** 概率阈值和自适应K、扩词表、新kernel、Source接力。全词表量化草稿头等本卡通过后另开一张卡。

### M1-10e Q4子集头放大行数｜已完成
- **完成：** `2bf864ba8`。固定三档n1后选最小114688行；正式七臂各n3及独立资源七臂各n3。K3合并154.2553 tok/s，为默认K1的116.36%，最慢轮106.25%，ready增量66.7ms；图调用238/226、后继提交200，512项同前缀b12063 BF16 teacher top1全同。GPU峰5101.082MiB，低于全Q4头5251.086MiB。默认仍为K1；新默认K1中位数比原main低1.24%且范围重叠，不声称严格无退化。详见[性能表](performance/decode.md#固定-k3本轮复测)。
- **目标：** BF16目标、固定K3使用Q4_K子集头；八轮合并decode ≥ 同窗默认K1的105%，每轮 ≥ 97%。使用M1-10d已合入的设备长度、完整轮图和后继重叠。
- **范围：** 基于 `020ed7b62`（`experiment/mtp-q4-subset-head-20261105`），固定测试114688／131072／163840行。按校准词频降序、同频按原ID升序选行；校准仅用旧24提示和已归档64个train提示。validation、gating和quality8192不参与生成或选择。
- **先做：** 用 `scripts/prepare-mtp-draft-vocabulary.py --calibration-run <旧校准目录> --calibration-run <新增语料目录> --inputs <原准备目录> --rows N --work-dir <新目录> --output <映射文件>` 离线生成三份。核对元数据、编码字节数、运行时行数边界及与main的合入条件；这一步不启动GPU。保留原校准排序，复用原Q4编码行复制和资源退休机制。
- **验收命令：** `python -B .scratch/inference-mtp-q4-subset-20261105/large_heads.py diagnostic <新目录>`；三档均完成后，选同时过105%／97%的最小一档。该档再跑 `large_heads.py timing <新目录> --rows N` 和 `large_heads.py memory <新目录> --rows N`。
- **正式验收：** 八轮同窗，各臂3个独立进程；主对照为默认K1、全词表Q4头K3、所选子集K3及b12063，另列main默认K1与所选／全头K1供默认政策判断。图调用比≤2、后继提交>0、ready增量≤100ms。独立资源进程中GPU峰不高于全Q4头K3。所有计时含真实冷费用，并单列ready与首轮；报告逐轮、完整进程和私有提交。
- **正确性：** 同头实现改动保持512项严格一致；换头按已获批的同前缀b12063 BF16 teacher核验。沿用既有同精度近平局准则（上游top1−top2<0.1），逐项报告分叉与非top1项，不将近平局写成全部top1一致。核对状态、工作量、RNG与真实退休。
- **停止边界：** 三档都未过105%／97%则按实测汇报并停止；n1不作正式性能结论。
- **不做：** 词表重排、概率加权、上下文词表、概率阈值、自适应K、新kernel，以及未验证的默认头切换。

### M1-9b Prefill捕获复用｜已完成（8小时时限）
- **完成：** `7369894f0`。默认构建、同窗交错、每臂3进程：4K／16K／30K的pp512为b12063速度的98.58%／97.56%／102.73%（范围重叠）；相对起点 `6ef6d0bae` 提速5.40%／6.83%／5.66%。整段16K／32736为1418.535／3132.745ms，比起点提速6.17%／4.32%；decode为起点100.15%～100.24%。独立显存峰增量26.004～28.004MiB；1315项同精度teacher核验及133项相关测试通过。详见[性能表](performance/decode.md)。
- **目标：** Batch配置重置后安全复用已录好的prefill图，块间接力也使用捕获；不再为相同完整批次重录。沿用M1-9已有调查，不重做已提交的GDN／注意力工作。
- **范围：** 捕获／复用、接力选择及必要的固定图计划／工作区约束；设备kernel不改。计时窗口597个完整512行批次均复用；480行尾块改变真实行数，首次录图费用包含在请求中。
- **验收命令：** `.scratch/inference-prefill-capture-reuse-20261103/acceptance.py timing|memory --output <新目录>`，再运行同目录 `check_acceptance.py`；构建与相关测试为 `build_check.py 12`，teacher为 `run_teacher.py teacher-default1 teacher5`。原始日志只留该目录。
- **通过标准：** 默认构建，d=0／4K／16K／30K的pp512与tg（生成上限512，EOS如实报告），以及16K／32736整段prefill；同窗交错、每臂3个新进程。4K／16K／30K pp512≥上游97%；tg≥本卡起点98%；整段prefill不慢于起点；独立GPU峰不多32MiB以上。
- **正确性：** 按用户修订，参照b12063同精度BF16 teacher-forcing；argmax分歧仅允许上游top1−top2<0.1。撤销旧HEAD的0.1门槛。行数变化、无效控制、重置和新增重置后复用测试通过；权限、RNG、有序发布与真实退休保持。
- **边界／不做：** ready增加51～78ms、私有提交峰增加126～186MiB已如实记录。短prompt的约2ms、MTP、启动优化、其它kernel调优、资源瘦身与M2不做。本goal完成后停止。

### M1-7a 与最新上游 llama.cpp 正面对比｜已完成
- **完成：** `e23027045`。固定最新提供官方CUDA包的`b12063 / 2cbca389f`，官方／自编与FA开关筛选后选官方CUDA13.4、FA=on。54个正式进程（36计时＋18独立显存采样）完整性检查通过，各格3个独立进程；20项工具检查通过，无引擎改动。标准pp512为上游80.9%～88.6%；tg128为98.7%～105.4%，范围均重叠。真实八轮合并普通／MTP decode领先4.5%／14.0%，MTP第4轮慢23.1%。普通外部ready慢202ms，MTP ready范围重叠；显存多332／494MiB。逐轮原数、MTP接受／提案数与冻结第三方见[性能表](performance/decode.md)，原始日志在`.scratch/inference-upstream-head-to-head-20261103/window1/`。一日时限内结束，不现场修复落后项。
- **目标／前提：** 先测清当前默认实现与最新上游的胜负。3.B已停在提交节点并结项；测量期间GPU只运行当前测量进程，不同时运行3.B或其它GPU任务。
- **上游准备：** 固定开始时最新、实际提供官方Windows CUDA包的release tag/commit；获取官方包与同源码自编Release版，后者用`-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89`。记录CUDA／编译选项，两份各跑llama-bench并择快；flash attention开、关均测，保留原数与选择依据。GitHub发布包／源码获取已获用户授权；模型只用本地BF16 GGUF，不下载权重。
- **A 标准吞吐：** 上游`llama-bench -ngl 99 -r 5`分两次运行：`-p 512 -n 0 -d 0,4096,16384,30720`；`-p 0 -n 128 -d 512,4608,16896,31232`。Hyle C32768：先填入d行且不计时，再测实际512行prefill及128 token生成，其余默认。双方各3个独立进程、同窗交错。核对实际上下文及处理行数；注明llama-bench tg随机token／无采样，Hyle包含greedy采样。
- **B 真实多轮：** 现有8个不同prompt，每轮64 token。上游`llama-server -np 1 -c 32768 -ngl 99`，使用A选出的FA设置，token数组、`cache_prompt`、greedy，读服务端`timings`；服务默认端口2048并启用`--metrics`。Hyle沿用现有入口；冻结参考作为同窗第三臂。各臂至少3个独立进程，逐轮报告。
- **C MTP：** 从上游源码和实际入口确认Qwen3.5 MTP是否受支持；支持则用B的负载比较，逐轮报告接受／提案数；不支持则比较Hyle MTP与上游最快普通模式，并明确功能边界。
- **D 启动／显存：** 上游启动到能接受请求，对比Hyle ready；显存由独立进程采样，双方capacity相同，采样不进入速度计时进程。
- **验收命令／输出：** `python scripts/inference-upstream-compare.py --help`列出准备、计时、独立内存与离线分析入口；按本卡本地路径完成全矩阵。输出一张Hyle÷上游表，含各深度pp512／tg128、逐轮prefill毫秒／decode tok/s、MTP、启动和显存；各指标注明方向，标记范围重叠。附更快／更慢及差幅结论。脚本提交，提交信息写实际命令和版本，原始日志只留`.scratch/`。
- **不做：** 不修改任何引擎／kernel，不下载模型，不现场修复落后项；只记后续profile方向。到时限即报告实测与缺项，不以历史窗口补齐新结论。

### C0 收尾工作树里的 1.C 改动｜已完成
- **完成：** `975f03ffd`；驱动8组、CPU／CUDA普通接线检查及两项CTest通过；普通与MTP greedy两轮各128个输出ID均与10-31窗口一致。
- **目标：** 处理工作树里尚未提交的普通流水接线改动（目前约 15 个文件，在 `engine/models/qwen35/`、`engine/inference/`、`tests/` 等处）。
- **验收：** 相关目标编译通过，改动涉及的测试通过，Qwen3.5-2B 普通和 MTP greedy 两轮输出 ID 与 10-31 窗口记录一致，然后提交。做不到就把改动存成 `.scratch/c0-wip/wip.patch`、恢复这些文件，再汇报。
- **不做：** 不扩大 1.C 的范围，不做新的性能测量。

### M1-1 一条命令的按轮对照｜待做
- **目标：** 把 `.scratch/inference-default-product-window-20261031/window.py` 的做法整理成提交到仓库的 `scripts/inference-compare.py`，模型、参考程序和输出目录都由参数给出。
- **验收：** 一条命令在同一时段交错运行 Hyle 与参考的普通、MTP × 单轮、两轮，每臂 3 个计时进程和 1 个独立内存进程，输出一张表：按轮 decode 的中位数和范围、prefill、启动到 ready、显存采样峰、Hyle 每轮录图次数、双方版本。本机按M1-4的新协议跑一遍；旧预知schedule／缺少参考warmup的数字不能作为验收锚点。
- **不做：** 不改引擎代码，不生成证据 JSON 或 ZIP。

### M1-2 启动时完成录图和预热｜待做
- **目标：** 首轮 decode 不再付录图和流水预热的成本。10-31 日志里，普通和 MTP 都在首轮前 5 步录图（visible 155–164）；普通每次录图那步多约 4–8ms，流水接力到第 9 步才开始。
- **先做：** 确认 CUDA Graph 的复用条件不绑定具体会话的地址；如果绑定，停下汇报。
- **验收（用 M1-1 的命令）：** ready 之后 decode 的录图次数为 0；普通和 MTP 的单轮、两轮 decode 中位数都不低于同时段参考；启动到 ready 不慢于参考；显存采样峰增加不超过 64MiB；greedy 输出 ID 与改动前一致。
- **不做：** 不改 kernel 和稳态路径，不处理多请求合批。

### M1-3 MTP 后继门控｜验收未通过（候选保留）
- **目标：** 按接受历史／可用置信度决定是否提前提交后继 B，减少猜错后的无效执行；提交时保留原设备接力。
- **负载：** 固定8个不同的聊天、代码、中英文、摘要等 prompt；用模型本地 tokenizer 生成 ID，命令写入提交信息。原重复 prompt schedule 仅作专项回归，不能证明真实多轮收益。
- **策略：** 先记录信号与接受结果，从连续接受规则开始；不够准再用草稿 top1 概率或 top1／top2 差值。缺失信号如实记为 unavailable，不把历史次数当概率。
- **显存：** 启动仅预录 T2 Round／CatchupRound；T1 首次使用时录图，其冷成本计入实际 decode。停止追查分配内部字段。
- **验收命令：** `.scratch/inference-mtp-gating-20261101/run.py timing` 与独立的 `memory` 阶段，完整参数见同目录 `commands.txt`；正式计时每臂至少3个独立进程，同窗交错。
- **通过标准：** 不同 prompt 每轮 decode ≥ max(关后继, llama.cpp)；重复 prompt 第3–8轮 ≥ 原开后继的97%；greedy ID 不变；显存采样峰相对 `88e0f96b6` 增加 ≤64MiB。逐轮报告接受率、猜错次数及门控信号的真实含义。
- **不做：** 设备端接受、kernel、ready 时间。约94%的盈亏接受率仍是粗估，不能当已测门槛。
- **当前结果：** 固定token负载可用于功能回归；旧预知schedule测试的性能／显存结论作废，仅留排查日志。等待M1-4有效协议重测，不据旧结果结项。引擎候选未提交。

### M1-4 Prefill不依赖预知长度｜M1-4c按用户调整范围收尾
- **收尾：** 设备连续MTP `67c6a9450`、有界有效行prefill `774abccaa`、独立冷准备 `eb767e022`。短普通／MTP合并decode领先3.37%／8.2%，同配置显存增量0；原短普通prefill为参考93.9%、普通ready慢80ms，95%／+50ms门槛未通过。短prompt约2ms与ready差距保留，不纳入M1-5。长16K decode退化已由 `cf56700c2` 修复。新增MTP teacher唯一第13项argmax不同、oracle分差0.085544586<0.1，teacher反馈14—16项一致；状态字节差异未自行设容差。原目标和验收保留作边界参照，详见[性能表](performance/decode.md)。
- **目标：** 冷期只依据已到达首轮与显式容量配置准备；后续请求长度未知。旧整schedule预知测试作废，不能另立为有效工况。
- **先测：** Hyle只暴露首轮给冷规划，未来prompt／配额不影响family、载体、workspace、replay策略或主机缓冲预备；参考使用固定batch容量和默认启动warmup。每轮报告prefill／TTFT、shape admission和真实载体创建时间；启动加载／编译／准备／warmup单列并计入完整进程。
- **再改：** 首次启动预热归引擎初始化，清理临时KV／GDN／RNG后才ready，测试只观察费用。优先复用现有有效长度prefill路径，采用与未来输入无关的少量容量档位／分块；工作区按配置上界复用。填充行不写KV、不推进GDN／conv；复用原权限与真实退休机制。
- **验收：** 普通和MTP，同窗交错，每臂至少3个独立进程；冷期不知道后续长度时每轮prefill不低于做过启动warmup的参考，ready不慢于参考，显存增量≤64MiB，greedy ID不变；完整进程耗时及逐轮decode同时复核。新协议下未测项不得用旧数据补齐。
- **不做：** Prefill数学kernel或decode优化，不靠挪ready标记、扣局部耗时或预读未来形状达标。若共享workspace需新生命周期接口，先报告范围与真实限制。

### M1-5 长上下文准备收尾｜进行中
- **已提交基线：** `cf56700c2`。16K 普通／MTP decode 已恢复至与参考接近；C2048 短负载不低于旧版98%，同配置显存采样峰增量0 MiB。详见[性能表](performance/decode.md)。不再重做已完成的容量诊断。
- **当前：** 32K 普通 prefill 为参考的92.7%；主机 decode 准备从1K到32K分别增长约3.0倍／3.7倍。布局关系缓存和MTP候选页引用合并均未测出收益，已撤回；待处理旧页权限按bank常驻与增量验证。
- **目标：** 一天内让 bank 页表／布局常驻，权限增量处理；decode 每步、prefill 每块只处理新增 token／页。保留原有权限、版本、有序发布和真实退休。
- **固定负载：** 普通／MTP 四档 1024／4096／16384／32736 token，全部 C32768，每进程生成16 token；真实文本经本地原生 tokenizer 生成，参考核对实际 target／draft 容量并使用默认 warmup。正式计时同窗交错，每臂3个独立进程；不使用 profiler。
- **验收命令：** `python .scratch/inference-long-context-20261101/acceptance.py timing --output <新目录> --hyle-version <实际版本> --baseline <cf56700c2构建> --baseline-version cf56700c2 --capacity 32768 --lengths 1024 4096 16384 32736 --modes both`；`python .scratch/inference-long-context-20261101/short_regression.py timing --output <新目录> --hyle-version <实际版本> --baseline <cf56700c2构建> --baseline-version cf56700c2 --modes both`；独立运行 `acceptance.py memory-run` 采样显存，并分别记录 1K／32K 的主机准备。
- **通过标准：** 两模式下 32K 的 decode 每步、prefill 每块主机准备均不超过 1K 的1.3倍；32K 普通 prefill 不低于参考97%；四档 decode 和 C2048 短负载不低于已提交基线98%；显存采样峰增量≤64MiB。
- **正确性：** 对同一已接受前缀比较 Round 与 Single 的 teacher-forcing；argmax 不同时 oracle top1／top2 分差须小于0.1。从两种状态各继续生成16 token，完全一致或只在近平局处分叉；核对版本、RNG、有序发布和真实退休。
- **不做：** attention kernel、短 prompt 的约2ms差距、ready 和 M2。到时限仍未达标就报告数据。

### M2-1 主场景实测｜待做
- **目标：** 用实测选出 M2 的场景。候选：(1) 混合架构加 MTP 的更大模型；(2) 12GB 显存加 64GB 内存跑 30B 级 MoE，专家放 CPU；(3) 混合模型的 agent 式多轮和分叉会话（重新生成、改写上一条、共享前缀）。
- **先做：** 为每个候选列出模型、量化和下载大小，交用户批准后再下载，遵守 AGENTS.md 的下载规则。参考尽量用上游最新发布的 llama.cpp。
- **验收：** 一张表，列出每个候选下 llama.cpp 的 decode、首 token 延迟、能否运行、显存和内存，以及版本；HYLE 现在能否跑、还缺什么的粗估；推荐一个场景并说明理由。
- **不做：** 不改 HYLE 代码，不做优化。
