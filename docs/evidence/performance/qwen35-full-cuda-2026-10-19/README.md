# Qwen3.5-2B 全 CUDA 实际对照

2026-10-19，Windows / RTX 4070 12GB / driver 617.14。双方完整执行官方同源2B文本模型24层，
不执行视觉或MTP，权重在推理前全部驻留GPU；同为BF16权重、BF16 KV、单请求、context容量256、GPU greedy。
llama.cpp为本地d9e03f1074dbd2979126d91dce1b5d304ec8394e，复用与其GGML pin 456相符的原CUDA数学闭包，
不是本地较新HEAD，也不把既有Hyle资源集成补丁隐藏为纯上游未修改构建。

以下首表保留修改前基线。本日后续先统一普通ggml CUDA区域的临时寿命复用，再接入共同Graph workspace，
并补齐BF16逐算子承载；当前128-token场景Hyle显存降至3.939 GiB。各阶段变更与同源A/B见文末，
不能把旧表6.431 GiB当作修复后的数值。

每个场景3对独立进程，按H/L、L/H、H/L顺序运行，表列中位数。输入前缀和31个decode输入逐token一致，
两边分别反馈自己的KV/conv/GDN状态；未读取对方状态、hidden或logits。所有吞吐测量未挂profiler。
21-token场景另有一次关闭内存监控的对照，两端预热后decode分别76.14和106.16 tokens/s，未作开销修正。

| 指标 | Hyle | llama.cpp |
|---|---:|---:|
| 128-token场景：权重/模型加载子段 | 3.086 s | 1.603 s |
| 有效载荷加载率 | 1.136 GiB/s | 2.188 GiB/s |
| 进程启动到可推理 | 3.892 s | 1.772 s |
| 进程启动到首token | 3.968 s | 1.877 s |
| 首次prefill，128 tokens | 1680.1 tokens/s | 1223.6 tokens/s |
| decode，全部31步，含原生预热 | 69.3 tokens/s | 104.1 tokens/s |
| decode7–31，预热后25步 | 77.4 tokens/s | 105.8 tokens/s |
| 进程Dedicated显存采样峰值 | 6.431 GiB | 3.946 GiB |
| 进程RAM Working Set峰值 | 1.501 GiB | 4.069 GiB |

短输入控制：21-token首次prefill为286.6 / 222.4 tokens/s，预热后decode为78.1 / 106.7 tokens/s，
Dedicated显存峰值为4.327 / 3.825 GiB，RAM Working Set峰值为1.501 / 4.069 GiB。
128-token场景中，Hyle预热后decode低约26.9%，进程启动到ready耗时约2.20倍，Dedicated显存多约2.484 GiB。
这是当前配置的实际输项；首次prefill更快不能抵销加载、decode和显存问题，也不能外推长prefill或质量等价。

加载子段的前端工作不同：Hyle从原safetensors按manifest读取、分配并同步上传320个张量，
llama从单份官方converter生成的text-only/no-MTP GGUF加载，含原格式的metadata和tokenizer处理。
原text payload为3,763,655,360B，GGUF的有效text payload为3,764,747,520B（187 BF16＋133 F32无损例外）。
因此同时保留从进程启动到ready和首token的完整端点，不单用加载子段宣称等工作胜负。
Hyle的ready计入prefill/T1原冷编译；双方都不额外预跑数学。首次prefill仍包含各自原生首次执行开销。

完整step从输入准备前开始，到GPU选出的token在host可读为止，包含布局、绑定、重放准备和完成等待。
仅使用旧Hyle submit→token的seconds会漏费用，本表使用step_seconds。
原生预热没有被删除：全31步与实际decode7–31窗口分别列出；后者不是长期运行或热稳定性承诺。
llama原公开tensor_buft_overrides将默认在CPU的input embedding置于CUDA0，并与tied output共享原tensor/backing；
实际25/25层、6个KV层、18个recurrent层均CUDA0，32次GPU sampler快路，没有CPU采样fallback。
其原框架仍额外回读248320个sampled logits，费用保留；未修改上游以消除该工作。

显存采用本PID的Windows GPU Process Memory Dedicated/Shared，50ms采样，峰值为采样下界；
本机WDDM的NVML进程显存为N/A，未把整卡占用当本进程VRAM。RAM为PSAPI Working Set，含映射/共享驻留页，
不能据此推导整机物理RAM节省相同数量。PrivateUsage/PeakPagefileUsage另存为提交量，不能与GPU Shared相加当物理内存。
短场景的完整独立memory-review保留阶段前后点、时间偏差和缺样；尤其不能将释放后的最近点冒充推理驻留。

提示21→128 tokens、context容量始终256时，Hyle显存峰值增加2.104 GiB，peer仅增加0.121 GiB。
128场景每组model_loaded→ready间均有9个有效样本，ready前已观察到约6.23–6.30 GiB；
这之前尚未首次提交数学。自loaded后首样本至峰值的增长中，92.8%–95.3%已在ready前存在，
该区间包含状态初始化和prefill/T1各两个执行槽，不能把全部费用归因于某一allocator。
源码确认ggml非tiled Invocation逐非view/nonexternal tensor独立分配buffer并随槽保留，没有原生liveness复用；
这是首要调查候选，尚未逐buffer定量。显式Graph workspace增长仅约33.3 MiB，
默认GDN仅保最终状态，不是误保所有token状态。逐样本与有界静态算术见128-prefill/memory-review.json。
128场景推理期间RAM Working Set中位数为1.108 / 4.069 GiB，Private Commit为8.254 / 4.882 GiB；
因此Hyle Working Set较低不能直接解释为主机内存总体代价更低。

双方中间精度仍有差异：Hyle多处保BF16边界，peer多处为F32。短场景预测top1为30/32相同（decode23、29不同），
128-token场景为32/32；这些都不是独立质量评测或整模型精度接受。
各场景源代码、命令、逐步计时、预测IDs与进程内存样本见[性能原始记录](qwen-full-cuda-comparison.zip)，
正常64-token greedy生成也在包内，达到步数上限而非EOS。模型转换/设备/ABI身份及原始peer构建见
[llama参考准备](llama-reference-preparation.zip)；
不包含SDK、object、image或额外模型副本。原生capture观测另见
[实际trace](../../correctness/inference-foundation-review-2026-10-18/qwen-native-graph-observation.zip)。

归档后通过根目录test.ps1精确删除20个已完成实验exe/map/lib/exp及1个写错checkout位置的证据zip副本，
共664,878,137B；可复用object、SDK、原模型及单份转换GGUF保留。[清理记录](qwen-benchmark-reclaim.zip)

## 原生临时缓冲寿命复用

同日沿原Invocation将普通区域与分块attention统一到原生gallocr：冷reserve临时空间，
真实绑定后仅安装已有地址；保持常量、校验输入、输出/root、各槽所有权及退休顺序。
移除了普通区域逐张量独立分配的分支，没有新增公共选项、执行器或数学实现。
只替换一个provider宿主object，未重编native/SDK。同一128-token场景重新运行3对独立进程B/A、A/B、B/A：

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| PID Dedicated显存峰值 | 6.431 GiB | 5.312 GiB |
| 首次prefill | 1647.6 tokens/s | 1653.0 tokens/s |
| decode7–31 | 77.32 tokens/s | 77.36 tokens/s |

显存三次均减少1,201,676,288B（1.119 GiB），吞吐无明显变化。21-token单对控制为4.327→4.159 GiB。
21/128-token各32阶段的logits和最后hidden共128份向量与旧实现逐字节相同；不是新的通用质量接受。
原24层Add长链只publish最终输出，实际native retained从184降到8 MiB，旧实现明确不满足128 MiB宽限，
新实现通过且精确输出25*x、原输入不变。published view/root、旧结果、checked失败不发布、双slot换地址等4项通过；
原TypedSlice负控错误地强制prepare成功，允许它在prepare按InvalidArgument提前拒绝坏对齐后，该项单独1/1通过。
原失败日志、初版fixture及更正后记录均保留，没有把006原4/5记成全过。

该阶段仍比原llama参考峰值高约1.365 GiB；每region/shape/slot的独立原生workspace当时尚未接入共同Graph liveness，
原生首次执行、融合地址条件及其他模型功能继续审查。本次不据此宣布架构goal完成。
[源码、原始A/B、决策数据与回归](ggml-cuda-liveness.zip)另含下一步沿既有caller workspace接线的只读设计，未将设计当作已实现。
本修复归档对相同决策payload去重，保存全部逻辑文件身份；经test.ps1精确清理16个已完成image/map/lib/exp，
回收599,196,703B，保留4个新增复用object及全部源码/日志/数据/SDK。[清理记录](ggml-cuda-liveness-reclaim.zip)

## 接入共同 workspace

第二阶段沿既有ProviderPrepared.workspace与create_invocation(BufferView)接线，不增加公共API或新执行器。
冷编译用空地址的真实buffer描述测量精确chunk span，不分配device payload、不上传常量、不执行数学；
普通Invocation将原gallocr chunk映射到真实caller workspace，逐项核对size/offset，禁止容量不足时另开私有池。
返回的buffer保留canonical CUDA类型；原安全ID/常量和原生BLAS/pool/cache仍独立持有。
Graph adapter转发真实需求，由现有HB/liveness规划复用有序region的空间。Direct调用按标准requirement提供并持有workspace，
Graph消费者入口不变。显式冷constant packing仍自行持有临时空间，未把它变成普通执行的隐藏fallback。

只替换provider与Graph adapter两个宿主object，native数学、共同Graph/Compute owner及SDK均复用。
同一128-token固定输入再次按B/A、A/B、B/A运行3对独立进程，性能运行均无trace/profiler：

| 指标 | 仅区域内复用 | 接入共同workspace |
|---|---:|---:|
| PID Dedicated显存峰值 | 5.312 GiB | 3.972 GiB |
| 首次prefill | 1702.0 tokens/s | 1706.4 tokens/s |
| decode7–31 | 77.47 tokens/s | 77.95 tokens/s |
| 进程启动到ready | 3.777 s | 3.772 s |

显存三次均减少1,438,658,560B（1.340 GiB）；吞吐与启动变化不足以宣称改善。21-token单对控制为4.159→3.872 GiB。
从首轮6.431 GiB累计降至3.972 GiB，仍比原llama参考的3.946 GiB高约26 MiB；peer本阶段未重跑。
21/128-token各32阶段共128份logits/hidden向量与上一阶段逐字节相同，数学与中间精度未调整。
新两region测试中，各原生需求8 MiB，加上8 MiB边界张量后Graph总空间16 MiB，证明共用而非相加。
8个相关测试TU鲜编，唯一过滤运行10/10通过，覆盖普通attention、GDN快照重绑、checked失败不发布/重建、
abandon/close、view/root、旧结果和双slot。两次首次编译配方/警告失败及修正均保留，未重编native或SDK。
多chunk与其他模型的完整旅程仍未因此实测验收，独立shape与并发slot仍各自持有backing，decode和加载输项继续处理。
[源码、精确命令、A/B原始数据与回归](ggml-cuda-caller-workspace.zip)。
本阶段归档后经根目录test.ps1精确清理12个已完成image/map/lib/exp，回收451,978,680B；
2个产品与8个测试object、全部数据/源码/日志和SDK保留。[清理记录](ggml-cuda-caller-workspace-reclaim.zip)

后续全库调用者扫描另迁移8个测试/fixture文件中的9处Direct调用：按实际prepared operation的
workspace_requirement分配并持有所选resource的缓冲，零需求仍传空view。8个CUDA测试TU增量编译成功，
8个代表用例全部通过、无跳过，83项实际依赖复核无漂移；CPU共享fixture分支未运行，未重建native或SDK。
[迁移源码、精确命令和回归记录](ggml-cuda-workspace-consumers.zip)保留扫描边界与未修改的零需求调用者。
归档核对后经根目录test.ps1精确删除4个已完成image/map/lib/exp，回收153,514,557B，保留8个新object。
[清理记录](ggml-cuda-workspace-consumers-reclaim.zip)

同包另含冷常量打包的只读审查：在已接纳原生工作后，若checked drain与析构drain均未证明退休，
当前Status工厂仍进入原生对象和失败产物清理，违反其返回前完成准备的合同。该缺口未修复，未复现UAF；
普通free可能同步、终态设备错误及实际context销毁的情况分别记录，不能把观察错误直接当作仍有活内核。

## Decode 成本复核

较早共同workspace候选（capture复用之前）的三轮128-token历史记录中，decode7–31每步耗时中位数为12.67–12.80ms。
上传、layout更新、绑定和重放准备合计中位数0.33–0.35ms，约2.7%；submit返回前为4.37–4.58ms，
约35%，包含驱动调用和可能的阻塞，与GPU执行重叠，不能直接当作可节省的CPU费用。
此前capacity64的软件trace中，预热后每步有401次原生Graph launch；它说明提交粒度较碎，
不能代替本阶段双方GPU关键路径对照，也未定量证明BF16、转换或路由是decode差距主因。
加载消费者则逐张量读取、分配并同步上传320份权重，尚未流水化；加载与decode问题分别调查。
[历史离线重算脚本、逐步统计及旧trace结构边界](qwen-decode-cost-review.zip)。

最新同窗三对里，Hyle PreferCapture的准备为0.499–0.726ms/步，submit调用0.128–0.133ms，
返回后到完成读回11.142–11.556ms。Peer的batch准备0.14–0.19µs，decode返回0.161–0.243ms，
同步10.205–10.360ms，token访问与采样合计0.018–0.027ms。准备、submit/decode调用、等待的边界不同，
调用期间还可能与GPU交叠；不能把等待全算作GPU内核，也不能把两端分段差全算作独占CPU成本。
每对在入队API返回前的墙钟差为0.454–0.693ms，占对应完整step差的27–43%；
因此准备/入队仍是当前剩余差距的重要候选，不能用历史105.8的跨时目标排除它。

## 原生向量路由实验已撤回

同一128-token输入和31步固定decode的软件trace显示，decode7–31每步Hyle为2400个kernel、401次原生Graph launch，
peer为558个kernel、1次Graph launch；187个Hyle投影实际走cuBLAS gemvx，peer为163个ggml mmvf。
peer的24次差值集中在6144输出行的融合kernel组，但未采集逐逻辑node参数，不能据此要求消除Hyle的BF16舍入边界。
[双方原始trace、分箱和精度边界](qwen-decode-native-route.zip)保留实际记录；软件profile时长不用于公平吞吐比较。

随后仅改两个宿主owner，将187个单向量投影交给原ggml mmvf，保留公共BF16输入/输出边界，未重编native或SDK。
候选实际全部走mmvf，却每步新增374个scalar cast（两向各187），kernel增至2774、Graph launch增至569；
冷Graph步骤减少19并不代表实际提交更少。[候选唯一trace](qwen-decode-native-route-candidate.zip)保留这一反例。
独立无profile的3对B/A、A/B、B/A中，首次prefill中位数1670.7→1503.4 tokens/s，
预热后decode中位数78.042→76.090 tokens/s，显存采样峰值3.972→4.009 GiB，因此撤回默认路由修改及实验测试源码。
这没有证明新增cast是全部回退原因，也没有证明mmvf在其他装配中普遍更慢。

两个定向数学/寿命测试通过，21/128-token决策旅程预测IDs一致，但128份hidden/logits向量仅2份逐字节相同；
这些结果不构成精度、质量或完整模型接受。四个源文件的精确恢复收据、冻结before/candidate/diff、全部失败编链日志、
原始性能数据及去重后的双方决策向量见[已否定实验归档](cuda-native-matvec-rejected.zip)。

## BF16 逐算子直接承载

同日为原ggml CUDA binary/unary模板补齐同BF16输入/输出，以及广播所需的typed Repeat，
消除原先逐节点先转F32、运算后再转回BF16的临时张量。每个算子仍有自己的BF16落存边界，
BF16连续Add/Mul禁用跨节点融合以保留逐节点RNE；未改变原F32/F16与Rsqrt数学。
仅重编3个原CUDA TU和1个Hyle宿主TU，unary保持原precise参数，无整套SDK重建。
原CUDA头的C4819警告保留；初次runtime-only源码派生漏掉IQ1扩展，在编译前被拒绝，
随后使用含IQ1的完整canonical派生，未把该初版用于运行。

同一128-token＋31个固定decode输入、context容量256、两个原槽，3对无profiler独立进程按B/A、A/B、B/A运行；
两端均使用原50ms PID显存采样，表列中位数：

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 首次prefill | 1589.1 tokens/s | 1765.1 tokens/s |
| decode7–31 | 77.32 tokens/s | 79.84 tokens/s |
| PID Dedicated显存采样峰值 | 3.972 GiB | 3.939 GiB |

首次prefill提高约11.1%，预热后decode提高约3.25%，显存三次均减少34 MiB。
加载和启动未见改善，llama参考本轮未重跑；采样峰值仍是下界。21/128-token各32阶段共128份
logits/hidden与旧caller-workspace产物逐字节相同，不构成独立质量或全平台验收。
回归有3个不同用例通过：原联合运行Direct与IQ1通过、Graph因fixture借入buffer寿命失败；
修正fixture重置顺序后，Graph单独重跑通过。初版BF16比较和Graph调用API编译失败也保留，未记作首轮全过。
[源码、派生纠正、精确构建、原始A/B与回归记录](cuda-bfloat-elementwise.zip)保存上述边界；
相同决策向量引用旧归档的精确payload哈希，scratch原始数据仍保留。

独立一次Nsight观测确认decode7–31每步实际执行BF16 Add 84、Mul 102、Sigmoid 24次，
BF16→F32 / F32→BF16转换从原801 / 427降为405 / 217；原生GraphLaunch仍为401次，187个cuBLAS投影仍在。
该trace只用于接线和调用计数，不用其时长代替上表吞吐；[完整原始trace](cuda-bfloat-elementwise-trace.zip)独立保留。

归档后通过根目录test.ps1精确清理22个已完成image/map/lib/exp与1个已撤回且未引用的实验header，共808,658,511B；保留10个新增object、8份原始Nsight/SQLite及全部决策数据、源码、SDK和模型。[清理记录](cuda-decode-investigation-reclaim.zip)

## 普通 Graph 的冷绑定与连续捕获

普通算子和区域共用原 Invocation 的冷绑定；连续合格命令可在原 segment 内一起录制，
保留不能捕获的命令及原 Completion 退休边界。ggml CUDA 复用原编码器直接录制，
不执行私有图 warmup 或嵌套 GraphLaunch；原 metadata 启用逻辑保持 Mean 的 CUB 数学选择。
10个不同用例分轮通过：2个Softplus跨provider/双槽用例、5个CPU合同控制、2个旧CUDA capture/checked控制、
1个宽度50000的Mean冷录制与逐bit对照。最初rank-2 MatMul fixture拒绝、CUB宏资格误判及旧owner链接失败均保留，
未记作首轮全过；调整过的旧TensorRegion helper本轮未另运行。

该轮完整Qwen仅运行1对功能探针：21-token prompt＋3个固定decode，Commands与PreferCapture各进程成功，
8份hidden/logits逐字节相同，实际路由分别为Commands和Hybrid。首次prefill冷准备为2.05与51.72ms，
三个decode的冷准备分别落在0.61–1.49与6.40–9.28ms。该轮每步reset/rebind会重新录制，
这不是摊销后的capture吞吐对照，不能宣称decode性能缺口已闭合，也不构成独立精度或质量验收。
[源码、失败经过、精确构建和原始结果](cuda-coarse-capture.zip)保存本轮证据；
决策向量引用既有归档中已逐SHA核实的相同payload，scratch原始数据全部保留。

根目录 test.ps1 已精确清理本轮12个 image/map/lib/exp，共402,821,651B；保留53个对象、16份原始向量，主ZIP不变。[已执行清理记录](cuda-coarse-capture-reclaim.zip)

后续已将同址录制保留在原Invocation：reset/rebind撤销执行权限，重新准备后可复用实际参数未变的录制；
配置或原生资源变化先销毁对应录制，失败也不例外。CommandCapture v2的身份能力默认未知，
由实际后端证明相同TensorView仍代表相同物理参数；不靠Provider或模型名单。ggml CUDA也复用既有指针绑定，
每次仍校验全部端口，仅省去地址未变时重复的Scope、bind与finish_bind。

CPU11与最终CUDA6共17个不同用例通过，两次初始CPU链接配方失败保留；未重编原生数学或SDK。
最终21-token prompt＋3步固定decode单对的8份hidden/logits逐字节等于旧caller归档；prefill及前两次decode各新录7段，
第三次decode新录0段、复用7段。最终同镜像128-token＋31步decode按B/A、A/B、B/A运行三对独立进程，
无profiler或显存监控；以下均按包含上传、冷准备、submit与读回的完整step计时，列三次中位数：

| 指标 | Commands | PreferCapture |
|---|---:|---:|
| 首次prefill | 1716.43 tokens/s | 1660.51 tokens/s |
| decode1–31 | 57.80 tokens/s | 74.06 tokens/s |
| decode7–31 | 62.32 tokens/s | 77.46 tokens/s |

六次定时运行的32个预测ID一致，只有上述独立功能对提供逐bit向量检查。Prefill仍有回退。
旧BF16的79.84同样按完整step的decode7–31统计；复原119个输入的旧闭包同窗控制为68.47/66.07，
该组发生最终绑定去重之前。初轮运行后观测到用户后台GPU竞争，不能据此归因各轮时长，
也不能宣称超过旧79.84或历史差异已经消除。本轮未重测llama、显存或独立质量。
[源码、17项回归、三组独立对照与原始记录](cuda-capture-reuse.zip)保留初轮、历史控制和最终结果；
32份scratch向量均核对旧ZIP的精确payload后引用，未重复复制。

根目录 test.ps1 已按两份归档及精确身份清理本轮33个文件（24个image/map/lib/exp与9个已替代对象），共878,181,587B；保留18个最终本轮对象、1个既有Direct测试对象与32份原始向量，主ZIP不变。[已执行清理记录](cuda-capture-reuse-reclaim.zip)

最新同窗按H/L、L/H、H/L完成三对无profiler、无外部显存监控的128＋31步对照：Hyle PreferCapture/peer的prefill中位1711.33/1278.20、decode1–31为78.60/93.32、decode7–31为83.36/95.16 tokens/s；32个预测ID一致。当前warm差约1.49ms/步（Hyle吞吐低12.4%），不是历史105.8对77.5的跨时差；本轮只确认完整step差距，不宣称内核主因或质量验收。

独立Nsight Systems观测确认当前每步1794/558个kernel与7/1次GraphLaunch，Hyle其中622个BF16↔F32转换；
capture已聚合提交，尚未消除数学节点。当前FFN的rank-2投影与Silu不被同一ggml region接纳，需真实融合lowering，
不能只删除支持谓词或舍去BF16中间舍入。Hyle准备约0.50–0.73ms/步仍是差距候选；软件profile时长不能加总解释未profile的1.49ms。
Nsight Compute各采同一LM-head权重形状的一颗kernel，两端DRAM吞吐约491/489GB/s、峰值占比97.55/97.04%；
该单核观测支持其已接近带宽上限，不代表全部投影或全模型。两端activation/output精度不同，缓存与时钟未锁定。

[当前同窗基线、Nsight/Ncu原始记录与融合边界](cuda-current-peer.zip)保留首次Ncu运行环境失败及peer辅助JSON被命令元数据覆盖的限制；单个LM-head计数器不是全模型性能结论，融合与cold-pack方案尚未实施。

根目录 test.ps1 已精确清理本轮6个image/map/lib/exp，共208,092,023B；保留91个对象、6份原始Nsight/SQLite/Ncu数据，主ZIP不变。[已执行清理记录](cuda-current-peer-reclaim.zip)

## 保留逐节点舍入的 BF16 连续融合

后续在原 ggml CUDA multi-Add/Mul 路径接通 BF16，2–8 个同类节点的每次结果在寄存器中执行 BF16 RNE，
检查广播、单用、发布输出与实际地址重叠，保留不合格后缀的普通执行。原 F32/F16 数学分支不变；
没有新增共同 Graph 路由或执行器。仅重编两个 CUDA TU、两个受当前头文件影响的测试 TU，复用依赖缓存。
三个定向用例通过，覆盖深度2/3/8、广播、双槽换址及发布中间值；3×17形状不覆盖unravel路径。
128-token prefill＋31步固定decode的64份hidden/logits与本轮旧实现逐字节相同，32个预测ID相同。

三对无profiler、无显存监控的独立进程按B/C、C/B、B/C执行，prefill中位1655.49→1668.36、
decode1–31为74.99→74.22、decode7–31为79.04→77.80 tokens/s。warm中位低约1.57%，
本轮未证明性能改善，也未据此归因差异。独立trace确认定向用例真实执行多源BF16 kernel；
完整Qwen没有命中该模式，每步仍有BF16 Add84、Mul102，不能把局部融合能力当作模型优化。
Nsight软件观测只用于调用计数，原driver/collector版本警告保留，不用于吞吐对照。

[实现、精确构建、原始结果及两次trace](cuda-bf16-chain-fusion.zip)记录上述边界。
首轮锁文件树哈希因排序方式错误被CMake拒绝，改为与CMake一致的排序后定向审计通过；
完整FFN等模型融合、准备成本及cold-pack来源/退休所有权缺口仍未修复。

根目录 test.ps1 已精确清理本轮12个image/map/lib/exp，共450,119,947B；保留4个新对象、
128份原始向量与两次原始trace/SQLite，主ZIP不变。[已执行清理记录](cuda-bf16-chain-fusion-reclaim.zip)

## 投影接线与输入/累加精度

尝试让ggml CUDA承接plain rank2与transpose-dense RHS后，Qwen的49个MatMul改路由：
18个in_proj_a、18个in_proj_b、6个q_proj、6个k_proj及LM head；72个FFN投影仍为cuBLAS。
公开step从1204降为1185，但初次候选warm探针仅5.66/9.15 tokens/s；紧邻旧路径为89.88。
这不是性能改善。32个预测ID相同，但64份hidden/logits均非bit-identical，最大绝对差0.25。

一次独立Nsight trace确认每步1977颗kernel（旧1794），GraphLaunch仍为7；新增98次BF16→F32原生转换。
其中一次尺寸恰好248320×2048的LM head转换，每步产生约1.89GiB F32数据；25步转换与后续F32 GEMV
占所记录kernel时长约88.2%。无GPU活动记录的空隙不足以解释主要退化；奇偶步长尾不能由此断言是纯数学或分页。

原生`GGML_PREC_F32`累加请求强制了F32 operands。修复仅限fast NVIDIA、双BF16来源、F32输出：
沿既有BF16 GEMM保留输入，仍用F32累加/输出和公开BF16 RNE；F32 strict、F16、mixed来源及ENV覆盖保持。
仅重编一份CUDA TU，产品patch与v9锁已更新；完整mixed/IQ1生成源码逐字节等于被测源码，CMake定向锁审计通过。
该候选64/64向量逐字节等于旧路径，6项定向测试通过；没有再采修复后的profile，不据此报告新kernel计数。

三对无profiler、无显存监控的新进程按B/C、C/B、B/C执行，128-token prompt、31个固定decode输入、容量256、
Direct进度与PreferCapture相同，按含prepare的完整step计时：

| 指标 | 旧路径中位数 | 扩大rank2接线＋累加修复候选中位数 |
|---|---:|---:|
| 首次prefill tokens/s | 1753.25 | 1239.61 |
| decode1–31 tokens/s | 86.35 | 81.10 |
| decode7–31 tokens/s | 91.39 | 91.03 |

warm基本持平，首次prepare/capture成本增加；未证明净收益。因此最终恢复原plain rank2准入边界，
保留native BF16输入/F32累加修复及batched transpose-dense RHS能力，不新增模型/Provider名单或修改共同Graph。
交付版本重编3个受影响host TU、6个定向用例通过；一次完整128＋31功能探针的64份向量、32个ID、路由与边界均逐字节等于旧路径。
不把该单次运行当性能结论，不代表完整FFN融合、独立质量验收或其他平台/设备验证。

[候选源码、初始失败、数值/路由对照、三对实测与原始trace](cuda-dense-projection.zip)保留推进及撤回依据。
最初Graph测试误把零拷贝View计为无step、修正时误用Result为指针，以及首轮把CUDA单层manifest哈希当mixed/IQ1哈希的集成断言失败均未记作通过。

归档后通过根目录test.ps1精确清理本轮36个image/map/lib/exp，共1,359,901,065B；
保留8个对象、原始trace/SQLite、全部256份向量和复现源码/命令。[已执行清理记录](cuda-dense-projection-reclaim.zip)

## 稳定 SiLU 融合与普通区域的冷偏好

单算子提案只发现一次，region回调借用当前target、实际Provider槽的叶排名。ggml普通组合不覆盖已知更优叶；
真实融合独立提案，不靠普通大岛固定高优先级或屏蔽rank2能力。删去SDPA的旧优先级拒绝，数学准入仍由原Provider负责。
默认cuBLAS投影及raw Context的GGML400/cuBLAS300同分反例均保留独立投影，原187个cuBLAS MatMul节点不变。
没有增加热路由、第二执行器或模型名单，也不承诺完整普通大岛作为另一份packing/workspace预算后备。

F32/BF16普通SiLU现在保留stable-tail和`-Inf→+0`，融合Mul时保留BF16中间RNE，不能替换成另一数值合同的GLU。
按def-use识别可收缩的非连续节点集，公开结果、额外读者、自乘及地址重叠仍受保护；F16该变体仍未接入。
仅重编2个CUDA TU及受本轮接口/源码影响的host TU，复用SDK与依赖。8个CPU、最终12个CUDA用例通过，无skip；
包含双槽换址与旧结果持有、published中间值、数值尾部/非有限值、rank2/batched转置RHS及自然路由。
两次初始缓存配方元数据字段假设失败发生在编译启动前，修正后定向构建通过；不是首轮全部成功。

128-token prompt＋31个固定decode输入、容量256、Direct进度与PreferCapture下，最终候选64份hidden/logits、32个ID
与旧路径逐字节相同。T1/T128均选42个融合候选，公开step1204→1180；capture均首3阶段各record7，随后29阶段各reuse7。
这只证明这条固定旅程的等价，不代替独立模型质量或跨平台验收。

三对无profiler、无外部显存监控的新进程按B/C、C/B、B/C运行，完整step包含准备、提交、等待及token读取：

| 指标 | 旧路径中位数 | 本轮候选中位数 |
|---|---:|---:|
| 首次prefill tokens/s | 1642.54 | 1604.92 |
| decode1–31 tokens/s | 76.99 | 77.19 |
| decode7–31 tokens/s | 81.23 | 80.21 |

本轮未证明提速；warm样本范围旧72.73–81.98、新72.80–84.12。prefill中位低2.29%、warm低1.26%，
不能忽略这些差异，也不能仅凭三对波动归因。现有日志未见冷prepare或submit一致增加；submit返回至token的区段
混合GPU执行、等待与读取，不是纯GPU时长。本轮没有重测llama、内存峰值或独立质量。

独立一次Nsight trace及官方SQLite/CSV确认：decode1–31每步实际执行24个BF16＋18个F32稳定SiLU/Mul融合kernel，
总kernel1794→1752，graph-node1777→1735，GraphLaunch仍7；prefill仅24个BF16物理融合，1932→1908。
全部32阶段除SiLU/Mul相关符号外的kernel次数均相同，转换与cuBLAS未增加；18个卷积后BF16 SiLU仍走SGLang。
driver13.4与collector13.3兼容警告原样保留，profile时长不用于上表吞吐比较。
prefill的18个F32逻辑候选未物理融合仍需处理。源码核对发现SSM的`[1,T,16,128]`在Unary前展成二维，
输出恢复四维后又被Binary展回二维；T128形成`UNARY→RESHAPE4→RESHAPE2→MUL`，破坏原生连续匹配及直接引用条件，
T1则省掉两个同值reshape。这个确定的lowering链与物理计数吻合，尚未逐实例观测guard返回；
后续应在共同lowering消除合法的逆reshape，不另加模型或融合名单。完整gate/up投影融合、其余量化接线、
准备与常量packing寿命缺口也未完成。

[源码、构建与回归、数值对照、三对实测和原始trace](cuda-stable-silu.zip)保留上述区别，避免把区域数当作功能或性能完成。

归档逐CRC/SHA核验后，通过根目录test.ps1精确清理本轮16个image/map/lib/exp，共557,035,045B；
保留11个对象、192份原始向量和原始Nsight/SQLite，不复制SDK或模型。[已执行清理记录](cuda-stable-silu-reclaim.zip)

## 相邻逆 reshape 与 prefill 融合

已消除上述prefill断边：ggml CUDA的`matrix()`仅检查相邻、零偏移、同dtype且连续的reshape，
若其直接源已是目标二维布局则复用该源；O(1)检查，不跨Cast/CONT/Transpose/非零View，不新增缓存或名单。
公开结果仍标记原tensor与root，多读者及原生地址重叠检查保留。只重编2个host TU，SDK、原生CUDA对象与其余链接输入均复用。

8个定向用例通过，无skip。新rank4消费case覆盖F32/BF16、公开SiLU与额外reader、双槽两轮新址及旧结果；
另核norm/unary、9种GGUF矩阵格式、Q8 grouped experts、转置RHS与原BF16逐节点舍入。
量化用例含原生CPU表初始化，整组约42.8秒，未据此扩大验证矩阵。
128-token prompt＋31个固定decode输入的64份hidden/logits与上一轮候选逐字节相同，32个ID、capture计数相同。
当前对照baseline重链接自上一轮最终对象闭包，先逐SHA核对所有输入；不是拿未融合的旧版冒作本轮baseline。

三对无profiler、无显存监控的新进程按B/C、C/B、B/C执行，同一完整step边界：

| 指标 | 修复前中位数 | 修复后中位数 |
|---|---:|---:|
| 首次prefill tokens/s | 1667.86 | 1707.41 |
| decode1–31 tokens/s | 76.54 | 77.46 |
| decode7–31 tokens/s | 81.85 | 81.08 |

prefill中位提高2.37%、完整decode提高1.20%，warm低0.94%；三对样本不证明统计显著性或decode改善。
warm原样本79.05–82.44、新80.15–82.33，不能跨轮拼接吞吐。未新增llama对照、显存峰值或独立质量验收。

独立一次Nsight和官方SQLite/CSV确认，prefill原18个F32独立SiLU及18个Mul被18个稳定gated kernel替代，
1908→1890；其余24个BF16融合保留。decode每步1752、graph-node1735和GraphLaunch7完全不变；
所有32阶段转换次数不变，没有新增多源Add/Mul融合。结合前轮，当前prefill/decode分别比原路径少42个kernel。
driver13.4/collector13.3兼容警告保留，profile只证明调用计数，不用于上表速度归因。

[最小源码改动、回归、逐bit对照、配对实测和原始trace](cuda-reshape-reuse.zip)保留完整依据。
完整投影融合、更多量化接线及冷packing失败退休所有权仍需继续处理；本轮没有修改公共ABI或共同执行器。

归档逐CRC/SHA核验后，根目录test.ps1精确清理12个image/map/lib/exp，共453,720,716B；
保留2个host对象、64份原始向量与原始Nsight/SQLite，复用的native/SDK输入不动。[已执行清理记录](cuda-reshape-reuse-reclaim.zip)

## BF16 向量投影的直接承载

[本轮源码、命令、离线分析与配对数据](cuda-bfloat-matvec.zip)先核原有1752-kernel trace：
warm7–31的kernel时长总和中，187次BF16 GEMV占83.20%，其中48次gate/up占26.88%、
24次down占12.13%、词表head占22.55%；622次BF16/F32转换占5.13%。24个FFN稳定SiLU/Mul
融合仅占0.238%，与gate/up之间没有额外cast。这里的角色按源码和顺序推断；原始CUDA活动可直接
证明计数与时长。该单次profile绑定reshape阶段镜像，不作为最新镜像的速度或差距主因证明。

在同一MMVF模板接通BF16权重、activation和output：精确load-widen、原F32累加/归约、BF16 RNE store，
限定单向量、偶数K、连续且无batch/IDs/fusion；其他形状和原F32实例保留原程序。
相应lowering让这条原生向量路径无需整份activation扩宽和output缩窄。公共MatMul描述不变，
默认Provider优先级未改，也没有添加模型名单、权重副本、GLU或普通SiLU epilogue。
CUDA补丁版本为v11，完整CPU/IQ1/CUDA组合重放source ID为`7caf606377daf938`。

仅增量编译2个native和3个host TU（含准确的Provider版本信息），未重编SDK；首个NVCC attempt因
子cmd未找到cl而失败并保留，缩短PATH并显式固定host compiler后完成。两个新增测试通过、无skip，
覆盖两种RHS布局、两方向ties-to-even、双槽两轮新地址与旧结果、奇数K及F32控制；5项既有CUDA
packing/专家/控制边回归亦通过。默认Qwen128prompt＋31fixed-decode的64份hidden/logits逐byte一致，
32个ID及capture计数相同；这不表示新MMVF已经成为该默认模型路径，也不是独立质量验收。

筛选使用公开Graph的真实RequireCapture和同一份benchmark镜像，每种形状3对AB/BA/AB独立进程。
gate/up和down各轮换3份25,165,824B权重，超过4070的L2；head单份1,017,118,720B。
每进程预热至少3个完整轮换周期，随后128步。下面为各进程中位数再取三轮中位数，单位µs：

| 形状K→N | cuBLAS完整调用 | typed MMVF完整调用 | cuBLAS stream区间 | typed MMVF stream区间 |
| --- | ---: | ---: | ---: | ---: |
| gate/up 2048→6144 | 66.00 | 65.90 | 57.344 | 55.296 |
| down 6144→2048 | 66.50 | 65.50 | 58.160 | 57.152 |
| head 2048→248320 | 2179.95 | 2188.55 | 2154.580 | 2169.340 |

完整调用含event API、submit及wait，stream区间含host接纳/launch间隙，都不是kernel-only时间；
它们也不是模型完整step。小差异与进程间抖动并存，尚未建立统计显著收益，因此保留cuBLAS默认路由。
新载体为后续保留各舍入点的投影融合提供基础；更多转换、归一化子图、冷常量表达式与完整投影融合仍待审查。

归档317项payload共15,490,900B，SHA256
`5bb79e2b6192c9f0d2d13f4e7c2e30c3b48168b1b3eab9a8ddef433013f68388`，CRC与逐项SHA核对通过。
随后经根目录test.ps1精确清理12个image/map/lib/exp，回收449,169,876B；
保留本轮objects、原SDK和64份raw vectors。[已执行清理回执](cuda-bfloat-matvec-reclaim.json)

## 常量表达式冷准备与 RMSNorm 融合

选定 CUDA provider 现沿实际不可变、owned、ready 根发现纯常量子图，仍用原生 Recipe 和原 QueueSequence
计算，保留 dtype、数值顺序与 BF16 RNE。多输出共享一次准备；相同只读 Cast 共享物理结果。
常量 key 增加由 cache 拷贝保活的精确 opaque identity，region extension 为29，共同核心不解释算子。
所有未发布输出均有合法 ready owner、且没有效果或运行工作时，零 dispatch region 保留逻辑/控制依赖，
无需 Queue、原生槽或占位 kernel。普通动态输入与公开 writable 输出保留原合同。

本轮29次host TU编译（含修正后的增量），没有重编native数学或SDK。CPU13与CUDA10个不同用例通过、无skip，
覆盖精确缓存身份/临时key寿命、跨target控制链、全图零Queue、双槽/旧结果、动态与published负控、原packing/专家及BF16载体。
CUDA首轮9/10：原重复Cast共享用例发现packed bytes翻倍；实现修复后原字节断言通过，没有放宽。
另保留旧CPU闭包混入过期Vulkan接线的3次链接失败；CPU测试仅消费CPU/fake provider，最终剔除无关GPU host对象。
这不构成本轮跨平台、完整Vulkan或安装构建验收。

完整Qwen128prompt＋31fixed decode的64份hidden/logits逐byte不变，32个ID不变。新Nsight官方CSV/SQLite一致，
warm7–31每步均为1528 kernels；保留的reshape trace为1752。实际差值是79次BF16→F32 Cast、61次F32 Add、
18次Exp和18次Neg移入冷准备，以及48次F32 Mul并入原有RMSNorm。1024-thread scaled RMS从1→49次/步，
unscaled从48→0；GraphLaunch仍为7次/步。该旧trace只对照物理计数，不能用profile时长声称提速。

单RTX4070/driver617.14、相同权重/输入/精度/容量、AB/BA/AB各3个独立进程，均无profiler或内存采样。
以下为进程中位数；tokens/s使用包含upload、layout、binding、capture准备、submit与wait的`step_seconds`：

| 项目 | 原实现 | 冷表达式准备后 |
| --- | ---: | ---: |
| 首次prefill tokens/s | 1761.42 | 1956.94（+11.10%） |
| 全31步decode tokens/s | 86.10 | 88.20（+2.43%） |
| warm7–31 decode tokens/s | 91.00 | 93.13（+2.33%） |
| 权重加载 seconds | 3.05771 | 3.05267 |
| 形状准备 seconds | 0.314309 | 0.408473 |
| 启动至ready seconds | 3.45598 | 3.54707 |
| 启动至首token seconds | 3.52850 | 3.61272 |
| T128 caller workspace bytes/slot | 35,553,792 | 37,052,416 |
| T1 caller workspace bytes/slot | 2,201,984 | 2,291,584 |

热执行获益伴随约94ms额外形状准备、约84ms首token回退及workspace小幅增加；这些费用没有摊掉或漏计。
仅3对，不宣称统计显著、独立质量验收、总显存峰值改善或新的llama.cpp差距。主要投影数学仍走原cuBLAS。
此前194次常量候选中已移走176次；外部metadata View后的18次conv weight Cast、typed RMS、
GDN L2逐舍入点融合与完整投影融合仍待处理，不能写成冷准备或模型总表已全覆盖。

[源码、失败过程、23项回归、64份向量、配对原始数据与官方trace](cuda-constant-expressions.zip)支持上述结论。
归档622项payload、19,624,311B，SHA256
`b0c8a22d636b618cf65553433c801361e72b86bf9978e2e11b2ede8d7eaebca3`；CRC及逐项SHA核验通过。
随后执行根目录test.ps1，精确删除37个image/map/lib/exp、回收962,971,205B；全部对象缓存、
原始向量与trace保留，源码冻结无漂移。[已执行清理回执](cuda-constant-expressions-reclaim.json)

## metadata View 常量来源与卷积权重冷转换

Graph 现向 region 提供不可变输入的实际物理绑定、真实常量根和根内偏移，owner/ready 资格逐物理副本核定；
声明域中尚待运行期确定偏移的 alias 保留原路径。缓存按边界、根、偏移与精确描述隔离，CUDA 多根表达式
也将全部实际使用的根及偏移纳入 opaque identity。extension 为30；外部 View 仍是原 metadata alias，
没有增加原生 View 算术或改动 Cast 数学、逐节点舍入、默认投影路由。

21次 host TU 编译（首轮19，随后只修两个测试TU），没有重编 native 数学或SDK。最终 CPU19＋CUDA12
共31项通过、无skip，覆盖相同描述的不同切片/根、finite variant、实际跨域副本owner、原borrow lease、
双槽/换址/旧结果、外部View单例Cast无需热Queue、公开结果与动态输入保护。原Cast准入也实际支持并检查了
非连续permute来源。CPU首轮16/19：共用factory读首float时漏取4B subview，另一个多队列fixture漏
prepare_replay后直接解引用失败结果；仅修夹具并重链CPU，产品来源与所有权检查未放宽。初失败保留。

完整128prompt＋31fixed decode的64份hidden/logits逐字节不变，32个ID不变。一次新Nsight trace与官方
CSV/SQLite计数一致；warm7–31的25步每步1510 kernels，前轮留存trace为1528。唯一symbol差值为
BF16→F32 Cast从326→308次/步，对应18份卷积权重转换；GraphLaunch仍为7次。当前Qwen镜像SHA256为
`8bae44757f12381ed34570755e1ffec576cad82ca4542b48be51b2f50535bda9`。
stats首尝试因导出文件mtime检查拒绝；直接读取已成功导出的SQLite完成报告，没有重跑GPU。

本轮没有新的无profiler配对计时，不由kernel减少推导tokens/s、加载时间、显存峰值或llama.cpp差距改善。
typed norm/GDN L2、完整投影融合及其他模型/量化/平台仍未完成。
[源码、构建和初失败、31项回归、原始向量与官方trace](cuda-constant-view-origins.zip)保留本轮依据。
归档466项payload、19,335,291B，SHA256
`15118c9dba4e67466ae96eafe0623f13347b70b176c7dee1c31b709217d8f2c8`，CRC及逐项SHA核验通过。
经根目录test.ps1精确删除16个image/map/lib/exp，回收412,338,576B；对象、SDK、原始向量和trace保留，
源码冻结无漂移。[已执行清理回执](cuda-constant-view-origins-reclaim.json)

## BF16 RMSNorm 的扩宽与写回融合

复用原CUDA RMSNorm模板的F32逐线程累加、256/1024线程归约及rsqrt，接通
BF16→F32 Cast→RMSNorm→可选F32 scale→BF16 Cast。BF16在读取时精确扩宽，归一化与缩放后显式RNE写回；
F32原分支不改。另有BF16 norm→F32 scale的链路仍保中间BF16舍入，不并入weighted变体。
匹配按实际边、dtype、布局、读者与地址，正确计入CPY self引用；公开中间值和额外读者保原计算，
输出与实际输入/scale（含OP_NONE叶）重叠时保守回退。没有新增公共descriptor、Provider或Graph特殊分支。

成功增量为2个native TU＋3个host TU，没有重建SDK。初次native编译因验证overlay缺少父级include路径，
误取原norm.cuh而失败；修验证脚本后编译通过。第一次CUDA执行实际15项全过，但exact筛选另需一个
未被原闭包链接的旧norm控制，检查器如实失败；追加该测试TU，仅重链CUDA后最终16项通过、无skip。
新3项含独立GPU F32 oracle、微小数/FTZ/Inf/NaN、两种中间舍入合同、双槽换址/旧结果与RequireCapture。
仅NaN按类别核定，其余逐bit；F32既有FMA合同不放宽。保留初失败和检查器拒绝。

完整Qwen128prompt＋31fixed decode的64份hidden/logits逐byte不变，32个ID和7段capture复用不变。
一次新Nsight trace的官方CSV/SQLite计数一致：warm7–31每步1414 kernels，前轮留存trace为1510。
48个1024线程weighted RMS改为BF16直接输入/输出，BF16→F32 Cast从308→260、反向从217→169次/步；
其余norm只是模板符号增加类型参数，物理次数不变。仍有1个1024线程weighted、12个256线程weighted、
18个256线程unscaled走原F32路径，未把这些31项写成已覆盖；未用role推断代替原生匹配原因证明。

无profiler或内存监控的AB/BA/AB共三对独立进程，对照为前轮View常量修复后的原镜像，权重、输入、精度、
容量、消费者与计时边界相同；tokens/s仍使用包含upload/layout/binding/capture准备/submit/wait的step_seconds：

| 项目 | 原路径中位 | 融合后中位 |
| --- | ---: | ---: |
| 首次prefill tokens/s | 1937.36 | 1909.53（−1.44%） |
| 全31步decode tokens/s | 87.47 | 88.04（+0.65%） |
| warm7–31 decode tokens/s | 91.82 | 92.06（+0.26%） |
| 权重加载 seconds | 3.08638 | 3.06454 |
| 形状准备 seconds | 0.416200 | 0.416687 |
| T128 caller workspace bytes/slot | 38,191,104 | 38,191,104 |
| T1 caller workspace bytes/slot | 2,115,712 | 2,115,712 |

prefill三个原样本为2015.17/1937.36/1880.71，新样本为1910.95/1871.00/1909.53 tokens/s，范围重叠且存在漂移。
本轮没有证明端到端改善或统计显著性，也未排除小幅prefill回退；保留原始差异继续调查。
减少kernel不能推导显存下降：原native allocator仍可为被省略的中间结果预留槽。本轮未测总显存峰值、
新llama.cpp差距或独立模型质量；GDN L2、完整投影融合及其余模型/量化/平台仍待处理。

补丁v12按生产mixed CPU＋CUDA IQ1工具重放，三份变化源码及vendor lock均核对；
source ID为`f2f72269789731e6`，实际Provider版本沿生产CMake形式绑定此ID。
[源码、构建和初失败、16项回归、原始向量、配对计时与官方trace](cuda-rms-cast-fusion.zip)保留本轮依据。
归档315项payload、18,713,667B，SHA256
`1406ad467a8c79ae2dd826f9be2bf06c4628d4321f7ff95aab2853e7b34c8893`，CRC及逐项SHA核验通过。
经根目录test.ps1精确删除16个image/map/lib/exp，回收607,916,618B；对象、SDK、原始向量和trace保留，
源码冻结无漂移。[已执行清理回执](cuda-rms-cast-fusion-reclaim.json)

## RMS 同址分配的实际拒绝原因与修复

对原v12镜像只增临时host元数据日志，完成一次128-token prefill＋1步decode诊断；每个shape精确覆盖
48个已命中和31个原路径RMS，无日志截断。31个中，18个完整BF16 widen→RMS→narrow因输入/输出同址被拒，
实际ne/nb/字节区间完全相同；其余13个原生region入口为F32，未追到Graph ValueId来源，不能归为同址或直接伪装BF16。

v13仅修输入alias资格：同址、同BF16 dtype、同shape/strides、两端连续、同字节范围且不溢出时允许原地执行。
原256/1024线程block归约同步全部初读，每线程随后只读写自己的列；其他row不相交，scale仍须严格disjoint。
部分重叠、不同映射、广播、公开中间值/额外读者等保护不放宽；原kernel、RNE、epsilon、分配及公共API不改。
生产补丁按mixed CPU＋CUDA IQ1重放，source ID为`94060dde81cf1bbd`；本轮只编1个native main和2个host TU。

17项CUDA通过、无skip，新增实图覆盖内部BF16 producer、两种block宽度、公开F32中间值对照、双槽换址/旧结果和RequireCapture；
原Direct GPU逐步运算作oracle。完整128＋31固定输入的64份hidden/logits逐byte不变，32个ID和7段capture复用不变。
一次新官方trace的CSV/SQLite逐symbol计数一致：25个warm步全部1414→1378 kernels，18个unscaled RMS换typed读写，
BF16→F32从260→242、反向169→151，其他物理kernel次数不变。workspace仍为T128的38,191,104B与T1的2,115,712B/slot。

无profiler/内存监控、与v12同闭包的AB/BA/AB三对独立进程：

| 项目 | v12中位 | v13中位 |
| --- | ---: | ---: |
| 首次prefill tokens/s | 1867.29 | 1862.15（−0.28%） |
| 全31步decode tokens/s | 77.68 | 77.37（−0.39%） |
| warm7–31 decode tokens/s | 81.49 | 82.00（+0.62%） |
| 权重加载 seconds | 3.19187 | 3.21990 |
| 形状准备 seconds | 0.449098 | 0.447860 |

warm原样本89.58/80.52/81.49，新样本87.13/80.29/82.00；有明显跨进程漂移，不能宣称稳定提速，
也不能将本轮绝对速度与前轮约92tokens/s直接比较。没有重测llama.cpp、总显存峰值或模型质量。
36组GDN L2及完整投影/FFN仍待融合；现有读写转换393次/步也不意味着都能直接删除。

[诊断日志、源码与审查、17项回归、64份向量、配对计时和官方trace](cuda-rms-inplace.zip)保留本轮依据。
验证脚本的封存过滤/字段名、首次两步参数与31份decode输入不匹配、筛选器去重数量未同步等初失败亦保留；
这些均发生在封存或消费者/检查器的执行前检查，没有重编SDK或把失败写成通过。

## GDN L2 的逐节点舍入融合与冷布局复用

一次128prompt＋1步decode元数据诊断完整记录72个L2原生图，无截断。每个核心含11个native节点、10个实际kernel：
BF16平方、扩宽、metadata reshape、F32 sum、BF16 sum、BF16加epsilon、扩宽、严格sqrt、fast div、BF16缩放值、BF16乘原输入。
REPEAT numerator与最后BF16→F32转换在核心外。T128还发现同一pitched输入被三次CONT/reshape；T1直接用相同leaf。

CUDA私有lowering现按本次build的ValueId缓存matrix结果，复用同一值的连续化；checked ID临时生产者同步清空/恢复缓存。
没有按裸地址或存储根合并不同值，也不引入热CSE、全局缓存或另一套执行槽。新native matcher只剥真正同源的连续零偏移reshape，
按完整数据流、CPY self引用、公开输出/额外读者及实际别名检查；读取最终Mul的真实x，拒绝部分重叠及epsilon/numerator重叠。

新核保留SUM_ROWS原512/32/128线程选择、八路部分和顺序和thread-zero结果。square、sum、加epsilon、整个Rsqrt、末乘共五处BF16 RNE；
明确的RN sqrt与原fast division分别执行，未换成rsqrtf或挪动epsilon。限定函数的SASS导出确认转换边界和独立division；
编译器可用RSQ及修正/特殊值路径实现RN sqrt，不能仅看某条MUFU或FTZ指令就判为近似替换。对象无PTX，
该结论结合源码合同、SASS和实际数值测试，不是全IEEE/NaN payload形式证明。

增量仅2个native＋3个host TU，未重编SDK。19项CUDA通过、无skip，含独立Direct GPU逐步oracle、T1/T128/T2、
1024列线程分支、正/零/负/最小正规BF16 epsilon、微小数/Inf/NaN、双槽换址/旧结果/RequireCapture，
以及全部中间值公开、额外sum reader和padded输入。完整Qwen64份hidden/logits逐byte不变，32个ID和7段capture复用保持。
生产mixed CPU＋CUDA IQ1补丁source ID为`5a1fced46d7403e2`，当前Qwen镜像SHA256为
`8070f3517db1c3a000ac04a58bd57f09cea07ab96d80c51a6b89c32df8bab285`。

官方trace/CSV逐symbol计数一致：25个warm步全部1378→1054 kernels，36个L2核心全部10→1；两向Cast各少72次。
含冷准备的prefill首窗口同样少324个kernel，另少72次512KiB D2D/2Dcopy、共36MiB，符合36条链各三份CONT变一份。
这项copy变化不计入kernel数。两个trace的部分重放节点bytes标签另有不一致，无法由SQLite离线确定真实descriptor；
因此不由整表SUM(bytes)推断总带宽改善或回退，保留逐节点调查和限制。未重新收集GPU trace以追这些metadata差异。

无profiler/外部内存监控的v13/v14 AB/BA/AB三对，同固定输入/精度/容量/消费者与full-step边界：

| 项目 | v13中位 | v14中位 |
| --- | ---: | ---: |
| 首次prefill tokens/s | 1866.13 | 1862.96（−0.17%） |
| 全31步decode tokens/s | 79.55 | 82.28（+3.43%） |
| warm7–31 decode tokens/s | 83.20 | 86.15（+3.55%） |
| 权重加载 seconds | 3.21579 | 3.16833 |
| 形状准备 seconds | 0.460686 | 0.440474 |
| T128 / T1 workspace bytes/slot | 38,191,104 / 2,115,712 | 相同 |

warm原样本83.20/82.63/84.95，新样本86.15/86.02/94.16；decode三个相邻配对均快，末对幅度也明显变大。
本轮观察到正向decode收益，但样本有漂移，不固定其长期提升幅度；prefill中位基本持平。
没有总显存峰值或独立质量验收；13处RMS区域切口、完整投影/FFN、其余模型/量化和平台仍未收完。
[源码、诊断、19项回归、64份向量、编译数学依据、配对计时和官方trace](cuda-gdn-l2-fusion.zip)保留本轮依据。

### 当前版本与 llama.cpp 的新同窗对照

参考仍为原d9e03f1、同源BF16转换GGUF与原公开GPU greedy；重新核实9个obj/lib及模型/输入身份，仅重链一次peer，
没有重新编译或改采样。首次六进程结果也保留，但发现Hyle `prepared` 的cerr输出落在step内，而peer报告在计时外。
测试消费者现保存准备时长，在token完成的finished之后输出日志；输入上传、batch/layout、绑定、replay、submit/wait/readback仍全部计时。
仅重编1个consumer TU，新镜像再核64份向量/32个ID相同；随后复用同一peer镜像，另开目录完成H/L、L/H、H/L三对，未覆盖首次记录。

| 项目 | Hyle | llama.cpp |
| --- | ---: | ---: |
| prefill step tokens/s | 1978.29 | 1289.08 |
| 全31步decode tokens/s | 81.82 | 91.96 |
| warm7–31 decode tokens/s | 85.74 | 93.31 |
| 权重/模型加载 seconds | 3.18993 | 1.83199 |
| 形状/ctx准备 seconds | 0.447684 | 0.007693 |
| 启动到ready seconds | 3.72945 | 1.91595 |
| 启动到首token seconds | 3.79363 | 2.01724 |

warm三样本Hyle86.10/85.74/84.29、peer93.31/95.25/90.32，中位Hyle仍低8.11%；全31步低11.03%。
Hyle在step前预建两种shape，peer的decode图维护在step内，因此首个prefill step数字不能宣称完整冷prefill更快；
启动到首token已含各自加载和准备，Hyle仍有明确输项。不同权重容器和加载API的细项不能混称为同一函数边界。
前后组有漂移，不能把Hyle首次组77.96到本组85.74全部归因于移出日志；原L2 v13/v14配对的两臂日志边界相同。
全部固定输入和32个预测相同，但中间数值边界并不bit等价，也没有独立质量targets。

另一次fresh peer trace确认warm每步558个kernel、1次GraphLaunch，官方CSV/SQLite一致且无graph API错误；
同轮Hyle数学镜像为1054个、7次。peer每步另有993280B sampled-logits回读，不能与32条4B token端点混成64步；
两类端点间无kernel，分别分窗得到相同计数。物理count仍不等于吞吐比例，也不由这些不同时间的trace计算同窗端到端差。
workspace之外的Hyle新总显存/内存峰值未重测。`peer-comparison-reporting/interpretation.json`、原组、peer trace及计时改动源码都在上述归档中。

## RMS 区域切口与映射加载

13处漏掉的typed RMS来自保守外部frontier：scale Cast/Add在先，激活的外部Gather/View引发切分，随后RMS又首次遇到scale。
Provider现按真实边补充私有Cast→RMS→Cast短链，复用原叶偏好、route/implementation和共同收缩检查；
已在同一普通区域的66条链不重复提案，scale仍独立冷准备，不改共同Graph或native数学。
新增回归覆盖正链、公开两个F32中间值、外部控制读者、冷scale及双槽换址/旧结果/RequireCapture。

消费者复用公共只读FileMapping，源范围检查后直接同步上传，替掉320份临时host vector及总3.763GB的分配、零初始化和文件复制。
仍是320次设备分配和同步上传；参考mmap分支也逐tensor同步，不能将加载差全部归为异步上传。
映射、上传和设备所有权沿现成合同，没有增加loader或异步pool。独立CMake显式链接Mem产品。

仅增量编译3个host TU，零native/SDK编译。首次Qwen手工link漏了platform_fs，保留失败后追加已有库并重链，未重编任何TU；
正常CMake的mem_vm公开依赖已含该库。20项CUDA零失败/skip，完整64份hidden/logits逐字节一致，32个ID及7段capture复用保持。
镜像SHA256为`6ebf57e026299db17b99ac68b55762f9f00b9fa96e320b51f3806e25c66a2d1a`。

一次独立官方trace确认每个warm步1054→1028 kernels，两向Cast各少13次，其他kernel次数不变；
typed RMS共61 weighted＋18 unscaled（49个1024线程weighted、12个256线程weighted）。T128/T1 workspace分别38,191,104/2,115,712→36,618,240/2,103,424B每槽。
workspace是报告值，不代表总显存峰值；物理count与吞吐仍分别判断。

随后同一窗口按旧H/新H/L、L/旧H/新H、新H/L/旧H交错九个独立进程，全部固定128prompt＋31decode、容量256，
无profiler或外部内存监控；旧新Hyle的日志都在step计时外。同源权重、固定输入和32预测全部一致，参考仍为原d9e03f1公开GPU greedy。

| 项目（3进程中位） | 改前Hyle | 当前Hyle | llama.cpp |
| --- | ---: | ---: | ---: |
| prefill step tokens/s | 1885.09 | 1960.17 | 1321.18 |
| 全31步decode tokens/s | 82.74 | 82.27 | 93.09 |
| warm7–31 decode tokens/s | 86.66 | 85.68 | 95.28 |
| 权重/模型加载秒 | 3.14169 | 1.23386 | 1.79653 |
| shape/ctx准备秒 | 0.433810 | 0.436926 | 0.007782 |
| 启动到ready秒 | 3.66737 | 2.13070 | 1.87501 |
| 启动到首token秒 | 3.73567 | 2.19494 | 1.97754 |

加载阶段中位少60.7%，完整启动到首token少41.2%，当前首token仍比参考慢约11.0%。加载细项API边界不同，完整首token时间已含各自准备。
warm旧样本86.66/85.18/86.71，新84.95/85.68/89.09，参考94.14/96.19/95.28；当前较旧中位低1.13%，
没有观察到本批decode收益，也不足以由这三点证明稳定回退。与同窗参考仍差10.07%，不能用减少26个kernel宣称解决。
prefill step仍有预建shape边界差异，不用该数字宣称完整冷prefill胜出。未测新内存/显存峰值，未独立验收模型质量或其他平台。

后续优先查原执行槽每token重复发现的6次全计划扫描及FFN双投影。peer少24次projection来自gate/up配对，
另30个projection epilogue融合来自24次down/residual及6次attention输出/residual；这由模型源码、顺序和grid交叉归因，trace本身无逻辑node标签。
不能将peer未保留中间BF16舍入的融合直接替换现数学。简单拼接权重会额外保留1.125GiB，且原bank和两种shape计划仍持有原租约，
因此不作为当前默认修复；无复制的双weight实现和复用成熟Lt的方案需要分别评估数学、寿命与实测收益。

[源码、20项回归、64份向量、九进程对照、原始trace及初次链接失败](cuda-rms-candidate-frontier.zip)保留本批依据。

## 原执行槽中的 batch 选择复用

普通prepare此前每token做六次全计划visitor，清理并重新发现固定的batch能力、执行槽和metadata槽。
现只在原Invocation资源payload中缓存一次成功的选择；当前Qwen把对1167个步骤的重复扫描改为遍历六个实际attention槽。
原reset的busy/结果/退休检查、owner、NodeId、完整layout/schema、Queue/options、native配置和metadata更新均保留。
只缓存不可变top-level身份，调用级QueueRef/layout/metadata-source在每次退出清除；失败不保留可用资格，pending模式继续原动态发现。
未增加公共API、Plan索引、执行器或payload副本。缓存向量保留已分配容量，不据此宣称整个prepare无分配。

仅鲜编1个CUDA配置的host batch TU；Qwen的64份hidden/logits逐字节一致、32个预测及capture旅程相同。
另用CPU配置鲜编batch和既有多Context测试，增加move、暖态重复layout/错误Queue拒绝、拒绝提交旧配置及数值恢复。
首跑发现手工闭包仍含旧CPU复制对象，只接受system_resource，实际错误与当前源码不同。未放宽fixture或产品检查；
鲜编当前mem/resource、batch_transfer和CPU复制测试后，两个定向case通过、零失败/skip，覆盖未证明映射拒绝及wrapper别名重叠。
合计6个host编译命令、零native/SDK编译。Vulkan旧闭包的Graph/RegionExtension ABI已漂移，该GPU用例本轮未运行。

同一九进程交错窗口，旧新Hyle的124个链接输入中仅batch.obj不同，其余123项逐SHA相同；所有固定输入和32个预测一致。
当前Hyle镜像SHA256为`d84bbdb0fd1bdbbcda5e8de2fb2d6f1fd2a695fc34981ccbff6d1e1514755602`。

| 项目（3进程中位） | 改前Hyle | 本批Hyle | llama.cpp |
| --- | ---: | ---: | ---: |
| prefill step tokens/s | 1958.42 | 1914.89 | 1263.72 |
| 全31步decode tokens/s | 81.75 | 82.61 | 93.88 |
| warm7–31 decode tokens/s | 85.03 | 86.77 | 95.50 |
| 权重/模型加载秒 | 1.24923 | 1.23381 | 1.80255 |
| shape/ctx准备秒 | 0.455602 | 0.448561 | 0.007486 |
| 启动到首token秒 | 2.24190 | 2.23026 | 1.99142 |

warm样本旧86.23/85.03/84.58、新87.78/85.74/86.77，中位观察到+2.04%，仍低于同窗参考9.15%；
prefill step中位回退2.22%，且它仍存在预建shape的边界差异。没有显著性或稳定幅度结论。
六个Hyle进程均32次prepare，recorded21/reused203；warm cold_batch每进程中位数之中位数0.3979→0.3863ms，
但每进程均值没有稳定下降。该字段还含layout、绑定与prepare_replay，不能当作被缓存扫描的独立CPU耗时。
本轮无新physical trace、显存/内存峰值或独立模型质量验收，不能从相同数学对象声称重新观测了kernel数。

[源码、完整模型核对、CPU先失败后通过、九进程计时和离线分析](cuda-hot-prepare-selection.zip)保留全部边界。

## BF16 FFN 双投影与实际原生图

本批在CUDA provider私有候选中接入单token、连续BF16激活与两份BF16权重的gate/up→普通稳定SiLU→Mul，
不拼接或复制整份权重，不接管prefill及其他量化域。原MMVF模板复用双投影累加与归约，新增epilogue在既有strict unary TU实例化；
gate投影G、up投影U、SiLU结果A和最终Y各保留BF16 RNE，SiLU仍用普通稳定公式。
F32点积的归约树允许与成熟Lt路径不同；逐节点舍入保留不等于完整MatMul逐bit相同，也不等于模型质量已验收。
Provider拒绝候选时保留原路由；已选候选若native matcher未命中，其普通ggml回退也不能称为Lt。

首稿按连续四个math节点匹配，遗漏真实原生图插入的权重TRANSPOSE：冷路由虽选中24个impl4，官方trace却全部零命中，
每步退为两次MMVF＋SiLU＋Mul，原SiLU/Mul融合也被切断，kernel从1028增至1052，首组测速回退。
修复只接纳四个math之间实际被另一投影读取的一个只读权重TRANSPOSE，核对同源、零offset、dtype、精确shape/strides、
唯一reader及无OUTPUT；实际跨度五节点返回skip4，数学值的真实usecount、公开输出、额外读者、别名和精度保护不放宽，L2闭包未变。
首稿零命中、原测速及全部失败过程保留，不能用冷选择或测试通过代替物理命中。

最终mixed CPU＋CUDA IQ1版本v16，source ID为`e0bf7135273ee756`，Qwen镜像SHA256为
`f1300408225479e73870a0b43cc655f43389c3499ee889ca74a57b439e068fef`。
修复仅鲜编native主TU与provider版本host TU后重链；28项CUDA全部通过、零skip，含三个新增数值/逃逸/长点积用例及25项既有回归。
最终官方CSV与SQLite逐symbol计数一致：31个decode步各实际命中24次dual epilogue，单步kernel从1028降至980，
原24个BF16稳定SiLU/Mul核被吸收，18个F32稳定SiLU/Mul核保持；prefill仍1554个kernel且无dual，GraphLaunch始终每步7次。
Nsight的driver13.4／collector13.3兼容警告原样保留，无graph API错误；trace用于核实物理结构，不用于推导无profiler吞吐。
T128/T1报告workspace每槽由36,618,240/2,103,424B变为36,618,240/2,054,272B，未重测总显存峰值。

同固定128prompt＋31份decode输入、容量256，在旧Hyle/新Hyle/llama.cpp交错九个独立进程中计时，无profiler或外部内存监控：

| 项目（3进程中位） | 改前Hyle | v16 Hyle | llama.cpp |
| --- | ---: | ---: | ---: |
| prefill step tokens/s | 1971.61 | 1887.66 | 1277.56 |
| 全31步decode tokens/s | 79.0476 | 79.9241 | 91.5089 |
| warm7–31 decode tokens/s | 82.8346 | 83.1370 | 91.2554 |
| 权重/模型加载秒 | 1.27240 | 1.28846 | 1.77992 |
| shape/ctx准备秒 | 0.452063 | 0.449415 | 0.007677 |
| 启动到ready秒 | 2.19348 | 2.20699 | 1.86193 |
| 启动到首token秒 | 2.25883 | 2.27865 | 1.96194 |

full中位观察到+1.11%、warm +0.365%，prefill step低4.26%；warm三样本旧81.68/83.91/82.83、新86.43/83.14/82.33，
参考89.07/101.83/91.26，存在明显漂移，未证明稳定提速。当前full仍比同窗参考低12.66%、warm低8.90%，首token慢16.14%，输项没有解决。
加载和准备细项的API边界不同，Hyle在step前准备两种shape，peer图维护仍在step内；不能用prefill step宣称完整冷prefill领先。
九进程固定输入和32个greedy ID均相同，不以kernel减少比例推算吞吐收益。

完整模型数值诊断中，baseline的64份hidden/logits与旧reference逐byte相同；新路径仅prefill两份相同，decode的62份有差异，
但所有输出有限、32个greedy ID相同。相对baseline，hidden/logits最大relative L2分别1.5200%/1.8023%，最大绝对差0.5/0.2109375；
每份向量的RMS、relative、ULP及原始数据均保留，不能用同32个ID宣布质量通过。
metadata修复后的64份向量与v15未命中的MMVF回退逐byte相同，说明本次命中未再引入该组输入上的数值差异；
这不消除相对Lt基线的变化，也不是全域等价证明。

同一组128prompt＋31份输入的独立Torch参考随后实际完成；eager与既有F32-RoPE诊断变体各从空cache独立跑32步，
只读取原权重和输入，不接纳Hyle的hidden、logits或state。两臂对两参考均32/32 top1相同，但误差指标并非一致改善：
对eager，logits平均relative L2旧1.1614%／新1.2634%，平均KL(reference→Hyle)旧0.00043108／新0.00045233；
对F32-RoPE，平均relative L2旧1.1904%／新1.1838%，平均KL旧0.00030783／新0.00052187。
31份固定下一输入的平均NLL旧1.5193117／新1.5204487；这是benchmark tape的诊断，末步没有目标，不是独立语料质量验收。
没有事后新增容差，也不能据同32个ID或某一个均值宣布模型质量通过。脚本、真实环境、逐phase指标与原始向量均保留。

本轨迹自身163次投影占kernel duration sum的88.85%，其中139次为普通cuBLASLt、24次为新dual MMVF；
FFN gate/up＋down占投影时间47.38%、head占26.51%。这只能定位本身热点，不能直接解释相对另一时间采集参考的差额。
既有三种真实尺寸的18进程独立筛选已含head K2048/N248320，且同为BF16直接输入／F32累加／BF16输出；
Lt与普通MMVF的head wall中位约2179.95／2188.55us，不能盲目重复“全部单投影改走GGML”的假设。

当前两边投影数均163，非投影实际为817／395，净多422次。按物理symbol家族对齐：

| 家族（每warm步） | Hyle | llama.cpp | 差额 |
| --- | ---: | ---: | ---: |
| BF16/F32 Cast | 223 | 0 | +223 |
| F32二元计算 | 138 | 36 | +102 |
| BF16二元计算 | 54 | 0 | +54 |
| Repeat | 48 | 0 | +48 |
| 单独激活 | 60 | 18 | +42 |
| Concat | 42 | 18 | +24 |
| I32/F32 Cast | 12 | 0 | +12 |
| RoPE或三角函数 | 24 | 12 | +12 |
| 其他家族合计 | 216 | 311 | -95 |
| 合计 | 817 | 395 | +422 |

其他家族合并了attention/merge、token选择、GDN、conv、cache、normalization、copy、gated activation、scale和gather，
完整正负项见`metadata-fix/non-projection-gap.json`。归一化把Hyle L2 core与RMS合并计数后双方均115，
但数学合同仍分开；peer projection epilogue也会吸收表外独立运算。计数差不是可直接删除的逻辑算子清单。
下一批先查这些转换／逐点／repeat的实际来源与寄存器内舍入承载，特别是显式频率RoPE的分解路径；不因其自身耗时占比小而略过。

原始源码、编译/测试日志、独立参考与数值向量、两次物理trace和两组九进程计时见[本批证据](cuda-bf16-dual-projection.zip)。

## 中间精度与转换的源码调研

本节仅检查锁定源码、输入清单与既有 trace，没有新增构建、GPU 执行或质量/性能通过结论。
用户要求先调研设计；消费者 BF16/F32 开关与混合 MatMul 都仅为 `.scratch/architecture-refactor-2026-10-18/cuda-activation-precision/`
下的未接入草稿。产品 `tools/qwen35_text/main.cc` 仍为 SHA256
`38ee18ff552ee4db15663469457d293219d4f6e4bbbef3ae4037af8c16350869`。

本地官方 Qwen3.5 Python 源 `modeling_qwen3_5.py` 的 SHA256 为
`0e7410f5251e5c324a5628c5340ee3551f7a6a505ce398569fcc3b5f3ac07f7a`。
config 的 BF16 不是全图要求：320 份文本权重中实际 284 BF16、36 F32，后者是18层的 A_log 与 GDN norm scale。
模型中的 norm、softmax、GDN、RoPE 和反馈状态有各自的升精度与舍入边界。
其中 HF 的 RoPE 将 F32 sin/cos 转回输入类型后旋转；Hyle 当前显式频率 RoPE 保持 F32 旋转、仅末尾窄化，
不能把它称为完全相同的逐步舍入。已选正式 Flash-0731 的官方实现又同时包含 FP4 权重、动态 FP8 操作数、
F32 累加与 BF16 输出，进一步说明权重格式不能推导所有计算精度。

实际对照的 `llama-matched-consumer/ggml-package/ggml-config.cmake` 指向既有静态库与
`.scratch/legacy-root/hyle-scratch/ggml-native-static/src/compute_ggml_runtime/source-4e1a7b2f39afe15d`，
ggml version 为 `456172ec733a135778adcd32d00e576a58232e45-hyle`；不是本批 Hyle v16 的源码。
转换清单中实际是187份 BF16、133份 F32，其中97份原 BF16 小权重精确扩宽为 F32；总数仍为320。

| 路径 | 实际行为与限制 |
| --- | --- |
| peer 激活图 | ggml GET_ROWS 与普通 MUL_MAT 输出 F32；FFN 不保留 Hyle G/U/SiLU 逐节点 BF16 RNE |
| peer decode | MMVF 直接读取 BF16 权重，逐项扩宽后与 F32 激活相乘并 F32 累加、输出 F32；不先生成整份 F32 权重 |
| peer T128 prefill | 此尺寸离开 MMVF/MMF，默认 cuBLAS 将 F32 激活临时转为 BF16，以 BF16 操作数、F32 累加产生 F32 输出 |
| peer KV/state | 本对照显式选择 BF16 KV；hybrid recurrent/conv history 保持 F32，与激活或权重格式独立 |
| Hyle 普通 F32 MatMul | 同 dtype 的输入/输出及 F32 累加；CUDA 另要求 F32 来源精度，不凭累加字段允许 TF32/BF16 乘法 |

源码定位：peer `src/ggml.c` 的 GET_ROWS/MUL_MAT 类型推断；`src/ggml-cuda/ggml-cuda.cu` 的
1822–1824/1884–1895（MMVF）、1520–1528/1561–1568/1672（cuBLAS），以及 `mmvf.cu` 的
280–292/347–379/657–659。prefill 与 decode 不能仅按输出 F32 这个名字认定为相同乘法精度。
既有 peer SQLite 中 F32→BF16 转换186次、BF16→half 12次均位于含冷准备的首个阶段窗口；
decode 1–31 两类均为零。首窗口不等于纯 prefill；warm 零外部转换核也不等于整个运行没有转换。

因此817／395的非投影核差额既包含融合/承载不足，也包含数值配方不同；不能把223个独立 Cast 全部称为冗余，
也不能用 peer 作为当前 BF16 逐节点舍入的 bit oracle。反过来，保留这些舍入并不要求独立 kernel 或显存往返。
现行 Hyle 的157次扩宽、66次窄化中，198次来自18层 conv/GDN/gate，24次来自RoPE、1次来自head；
48次 Repeat 中36次来自L2标量分子广播、12次来自RoPE频率广播。这里是源码与实际物理网格的数量对齐，
不是逐节点 NVTX 归因，也不证明全部可删除或贡献了同等比例的时间。

共同 MatMul 目前要求同 logical dtype，cuBLAS 与 ggml CUDA admission 也保留该限制；这不必然要求立即扩展公共 API。
`X:F32 × Cast(W:BF16→F32)` 已能表达数学，缺口可以在冷 lowering 融合精确扩宽、直接读取原权重。
但现有常量表达式/packing 会先物化 F32 结果，新的承载须在物化/分配前保留原源与真实读者，
不能只删除热 Cast。若直接放宽当前 ggml CUDA admission，严格 F32 的大矩阵路径还可能每次将 BF16 权重
扩入原生池；head 单份 F32 scratch 为2,034,237,440B，且池可按活跃槽保留，不能把这种草稿称为省显存方案。

设计收敛见[精度配方与冷编译](../../../architecture/compute-design.md#精度配方与冷编译)：
先分清等价的融合/转换承载与改变数值配方，再以实际路径检查质量、吞吐和资源代价。
本轮尚未选择另一精度配方作为默认，也未证明全 F32 或全 BF16 更快、更准。

## 保持精度的显式频率 RoPE 承载

本批保留现有 BF16 激活/端口与 F32 旋转数学，CUDA 私有 lowering 将原 RoPE 分解改为单个 native 节点。
位置先 I32→F32 RN，再乘实际频率，独立 strict sin/cos、四个 RN/FTZ 乘积和独立加减，最后回原输出类型；
partial tail 保留原宽化/窄化语义。没有切换为 HF 的另一套中间舍入、逆频率或近似三角函数。
频率 shape 本身决定旋转宽度，避免重复 int32 参数缩小已有维数域；同 phase 的三角函数供各 head/batch 复用，
单 launch 不申请临时 payload。内部完全相同的 dense 输入映射可复用，部分重叠及覆盖位置/频率拒绝，公共 alias 不变。
私有 marker 不进入原 YaRN/逆频率融合；新的绑定失败通过既有 native failure 边界返回。

锁定版本为 `...-bf16-dual-explicit-rope-v17`，source ID `96da4225333d2bd6`，
vendor tree SHA256 为 `d0d2a342d335334c21bdb83616bf0bebe73b34fc60f746d19814bd9c36088c2c`。
所有 mixed CPU＋CUDA IQ1 patch 均逐条唯一重放并核 SHA，CMake dependency lock audit 通过。
增量仅2个 native（main/unary）和3个 host（tensor/provider/test）TU，没有 SDK 或全量构建；
首编发现 initializer_list 混入 const/nonconst 指针不能推导，改为明确类型的数组后全闭包编译/链接通过，失败日志保留。
主测试消费者源码未改；当前镜像 SHA256 为 `a6b8fa6ac6997a92d956c3976678ece4c2d2112d412da9eb83f1d6a33dd3ddf2`。

30项相关 CUDA 测试全部通过、零失败/skip，其中两项新增测试含6个定向 tuple：F16/BF16/F32、两种布局、
共享/批次位置、静态/逐序列频率、partial tail、超过2^24的I32位置、负值、subnormal、零、Inf/NaN。
GPU oracle 直接运行旧 `explicit_rope.h` primitive 图，检查不含新 RoPE，公开中间值以保留独立计算边界；
有限值、Inf、符号零逐 bit 比较，NaN比较分类。候选含双槽、换址、Commands/RequireCapture、
内部单读者 producer、公开 producer 和旧结果检查；没有新增公共测试数学接口或 CMake 依赖。
实际 unary 对象的 SASS 保留独立 FMUL.FTZ 与 FADD.FTZ 序列；sin/cos 内部仍可包含其合法 FMA，不能声称全函数无FMA。

完整128prompt＋31固定decode输入的两次独立旅程中，改前镜像使用原 v16 的124项对象闭包；
两臂各64份 hidden/logits 均与保留的 v16 current 向量逐 byte 相同，所有结果有限、32个预测及7段capture旅程一致。
这只证明本次输入上的数值保持，不消除上批FFN点积树相对Lt/Torch的已有差异，也不替代独立语料质量判断。

一次官方 Nsight CSV/SQLite 计数一致，prefill和31个decode阶段各实际执行12个新RoPE kernel，GraphLaunch均保持7次：

| 每warm步实际kernel | v16 | 本批 |
| --- | ---: | ---: |
| 全部 | 980 | 812 |
| BF16/F32独立转换 | 223 | 199 |
| I32/F32独立转换 | 12 | 0 |
| Repeat | 48 | 36 |
| F32二元计算 | 138 | 54 |
| Concat | 42 | 18 |
| 独立sin/cos | 24 | 0 |
| 新显式RoPE | 0 | 12 |

prefill实际1554→1362 kernels。投影163次保持，非投影817→649；peer历史同输入轨迹非投影395，
剩余差额仍含数学配方差异，不能视为全可删除。T128/T1 workspace每槽
36,618,240/2,054,272→33,964,032/2,033,408B；未重测总显存/内存峰值。
driver13.4／collector13.3兼容警告保留，未见graph API错误；profile耗时不进入以下吞吐结果。

同窗旧Hyle／新Hyle／llama各3个fresh进程交错，无profiler或外部内存监控：

| 3进程中位 | v16 Hyle | 本批 Hyle | llama.cpp |
| --- | ---: | ---: | ---: |
| prefill step tokens/s | 1888.11 | 2021.39 | 1290.57 |
| 全31步decode tokens/s | 80.5688 | 82.8350 | 88.3265 |
| warm7–31 decode tokens/s | 83.7385 | 85.6576 | 90.3880 |
| 启动到首token秒 | 2.23753 | 2.24595 | 2.00046 |

观察到full +2.81%、warm +2.29%、prefill step +7.06%，但每臂仅3进程；warm样本旧83.74/82.17/84.44、
新86.52/83.98/85.66、参考90.39/87.06/91.68，仍有漂移，没有稳定幅度或显著性结论。
当前full低同窗参考6.22%、warm低5.23%；不能与上一窗口12.66%直接相减，将全部差额变化归因于本批。
Hyle仍在step前准备两种shape，prefill step与peer不是完整冷工作对齐；首token比较也仍输。
九进程固定输入、预测和Hyle捕获旅程一致；保留peer原采样与readback，没有改变输出工作量。

接下来仍需处理199个BF16/F32独立转换中的conv/GDN/gate边界与head、36个L2分子广播，以及投影/首次录图成本。
不把这批CUDA结果推导为其他模型、量化或平台已经补全。[源码、失败、对象身份、全部向量及原始实测](cuda-explicit-rope.zip)。

## 标量广播与保留舍入的 L2 输出融合

本批没有新增模型特例、公共算子或另一套执行器。CUDA 私有 lowering 对 F32 首标量 Divide 和 Rsqrt 分子
构造真实只读 scalar View，再冷设置广播 shape 与零步幅；底层 owner、依赖与读取范围仍是一个元素。
没有要求 Repeat 在原生节点序中紧邻 L2，也不物化重复数组。其他 Binary 仍保留原承载：原生普通二元核
支持零步幅，不代表旧 MatMul＋bias/GLU 融合的 dense 读取同样合法，旧 bias matcher 的布局资格仍待补全。

L2 原核增加实际 scalar numerator 读取与 F32 输出变体，保留原 SUM_ROWS 的八路局部累加、block/tree、
严格 sqrt、fast div/FTZ 及五处 BF16 RNE。最后先做 BF16 RNE，再精确扩宽到 F32；不是直接写未舍入 F32。
只接受真实单元素 source、零偏移与零步幅的只读 View，动态 numerator 从实际地址读取；原 dense row numerator 路径保留。
最终 BF16 值私有时可融合 BF16→F32 CPY；公开值或额外读者保留原可见边界。
F32 目的须与输入、epsilon、numerator 不重叠，原 BF16 完全同址 dense 映射继续按原合同处理。

版本 `hyle-ggml-cuda-resource-iq1-inference-v18`，source ID `971b3d4968812a0f`，vendor tree
`82b4452dae9332bdb5a1f294c3d62f0cc32d7b8b0aaaa27a984001064748e815`。
mixed CPU＋CUDA IQ1 manifest 逐条唯一重放、全部 SHA 与 CMake lock audit 通过。
仅增量编译2个 native TU（main/norm）及3个 host TU（tensor/provider/既有 L2 测试），两次链接通过；
复用旧 v17 baseline 和 peer 镜像，没有另复制 SDK、权重或重编公共 Graph/消费者。
新消费者镜像 SHA256 `b1c4f5d2c9740527473a9c4fc7b18443b6ddff0285330d49079725b4cf72b893`。

32项相关 CUDA 测试全部通过，零失败/skip。新增覆盖 T1/T128、逐行非均匀 numerator、BF16 公开中间值及额外读者、
不带扩宽的 BF16 输出；动态 scalar 为 −0.75、2.5、Inf/NaN，并含 F32/F16 普通标量左除法。
oracle 在 host 独立复制 dense numerator，GPU 仍执行原分阶段数学，未复用新广播 View；有限 witness 检查最后 BF16 舍入。
候选检查双槽、两轮换址、Commands/RequireCapture 和旧结果。完整128prompt＋31固定decode输入的两个新旅程中，
两臂各64份 hidden/logits 与保留 v17 current 逐 byte 相同、全部有限，32预测、31输入及32阶段×7段capture一致。
这不消除早先 FFN 点积树相对 Lt/Torch 的已有差异，也不代替独立模型质量判断。

单次官方 Nsight 采集的 CSV/SQLite 计数一致，无 graph API 错误；每个阶段实际命中36个
`hyle_bf16_l2_core<float, (bool)1>`，12个显式RoPE保持。首次离线解析只匹配字面 `true`，
漏掉 demangle 的 `(bool)1`；保存旧解析与结果后只重跑离线分析，未重采 GPU。

| 每warm步实际kernel | v17 | 本批 |
| --- | ---: | ---: |
| 全部 | 812 | 740 |
| BF16/F32独立转换 | 199 | 163 |
| 其中 BF16→F32 | 145 | 109 |
| Repeat | 36 | 0 |
| L2 core（本批为scalar/F32变体） | 36 | 36 |
| 投影 | 163 | 163 |
| GraphLaunch | 7 | 7 |

prefill 1362→1290 kernels；非投影649→577，peer历史同输入非投影395仍不是等价工作量。
T128/T1 workspace每槽33,964,032/2,033,408→33,947,648/2,033,152B；未重测总显存／内存峰值。
driver13.4／collector13.3兼容警告保留。profile耗时不进入吞吐比较。

同窗旧Hyle／新Hyle／llama各3个fresh进程交错，无profiler或外部监控：

| 3进程中位 | v17 Hyle | 本批 Hyle | llama.cpp |
| --- | ---: | ---: | ---: |
| prefill step tokens/s | 2059.00 | 2044.25 | 1280.72 |
| 全31步decode tokens/s | 93.8228 | 94.7909 | 103.1100 |
| warm7–31 decode tokens/s | 98.3019 | 98.4952 | 105.6618 |
| 启动到首token秒 | 2.04139 | 2.02101 | 1.81920 |

观察到full +1.03%、warm +0.20%、prefill step −0.72%，每臂仅3进程；warm旧98.65/98.30/97.71、
新100.18/98.50/97.94、参考105.66/104.37/106.80，样本重叠，未证明稳定提速。
当前full低同窗参考8.07%、warm低6.78%。三个引擎均比上一窗口更快，不能把跨窗口速度或差距变化归因于本批。
Hyle仍在step前准备两种shape，prefill step不代表完整冷工作领先，首token仍输。
九进程输入、32预测和Hyle捕获旅程一致；peer原采样/readback不变。

离线复查旧 v17 实际轨迹，投影占其自身kernel时长和89.64%，199个转换占2.00%、36个Repeat占0.57%；
这是同一轨迹内的耗时构成，不是端到端占比，也不能与peer另一轨迹相减估算可提速幅度。
下一步更多调查实际投影与准备／提交关键路径。现有host区间可含等待并与GPU重叠，尚无函数级CPU归因，
不能把7次GraphLaunch或Completion wait全算成可删除CPU成本。

本次静态架构复查未在共同Graph/graph_batch发现按Qwen模型名或尺寸准入的分支；RoPE/FFN/L2按数学、dtype、
真实边、布局和读者匹配。CPU默认排名仍含实测精确shape偏好，实际能力须先由实现承认，不等于模型支持名单。
主要剩余是原生融合布局资格、共同编译/执行代码的职责收拢，以及跨模型、量化、状态和平台的真实调用链；
当前Qwen成功不能证明八模型齐备，也不足以宣布架构最终稳定。继续复用现有合同和总表，不增加审查框架。
[本批源码、构建、回归、静态审查及原始实测](cuda-gdn-l2-epilogue.zip)。

## 原生融合的布局、读集与量化尾行界

这批修复通用后端合同，未加入Qwen模型名、尺寸、CMake选项或新公共算子。九处旧MMVF/MMVQ的
MatMul/bias/scale/GLU承载共用一处资格：bias实际活跃轴须符合最终输出索引，ADD_ID另核专家轴；
gate bank、scale、dtype、元素步长和对齐按真实kernel读取检查。保留合法offset、padding和singleton无关stride，
不以全contiguous代替实际合同。真实读取包含OP_NONE叶及view祖先，不能来自被消除的producer，
也不能与dst重叠；即使逐元素bias看似可原地，载体的restrict合同仍要求拒绝该融合，保留独立计算。

另发现MMVQ standard核用`stride_col_dst`作尾行掩码；列pitch不等于行数，普通padded输出或odd-N indexed
输出可多预取／写一行。现将已有host真实`nrows_x`传入原kernel，修正bias预取和最终store两处掩码；
不复制设备调参表、不禁用所有padding、不改归约或launch选择，多token专用MoE核原真N路径不变。

版本`hyle-ggml-cuda-resource-iq1-inference-v19`，source ID `470dd6082f0c063e`，vendor tree
`4d9b09e716d5f1e7f6443d124df0729151db9b66810b6ebcf2397257e18d17bd`。
新增MMVQ edits在CUDA manifest先应用，再与原8项IQ1 edits唯一重放，全部mixed SHA及CMake lock audit通过。
复用原SDK/库和对象，仅鲜编main/MMVQ两个native TU、provider身份和既有stable_silu测试两个host TU；
新MMVQ对象在原archive解析前提供同一外部入口，没有重建SDK。当前Qwen125、CUDA133输入闭包，
旧反例镜像为原132项CUDA闭包只换同一新测试对象。主消费者源码未改。

新增3个测试，包含F32普通bias的dense offset／隔列／零步幅，GLU的dense及单lane隔列bias，
indexed bias的dense及padded专家stride；Q8_0另查K32/N7普通padded目标与单token两slot专家选择。
各两组输入。原生metadata先展开，断言真实MM→ADD等相邻序列；公共lowering会先CONT非连续bias，
且没有普通ADD_ID primitive，因此这些负控使用既有raw ggml测试接缝，不声称公有入口直接触发。
F32 oracle公开原MM/ADD，仍运行原GPU GLU；Q8使用相同packed bytes的普通dense MM计算全部专家行后
按IDs选取，不把旧odd-N ID作为oracle。真实backing保留额外权重行和输出哨兵，旧错误仍落在分配内。

首次新实现35项中34通过，GLU fixture在执行后读取bias根，却仅标INPUT。gallocr的INPUT只提前分配，
首bias最后读者结束后可合法回收给later upMM；首异常0.0625恰为upMM首元素。
修正测试给两个实际后读根OUTPUT寿命，不公开数学中间值、不放宽根未变和最终bit断言；
失败源码、日志和修正diff均保留，native代码未因此更改。随后同一修正测试下：

| 实际执行 | 结果 |
| --- | --- |
| 原native＋同一新测试 | 3项全部assert失败，正常GoogleTest exit1，无skip/设备崩溃 |
| 修复native＋相关闭包 | 35项全通过，零失败/skip |
| 一次fresh完整128prompt＋31固定输入 | 64份hidden/logits与保留v18逐byte相同，全部有限，32预测／31输入／32×7capture一致 |

当前Qwen镜像SHA256 `e0d11c1eafa5bd92cb54b61697c9f949dcbcea9a377cc1028f00378ff9a06fb2`。
这批没有重跑九进程吞吐、完整模型profile或总内存峰值，不据正确性变化宣称提速或独立质量验收。

只对前两个小测试采集一次官方Nsight：CSV/SQLite逐symbol一致，共88个kernel，其中MMVF fused10、
unfused32、独立GLU8、Add20、ADD_ID6、gather12。driver13.4／collector13.3警告保留，无graph API错误。
初始静态预期8次融合假定dense GLU整段命中，实际仍拆分为合法MM＋bias；保存原失败分析后仅离线修正解释，
未重采GPU。真实10次证明融合没有全部关闭，不能据此宣布完整GLU命中或所有机会都已覆盖。
原未融合分配可回收已消费的weight/input，完整融合却仍需读取它们；新增range guard须拒绝重叠。
这与当前分解一致，但本trace没有tensor地址，具体复用哪个backing尚未直接观测。需要时应在分配前表达真实融合依赖，
不能放宽读写保护或复制影子buffer来追命中数。

当时另两项源码审查形成后续草稿：F16 MMVF融合误从最终ADD/GLU参数读取精度，
应显式传原MatMul精度；非默认OAI alpha/limit不能被默认融合吞掉。二者随后已修复，见下一节。共同warm输出bind在每token reset后
反复扫描常量，当前消费者源码有51输出与418常量，即每轮至少21,318次常量check迭代，另加51P；
这是迭代数，不是耗时。冷索引草稿保持原动态权限、输入/输出和planned alias检查，尚不能宣称已省去这些开销。
[本批源码、失败、红绿结果及原始小trace](cuda-fusion-layout.zip)。

## 原投影精度与 OAI 参数传播

MMVF现显式接收被覆盖MatMul的原累加精度，普通与融合调用共用原入口，不从ADD/GLU的同槽参数猜测。
F16两投影的实际累加域不同时拒绝完整融合，保留原数学；原硬件能力归一化不变。
OAI的alpha/limit完整传入MMVF、普通MMVQ和多token MoE epilogue，支持合法非默认值，不仅拒绝非默认融合。
这些修改不新增模型名、公共精度开关、额外buffer或执行器。

版本`hyle-ggml-cuda-resource-iq1-inference-v20`，source ID `866a383d696d3931`。私有融合参数结构由48变56B，
因此main/MMVF/MMVQ/unary四个实际使用TU全部鲜编，另编provider身份与测试；沿用原SDK、sm89和各TU数学flags。
首编因新旧物理路径同时引入`#pragma once`头而exit4；按五份实际依赖文件整理89个GGML头、约1.14MB一致闭包后通过，
不修改共享SDK或复制整套源码。首次失败、修正脚本、依赖和源身份均保留，CMake lock audit通过。

同一三项新测试使用原native时全部普通断言失败、exit1；新native共38项相关CUDA通过，无skip/crash。
测试覆盖F16原F32累加的溢出与低位反例、双投影不同累加域，以及F32普通／Q8_0普通／Q8_0 indexed多token OAI。
OAI使用alpha=0和limit=7/2隔离参数传播，原GPU独立路径为oracle，实际原始读取根显式保活。
一次fresh完整128prompt＋31固定输入的64份hidden/logits与v19逐byte相同、全部有限；预测与32×7 capture旅程一致。

仅OAI单测试采集一次短Nsight：官方CSV与SQLite逐symbol一致，MMVF、MMVQ standard、MMVQ MoE各4次真实融合，
共12次；原始driver13.4/collector13.3警告保留，无graph API错误。该证据不代替F16完整GLU的物理命中证明。
未重测本批吞吐、完整模型profile、峰值内存或独立模型质量。
[源码、首失败、红绿结果、64向量与原始trace](cuda-fusion-precision.zip)包含341个payload，CRC及逐文件SHA均核对。
归档后通过根`test.ps1`精确清理12个已替代镜像及map/lib/exp，共462,758,701B；对象、配方、v20当前镜像与共享参考保留。

## 不可变 allocation 的共同绑定索引

本批仅修改共同Graph的`executable.cc`与私有`plan_state.h`，不改模型、数学或Provider。
冷编译在常量和prepared输出确定后，将原绑定位置按实际MemoryResource对象和opaque allocation handle排序；
warm输出bind用lower_bound找到同allocation组，再沿原逻辑检查offset/size、same-value、bound及planned alias。
不同资源或handle不猜测物理别名，动态输入/输出、effects、权限与旧结果/退休gate保持原路径。
编译重试清除旧索引，分配沿原Plan allocator计费与失败释放；无另一个payload、权限cache或执行槽。

不可变部分冷成本O(R log R)、metadata约capacity×sizeof(size_t)，查询O(log R+K)。同bank的全部根仍可进入K，
因此不能宣称所有布局都是O(log R)。当前消费者51输出、418常量原有每轮21,318次C循环，另加51P；
消除全常量扫描不等于同量级端到端收益，尚无函数级CPU耗时归因。

私有PlanState布局改变，鲜编8个核心owner、2个已链接私有头fixture及公共executable fixture，共11个host TU；
CPU112与Qwen125输入闭包重新链接，未编native数学或SDK。9项相关CPU通过，包括同allocation多offset、
部分/跨根重叠拒绝、空闲尾部与其他allocation正控、reset后重绑、prepared真实范围、双槽/旧结果及metadata OOM回滚。
一次fresh完整128prompt＋31固定输入：64份hidden/logits与v20逐byte一致、有限，32预测与capture旅程一致。

无profiler的同窗旧Hyle／新Hyle／llama各3个独立进程交错，复用相同权重、128＋31固定输入和原参考镜像：

| 边界 | 旧Hyle中位 | 索引Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 82.93 | 82.26 | 89.84 |
| warm第7–31步tokens/s | 85.65 | 86.27 | 94.05 |
| prefill step tokens/s | 1969.10 | 1987.00 | 1273.08 |
| 进程入口至首token秒 | 2.270 | 2.245 | 2.011 |

旧warm84.37/85.65/86.75，新86.27/94.80/84.32，范围重叠；full −0.80%、warm +0.73%，未证明稳定提速。
本窗口新Hyle完整decode低参考约8.43%、warm低约8.28%；三方绝对速度均与先前窗口不同，不跨窗归因。
Hyle仍在prefill step前准备两种shape，step吞吐不能替代完整冷工作对比；加载与准备的两引擎子边界也不同。
9进程均exit0，固定输入与预测一致。未测本批完整物理kernel数、峰值内存、其他平台或独立模型质量。
[源码、别名与OOM回归、完整向量及九进程原始记录](immutable-alias-index.zip)共268个payload，CRC及逐文件SHA已核。
归档后按根`test.ps1`清理8个可重建镜像和map/lib/exp，共207,617,040B，保留对象、配方、当前Qwen/CUDA与共享参考。

## 融合读集进入冷分配

九处既有MMVF/MMVQ融合的matcher和八项实际读集改为一份定义，同时供冷规划和原执行器使用。
冷阶段将weight、activation、IDs、bias、scale及gate的真实storage root延寿到潜在融合输出；
Invocation沿ggml既有分配依赖规则生成allocation-only metadata图，reserve和实际alloc使用相同序列，
执行、capture、原src/读者/flags和数学顺序保持不变。隐藏的合法NONE view根计入leaf/hash容量，
新增metadata沿Invocation allocator；不公开输入来强制保活，不加payload副本或另一套executor。
原upstream weighted-MoE optimizer行为保留，本批未额外启用它。

冷匹配只推迟需要真实buffer/地址的检查，热执行保留完整layout/alignment/padding/range/RESTRICT资格。
因此显式重叠仍须回退或拒绝，保守延寿也可能增加workspace而最终未融合；不宣称全局最优分配。
版本`hyle-ggml-cuda-resource-iq1-inference-v21`，source ID `52e46f2bb4908ab3`。
仅鲜编main CUDA、provider和测试三个TU，复用89头的一致闭包及既有数学对象，CUDA/Qwen链接输入仍为133/125。

首编发现搬移遗漏`ids`局部引用，改为同一原MatMul的`src[2]`；此前规范化源码相等核对未覆盖def-use，不能代替编译。
第二次host编译暴露新引用ggml私有头在`/W4 /WX`下的第三方警告；两处CMake将该私有头目录标记SYSTEM，
增量配方对应使用external include，Hyle本身仍保持原警告级别。两次失败及修正源码/命令均保留。

10项相关CUDA全部通过、零skip，覆盖普通与完整GLU的dense/VIEW/NONE根、显式重叠，以及公共Graph内部Copy输入、
双Invocation、两轮重绑、commands/capture及旧输出。公共Graph测量workspace为1408B，精确预算成功、少1B拒绝；
该公共case不据此宣称完整融合命中。一次fresh完整128prompt＋31固定输入的64份hidden/logits与GDN批次逐byte相同、
全部有限，32预测及32×7 capture旅程保持。

独立raw完整GLU用例只采一次短Nsight：6候选观察到6次fused MMVF，6数学对照另有12次unfused MMVF和6次独立OAI GLU，
总36个kernel，官方CSV/SQLite逐symbol一致。因此本用例的完整GLU已物理命中，未退成两段MM＋bias。
driver13.4/collector13.3警告保留；该trace未执行GraphLaunch，不冒充capture链路性能证据。
未测本批吞吐、完整模型物理kernel数、峰值内存、其他平台或独立模型质量。
[源码、首失败、10项回归、64向量与原始小trace](cuda-fusion-allocation.zip)含415个payload，CRC及逐文件SHA均核对。
归档后按根`test.ps1`精确清理旧GDN两镜像和本批map/lib/exp共8个文件、307,143,565B；
当前Qwen/CUDA、共享参考、对象与配方保留，[清理收据](cuda-fusion-allocation-reclaim.json)单独保存。

## 批次复用计划的只读 sizing

此前真实消费者六段attention通常已复用原native计划，但每轮仍先在workspace callback调用原生host scheduler查询。
现在私有callback显式接收本次allow_update意图，FlashInfer用同一原chunk/CTA拓扑判据完成只读sizing与后续update。
合法复用直接报告原计划的实际float/int需求，Queue/context检查保留；公开API、数学核、workspace owner及执行器不变。
Batch/MLA私有扩展版本从6/4升为7/5，全部9个相关header owner对象增量鲜编（8个源，batch.cc两种配置）。

强制fresh不能用旧报告代替：保留的纯host scheduler反例中，old/new KV在旧chunk下均为[7,7]，
fresh搜索却从1000-row/14tiles改为875-row/15tiles，所需partial workspace扩大。共同层还要求上轮完整prepare或已退休run，
并用同一资格同时控制sizing和update；workspace扩容中途OOM清除两种完成状态，重试必须fresh sizing＋fresh plan。
不以变动式update作为sizing探针，不引入另一个计划cache、持续权限cache或产品计数器。

4项公共BatchWorkspace通过，包含强制fresh超cap时保留旧episode，以及扩容下一次metadata分配定点OOM后正确重试，
最后零泄漏。真实私有查询wrapper另用1个case执行ragged/paged两域各5轮：64cold、96update、96强制fresh、
129跨chunk、130update；两次合法update不调用native query，强制fresh和跨chunk仍查询，所有输出按独立均值oracle核对。
wrapper只在独立测试provider对象重命名两个私有符号，始终委托原数学实现；生产Qwen没有计数或测试宏。
一次fresh完整128prompt＋31固定输入的64份hidden/logits与allocation批次逐byte相同、有限，预测和32×7 capture旅程保持。

总计13个host TU，无native CUDA/SDK重编。首编只有独立FI测试缺实际能力宏而失败；保留原命令后补齐该测试定义，
复用11个已成功对象，只补编测试与观察provider两TU，再完成CPU115、CUDA133、Qwen125与probe135输入链接。
paged测试权限过多在编译前静态审查时已改为精确页集，原草稿与修正保留；未更改产品权限合同。

同窗旧Hyle／新Hyle／llama各3个独立进程交错，无profiler：

| 边界 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 85.15 | 84.80 | 94.17 |
| warm第7–31步tokens/s | 88.35 | 88.86 | 95.14 |
| prefill step tokens/s | 1867.59 | 2004.74 | 1277.11 |
| 进程入口至首token秒 | 2.206 | 2.232 | 1.991 |

旧warm86.55/88.45/88.35，新91.62/88.08/88.86，范围重叠；full −0.42%、warm +0.58%，未证明稳定提速。
本窗口新Hyle完整decode低参考约9.95%、warm低约6.61%，不与历史窗口相减归因。9进程exit0，固定输入和预测一致。
生产Hyle实际link inputs加peer构造运行环境，不混入CPU测试SDK。引擎加载/准备子边界、prefill前准备两种shape及
[现行中间精度差异](#中间精度与转换的源码调研)仍须分别解释，step吞吐不是完全相同数学图的端到端结论。
未测本批整模型物理kernel数、峰值内存、MLA/其他平台runtime或独立模型质量。
[源码、准备反例、失败恢复、私有查询观察、64向量与九进程记录](batch-layout-query-reuse.zip)含458个payload，CRC与逐文件SHA已核。
归档后按根`test.ps1`清理旧allocation镜像、本批CPU/probe镜像及map/lib/exp共16文件、525,822,744B；
当前Qwen/CUDA、共享peer及所有对象/配方保留，[清理收据](batch-layout-query-reuse-reclaim.json)单独保存。

## 分组专家投影的原生融合接线

冷lowering按实际tensor/rank/extents共享matrix/dense reshape，并只对相同原IDs/count/extent共享安全tensor；
每个consumer的检查保留。原生MMID producer与公开shape分别保存，合格的separate GLU直接使用producer边，
公开gate/up、额外读者、原精度与最终地址保护保持。plain重复ID仍按槽展开，没有扩大浮点MMVF的单列域。
原生量化MMVQ另修复batch轴：token在ne2，Top-K槽在ne1；两处grid.y使用原device-info初始化所得真实上限。
supports只拒绝实际选MMVQ且超限的域，不缩大token的MMQ资格，无新热CUDA查询或共享CUDA ABI变更。
版本v22、source ID `a80c353ef5972ed2`；仅编1个native与3个host TU，复用89头与最新133/125链接闭包。

25项相邻融合/布局/精度及新grouped回归、3项既有plain F32/BF16重复ID与checked恢复检查均通过，零skip。
新增公开Graph覆盖Q8_0/MXFP4、S10、T1/T2、重复IDs、双实例两轮重绑、公开gate/up/Neg与混格式，
另核E11/E12校验域、非法[-1,11,12]保留输出与原IDs，以及失败槽重建后恢复。
首两次fixture编译缺显式low_precision/model_ops头；补头后运行又揭示新测试误用退休槽，按已有公共合同修正。
实现对象均复用；原编译失败、24/25首轮结果和修正源码保留，未放宽执行失败语义。
设备真实grid.y上限65535；旧native在同一metadata-only反例错误接受65536，新native拒绝，未提交非法grid。
一次fresh完整Qwen的64份hidden/logits与既有基线逐byte相同、有限，预测和32×7 capture旅程保持。

相同Q8公开Graph短用例，旧/新各采一次Nsight，包含8组候选和8个独立单算子GLU数学对照：

| 实际执行 | 旧 | 新 |
| --- | ---: | ---: |
| 未融合MMVQ（标准／MoE） | 8／8 | 0／0 |
| 完整融合MMVQ（标准／MoE） | 0／0 | 4／4 |
| 独立GLU（包括8个oracle） | 16 | 8 |
| Q8_1 activation量化 | 16 | 8 |
| 检查／发布guard | 16／8 | 16／8 |
| 总物理kernel（含oracle与状态合并） | 80 | 56 |

T1为标准MMVQ，T2为MoE kernel，后者grid[4,10,1]/block[32,2,1]。两份官方CSV与SQLite逐symbol一致，
两次target成功、各8次GraphLaunch无API错误；保留driver13.4/collector13.3警告。
本用例证实完整GLU物理命中与共同activation量化复用；未证明MXFP4物理命中、K/IQ覆盖、官方grouped FP8 Top10、
其他平台、峰值内存、整模型新kernel数、吞吐提升或模型质量。
总表另补前8/33条扩展制品的静态provider合同，条件入口与格式缺口分开，全部保持未验证制品执行。
[源码、失败与修正、28项回归、64向量、逐制品静态映射及两份原始trace](cuda-grouped-fusion.zip)。
归档含513个payload，CRC与逐文件SHA已核；随后按根`test.ps1`精确清理12个可重建文件、464,641,229B。
当前Qwen/CUDA、共享peer、对象与配方保留，[清理收据](cuda-grouped-fusion-reclaim.json)单独保存。

## 当前完整 decode 的实际耗时分解

在上述grouped融合后的保留镜像上，双方各采一次完整128prompt＋31固定decode的Nsight trace。
warm每步Hyle/llama为740/558个kernel，投影均为163次；577/395是非投影数，395不是参考总数。
Hyle投影占kernel耗时约91%，其中FFN gate/up约3.332ms、down约1.576ms、LM head约2.502ms；
参考分别约3.158/1.508/2.514ms。163个转换核合计约0.154ms，纯GDN约0.067ms。
逻辑矩阵字节数每步约3.763GB，FFN与head占75.2%；这不是实测DRAM流量。
head等主要投影的中位耗时接近参考，本轮没有支持重新替换全部GEMV的证据。

Hyle/参考内部设备空隙约0.351/0.066ms；七段Graph与六次attention之间的prepare/bind/submit值得继续查。
73次D2D合计约0.088ms，但trace未识别其具体storage角色。CPU API耗时与GPU重叠，不能直接相加。
投影权重均为BF16，中间activation/output精度仍不同；保留driver/collector版本警告与单trace长尾限制。
这轮用于定位，不是新吞吐结论。[完整报告、逐角色样本、命令与两份原始trace](decode-critical-path.zip)，
58个payload的CRC和逐文件SHA已核。

## 固定资源绑定的冷索引

上述prepare/bind审查确认：资源episode在每个borrowed/owned重绑中反复扫描固定外部binding表，
Qwen约百次绑定使这部分管理工作呈平方级。现在只在准备时创建ValueId到固定槽位的索引，
每次仍核对完整ValueId；额外投影页guard只追加在尾部，替换和reset保留原权限与退休语义。
没有缓存旧权限/ready结果，没有公共或私有头ABI变化，也没有引入另一套执行器。

仅鲜编resources及既有CPU测试两个host TU，复用原生数学与SDK；CPU115、CUDA134、Qwen125链接闭包保持。
5项CPU回归通过，包括16路反序重绑、不同generation、配置输入、旧结果与独立storage lease、三页投影suffix；
首fixture编译及Provider/冻结输入构造失败与修正保留，未放宽公共合同。
一次fresh完整Qwen的64份hidden/logits与既有基线逐byte相同、有限，固定输入、预测和32×7capture旅程保持。
FP8实际S修复亦在该最终镜像中；本模型不使用grouped FP8，不能据此代签量化制品。
本批未采新trace、吞吐或峰值内存，复杂度改善不能直接当作decode速度提升。
[源码、失败与最终回归、逐向量比较及命令](decode-resource-binding-index.zip)。
归档后按根`test.ps1`核对精确路径/尺寸/SHA并清理Windows 64个旧镜像及链接产物2,094,515,455B，
Apple另清6个私有镜像/map/资源副本29,979,401B；当前CUDA/Qwen、共享参考、所有对象/配方及最新64向量保留，
Apple的139项增量缓存SHA保持。[本轮清理收据](current-batch-reclaim.json)。

## 冷索引修复后的同窗吞吐对照

2026-10-19北京时间22:12:40–22:13:07，旧资源查找／最新Hyle／llama各3个独立进程交错，无profiler。
最新CUDA镜像已经通过64份完整hidden/logits逐字节回归；旧Hyle只恢复同一127输入链接中的resources.obj，
因此本次旧新对照隔离固定binding冷索引，未混入另一套数学实现、SDK或消费者。

| 范围 | 旧资源查找中位 | 最新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 95.71 | 96.34 | 104.03 |
| warm第7–31步tokens/s | 99.71 | 100.36 | 105.96 |
| prefill step tokens/s | 2068.89 | 2084.13 | 1375.41 |
| 进程入口至首token秒 | 2.012 | 2.007 | 1.766 |

旧新完整decode约+0.67%、warm约+0.65%；warm样本99.71/100.22/99.32与100.36/100.51/98.96重叠，
未证明稳定提速。本窗最新Hyle完整decode低参考7.39%、warm低5.29%；不与20:10窗口绝对速度相减归因。
九进程均exit0、固定输入和预测一致。双方权重同源BF16，中间精度配方仍不同；Hyle的两shape准备在prefill前，
两引擎加载/准备子边界不同，step吞吐不能代替冷端到端结论；本批未测峰值内存。

仅重链接一个旧资源查找对照，无源码/SDK编译；两次本地准备脚本错误均发生在模型进程启动前并保留记录。
[精确链接差异、命令/PID/退出码、九进程原始日志与计时汇总](decode-latest-comparison.zip)含57个payload，CRC与逐文件SHA已核。

## 静态 capture 分段移入图冷编译

原prepare_replay每轮重新扫描最大兼容区间、统计命令并按步骤核Queue身份；现在Plan在最终分段时保存同一组候选区间与计数，
重排后重建，Queue未变时只检查lane。View/ready-only metadata、delegated命令及不兼容命令的边界保持；实际Queue支持、
地址、绑定、权限、录制/失效/回退和退休仍属于Invocation，不缓存旧请求的动态准备事实。

仅鲜编8个私有PlanState实际核心owner及2个测试fixture，无CUDA数学或SDK重编；CPU114/Qwen127链接成功。
12项既有定向回归全部通过、零skip，含兼容区间中的不可录制命令、Require拒绝、换Queue/关闭Context、
delegated/ready-only/View无Queue、重排与编译分配失败回滚。一次完整Qwen的64份hidden/logits逐byte相同、有限，
固定输入、32预测和七段capture旅程保持；尚未运行本批Linux/macOS Graph回归。

2026-10-19北京时间22:25:55–22:26:21，旧／新／参考各3进程交错，无profiler：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 94.78 | 94.99 | 103.21 |
| warm第7–31步tokens/s | 98.95 | 99.38 | 105.15 |
| prefill step tokens/s | 2093.41 | 2102.50 | 1340.16 |
| 进程入口至首token秒 | 2.005 | 1.981 | 1.763 |

旧新完整decode约+0.22%、warm约+0.43%，样本范围重叠，未证明稳定提速。本窗新Hyle低参考约7.97%／5.50%，
不与前窗绝对速度相减归因。每进程warm cold_batch_seconds均值的中位为281.824→256.680微秒；
该汇总包含layout、batch prepare、bind与prepare_replay，不能归为单个函数的纯CPU节省，也不与GPU时长相加。
九进程exit0且预测一致；双方中间精度、冷准备/加载边界仍不同，本批未测峰值内存、独立模型质量或新物理kernel数。

[源码、独立审查、实际依赖、12项回归、64向量比较与九进程记录](cold-capture-topology.zip)。
归档后按根`test.ps1`清理18个此前已归档的消费者镜像与链接产物，共479,714,509B；
当前/旧Qwen、CUDA、共享参考、CPU验证镜像及所有源码/对象/依赖保留，[清理收据](cold-capture-topology-reclaim.json)。

## FFN 投影的单节点硬件计数器

同一固定输入的第7步、首层 FFN gate/up，分别采 Hyle、参考和 Hyle store 指标；每次只匹配一个
6144×256 节点，模型进程均正常结束。Hyle/参考 DRAM read 为50.83/51.90MB，Hyle逻辑BF16权重为50.33MB；
本次没有明显重复读取整份权重的证据。DRAM吞吐指标92.71%/93.96%包含读写，不能解释为有效权重读取达到同等峰值比例。
Hyle store专项中local load/store sectors均为0，未观察到local-memory spill；global store为6144 sectors，
与源码每CTA一次输出store一致，不能将sector请求数当作有效输出字节。

主采样和store专项使用5/3次kernel replay，DRAM write分别7.37/2.98MB；这些计数受回放、缓存及非隔离流量影响，
差值不能归为多余中间张量写入，也没有证明其具体来源。该单节点结果不替代无profiler的完整吞吐，双方中间精度仍不同。
[三份原始报告、精确命令、CSV、独立核读与限制](projection-counters.zip)，CRC及58个payload SHA已核。

## attention 加入原 Graph 录制

实际选中的共享批次实现现可在原QueueSequence中录制，与前后普通算子形成同一区间；没有新执行器、嵌套capture或影子tensor。
每次准备仍撤销当前就绪状态，稳定地址不能跳过原生槽检查；工作区、计划、metadata载体或其所有者变化前先销毁相关录制。
相同载体的内容更新重新取得本轮权限，再由原Graph检查物理绑定并决定复用。独立审查补出了普通TensorView缓存绑定跳过seal的共同层缺口，已修复失败撤回旧route。

最终12项CPU通过、零skip；1项真实CUDA小图完成初次录制、页ID内容更新复用、强制重规划与所有者替换四轮，
并检查旧结果阻止覆盖、失败准备后即使补齐普通端口也不能提交。最终完整Qwen的64份hidden/logits逐byte相同、有限，32预测及固定输入保持。
32阶段均由七段变一段，仍是hybrid，末尾采样没有被伪报为捕获。首次分析脚本误要求全图captured，以及测试编译/配置修正的原失败均保留。
仅增量编译相关host owner，复用原CUDA数学及SDK；未验证本批Linux/macOS旅程、独立模型质量或峰值内存。

2026-10-19北京时间23:16:19–23:16:46，旧／新／llama各3独立进程交错，无profiler：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 95.79 | 96.68 | 104.28 |
| warm第7–31步tokens/s | 99.41 | 100.13 | 106.26 |
| 首次prefill step tokens/s | 2111.35 | 1969.06 | 1337.24 |
| 进程入口至首token秒 | 1.997 | 2.007 | 1.787 |

完整decode约+0.93%、warm约+0.73%，仍是小窗口观察，warm样本重叠；当前低参考7.28%／5.77%。
首次prefill反慢6.74%：cold阶段中位39.175→45.881ms，执行/等待/读回21.352→19.091ms，总step60.625→65.006ms。
不能只报告decode改善；首cold回退的具体内部主因尚未证实。

随后旧新各一次Nsight trace确认warm仍740个kernel，GraphLaunch 7→1，录制kernel 723→735，原12个eager attention/merge已进入外层录制。
这次改变提交边界，没有减少数学kernel。profile没有复现首prefill回退，首GraphInstantiate总和反而4.946→3.817ms；
因此不能认定大图实例化是根因，也不把profile时长替代上述吞吐。首次原生dispatch的主机工作移入录制、减少执行重叠是待查方向。
同一最终镜像使用已有Commands选项的一次完整诊断，首次prefill step56.790ms、预测/输入保持；它同时改变全部区间的录制策略，
不能当作FI单因或首token改善证明，也未据单样本修改消费者默认政策。一次性prefill与重复decode的策略可由引擎通过现有API分别选择。

九进程全部exit0且预测一致；双方中间精度、加载/准备子边界仍不同。源码、独立审查、原始失败、数值比较、九进程日志及两份trace见
[本批证据](batch-graph-capture.zip)。

首次FI调用随后已定位到具体API：旧/新host-wrapper探针的原生调用分别约5.377/5.292ms，端口绑定与Context检查不足1μs；
未改产品的完整CUDA API trace中，首个`cudaFuncSetAttribute`为4.974ms，后五次约1–1.3μs。
原始Graph节点→克隆前节点→录制时launch的映射确认它属于首次prefill FI；默认Nsight未采全部API，先前空白不能解释为纯CPU工作。
这确认首次函数配置成本，尚未证明内部module/JIT机制或上窗6.7ms回退的全部来源；将它移到prepare也不能直接宣称TTFT改善。
[探针、完整trace、映射重放与限制](first-fi-startup.zip)保留原始日志，未据诊断改默认策略。

## 动态别名索引与路由数值修正

Invocation现为动态输入/输出预分配allocation候选索引，成功bind增量更新全部物理成员；内部生成/View范围变化后重建，原检查顺序、权限、失败回滚、reset与退休保持。
避免每次bind扫描全部动态表；共享allocation和碰撞仍可能检查较多成员，源码推导的旧表访问数不能当作实测耗时。
另将普通/hash sqrt-softplus路由改为GA的真实分数和归一化，移除epsilon/零和回退并更新OpID；该修复不参与Qwen数学。

17项CPU、9项CUDA Direct/Graph通过且零skip；两项旧内核对照按预期失败，检出极小分数与全零行的差异。
完整Qwen一次运行的64份hidden/logits逐byte相同、有限，32预测、固定输入与每阶段一段hybrid录制保持。
只增量编译45个host owner及1个CUDA TU，复用SDK；旧Ninja测试配方与合并route测试的原失败、修正和缓存复用均保留。
自定义路由尚无CPU/Apple/Vulkan实现；本批未运行Linux/macOS、完整GA模型、独立质量或峰值内存检查。

2026-10-20北京时间00:12:41–00:13:07，旧／新／llama各3独立进程交错，无profiler：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 96.91 | 96.93 | 104.96 |
| warm第7–31步tokens/s | 100.40 | 100.86 | 106.83 |
| 首次prefill step tokens/s | 2019.08 | 2005.63 | 1391.48 |
| 进程入口至首token秒 | 1.998 | 2.014 | 1.753 |

完整decode仅约+0.02%、warm约+0.45%，不宣称稳定收益；新Hyle低参考7.65%／5.59%。
warm每进程cold_batch均值的中位为247.264→248.084μs，未显示该汇总稳定下降；它含layout、batch prepare、bind和重放准备，不能作单函数CPU归因。
shape prepare中位423.281→438.394ms，TTFT增加约16.5ms，尚未归因；首次prefill step亦慢约0.67%，没有隐藏这些回退或追加窗口挑选结果。
九进程exit0且预测一致；双方中间精度、加载/准备子边界仍不同，不与跨窗绝对速度相减归因。
[源码、独立审查、原始失败、精确测试、64向量SHA与九进程日志](dynamic-alias-routing.zip)。
归档后按根`test.ps1`清理90个临时镜像/旧验证产物与重复向量，回收938,986,967B（895.49MiB）；
当前与对照镜像、全部对象/SDK及64份参考向量保留，[清理收据](dynamic-alias-routing-reclaim.json)。

[剩余转换的实际邻居](../../research/remaining-conversions-2026-10-20.json)将163核分为109次BF16→F32和54次反向；
[当前投影已直接输出BF16](../../research/projection-typed-epilogue-2026-10-20.json)，不能再次假设投影尾部存在F32→BF16 Cast。
下文记录GDN的V/beta typed load与conv的typed/strided sample读取；保留F32数学、state和真实BF16舍入，不以减少launch数预报吞吐。

## GDN typed load 与冷图候选组合

GDN 原生 kernel 现可独立读取 F32/BF16 V 与 beta，在原 load 处扩宽，保持 F32 归约、decay/state、launch geometry 及全部已有 BF16 舍入。
只折叠未发布、单读者的 Cast；实际 BF16 View、strides 和存储根进入原生图。recurrent raw beta 仍经过原 F32 sigmoid。
首次小测试与64向量虽通过，完整 trace 却仍为740核、109扩宽＋54窄化，18个GDN全部F32/F32；原生支持没有变成实际模型接线。
首版[原生源码、命令、测试、计时及未命中trace](gdn-typed-load.zip)独立保留，没有把吞吐约+0.10%写成收益。

根因有三层：普通分段因新的外部producer切开Cast/GDN；仅追加后缀候选仍会被较早普通前缀占走Cast；仅预留Cast又让重叠普通候选挤掉已有18组RMS融合。
第二版trace虽已命中18个BF16/BF16 GDN，RMS却退回F32/F32，仍740核、91扩宽＋72窄化，未继续跑吞吐。
最终由Provider先提出闭合Cast＋GDN，writer成功接纳后将整组排除出普通前缀；原叶候选、rank、cost和Graph共同调度保持。
新增实际NodeOutput/View前沿与CompileReport节点集断言，已检出中途两项失败；保护发布值、额外读者和recurrent beta的断言保持。

最终3项真实CUDA测试含10个内部case通过、零skip；完整模型64份hidden/logits逐byte相同、有限，固定输入与32预测保持。
单次all-API trace确认18层BF16/BF16 GDN及原18组typed RMS均命中，warm **740→704核、163→127转换**（73扩宽＋54窄化），每阶段仍一段hybrid录制。
官方CSV与SQLite逐symbol计数一致；Nsight报告驱动13.4使用13.3采集库，原警告保留。trace时长不作吞吐证据。
T128 workspace/slot为33,947,648→32,374,784B（少1.5MiB），T1为2,033,152→2,024,704B；这不是进程/GPU峰值测量。

2026-10-20北京时间00:58:00–00:58:27，旧／新／llama各3个独立进程交错，无profiler：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 96.49 | 97.64 | 103.99 |
| warm第7–31步tokens/s | 100.03 | 101.46 | 106.23 |
| 首次prefill step tokens/s | 1948.05 | 1936.98 | 1317.55 |
| 进程入口至首token秒 | 2.028 | 2.011 | 1.778 |

本窗完整decode约+1.19%、warm约+1.43%，样本范围仍重叠；新Hyle低参考6.10%／4.50%，不宣称稳定提速或已追平。
首次prefill慢0.57%；cold中位46.371→47.249ms，其他step部分逐进程相减后的中位19.813→18.833ms。
warm cold_batch均值的中位247.512→233.000μs，含layout/prepare/bind/replay，不能归因某一个CPU函数。
首版未命中trace还将19.52ms的首次launch定位到ATen Softplus，首次GDN/FI属性调用另为约1.34/4.86ms；它们不是同一次调用，也未证明无profiler回退的全部来源。
九进程exit0且预测一致；双方中间精度、加载/准备子边界仍不同，未测独立模型质量、峰值内存或其他平台。

[最终源码、独审、全部中途失败、两次后缀trace、64向量SHA与九进程原始日志](gdn-region-suffix.zip)保留可复现材料。
归档后按根`test.ps1`逐文件清理224项临时镜像、副产物和192份重复向量，回收1,290,552,336B（1.20GiB）；
当前/对照/参考/CPU镜像、实际链接对象、SDK及64份原始向量保留，[精确目标与清理收据](gdn-region-suffix-reclaim.json)。

## Conv typed load 与真实布局

CUDA SSM conv现沿真实BF16 sample/View的time/channel/batch strides读取，在原load处扩宽；F32权重、窗口、j顺序MAC、bias和输出保持。
输出BF16 RNE、普通稳定SiLU与history Copy未消除，发布/额外F32读者保留原Cast。共同长核限制末块预取范围，修复原F32亦存在的越界；
冷支持按实际K模板、跨度与设备geometry拒绝无法执行的描述，未新增公共混合精度Conv或模型形状规则。
6个精确CUDA用例零skip，完整64份hidden/logits逐byte相同、有限，固定输入、32预测与每阶段一段hybrid录制保持。
单次all-API trace确认18层typed conv及原GDN/RMS保持：**704→668核，转换127→109（55扩宽＋54窄化），18次F32实体转置消失**。
每slot workspace的T128为32,374,784→28,573,696B，T1为2,024,704→1,824,256B；这是计划workspace，不是进程/GPU峰值。

2026-10-20北京时间01:16:12–01:16:39，旧／新／llama各3个独立进程交错，无profiler：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 97.9626 | 97.2748 | 103.5217 |
| warm第7–31步tokens/s | 101.6379 | 100.4901 | 106.0403 |
| 首次prefill step tokens/s | 2005.70 | 1973.69 | 1347.98 |
| 进程入口至首token秒 | 2.00836 | 2.03042 | 1.78414 |

本窗warm回退1.13%、完整decode回退0.70%、prefill回退1.60%；新Hyle低参考5.23%／6.03%，不宣称速度成功。
warm cold_batch均值的中位增加26.52μs，尚未归因。现有两个独立trace中，短conv每步总时长中位只增约0.288μs；
寄存器32→39但无spill，不能据此将整步回退归因conv。临时caller小探针的prepare_replay旧/新约150.78／151.14μs，
只定位待拆解范围，不作吞吐或单函数根因证明。九进程exit0且预测一致；双方精度、加载/准备子边界仍不同，未测独立模型质量或其他平台。
[源码、独审、6项测试、64向量SHA、实际workspace、原始trace与九进程日志](conv-typed-load.zip)保留本批证据。
归档后按根`test.ps1`逐文件清理80项临时镜像、副产物与重复向量，回收631,477,684B（602.22MiB）；
当前/对照/参考/CPU镜像、实际链接对象、SDK及64份原始向量保留，[清理收据](conv-typed-load-reclaim.json)。

## 冷重放准备索引与宿主分段诊断

seal hook、Copy步骤和segment静态`blocks_host`基值现随最终计划冷建索引，重排后重建，准备时不再做三遍全计划分类扫描。
逐episode仍调用实际hook、验证Copy资源/范围与transfer mode，并核对当前Queue、槽、owner、权限与退休状态；没有按地址缓存权限。
产品批次增量编译9个host TU（8个产品owner＋1个私有fixture），0 native、0 SDK重编；20项CPU零skip，完整64份hidden/logits逐byte相同。
本批没有新物理trace；当前668核沿用前Conv记录。重核当前同口径，延续前述说明：总数为**668／558**、非投影为**505／395**，395不是peer总kernel数。
双方均163次投影、BF16权重与F32累加，但Hyle投影激活/输出为BF16、peer为F32；次数相同不代表完整舍入合同相同。

2026-10-20北京时间01:40:17–01:40:44，旧／新／llama各3个独立进程交错，无profiler：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 96.7053 | 96.4536 | 104.9897 |
| warm第7–31步tokens/s | 100.6261 | 100.3800 | 107.0536 |
| 首次prefill step tokens/s | 2001.989 | 1976.987 | 1363.950 |
| 进程入口至首token秒 | 2.00116 | 2.00906 | 1.77265 |

本窗warm约−0.2445%、完整decode约−0.2603%，样本重叠，未证明稳定退化或收益；新Hyle低参考6.23%／8.13%。
prefill约慢1.249%，TTFT约多7.9ms；warm每进程cold_batch均值的中位254.888→241.868μs，包含layout、batch prepare、bind和replay，不能作单函数归因。

另一次基于改动前Conv源码副本的临时分段probe保留原调用顺序，32预测及每阶段录制/复用旅程保持。warm25总计198.372μs、binding189.480μs，
其中materialized端口处理131.104μs、View28.804μs；provider hooks0.932μs与seal5.684μs是嵌套时间，不能再次相加。
逐step时钟增加扰动，因此不是原caller-only约151μs的无扰动分解，也未证明这些比较或拷贝是单一根因。
实际每轮1119steps、518View、1941次物理绑定比较；下一批研究同Invocation固定描述下的引用比较与条件赋值，不跳过实际权限、owner或准备资格。

[源码、独审、实际owner/deps与命令、CPU/64向量结果、原始probe及九进程日志](prepare-replay-cold-index.zip)保留本批证据。
归档后按根`test.ps1`逐文件清理76项临时/旧镜像、副产物与重复向量，回收392,528,925B（374.34MiB）；
当前Qwen/CPU、Conv对照、参考镜像、全部对象/SDK及64份原始向量保留，[清理收据](prepare-replay-cold-index-reclaim.json)。

## 提交端口减少重复复制

共同绑定循环现以引用比较完整TensorDesc与资源/handle/offset/span，仅在端口值变化时更新原submission数组；输出不再生成两份as_const大值。
revision/native-input失效、alignment、租约/载体刷新、generated输出与seal均保持逐轮处理，没有新缓存、公共API或私有header变化。
仅1个host TU增量及原CPU115/Qwen129链接；21项CPU零skip，完整64份hidden/logits逐byte相同、有限，32预测、固定输入和每阶段一段hybrid录制保持。
CUDA数学对象未变，没有新物理trace；668核仍引用前Conv证据。未运行本批Linux/macOS、独立质量或峰值内存检查。

2026-10-20北京时间01:50:15–01:50:41，旧／新／llama各3个独立进程交错，无profiler：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 97.3421 | 97.5731 | 103.5391 |
| warm第7–31步tokens/s | 100.8581 | 101.5014 | 105.6759 |
| 首次prefill step tokens/s | 1984.188 | 1898.718 | 1338.089 |
| 进程入口至首token秒 | 2.00182 | 1.98380 | 1.77937 |

本窗warm约+0.64%、完整约+0.24%，样本重叠，未证明稳定提速；新Hyle低参考3.95%／5.76%。
首prefill反慢4.31%：cold中位45.361→48.493ms，逐进程step减cold的其他部分中位19.269→19.633ms；具体主因未证实。
shape prepare中位426.602→414.827ms、TTFT少18.02ms，不能将子边界与参考引擎视为相同工作，亦不跨窗相减归因。
warm每进程cold_batch均值的中位239.240→207.868μs，含layout/prepare/bind/replay，不能作单函数节省或与GPU核时长相加。
九进程exit0且预测一致；双方中间精度和加载/准备子边界仍不同。

[源码、独审、实际依赖、21项CPU/64向量结果与九进程日志](binding-port-reuse.zip)保留本批证据。
[下一批Conv输出融合设计](../../research/conv-output-fusion-design-2026-10-20.json)核清实际激活来自SGLang；源码公式相近不替代严格编译数学与真实路由验证。

## Cast 与稳定 SiLU 融合

原F32 Conv不变。下游Cast(F32→BF16)＋普通稳定BF16 SiLU在strict unary TU内融合，保留原Cast RNE、稳定exp/div及最终BF16 RNE。
Graph按实际闭包提出具体两节点候选，重核未发布Cast及其读者/控制边；F32来源与最终输出仍可发布，普通SiLU叶偏好保持。
同一原生程序/Queue执行，冷计划保实际F32读集至末节点；执行前核owner/device/span/alignment和禁止跨宽度别名，不新增公共算子或共同执行器。
Cast→SiLU→Mul若同时合格，可有两种各2dispatch的候选；本批没有承诺强制保原SiLU/Mul carrier，也未加模式优先白名单。

首编3host＋2native均成功；首轮5测试失败源于夹具把`output()`借用视图误当结果租约，只重编1个test TU并重链测试，原Qwen镜像未重链。
原失败源码/XML/日志保留。最终5项CUDA全通过、零skip：65536种BF16位型对应输入、F32舍入ties/邻居与双RNE见证、发布/额外读者/View/control保护、2槽×2轮换址与借用旧视图、失败准备恢复。
真实Direct Cast＋默认SGLang unary作原链参考；有限/零/Inf按位比较，NaN核分类，不把借用视图检查写成GraphOutput租约验证。
完整64份hidden/logits逐byte相同且有限，32预测、固定输入及每阶段一段hybrid录制保持；独立质量、其他平台及总峰值未测。

单次all-API current trace确认18处新kernel，原18层BF16 Conv/GDN与typed RMS保持：**668→650核，转换109→91（55扩宽＋36窄化），实体转置0**。
官方CSV与SQLite计数一致；Nsight旧采集库警告保留，trace时长不作吞吐证据。
T128每slot workspace为28,573,696→27,435,008B，T1为1,824,256→1,815,808B；F32 Conv仍物化，这不是进程/GPU峰值。
两份既有trace各自D7–31的450个Conv，均按本进程/device/context/stream定位邻接后继；原18Cast＋18SGLang SiLU每步时长和中位37.727μs，新18融合15.840μs。
仅配对Conv后实际18个Cast，未把旧54个窄化全算作消除；独立profile窗口的约21.887μs差值不能直接归因完整吞吐。

2026-10-20北京时间02:06:05–02:06:31，旧／新／llama各3个独立进程交错，无profiler：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 98.0949 | 97.9274 | 103.5339 |
| warm第7–31步tokens/s | 101.3918 | 101.5118 | 105.3241 |
| 首次prefill step tokens/s | 2031.556 | 2019.587 | 1320.969 |
| 进程入口至首token秒 | 1.98712 | 2.02358 | 1.79949 |

本窗warm约+0.12%、完整约−0.17%，样本重叠，未证明稳定收益；新Hyle低参考3.62%／5.42%。
prefill慢0.59%，TTFT多36.46ms、shape prepare多8.07ms，尚未归因；warm cold_batch均值的中位206.028→204.060μs，不作单函数归因。
九进程全部exit0、预测一致，双方精度和加载/准备子边界仍不同，不把跨窗速度变化归为本批。
[实际源码与依赖、独审、首轮失败与修正、5项CUDA/64向量、物理trace及九进程日志](cast-stable-silu.zip)保留本批证据。

本轮已按根目录 `test.ps1` 精确清理 148 个已归档临时镜像、链接副产物及重复向量，共 725,425,378 字节（691.82 MiB）；保留当前与基线镜像、CPU/参考镜像、源码/对象/SDK 和 64 份原始比较向量。见 [清理记录](cast-stable-silu-reclaim.json)。

## 后续功能覆盖与热点方向

CUDA dense BF16 index scorer 现已复用原公共 MatMul／Relu／Mul／Reduce 图及实际 K/W View。
候选只覆盖数学与可选 mask／扩宽，修正免费 metadata 先被选择而挡住复合候选的问题，原排名与共同规划器保持。
原 Lt＋typed tile 沿同一 Provider、Invocation、QueueSequence 和调用者 workspace 执行；无整历史 K 解码或复制，显式保留原舍入阶段。
11项CUDA最终全通过、零skip，含双槽两轮/真实batch gap＋2B、旧结果、8次RequireCapture、逃逸/空轴，以及原packed/paged路线。
小trace实录8次GraphLaunch、160颗kernel：32个实际key tile各执行Lt、Relu、Mul、head sum、typed publish五颗kernel。
首次误选退回普通算子、旧测试对象引用已删除API、夹具类型名与漏prepare等失败均保留；最终完整Qwen64份向量逐byte相同。
该批没有测吞吐、总峰值或完整GA模型；[完整记录](../../correctness/ga-dense-indexer-cuda-2026-10-20.json)与上述Qwen速度窗口分开。

同时只读重查既有650核trace：D7–31逐步GPU kernel时长和的中位，163次投影约8,740μs，所有kernel约9,531μs，91个转换核约73.5μs。
投影已占该口径约九成；下一轮需优先核对其实际布局、带宽与执行成本。此处没有重采peer，也不把profile时间和解释为完整decode或可保证的提速空间。

本批已按 `test.ps1` 清理152个已归档临时/过时镜像、链接副产物及重复向量，共969,563,035字节（924.65 MiB）；当前Qwen、其直接性能基线、CPU/参考镜像、源码/对象/SDK和64份原始向量保留。[精确清理记录](../../correctness/ga-dense-indexer-cuda-2026-10-20-reclaim.json)。

## 稠密单行 TopK 的地址计算开销

650核trace中的词表TopK仍走通用布局iterator，每元素重复64位除模和布局映射。
现仅对实际稠密、inner=1且count=width的单行归约直接按字节线性读取；仍用原CCCL DeviceReduce、ArgPair/ArgBest及输出原bits发布，非连续/多行路径保持。
F32/I32/F16/BF16的双方向stable ties、全NaN payload、两种首±0及generic邻界由独立稳定排序oracle核对；6项相关CUDA测试全过、零skip。
完整模型64份hidden/logits逐byte相同且有限，固定输入、32预测及一段hybrid录制保持。

同一benchmark main仅换CCCL CUDA对象，Commands路线上AB/BA/AB三个独立进程对：
提交至完成的墙钟进程中位129.90→36.55μs，CUDA事件区间121.856→19.968μs。
固定约970KiB输入热复用，事件包含提交空隙，不是kernel-only或DRAM上限测试。
一次完整模型trace确认D7–31每步一颗`SingleRowArgInput`归约、零颗原generic归约，仍是650颗kernel；归约核本trace中位3.392μs。
旧trace约104.927μs来自另一采集窗口，不能直接将该差值当作完整模型收益。

2026-10-20北京时间03:01:54–03:02:20，旧／新／llama各3个独立进程交错，无profiler或构建：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 98.5127 | 99.3538 | 103.7939 |
| warm第7–31步tokens/s | 102.0579 | 103.2696 | 106.0302 |
| 首次prefill step tokens/s | 2027.206 | 1963.362 | 1361.722 |
| 进程入口至首token秒 | 1.99737 | 1.97458 | 1.76369 |

本窗完整／warm提升0.85%／1.19%，新Hyle仍低参考4.28%／2.60%；prefill回退3.15%，样本范围重叠，未归因。
九进程全部exit0且预测一致，保留原精度与加载/准备子边界差异；未测新的总内存峰值或独立质量，也不由三次进程宣称长期统计显著。
[源码、命令、审查、微测、模型回归和实际trace](../cccl-single-row-topk-cuda-2026-10-20.zip)同时保留首稿误设capture但未编未跑的纠正记录；Provider仍是Commands承载，没有改capture资格。

投影审查确认双方均163次投影，但中间精度不同；既有NCU中head已约97% DRAM峰值，dual约93%，不是可保证的剩余提速比例。
普通投影整体换MMVF的历史实验近似持平，不重复接入。
随后实测双投影两行/CTA：两份host/native TU增量构建，三对共享权重共150,994,944B，超过实际L2 37,748,736B；
每进程先核三组K2048/N6144与N1/3/33特殊值的完整BF16 bits及16元素尾哨兵，六进程全部通过。
AB/BA/AB三个进程对的原一行／新两行中位事件区间均107.52μs，墙钟115.6／115.9μs，未测出收益，未推广到产品或重跑整模型。
实际ptxas为39→48寄存器/线程，dynamic shared 256→512B，stack/spill均0；不能据此单独归因占用率或性能。
该检查以同一strict TU保留的原一行入口为差分参考，不是独立数学oracle；[候选、真实命令/依赖、独审、资源与六进程记录](../dual-matvec-row-tiles-2026-10-20.zip)保留了边界。

本批已先写根目录`test.ps1`再执行精确清理：76个临时镜像/链接副产物及重复向量，共483,112,857字节（460.73 MiB）。
当前/基线Qwen、CPU/参考镜像、全部源码/对象/SDK与64份原始向量保留；[清理证据](../cccl-single-row-topk-cuda-2026-10-20-reclaim.json)已校验CRC及逐项SHA。

## View 投影元信息复用

单次scratch消费者只加4处阶段时钟，不改数学或runtime：warm准备中位约209.4μs，其中layout11.7、batch49.5、101项bind14.8、replay138.2μs。
这不是core子函数采样，不能把replay全归某个循环。现有实际冷路由含518个View，每轮会重复算extent和构造相同TensorView。
现沿原Invocation readable值复用canonical精确span；描述不同回原工厂，始终先从当前parent重算实际subview/offset/bounds/overflow，
仅resource/handle/offset/size全等时复用原metadata。原alignment、发布精确range、alias、native hook、seal与租约检查原位保留；无新状态字段或ABI。
两份host TU增量构建，14项相关CPU全过/零skip；唯一新case含三层View、较大backing裁剪、offset0→2→2和内容改变、错误shape准备失败后重试。
完整模型64份hidden/logits逐byte相同且有限，32预测/固定输入及一段hybrid旅程保持；未新增native或SDK编译。

同一diagnostic main仅换共同Graph对象，AB/BA/AB三对进程：replay准备的进程中位134.4→125.1μs，但总准备197.2→200.0μs，未证明总准备下降。
仍遍历图和ports，精确描述比较仍O(rank)；不是消除全图扫描或跳过当前episode的权限检查。

2026-10-20北京时间03:19:06–03:19:32，旧／新／llama各3个独立进程交错，无时钟插桩、profiler或构建：

| 范围 | 旧Hyle中位 | 新Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 99.2798 | 99.8293 | 103.8388 |
| warm第7–31步tokens/s | 102.5069 | 103.1033 | 105.6764 |
| 首次prefill step tokens/s | 2039.711 | 1989.221 | 1356.319 |
| 进程入口至首token秒 | 1.97837 | 1.98140 | 1.77480 |

本窗完整／warm约+0.55%／+0.58%，样本重叠，不宣称稳定收益；新Hyle低参考3.86%／2.43%。
prefill约−2.48%，样本有重叠；[原日志的边界拆分](../view-projection-reuse-prefill-2026-10-20.json)显示首次冷准备中位43.7445→45.4649ms，submit至token为18.8456→18.8501ms。
差异主要落在首次准备区间，区间内部仍未归因；不把它写成已解决或把跨窗变化归因给本补丁。各分量中位数不能直接相加当总中位数。
九进程全部exit0且预测一致；中间精度、加载/准备子边界差异仍保留，本批未重采native trace、总内存峰值、其它平台或独立模型质量。
[源码、当前父存储/失败/租约审查、14项CPU/64向量、阶段时钟和九进程记录](../view-projection-reuse-2026-10-20.zip)保留本批依据。

本轮已按根目录`test.ps1`核对并清理94个可重建临时/过时镜像、链接副产物及重复向量，共911,152,409字节（868.94 MiB）。
保留当前CPU/Qwen、直接Qwen基线/参考、全部实际链接对象/库/源码/SDK与64份原始比较向量；[精确清理记录](../view-projection-reuse-2026-10-20-reclaim.json)已校验CRC和逐项SHA。

## 当前实现重测

2026-10-20北京时间07:50:41–07:51:08，使用当前产品闭包、原同源无插桩消费者，旧／当前／llama各3进程交错。
仅重编1个消费者C++文件再链接；复用既有CUDA、SDK与产品对象，删除旧best-of对象及测试依赖。
完整模型64份hidden/logits逐byte相同，九进程均exit0且预测一致；保留原BF16合同，未启用F32消费者草稿。

| 范围 | 旧Hyle中位 | 当前Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 98.8722 | 100.2040 | 103.5256 |
| warm第7–31步tokens/s | 102.4731 | 103.7081 | 106.1450 |
| 首次prefill step tokens/s | 2035.581 | 2001.229 | 1339.645 |
| 进程入口至首token秒 | 2.00256 | 1.99879 | 1.75549 |

本窗warm提升1.21%，三个进程中位范围未重叠；完整提升1.35%，范围有重叠。当前仍低参考2.30%／3.21%。
prefill低1.69%，范围有重叠；双shape准备中位424.735→429.814ms，不能据此归因其中某个函数。
当前权重加载1.04332s、shape准备0.429814s；参考模型加载1.578979s、context准备0.007435s属于不同工作边界，不能逐项判输赢。
三次进程不代表长期显著；未重采整模型trace或独立质量，不把本窗变化归于未被该BF16模型执行的FP8共享量化。
[当前闭包、实际依赖、64向量与九进程原始记录](../qwen-current-retest-2026-10-20.json)保留本轮依据。

随后额外各运行1个相同命令，使用PSAPI和持久PDH per-PID计数器，请求40ms采样；两个进程均exit0。
这些采样运行不进入上述吞吐统计。Windows归因的最大观察值如下，单位GiB：

| 计数 | 当前Hyle | llama |
| --- | ---: | ---: |
| CPU working set | 4.0182 | 4.0687 |
| CPU private commit | 5.4767 | 4.8837 |
| GPU dedicated | 3.8057 | 3.9464 |
| GPU shared | 0.0762 | 0.0801 |

private commit不是物理驻留量，以上列不能相加当总物理内存；PDH存在刷新延迟，GPU最大观察值也不等于瞬时真实峰值或CUDA allocator总量。
Hyle／llama分别采到71／66条记录，其中GPU有效记录62／63条；完整时间序列、OS高水位及实际采样间隔见同一实测报告。

[随后单对象计数诊断](../qwen-lease-lookup-probe-2026-10-20.json)保持原模型64向量，实际确认原生region输入租约查找
每次扫描102条绑定，而只有1条持有可用读租约；decode每步680次查询、69360次记录访问。
插桩只证明重复工作数量，不用于耗时归因；准备阶段的过滤设计仍须保留当前权限、覆盖范围与原顺序。

本批已先写根目录`test.ps1`再执行精确清理，删除77个可重建文件，共513,800,205字节（约490MiB）。
当前Qwen／测试镜像、直接基线／参考、全部链接对象与库、当前64向量及源记录保留，见[清理记录](../qwen-current-retest-2026-10-20.cleanup.json)。

## 同次准备的读租约候选

通用ResourceEpisode现在只在一次受guard保护的绑定准备中按原序收集非独占读记录，
原生输入端口沿候选序号逐次检查当前view、资源与覆盖范围；候选退出或失败即清空，不缓存权限或命中。
序号空间使用原metadata allocator冷预留；有效零字节读、first covering、独占writer排除及同址撤权语义保持。
2个公共Graph C++文件和1个已有测试文件增量编译，7项CPU回归全过；正式与隔离计数镜像各64向量逐byte相同。
本旅程实际只有1次collect/step：完整访问2217072→25000，decode每步69360→782；最坏R=B时仍为O(P*B)。

2026-10-20北京时间08:11:49–08:12:16，同main对象、只换两个Graph对象，旧／当前／llama各3进程交错：

| 范围 | 旧Hyle中位 | 当前Hyle中位 | llama中位 |
| --- | ---: | ---: | ---: |
| 完整31步decode tokens/s | 98.3632 | 98.2742 | 103.8084 |
| warm第7–31步tokens/s | 102.3881 | 102.8047 | 106.2615 |
| 首次prefill step tokens/s | 1954.595 | 1941.097 | 1336.616 |
| 进程入口至首token秒 | 2.02775 | 2.02235 | 1.77792 |

warm约+0.41%、完整约−0.09%，样本范围明显重叠，未证明稳定提速；当前低参考3.25%／5.33%。
原消费者的整个cold_batch准备边界中位201.9→196.3μs，同样重叠；它包括layout、batch prepare/reset、101次绑定及prepare_replay，
不能将其全部归于租约查找，亦不能用独立计数镜像的时间或跨窗差值作收益。
九进程均exit0、预测一致，中间精度及加载／准备边界差异保持；未新增整模型trace、内存采样、质量或跨平台验收。
[源码、独审、实际计数和九进程记录](../lease-read-filter-2026-10-20.json)保留本批依据。

## BF16 输入、F32 门控与 BF16 存储融合

原稳定 SiLU×Mul 现在可直接读 BF16、保原 F32 SiLU／乘法并最终写 BF16，四节点复用原融合候选。
13项相关CUDA回归及完整64向量通过；实际25个warm阶段每步650→614核、转换91→55，typed门控18处。
每步仍只有1次GraphLaunch，含609个图内核，另有5个Gather／ArgMax辅助核。
同一微基准对象、18组独立后缀、六进程交错，捕获后提交至完成中位54.0→42.4μs；输入／输出校验和相同。
进程中位呈双档漂移，不能把21.5%幅度推广到整模型，也不能用历史Cast时间和推导收益。

2026-10-20北京时间08:31:44–08:32:10，旧／当前／llama各3进程同窗：warm为102.74／103.23／106.24 tokens/s，
完整31步为98.02／99.57／104.48，首次prefill为1914.52／1995.95／1387.23，首token为1.995／1.976／1.752秒。
旧→当前warm约+0.47%、完整约+1.59%，范围重叠，未证明稳定提速；当前低参考2.83%／4.69%。
九进程均exit0且预测一致；BF16/F32中间计算及加载／准备边界差异仍在，未重测峰值内存或独立质量。
[本批源码、数值与寿命回归、物理trace及公平性审查](../typed-gating-suffix-2026-10-20.json)保留原始依据。

按根`test.ps1`核对后，本轮合计回收997,988,851字节重复向量、微基准和过时镜像；保留当前CPU／CUDA镜像、
即时基线、参考、全部对象与依赖缓存。两次精确清理见[租约批次](../lease-read-filter-2026-10-20.cleanup.json)与
[门控批次](../typed-gating-suffix-2026-10-20.cleanup.json)，已执行的删除脚本不可直接重跑。

## 冷常量准备队列复用

2026-10-20北京时间09:04:21–09:04:47，旧／当前／llama各3进程：双形状准备中位437.308→332.805ms，
首token2.034→1.917秒（llama1.802）；当前prefill2064.84、完整31步99.44、warm102.44 tokens/s，
参考分别1317.10、103.51、106.27。该窗warm旧104.17→新102.44，范围未重叠；一次后续B/C/C/B复测为
103.09→104.44，完整99.75→100.77，方向反转。分别保留两窗，尚未证明稳定decode提速或回归。

原Queue按实际Context在同次冷编译复用，逐份Completion等待和事务发布不变；两形状合计stream／pinned创建182→2、
event创建2912→32，各形状194核、170小上传和等待次数不变。16项相关回归与最终64向量通过。
原位置映射释放的单次诊断为372.975ms，未把它移出TTFT；没有flush，也不能由此指认OS unmap为库缺陷。
小投影原权重实验的576个BF16值相同，18组捕获片段90.9→50.4μs，但权重仅2.25MiB、可容于36MiB L2，
尚未产品化或证明整模型收益。[本批源码、失败、实际trace与两个独立计时窗](../cold-preparation-queues-2026-10-20.json)保留依据。

## 同激活双输出小投影

随后将两条普通BF16 MatMul接为通用多输出冷候选，保F32累加及两个独立BF16输出，不复制权重，完整FFN优先。
5项新CUDA、7项原生邻居及8项共同CPU回归通过，完整64向量逐byte相同；实际25个warm阶段每步614→596核，
18个pair、55个Cast、591个图内核加5个辅助核、1次GraphLaunch。T1显式workspace每槽1811712→1807616B。
同批把共同收缩查询的scratch改为候选拓扑跨度，并补完整FFN的冷输入root延寿；未重测整模型峰值。

2026-10-20北京时间09:29:46–09:30:16九进程，旧／当前／参考warm90.61／91.59／92.96 tokens/s，
完整87.37／87.74／91.10；旧→当前中位+1.08%／+0.42%，但后段各臂均有明显慢样本。
另一次带粗粒度GPU时钟旁采的B/C/C/B四进程为warm84.48→84.70、完整81.98→81.95，
不合池、不删慢样本，未证明稳定decode提速。全GPU采样不能作进程归因或精确decode时钟归一化。
[本批源码、数值、物理trace及两窗原始记录](../paired-small-projections-2026-10-20.json)保留全部边界。

## 有界 pinned 文件加载

2026-10-20北京时间11:17:36–11:18:03，旧加载／新加载／llama各3个新进程交错：
TTFT中位2.139→1.446s（-32.39%，参考2.006s），ready2.069→1.376s。
新Hyle／参考prefill为1865.14／1278.01、完整decode88.36／90.72、warm90.19／91.78 tokens/s；
该窗新旧decode完整／warm为-2.02%／-1.58%。另一次11:19:25–11:19:36的C/B/B/C四进程，
TTFT2.106→1.429s（-32.15%），完整decode+0.71%、warm-0.44%，prefill两窗约-0.6%／-0.4%。
两窗分别保留，不混合后窗Hyle与前窗参考；小负差未同向复现，不能证明稳态收益或无回归。

加载复用现有FileResourcePipeline、原Queue和两个8MiB pinned slot，保原320个device分配及权重字节；
文件关闭、管线／native退休、临时运行时释放均在TTFT内。64份hidden/logits逐byte相同，13个计时进程均exit0、预测一致；
实际采纳源码另经1次增量编译及固定128＋31旅程核对，不加入性能样本。
新进程未清OS页缓存，不能称冷盘加载；旧load子计时不含mmap释放，新load含管线清理且lease域创建前移，净收益只按完整启动判断。
约32%是TTFT改善，本批未改数学或证明decode提速，未重测峰值内存或独立质量。
[源码、初次编译失败、64向量、两窗原始计时与采纳收据](../qwen-pinned-weight-load-2026-10-20.json)一并保留。
