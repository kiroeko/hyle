# 推理架构 Review 进度

更新：2026-10-27 05:15（北京时间）。本轮目标：**处理推理架构 Review，补齐功能、收敛架构，再改善性能**。整体约 **77%**；总数与各分支按实际交付和剩余实现工作量一起复估，不表示测试通过率。先共同引擎核心，再稳态状态协议，最后重叠执行。当前仍以底层支持为主，HTTP／通用 loader／tokenizer 不是前置。

## 任务树

- Review 事实核实与范围收敛：100%（已完成）
  - 源码／性能／依赖／第二消费者核查：100%（已完成）
  - 因果修正与实施顺序：100%（已完成）
- 首批功能与接线修正：100%（已完成）
  - 空闲新形状预热与真实接纳计时：100%（已完成）
  - 普通合批单成员取消与明确结果：100%（有限 CPU／CUDA 检查通过）
  - 模型 EOS 配置集中与诊断迁移：100%（已完成）
- 模型无关的共同引擎核心：约65%（实现中）
  - 请求／兼容合批／接纳与缓存编排：约80%（共同循环与取消／交付已实现，有限检查通过）
  - 模型适配、错误类别与输出合同：约60%（取消／交付已接线，有限检查通过）
  - SDK／Provider／依赖与诊断组织：约65%（已补公开页导出、冻结、Completion／CopyBatch保活及Context身份，有限检查通过）
- 稳态状态协议：约95%（实现中）
  - 持续工作态与按需导出不可变快照：约90%（KV已接持续工作帧与按需封存，有限检查通过）
  - KV持续工作帧及无失败事务发布：100%（本批实现完成，Windows／Apple有限功能检查及B2对照完成）
  - 原生 GDN／conv 的版本化双缓冲与索引状态：约95%（原池接线、CPU／CUDA有限功能及真实CUDA融合通过；其它平台尾部待完成）
  - MTP 稳定状态槽及接受／回滚接线：100%（本批CPU／CUDA异成员接受通过；性能和其它平台边界另列）
- 其它底层功能补齐：约60%（实现中）
  - 长度分桶、启动预热政策与异长批次：约30%（普通真实分块已接CLI并检查；预热政策及全模型异长承载仍待完成）
  - 完整会话主机换出／换入：约95%（普通／MTP／lag1及受控组件故障已有限检查，原生驱动故障未查）
  - 状态搬运非阻塞观察／取消：100%（CPU受控合同及CPU／Vulkan／Apple实际关闭后回收通过有限检查）
  - Vulkan非4字节权重片段接入：100%（F3／Tensor／Graph／Vulkan已接通，原tiny普通／MTP／lag1通过有限检查）
  - CUDA 小通道因果卷积承载：100%（已补，定向数值与原失败消费者通过）
- 重叠执行与完整性能对照：约15%（设计中）
  - 提前准备、设备 token 接力与有序发布：约5%（设计中）
  - CPU／GPU／混合参考与完整四项对照：约25%（CUDA普通／MTP最新四臂完成，其它工况仍待完成）
- 遗漏复查与资料收敛：约20%（进行中）
  - 文档恢复、事实校正与阅读入口：100%（本轮修正完成）
  - 剩余功能与迁移遗漏复查：0%（待完成）

## 正在做

- Vulkan 16位状态尾部：约50%（所属：稳态状态协议／原生索引状态；精确Store及pipeline保活已完成，正在准备原码16位Gather承载）。

剩余大项包括索引状态的其它平台尾部、形状／预热政策、共同核心边界及重叠执行，不能以本批CUDA通过代签这些工作。

## 本批结果与下一步

[普通prefill分块CLI](../evidence/correctness/inference-prefill-chunk-cli-2026-10-27.json)三条tiny CUDA旅程通过：中间块不采样、异长B2多轮合批／pending保真、prefix复用及N1淘汰重建；三个非法组合明确拒绝。只编一个Model单元、复用223输入，无SDK重建。默认保持无限块长；MTP有限块长明确拒绝，不称已实现varlen、完整预热或吞吐提升。

[新32进程四臂窗口](../evidence/performance/inference-indexed-current-window-2026-10-27.json)已完成，普通decode单轮／两轮慢5.42%／4.64%，MTP慢7.46%／6.98%；最新完整表在下方。仅更新本窗，不把跨日速度变化归因某次优化。
[MTP异成员检查](../evidence/correctness/inference-mtp-batch-slots-2026-10-27.json)两轮B2/K3分别取1／3、3／1前缀，CPU／CUDA均通过；原Store候选字节、KV／draft续算、旧快照和未接受槽回收符合原合同，第二轮两池目的ID实际不同。受控接受不是自然质量证明。
[Vulkan半字Store](../evidence/correctness/inference-vulkan-halfword-store-2026-10-27.json)五项有限检查通过：精确2／6字节、原码／gap、换序／reset及外壳先释放；Completion沿原Run保活pipeline。非4字节Gather仍缺原生承载，不称整链完成。

[引擎原池状态接线](../evidence/correctness/inference-engine-slot-state-2026-10-27.json)已完成本批CPU／CUDA有限检查：tiny正式CLI普通／MTP各生成8 token；两后端的镜像、lag1、状态复用共6进程通过。原N2／R1三槽容量内两块交替，旧冻结快照字节不变；真实Qwen3.5-2B另通过17个接受前缀、7组后缀扰动，五种冷子图各选中18个索引GDN及18个索引卷积融合。实际6次C++编译、4个不同TU配置、4次链接；复用SDK及原生内核，无下载。保留一次源码并发变化的不确定编译、一次参数拒绝和一次真实append取回顺序失败；后者修正后另冻源码重新编译。没有模型质量或全平台完成结论。

本批[普通B2同窗A/B](../evidence/performance/inference-engine-slot-state-2026-10-27.json)完整非时间报告及198个ID一致，1728输入行／99次执行相同。decode＋2.20%、综合＋1.51%；prefill−0.80%且范围重叠，显存采样峰低48MiB。ready中位增加44.32ms，冷形状准备增加52.90ms；收益与代价分别保留。该A/B未包含llama.cpp或MTP；本批新四臂窗口另列，不能拿普通B2收益代签MTP提速。

索引状态的三个前提已落地：[原池槽资源](../evidence/correctness/inference-lease-slots-2026-10-26.json)25项、[融合写资格](../evidence/correctness/inference-fused-writer-2026-10-26.json)4项、[CUDA浮点Gather](../evidence/correctness/inference-cuda-slot-gather-2026-10-26.json)3组有限检查通过。后者保持F32／F16／BF16原码并支持行padding；当次尚未包含下述GDN融合，conv及完整模型性能仍未完成。
模型无关的Store描述及[槽绑定事务](../evidence/correctness/inference-slot-bindings-2026-10-27.json)另有6组检查通过，覆盖元数据失败不消费writer、寿命、真实地址与预算；这些资源类型不代表实际Store执行已接通。
[读权限撤回](../evidence/correctness/inference-projected-read-release-2026-10-27.json)11项检查通过：真实退休后只撤回指定源及其alias与guard，旧源可重获写权，其它绑定仍保留；包含三项原unbind回归。该次检查尚未包含下述实际Store执行，原生索引GDN／conv仍是剩余实现的大头。

[Store实际执行](../evidence/correctness/inference-slot-store-2026-10-27.json)已接默认CPU／CUDA装配：5个新实际写入检查及8个相邻权限合同通过，覆盖固定索引载体复用、非连续目标槽、未选择槽、原结果寿命、非法重绑和reset后失权拒绝。本批只成功编译6个不同单元、链接1个镜像，保留首次测试命令缺Toolkit头路径的失败；未重建SDK、未接引擎或测性能。该fallback不支持capture、冷索引上传可能等待，下一步仍是原生索引融合。

[私有索引GDN](../evidence/correctness/inference-indexed-gdn-native-2026-10-27.json)三组原生检查通过：K1、T4>K2最后K个状态及BF16 V／beta，与原稠密F32路径逐位一致，另核源槽、未选槽及padding不变。首次BF16失败来自旧融合预检漏排除新模式，已窄修并复用成功内核／测试对象；实际只编两个不同NVCC单元和一个测试单元，dispatcher因该修正补编一次。保持原F32运算体及旧fused-cache布局；未证明公开Provider／Graph／引擎接通或完整提速。

[CUDA GDN的Graph融合](../evidence/correctness/inference-indexed-gdn-graph-2026-10-27.json)15项有限检查通过：原13项Store／权限合同，加上Y发布、仅写状态两种实际融合；后者覆盖reset后拒绝旧权限及取回原事务。共用唯一StoreBinding及一次小索引上传，原Recipe的effect roots保留无Y输出的状态写入；不复制状态载荷。另修权限预留误用旧extra计数的问题，扩容失败仍保留原投影。7个不同C++单元增量编译，5个产品对象全程复用，追加编译只修测试；各通过结果的实际镜像、DLL装配及测试预算／alignment失败均在记录中。尚未接引擎或conv，未测完整性能，也未验证全部平台与图形。

[可移植源绑定](../evidence/correctness/inference-slot-source-fallback-2026-10-27.json)本批11项有限检查通过：共同冷准备按原Graph关联Store／Gather，CPU稠密及padding源槽可取回重用，替换源ID前先确认合法准备成功、替换后拒绝；CUDA原两融合及公开Gather结果后的默认fallback均与原稠密状态逐位一致。默认CCCL叶子缺失的读取资格已补，未强制换Provider。首次CPU缺输出绑定、CUDA测试误以为无用节点会裁掉，以及缺资格的原失败均保留；通过结果分属实际镜像，不声称同一最终镜像重跑全组。ggml叶子资格已补但本批未单独命中，尚未接引擎／conv或证明提速。

[原生槽写入](../evidence/correctness/inference-native-slot-store-2026-10-27.json)窗口中，新增Apple 3项（4种载荷）与Vulkan 3项有限检查通过，共同层另有原CPU／CUDA 11项回归通过。三种实际镜像分别记录；共同层回归后的默认初始化拼写修正另经两原生镜像编译，不冒称全组在同一源码镜像重跑。原生路径复用唯一槽事务、原ExecutionSource、FIFO／Completion及真实退休，不伪造Compute Queue；普通Gather保留实际stride、原精度及选行权限。该窗口覆盖生成结果、结果保活、未选槽／padding和reset回收，只检查4字节对齐item／offset；其后CPU与Apple补充见下，conv与完整引擎仍未完成。

[CPU默认选行](../evidence/correctness/inference-cpu-storage-gather-2026-10-27.json)已补上述CPU缺口，11项有限检查通过：普通Compute／Graph按存储格式复用Native原码选行，保留padding行stride，避免整池CONT及F16／BF16经F32往返；新规则不依赖模型、硬件或Store注解，稠密F32和已有实测规则保留。检查包含默认槽投影、两种普通入口、原7项槽合同和一项既有路由回归，未测模型或速度。仅增量编译5个单元；保留测试编译失败与离线XML汇总修正，不重跑已通过进程。

[Apple精确字节范围](../evidence/correctness/inference-apple-slot-byte-ranges-2026-10-27.json)5项有限检查通过：原3项Graph回归，加生成的BF16 2／6字节行及原FIFO拷贝合同。实际对齐范围仍有2次native blit，2字节范围为精确host copy；全部实际范围和别名先验后才写，源失败不重算。保留原事务、结果保活、真实退休及reset回收，不读写额外padding。只增量编译3个单元，保留两次验证配方缺头文件／搜索路径失败；未跑模型、性能或原生故障注入。Vulkan尾部仍需真正的原码索引／copy承载，不能仅放宽4字节transfer门。

[CUDA索引卷积及共用融合](../evidence/correctness/inference-indexed-conv-graph-2026-10-27.json)7项有限GPU检查通过：4项新conv检查及3项原GDN回归。普通图的Gather／Concat／Cast／Conv与最后K个原精度history窗口共用原槽事务；实际覆盖F32多成员、BF16长chunk、F16短chunk及逃逸读者fallback。更换目标槽ID后的第二轮实际复用原CUDA录制，历史原码、未选槽及padding保持；不复制状态载荷。只编译6个不同单元，共7次编译、2次链接；首镜像未运行，最终镜像只运行一次。没有完整引擎、模型性能或原生故障注入结论；当时尚未包含下述共同批量控制。

[共同批量控制与事务复用](../evidence/correctness/inference-batched-slot-controls-2026-10-27.json)取得29项有限CPU／CUDA通过结果：按真实执行源／Queue／资源共用固定控制载体，独立池各保真实槽号；覆盖两轮换ID、未全部take拒绝重绑、reset及真实晚期权限失败回滚。clear／refill保留原元数据并重新核租约；两处权限预留已去掉重复计数。6个不同C++单元、9次编译、2次链接，原生CUDA与SDK全部复用。首镜像26项通过、3项因测试漏Store控制边而拒绝构图；只改两测试单元，第二镜像仅重跑这3项并通过，未重复原通过项。首次重载解析编译失败与两镜像结果均保留。Apple／Vulkan本批未新运行，其原生copy内部仍可临时分配；没有模型或速度结论。下一步接入真实引擎，不能用组件通过代签MTP接受／回滚及完整提速。

KV工作帧已接通Compute、Graph及引擎：41项页存储检查、23组前缀／工作态检查、7个CPU／Vulkan／CUDA功能进程和4个递归MTP对照进程通过。Apple复用缓存增量编译11个单元，tiny普通／MTP及lag1两次运行通过，状态字节、续算、旧快照及最终资源归零一致。旧快照、失败边界、封存失败与较短接受前缀均有有限覆盖；[Windows功能记录](../evidence/correctness/inference-paged-frame-2026-10-26.json)和[Apple功能记录](../evidence/correctness/inference-paged-frame-macos-2026-10-26.json)保留原失败和未验证项。本批没有新Linux运行，4个诊断probe虽已适配但未单独编译；有限检查不代表全平台与模型质量穷尽。

[同窗Hyle控制／候选B2](../evidence/performance/inference-paged-frame-2026-10-26.json)的六计时＋两独立内存进程全部成功：prefill7509.27→7401.39、decode156.72→158.30、综合138.47→139.75 tok/s；分别−1.44%（范围重叠）、＋1.01%／＋0.92%（范围不重叠），三样本不代表统计显著性。显存采样峰相同，未证明节省。198个输出ID、1728输入行、99次执行及完整非时间报告一致；本批没有新llama.cpp进程，不刷新旧跨引擎对照表。

### 此前交付依据

- 文档：补齐16份缺页并校正范围与合同；入口见[架构索引](../architecture/README.md)。
- 共同请求循环、工作态和取消交付：[共同循环](../evidence/correctness/inference-request-loop-2026-10-26.json)、[工作态迁移](../evidence/correctness/inference-working-state-2026-10-26.json)、[请求控制](../evidence/correctness/inference-request-controls-2026-10-26.json)。
- 状态块复用与批量权限撤回：[状态复用](../evidence/correctness/inference-reusable-state-2026-10-26.json)、[批量append](../evidence/performance/inference-bulk-append-2026-10-26.json)。后者未证明提速。
- 完整主机镜像、普通／MTP／lag1：[镜像](../evidence/correctness/inference-host-image-2026-10-26.json)、[lag1](../evidence/correctness/inference-host-image-lag1-2026-10-26.json)、[原生平台](../evidence/correctness/inference-native-host-image-2026-10-26.json)。固定bank归还槽不等于释放显存；模型搬运仍同步。
- 副本保活、非阻塞观察及关闭后回收：[退休](../evidence/correctness/inference-transfer-retirement-2026-10-26.json)、[公开副本接口](../evidence/correctness/inference-copy-public-retention-2026-10-26.json)。观察错误不等于退休。
- Vulkan片段对齐与CUDA小通道卷积：[片段接入](../evidence/correctness/inference-stream-padding-2026-10-26.json)、[卷积承载](../evidence/correctness/inference-cuda-causal-conv-2026-10-26.json)。
- SDK权重冻结：[所有权边界](../evidence/correctness/inference-storage-freeze-2026-10-26.json)。

单carrier不是已证实的落后主因。[同设置CPU诊断](../evidence/performance/inference-b2-peer-cpu-critical-2026-10-26.json)支持减少主机协议工作，未证明整个区间都可消除。剩余职责见[整体设计](../architecture/inference-design.md)，原始性能与候选取舍见[性能资料](performance/decode.md)。

## 最新完整性能对照

最新两版Hyle同窗普通B2对照（各3次计时，另各1次内存采样）：

| 指标 | 改动前 | 当前 | 变化 |
|---|---:|---:|---:|
| prefill tok/s | 7668.03 | 7606.82 | −0.80%，范围重叠 |
| decode tok/s | 158.48 | 161.96 | ＋2.20% |
| 多轮综合 tok/s | 140.55 | 142.67 | ＋1.51% |
| dedicated显存峰 GiB | 3.999 | 3.952 | −48MiB，采样下界 |

这是Hyle版本对照，不与下列旧双引擎窗口拼接。三样本不代表统计显著性；ready显存基线未知。

以下各格为 **Hyle／llama.cpp**。速度单位 tok/s，显存为独立采样 dedicated 峰 GiB。B2保留此前跨引擎窗口，单用户普通／MTP使用新32进程窗口；上方普通B2的两版Hyle A/B不混入跨引擎表。

| CUDA 工况 | prefill | decode | 综合速度 | 显存峰 GiB |
|---|---:|---:|---:|---:|
| 单用户单轮普通 | 2160.112／1559.773 | 91.483／96.723 | 84.200／85.325 | 3.876／3.728 |
| 单用户单轮 MTP | 2055.844／1476.649 | 120.143／129.830 | 105.853／109.071 | 4.013／3.896 |
| 单用户两轮普通 | 2613.800／2422.222 | 92.057／96.537 | 86.083／89.395 | 4.009／3.728 |
| 单用户两轮 MTP | 2481.770／2297.157 | 123.706／132.991 | 112.146／117.957 | 4.151／3.896 |
| 普通 B2 三轮固定输入 | 7469.731／6086.119 | 154.941／170.683 | 136.981／143.737 | 3.999／4.743 |

前四行来自2026-10-27的[32进程窗口](../evidence/performance/inference-indexed-current-window-2026-10-27.json)：索引状态接线镜像，不含随后CLI分块政策。各臂内稳定，16份Hyle非计时报告与此前对应窗口一致；自然续文跨引擎不同，旧时间不混算。最后一行来自此前状态复用后的[8进程B2窗口](../evidence/performance/inference-b2-reusable-state-2026-10-26.json)：C1024、N2／R2、prefix off、无MTP，每请求三轮P256＋32固定输入；1728输入行、99物理执行和198输出ID一致。prefill＋22.73%、decode−9.22%、综合−4.70%；工作集峰Hyle／参考1.080／4.106GiB，private commit5.624／5.926GiB，ready内存基线未知。参考为实际保留版本，不声称最新上游。各窗口不混算，ID一致也不代表中间数学逐位相同；不能用跨窗绝对速度变化归因工作帧收益或退步。

CPU最新[同窗对照](../evidence/performance/inference-cpu-constant-view-regions-2026-10-26.json)：i9-13900K、原生4／ATen4线程、调用线程、P154＋16、C256、N1／R1、无MTP；参考为同版本AVX2／FMA／F16C。

| CPU 配置 | prefill | decode | 综合速度 | 工作集峰 GiB |
|---|---:|---:|---:|---:|
| Hyle／llama.cpp | 43.953／42.937 | 11.437／12.027 | 3.323／3.314 | 5.002／3.828 |

CPU decode仍慢约4.9%，private commit为5.740／0.606GiB；映射和私有分配不能当权重大小。内存峰均为有限采样下界，缺样不记零；加载口径和OS缓存不同，不排名冷盘。CPU＋GPU、Apple／Vulkan仍缺当前完整消费者公平基线。
