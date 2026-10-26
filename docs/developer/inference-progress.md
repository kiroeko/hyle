# 推理架构 Review 进度

更新：2026-10-27 01:10（北京时间）。本轮目标：**处理推理架构 Review，补齐功能、收敛架构，再改善性能**。整体约 **61%**；总数与各分支按实际交付和剩余实现工作量一起复估，不表示测试通过率。先共同引擎核心，再稳态状态协议，最后重叠执行。当前仍以底层支持为主，HTTP／通用 loader／tokenizer 不是前置。

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
- 稳态状态协议：约60%（实现中）
  - 持续工作态与按需导出不可变快照：约90%（KV已接持续工作帧与按需封存，有限检查通过）
  - KV持续工作帧及无失败事务发布：100%（本批实现完成，Windows／Apple有限功能检查及B2对照完成）
  - 原生 GDN／conv 的版本化双缓冲与索引状态：约55%（Store和私有索引GDN已有限检查，正在接共用绑定及Graph融合；conv／引擎待完成）
  - MTP 稳定状态槽及接受／回滚接线：0%（待实现）
- 其它底层功能补齐：约50%（实现中）
  - 长度分桶、启动预热政策与异长批次：约10%（设计中，已确认真实分块与活跃长度边界）
  - 完整会话主机换出／换入：约95%（普通／MTP／lag1及受控组件故障已有限检查，原生驱动故障未查）
  - 状态搬运非阻塞观察／取消：100%（CPU受控合同及CPU／Vulkan／Apple实际关闭后回收通过有限检查）
  - Vulkan非4字节权重片段接入：100%（F3／Tensor／Graph／Vulkan已接通，原tiny普通／MTP／lag1通过有限检查）
  - CUDA 小通道因果卷积承载：100%（已补，定向数值与原失败消费者通过）
- 重叠执行与完整性能对照：约5%（设计中）
  - 提前准备、设备 token 接力与有序发布：约5%（设计中）
  - CPU／GPU／混合参考与完整四项对照：0%（待执行）
- 遗漏复查与资料收敛：约20%（进行中）
  - 文档恢复、事实校正与阅读入口：100%（本轮修正完成）
  - 剩余功能与迁移遗漏复查：0%（待完成）

## 正在做

- GDN的Graph索引融合：约50%（所属：稳态状态协议／索引状态；原生三组逐位检查通过，正在共用槽事务并接冷匹配、真实读写权限和无Y输出的状态写入；Graph及引擎未完成）。

剩余大项仍包括原生GDN／conv稳定状态、MTP稳定状态槽、形状政策及重叠执行，不能以KV完成代签这些工作。

## 本批结果与下一步

索引状态的三个前提已落地：[原池槽资源](../evidence/correctness/inference-lease-slots-2026-10-26.json)25项、[融合写资格](../evidence/correctness/inference-fused-writer-2026-10-26.json)4项、[CUDA浮点Gather](../evidence/correctness/inference-cuda-slot-gather-2026-10-26.json)3组有限检查通过。后者保持F32／F16／BF16原码并支持行padding；未完成GDN／conv整段融合，也未重测完整模型性能。
模型无关的Store描述及[槽绑定事务](../evidence/correctness/inference-slot-bindings-2026-10-27.json)另有6组检查通过，覆盖元数据失败不消费writer、寿命、真实地址与预算；这些资源类型不代表实际Store执行已接通。
[读权限撤回](../evidence/correctness/inference-projected-read-release-2026-10-27.json)11项检查通过：真实退休后只撤回指定源及其alias与guard，旧源可重获写权，其它绑定仍保留；包含三项原unbind回归。该次检查尚未包含下述实际Store执行，原生索引GDN／conv仍是剩余实现的大头。

[Store实际执行](../evidence/correctness/inference-slot-store-2026-10-27.json)已接默认CPU／CUDA装配：5个新实际写入检查及8个相邻权限合同通过，覆盖固定索引载体复用、非连续目标槽、未选择槽、原结果寿命、非法重绑和reset后失权拒绝。本批只成功编译6个不同单元、链接1个镜像，保留首次测试命令缺Toolkit头路径的失败；未重建SDK、未接引擎或测性能。该fallback不支持capture、冷索引上传可能等待，下一步仍是原生索引融合。

[私有索引GDN](../evidence/correctness/inference-indexed-gdn-native-2026-10-27.json)三组原生检查通过：K1、T4>K2最后K个状态及BF16 V／beta，与原稠密F32路径逐位一致，另核源槽、未选槽及padding不变。首次BF16失败来自旧融合预检漏排除新模式，已窄修并复用成功内核／测试对象；实际只编两个不同NVCC单元和一个测试单元，dispatcher因该修正补编一次。保持原F32运算体及旧fused-cache布局；未证明公开Provider／Graph／引擎接通或完整提速。

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

以下各格为 **Hyle／llama.cpp**。速度单位 tok/s，显存为独立采样 dedicated 峰 GiB。B2来自状态复用后的最近跨引擎窗口，单用户普通／MTP来自更早窗口；本批工作帧只做Hyle控制／候选A/B，未重新运行llama.cpp，以下表格不混入新样本。

| CUDA 工况 | prefill | decode | 综合速度 | 显存峰 GiB |
|---|---:|---:|---:|---:|
| 单用户单轮普通 | 2094.269／1578.977 | 98.474／102.141 | 89.724／89.680 | 3.888／3.728 |
| 单用户单轮 MTP | 2229.803／1462.412 | 122.475／134.439 | 109.691／111.814 | 4.048／3.896 |
| 单用户两轮普通 | 2894.643／2453.351 | 98.605／102.124 | 92.351／94.137 | 4.023／3.728 |
| 单用户两轮 MTP | 2797.193／2420.500 | 127.091／140.805 | 116.165／125.083 | 4.191／3.896 |
| 普通 B2 三轮固定输入 | 7469.731／6086.119 | 154.941／170.683 | 136.981／143.737 | 3.999／4.743 |

前四行来自2026-10-26的[32进程窗口](../evidence/performance/inference-current-window-2026-10-26.json)，自然续文跨引擎不同。最后一行来自此前状态复用后的[8进程B2窗口](../evidence/performance/inference-b2-reusable-state-2026-10-26.json)：C1024、N2／R2、prefix off、无MTP，每请求三轮P256＋32固定输入；1728输入行、99物理执行和198输出ID一致。prefill＋22.73%、decode−9.22%、综合−4.70%；工作集峰Hyle／参考1.080／4.106GiB，private commit5.624／5.926GiB，ready内存基线未知。参考为实际保留版本，不声称最新上游。各窗口不混算，ID一致也不代表中间数学逐位相同；不能用跨窗绝对速度变化归因工作帧收益或退步。

CPU最新[同窗对照](../evidence/performance/inference-cpu-constant-view-regions-2026-10-26.json)：i9-13900K、原生4／ATen4线程、调用线程、P154＋16、C256、N1／R1、无MTP；参考为同版本AVX2／FMA／F16C。

| CPU 配置 | prefill | decode | 综合速度 | 工作集峰 GiB |
|---|---:|---:|---:|---:|
| Hyle／llama.cpp | 43.953／42.937 | 11.437／12.027 | 3.323／3.314 | 5.002／3.828 |

CPU decode仍慢约4.9%，private commit为5.740／0.606GiB；映射和私有分配不能当权重大小。内存峰均为有限采样下界，缺样不记零；加载口径和OS缓存不同，不排名冷盘。CPU＋GPU、Apple／Vulkan仍缺当前完整消费者公平基线。
