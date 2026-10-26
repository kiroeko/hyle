# 推理架构 Review 进度

更新：2026-10-26。本轮目标：**处理推理架构 Review，补齐功能、收敛架构，再改善性能**。整体约 **48%**；不沿用上一轮完成比例。先共同引擎核心，再稳态状态协议，最后重叠执行。当前仍以底层支持为主，HTTP／通用 loader／tokenizer 不是前置。百分比估计开发工作量，不表示测试通过率。

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
- 稳态状态协议：约30%（实现中）
  - 持续工作态与按需导出不可变快照：约80%（已实现，有限检查通过）
  - 原生 GDN／conv 的版本化双缓冲与索引状态：约10%（设计中）
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
- 遗漏复查与资料收敛：约10%（进行中）

## 正在做

- 稳态状态协议：约30%（所属：稳态状态协议；核对持续工作态仍需的导出／reframe／绑定／发布，设计原kernel直接写固定GDN／conv状态槽，尚未启动下一批构建）。

## 本批结果与下一步

已补形状预热／接纳计时、EOS配置、普通合批单成员取消、输出credit／ACK、共同请求循环与持续工作态。有限检查和保留的失败见[工作态迁移](../evidence/correctness/inference-working-state-2026-10-26.json)、[共同循环](../evidence/correctness/inference-request-loop-2026-10-26.json)及[取消与交付](../evidence/correctness/inference-request-controls-2026-10-26.json)。原生状态导出、KV reframe和复制仍在；不据元数据复用宣称提速。

完整主机镜像已接CPU／CUDA／Apple／Vulkan，带独立数量／字节预算、真实CopyBatch退休和全状态恢复；见[镜像实现](../evidence/correctness/inference-host-image-2026-10-26.json)、[lag1边界](../evidence/correctness/inference-host-image-lag1-2026-10-26.json)及[原生平台检查](../evidence/correctness/inference-native-host-image-2026-10-26.json)。前批三类GPU原始335角色的普通／MTP、lag1／非零prior、同形状续算、旧快照和资源释放均通过有限检查。lag1使用明确验证前缀，不冒充自然接受或任务质量；受控pending／原生故障仍待补。固定bank归还槽不代表显存下降。

原tiny Vulkan加载失败已修：复用F3显式padding，原文件字节／形状不变，实际载体与预算按四字节补齐；Tensor／Graph保留原已初始化的只读范围，普通数学复制仍按逻辑长度。56项定向检查及原tiny Vulkan普通／MTP／lag1、同一程序CPU lag1通过；见[片段接入](../evidence/correctness/inference-stream-padding-2026-10-26.json)。仅两个BF16单元素角色各补2B，普通／MTP实际传输为191240／298632B。首次链接重复输入、3个测试误用提交入口、CPU测试缺DLL位置均保留原记录并修正；没有放宽产品提交、所有权或设备对齐合同。本批未重跑原335或Apple／CUDA新片段消费者，未测性能。

公开`CompletionRetention`、搬运非阻塞poll／原端点取消、关闭后成功writer归还已通过[退休与回收检查](../evidence/correctness/inference-transfer-retirement-2026-10-26.json)。后续已把CopyBatch源查询／附加和Apple观察锁收回原实现，搬运及hidden-output副本移除原Completion借用，Context身份改用公开查询；9项Device、两系统CPU受控合同、实际CPU／Vulkan／Apple关闭后回收及tiny Vulkan普通／MTP／lag1通过，见[公开副本接口](../evidence/correctness/inference-copy-public-retention-2026-10-26.json)。三份copy实现近期修改另经独立源码复审，未发现确定正确性缺陷；源清单按已接受phase引用的producer保守保活，不冒充精确accepted wait清单。首次头文件类型限定错误已修，原失败记录保留。模型换出／换入仍同步，未注入GPU驱动故障，无新性能结果。

CUDA小通道causal conv已由原native provider补F32后备，ggml能力声明同步收窄，默认优先级不变；三项定向检查和原失败消费者通过，见[算子承载](../evidence/correctness/inference-cuda-causal-conv-2026-10-26.json)。本轮只做相关增量编译，累计经`test.ps1`清理已替代验证二进制约1.82GB，没有新完整性能结果。

SDK现提供受检零拷贝`StorageOwner::freeze`，权重层已移除直接LeaseAccess调用；8项所有权检查及新权重入口的原始Qwen普通／MTP CUDA消费者通过，见[SDK边界](../evidence/correctness/inference-storage-freeze-2026-10-26.json)。下一步优先稳态状态协议、版本化状态槽，再处理形状政策和重叠执行。供应商命名、已安装detail头和按数学形状匹配的CPU偏好表本身不是模型白名单问题，不机械改名或删表。

单carrier不是落后主因。[同设置CPU诊断](../evidence/performance/inference-b2-peer-cpu-critical-2026-10-26.json)的共同90稳态根，GPU／CUDA API外运行区间为99.701／14.365ms；差85.337ms支持减少主机协议工作，但不是全额可删成本。架构合同见[整体设计](../architecture/inference-design.md)，范围见[工程计划](../implementation/inference-roadmap.md)，口径和关闭候选见[性能资料](performance/decode.md)。

## 最新完整性能对照

以下各格为 **Hyle／llama.cpp**。速度单位 tok/s，显存为独立采样 dedicated 峰 GiB。正式窗口尚未重测本批修改；单次诊断不替换速度表。

| CUDA 工况 | prefill | decode | 综合速度 | 显存峰 GiB |
|---|---:|---:|---:|---:|
| 单用户单轮普通 | 2094.269／1578.977 | 98.474／102.141 | 89.724／89.680 | 3.888／3.728 |
| 单用户单轮 MTP | 2229.803／1462.412 | 122.475／134.439 | 109.691／111.814 | 4.048／3.896 |
| 单用户两轮普通 | 2894.643／2453.351 | 98.605／102.124 | 92.351／94.137 | 4.023／3.728 |
| 单用户两轮 MTP | 2797.193／2420.500 | 127.091／140.805 | 116.165／125.083 | 4.191／3.896 |
| 普通 B2 三轮固定输入 | 7791.635／6450.984 | 163.452／181.111 | 144.200／152.128 | 3.999／4.743 |

前四行来自2026-10-26的[32进程窗口](../evidence/performance/inference-current-window-2026-10-26.json)，自然续文跨引擎不同。最后一行来自另一个[8进程B2窗口](../evidence/performance/inference-b2-current-window-2026-10-26.json)：C1024、N2／R2、prefix off、无MTP，每请求三轮P256＋32固定输入；1728输入行、99物理执行和198输出ID一致。参考为实际保留版本，不声称最新上游。各窗口不混算，ID一致也不代表中间数学逐位相同。

CPU最新[同窗对照](../evidence/performance/inference-cpu-constant-view-regions-2026-10-26.json)：i9-13900K、原生4／ATen4线程、调用线程、P154＋16、C256、N1／R1、无MTP；参考为同版本AVX2／FMA／F16C。

| CPU 配置 | prefill | decode | 综合速度 | 工作集峰 GiB |
|---|---:|---:|---:|---:|
| Hyle／llama.cpp | 43.953／42.937 | 11.437／12.027 | 3.323／3.314 | 5.002／3.828 |

CPU decode仍慢约4.9%，private commit为5.740／0.606GiB；映射和私有分配不能当权重大小。内存峰均为有限采样下界，缺样不记零；加载口径和OS缓存不同，不排名冷盘。CPU＋GPU、Apple／Vulkan仍缺当前完整消费者公平基线。
