# D5-A6：CPU / Apple / Vulkan 真实片段性能批次

<!-- MODULE REVIEW: AUDITED -->

baseline `51da0f8d6`。承接当前D5的A6；产品FIX为零，只接性能消费者，复用各路线已验收的
源码/static/shared SDK。CUDA保持原独立合同，不换掉其官方FP4/FP8 MoE；GPU验证尊重用户正常桌面使用，不要求计数器绝对稳定。

## 固定消费者与有限FIX

1. 共用已验收的Qwen3.5-2B官方BF16完整FFN：本地权重、M1/H2048/N6144、两个变化输入。
2. 共用已验收的DeepSeek Flash UD-Q4_K_XL完整hash-MoE：H4096/N2048、top6、七个驻留专家，
   MXFP4 routed与Q8_0 shared；三个变化activation/token/table，包含原sqrt-softplus归一化。
   保留真实256路打分宽及原容差，只发布最终合成输出，不让诊断中间结果约束融合。
3. CPU、Apple、Vulkan各一份原生执行适配，共用测试配方和计时入口；不是产品万能执行器。
   冷准备、资源和实际Provider报告单列；热路径包括完整提交、完成、失败检查和本轮所有权回收。
4. 三路线批量接线后，按低并行2、重型串行验证两个形状的全部变化输入及两次重复；只复用
   已建SDK，不重建产品，不重复A5。消费者若有编译/接缝错误只按具体失败修正。
5. 消费者稳定后固定不可变输入、公共仪器合同与对照源码，再启动适用P1；未完成资格不采样。
   本批不改公共统计核心、不增产品算子、Provider、模型或精度例外。

## 比较边界预审

比较一个阻塞到最终输出/失败可见的完整片段，输入/权重常驻；两侧都只允许完成后复用槽位。
不是裸叶函数开销，也不把Graph/Direct比值解释成数学SOTA或整模型吞吐。

- CPU：使用普通默认Context、一个native线程和调用者线程；Direct持有已准备的完整调用序列，
  提交后收回全部Completion；Graph使用原生region/资源规划和prepared replay。固定同一CPU资源。
- Vulkan：Direct使用公开Recorder组合准备程序，预绑调用者所有的最终/中间目标，一次提交并
  等待终止，不人为逐节点host wait。Graph保留普通Program/Invocation和结果租约；本消费者不将
  租约跨次保留，所有租约/完成释放都计入调用。Recorder不会拆开成熟Provider自己的region。
- Apple：纯MLX FFN的Direct使用Context.compile组合完整原版MLX表达式，保留BF16节点舍入和
  F32 norm/gate计算；不能以逐叶wait作该形状的对照。混合ggml Metal MoE按完整原生边界组合，
  使用公开`lower()`和`Context.compile`合并连续MLX节点，仅发布后续region需要的端口；
  ggml整段使用公开prepare。输入必须ready的跨region同步计费；不把完整任务比率声称为纯调度开销。
  Graph保留MLX融合与完整ggml region。两侧均保有所有输入/常量所有者，并收回所有结果。

固定消费边界下的额外检查、布局转换和同步全部计费；不关闭某一侧错误检查或绕过公开API。
若某一API不能表达上述比较，先记录具体接缝并停止该对照，不制造必胜的劣化reference。

## 验证预算与停止

复用W1-WIN及MacBook Air M1-MAC，CPU锁既定P核，GPU不改变系统电源/权限或关闭用户应用。
先输出/变化输入/终止/资源核对，再按各路线实际实例拓扑做A/A，执行与分配双顺序。
通过后每路线两个固定形状一次P1，工程非劣门5%，n15，至多一次n31。环境或A/A失败不记产品FAIL，
不加样本、不换形状；有界定位后记录恢复条件。已有正确性/三交付不因文档改变重跑。

当前：CPU、Apple、Vulkan消费者均已编译并通过两个完整片段的输出、变化输入与完成检查，合计60次；[原始记录](../../../../evidence/correctness/llm-moe-d5-real-fragments/portable-performance-consumers/README.md)保留初次失败及修正。CPU和Apple现已分别完成8进程A/A、60进程正式比较和30进程公共控制，两个形状均为PARITY/5%非劣PASS，停在n15；Vulkan的8进程A/A已通过，n15的60进程正式比较与30进程控制已完成；自动n31因用户需要恢复桌面使用而中止，最终P1仍为INCONCLUSIVE。全部结果进入D5同一批次汇总，D6不首次补测。

`784819ca`的CPU两个完整形状和两入口20次输出/完成检查已通过。Vulkan首个FFN输出通过后，
消费者错误地将冷准备字节数当作ggml已热身资源；既有`managed_region_test.cc`已明确上游首次
逐槽分配辅助buffer。仅修消费者：保留冷值，首轮完整槽位预热后记录实际值，第二轮及所有后续
计时前后必须保持该值；全部成本仍记录，不删除资源检查、不改产品。CPU不因该修正重跑。

Apple首轮完整MLX片段输出检查失败。复核原普通Apple消费者的Qwen门为`0.008*abs(expected)+3e-8`，
新共用消费者误用了CUDA的`+1e-8`；纠正为原Apple门，并补失败元素/实际值日志。完整MLX区域
仍须实际通过，不能以原先串联小Graph的通过代签融合后结果；产品不变，原失败保留。

## 固定执行批次

三路线复用同一份 `bench/graph_route_cpu/external` 适配与公共仪器；不改仪器统计核心。
源码文本按 Git 跨平台换行归一后绑定，SDK 和输入按原始字节绑定；复用上批逐文件 SDK 清单。
Apple 补公共 QoS 设置与遥测，Windows 保留 P 核调用者和 native-runtime-pair 线程边界。

每路线 A/A 固定 8 个进程：Graph/Graph 与 Direct/Direct，各执行顺序 × 分配顺序四组合。
同 CUDA 已定规则：单进程偏差不得超过 10%，总体及每因素分组中位偏差不得超过 5%。
失败不加样本、不启动产品性能裁决；通过才由原公共仪器执行固定 n15→至多一次 n31。
使用 fresh 消费者和公共 control 编译目录；SDK 不重建。各路线串行执行，结果分别裁决，
单路线环境问题不阻塞其他路线。A/A 和 P1 都保留最终输出、变化输入与完成检查。

## CPU / Apple 已完成

[CPU P1](../../../../evidence/performance/graph-cpu-d5-fragments-p1/README.md)与
[Apple P1](../../../../evidence/performance/graph-apple-d5-fragments-p1/README.md)均已完成，
公共仪器严格重放通过。候选 `1bf0b556`，产品 SDK 未变；A6 不把这些结论外推到全部算子或整模型。
CPU 两形状四顺序中位耗时差在约±0.4%内；Apple FFN约-0.65%至-0.19%、MoE约-3.64%至-3.50%，
均按预定5%等价带裁决为PARITY，没有扩样。原始日志、控制项与全部区间已归档。

Windows初次预检因reference稀疏checkout漏了测试头停止，补齐同一不可变reference的目录后才启动
真正采样；原失败保留。CPU旧验证构建和失败预检构建已清理约91.6MiB，合格构建与SDK继续保留。
Vulkan继续使用已固定同一合同；用户已结束GameViewer串流并关闭其它应用，记录变化后的GPU背景，
以一次固定A/A裁决稳定性；不要求桌面计数器绝对为零，不重复调门或无限换配置。

## 用户桌面窗口结束：Vulkan 本次采样终止

Vulkan 固定 A/A 八进程通过，n15 的60个产品进程和30个公共控制进程完整保留。
用原公共仪器离线解析，四项公共控制全部通过；Qwen FFN 的非劣区间仍未裁决，
DeepSeek GGUF MoE 的诊断非劣门通过，但不能代替本次完整资格化。
仪器自动进入 n31，已持久化8个产品日志时按用户桌面时间限制停止 launcher 和子进程；
确认无 HYLE 测试进程继续使用GPU，本轮不再启动 CUDA 或 Vulkan 测量。

[中断收据](../../../../evidence/performance/graph-vulkan-d5-fragments-interrupted/README.md)
保存完整n15、部分n31、原A/A、固定合同及SHA256。缺少末尾环境/产物资格核对和正式manifest，
本次裁决为 **INTERRUPTED / INCONCLUSIVE**，不记产品性能FAIL，也不以诊断结果签PASS。
本次采样停止，退回本卡AUDITED的验证计划，产品FIX仍为零；未来GPU验证前先核对已有完整
数据与有限剩余预算，不重跑CPU/Apple/A5，不要求用户继续关闭应用或提供绝对平稳GPU。

## 正常桌面环境下的一次中断恢复

用户随后明确授权继续测试，接受当前占用和波动，不再要求进一步静默桌面。
baseline `15dae3023b94c97b8a538347be623384bc40e930`；承接本卡当前owner/AUDITED，
产品、消费者、输入、精度合同和公共仪器FIX均为零。

恢复前已核对旧n15与部分n31：旧完整n15继续作为诊断证据保留，8个n31日志也完整保留，
不丢弃不利样本或合并不同采样阶段。公共runner只在末尾写出host/source/artifact资格记录，
上次中止没有这些记录，也没有可恢复的checkpoint；不能凭事后当前状态伪造旧批完整资格。
本次不改公共runner或统计核心，采用原合同**一次完整恢复批次**：复用静态产品SDK，
只新建外部consumer/control构建；8进程A/A通过后n15，只有原停止规则触发才升级一次n31。
这是用户中断后的证据恢复，不是因性能结果重试；旧批不计入新批统计，新旧结果分别归档。
本批无论PASS、FAIL或INCONCLUSIVE都停止采样，不再换顺序、配置或扩大样本。

CPU/Apple已通过证据、四路线A5和所有产品正确性结果继续复用。CUDA按独立原卡完成其既定恢复；
两GPU路线重型验证串行，DLL邻置、隐藏子进程，结束后立即告知用户可正常使用GPU。
