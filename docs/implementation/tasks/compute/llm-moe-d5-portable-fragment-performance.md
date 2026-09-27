# D5-A6：CPU / Apple / Vulkan 真实片段性能批次

<!-- MODULE REVIEW: AUDITED -->

baseline `51da0f8d6`。承接当前D5的A6；产品FIX为零，只接性能消费者，复用各路线已验收的
源码/static/shared SDK。CUDA保持原独立合同，不换掉其官方FP4/FP8 MoE；GPU未空闲不重启采样。

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

当前：消费者接线尚未验证；A/A、P1均未运行。全部结果进入D5同一批次汇总，D6不首次补测。

`784819ca`的CPU两个完整形状和两入口20次输出/完成检查已通过。Vulkan首个FFN输出通过后，
消费者错误地将冷准备字节数当作ggml已热身资源；既有`managed_region_test.cc`已明确上游首次
逐槽分配辅助buffer。仅修消费者：保留冷值，首轮完整槽位预热后记录实际值，第二轮及所有后续
计时前后必须保持该值；全部成本仍记录，不删除资源检查、不改产品。CPU不因该修正重跑。

Apple首轮完整MLX片段输出检查失败。复核原普通Apple消费者的Qwen门为`0.008*abs(expected)+3e-8`，
新共用消费者误用了CUDA的`+1e-8`；纠正为原Apple门，并补失败元素/实际值日志。完整MLX区域
仍须实际通过，不能以原先串联小Graph的通过代签融合后结果；产品不变，原失败保留。
