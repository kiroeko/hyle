# CPU 新官方低精度完整片段 P1

当前为固定消费者输入，尚无 P1 PASS。产品及安装 SDK 保持
`3c22b7b9f230abc86e749225174d86b59c79881e`，不重编产品。

两个代表消费者为 GLM 5.3 Flash 官方 FP8 完整 MoE 和 DeepSeek V4 Flash 官方
FP4 routed / FP8 shared 完整 MoE，保留真实 N/K/topK、A8 与 BF16/F32 边界。
每个计时调用执行全部变化输入一次，分别两个和三个片段；比较相同数学的普通 managed
Graph / Direct 完整周期。统计、正反执行/分配顺序、A/A 和线程政策复用现有仪器。

[消费者来源](consumer/consumer.json)与[输入身份](consumer/inputs.json)绑定普通接口、
源文件、已验搬迁 SDK 和 CPU-only LibTorch 链接/runtime 文件。无模型加载、整模型吞吐
或自有算子与第三方库胜负声明；Pro 的正确性结果不冒充 Pro 的独立性能验收。

预算为一次八进程 A/A，通过后四顺序各 n=15、5%非劣门，无样本扩展。
代码使用既有 `measure` 和公共仪器；仅增加消费者与其固定输入，不重写统计核心。
