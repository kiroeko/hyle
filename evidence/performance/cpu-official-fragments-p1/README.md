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
代码使用既有公共仪器；仅增加消费者与其固定输入，不重写统计核心。

## 一次重型消费者包装纠正

首次 A/A 只完成一个进程，未完成 A/A裁决且正式产品样本为零。
[中止记录](initial-heavy-consumer/interruption.json)及原日志保留。默认 nanobench compare
生成每臂12轮；DeepSeek 一周期约3.4秒，使单进程约两分钟，四层n=15会重复执行超过两小时。
原表 op/s 只保留两位小数，0.29/0.30还会引入约3%的人为比率量化。

仅纠正重型消费者：仍用 nanobench 原计时，`epochs(1)/epochIterations(1)`，每臂一完整
变化输入周期；原 oracle 每进程前后各一完整周期，普通 correctness 已检查多次 replay。
从原 `Result::elapsed` 输出17位精度，继续使用相同公共表解析器。无需进程内的第二套CI；
结论仍由四顺序各15个独立进程、相同公共统计/控制/5%门给出。未改变产品或数学，也未
签发首个不完整 A/A 的 PASS。重新冻结消费者后只允许这一批 A/A/P1，不继续换姿势重测。
