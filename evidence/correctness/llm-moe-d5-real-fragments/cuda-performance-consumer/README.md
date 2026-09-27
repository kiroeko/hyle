# CUDA A6 消费者正确性

**verify-only PASS；性能和 A/A 未运行。** 使用 A5 已通过的 `9273bc71` 搬迁静态 SDK，
benchmark 源码/fixture/binary 的 SHA256 见 [收据](receipt.json)。仅编译外部消费者与 host oracle，
未重编 CUDA 产品、未新增产品数学。候选准备见
[任务卡](../../../../docs/implementation/tasks/compute/llm-moe-d5-cuda-fragment-performance.md)。

两个入口均使用普通 API、固定相同 recipe 与原容差，先完整提交再等全部完成。
两轮遍历全部输入，共 20 次完整片段运行；最终输出及返回后不 in-flight 检查通过，
CTest 1/1、零跳过，见 [原始运行](raw/verify.log)。

| 固定消费者 | 变化输入 | Graph 实际计划 | Graph workspace |
|---|---:|---|---:|
| Qwen3.5-2B 官方真实 BF16 FFN，2048→6144→2048 | 2 | 5 steps、0 regions | 40,960 bytes |
| DeepSeek V4 Flash 官方 FP4 experts/FP8 shared MoE，H4096、N2048、top-6 | 3 | 40 steps、0 regions | 415,360 bytes |

DeepSeek 保留真实格式、模型内宽与原动态量化边界；驻留七个专家，payload 为既有 header-bound
非零 fixture，不能称下载了该大模型权重。两侧仅发布最终结果，不为调试中间输出阻止正常融合。
这里实际没有形成多节点 ggml region；不把 Graph 入口或步骤减少误写成成熟库融合。

初次编译的 backend include/BF16 host decode 问题与 workload region 身份初始化错误仅在新
consumer 中纠正；保留 [原编译失败](raw/first-compile.log)、[原验证失败](raw/first-verify.log)。
随后的 [编译](raw/build.log) 与 [CTest](raw/ctest.log) 通过。未改变产品或 oracle。

selector 选择 docs/closure，见 [选择计划](raw/selection-plan.json)；本卡另预声明外部 consumer
编译/verify-only，已经实际执行。已通过的 A5 交付和 459 项产品测试未重复。
保留唯一 `C:/hv/d5a6` 外部构建目录供后续不可变 P1 使用，不增建各入口 producer。

最终消费者调用 `nanobench::compare()`；当前采样必须显式启用 diagnostic 模式，不能签 P1。
准备毫秒值包含首次进程初始化，设备内存数不含 Provider 私有池，不据此作冷启动或总显存结论。
性能/仪器资格完成前，A6 仍未通过，完整画像 ledger 仍为 152/176。
