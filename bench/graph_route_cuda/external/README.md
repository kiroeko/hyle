# CUDA D5 外部性能消费者

本工程消费已安装的同候选 HYLE CUDA 静态 SDK，复用已资格化 D5 模型 recipe 和原 oracle。
不配置或重建 HYLE 产品，也不实现模型数学。

在开发者编译环境中，将 `CMAKE_PREFIX_PATH` 指向 SDK；`D5_GGML_SOURCE` 和
`D5_GTEST_SOURCE` 指向 dependency lock 对应的两个本地源码目录。只构建
`hyle_graph_route_cuda_bench`，并行上限为 2。SDK 的公开 runtime staging 自动把 DLL 放到 EXE 旁。

运行前设置 `HYLE_D5_QWEN35_FIXTURE` 为已准备的本地 Qwen3.5-2B fixture 目录；缺失直接失败，
不会跳过。`ctest -R ^d5_cuda_fragment_verify$ --output-on-failure` 只执行完整片段正确性。

消费者固定 Qwen BF16 FFN 和 DeepSeek 官方 FP4/FP8 MoE，两个入口共 20 次完整运行检查。
Graph 使用普通默认规划；Direct 使用普通 prepared operation，在同一 queue 上提交整个片段后
drain 全部 completion。两侧计时区都不重新建描述、选 Provider 或分配消费者容器。

`HYLE_D5_DIAGNOSTIC=1` 仅允许开发诊断，**不能产生 P1 PASS**。统一仪器的正式采样还要求
本 artifact 已生成通过的 A/A 收据；没有该收据时消费者拒绝启动产品采样。
按 [A6 任务卡](../../../docs/implementation/tasks/compute/llm-moe-d5-cuda-fragment-performance.md)
绑定不可变候选、`inputs.json` 的 SDK/fixture/DLL 字节身份及 `consumer.json` 的消费边界。

沿用已有环境约定：`HYLE_BENCH_AA_ROUTE=selected|peer` 替换两侧为 Graph 或 Direct；
`HYLE_BENCH_RIVAL_FIRST=1` 交换执行顺序；`HYLE_BENCH_PEER_ALLOCATION_FIRST=1` 交换分配顺序。
`HYLE_PERF_CALLER_CPU=2` 使用 W1 已登记调用线程亲和。不得改系统电源计划或请求提权。
每轮输出实际 Provider、Graph region/workspace 与准备信息；准备时间包含进程首次初始化，
不能单独据此判定一侧冷启动性能更好。设备字节报告只统计 consumer 显式分配，Graph 内部
workspace 另列，Provider 私有池未计入时不能声称总显存峰值。
