# CUDA 片段性能消费者

本工程消费已安装的 HYLE CUDA 静态 SDK，复用模型片段和独立 oracle；不重建产品或实现新模型数学。

在开发者编译环境中，`CMAKE_PREFIX_PATH` 指向 SDK，`D5_GGML_SOURCE` 与 `D5_GTEST_SOURCE` 指向
dependency lock 对应的本地源码。构建 `hyle_graph_route_cuda_bench`；SDK 的公开 runtime staging
把 DLL 放到 EXE 旁。运行前设置 `HYLE_D5_QWEN35_FIXTURE` 为本地 Qwen3.5-2B fixture 目录，缺失会失败。

`ctest -R ^d5_cuda_fragment_verify$ --output-on-failure` 在本消费者 build 目录中检查完整片段正确性。
直接运行构建出的 benchmark 可采集计时。

消费者固定 Qwen BF16 FFN 和 DeepSeek 官方 FP4/FP8 MoE，两入口共 20 次完整运行检查。
Graph 使用默认规划；Direct 使用 prepared operation，在同一 queue 提交整个片段后 drain 全部
Completion。计时区内不重建描述、选 Provider 或分配消费者容器。

- `HYLE_BENCH_VERIFY_ONLY=1` 只做正确性检查。
- `HYLE_BENCH_AA_ROUTE=selected|peer` 将两侧替换为 Graph 或 Direct，检查相同路径的测量偏差。
- `HYLE_BENCH_RIVAL_FIRST=1` 交换执行顺序，`HYLE_BENCH_PEER_ALLOCATION_FIRST=1` 交换分配顺序。
- `HYLE_PERF_CALLER_CPU` 可设置调用线程亲和；按本次机器拓扑选择并核对实际落点，不假设 CPU 2
  在任意机器上等价。

保留完整输出并按[性能比较方法](../../../docs/developer/performance/README.md)分析重复运行、A/A
和顺序差异。报告包含实际 Provider、Graph region/workspace 与准备信息。准备时间包含进程首次
初始化，不能单据此判定冷启动优劣。设备字节只统计显式分配，Graph workspace 另列；Provider 私有
池未计入时不能声称总显存峰值。

原始实验身份和历史结果见[实现记录](../../../docs/implementation/tasks/compute/llm-moe-d5-cuda-fragment-performance.md)。
