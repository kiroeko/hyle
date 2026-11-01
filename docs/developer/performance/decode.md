# CUDA 推理性能

更新：2026-11-02。**M1-5 尚未完成；以下为已提交基线的数据。**

RTX 4070 12GB，Qwen3.5-2B BF16，greedy、纯 CUDA、默认参数。真实文本由原生 tokenizer 生成；双方 capacity 与参考实际 target/draft `n_ctx` 均为 32768。参考按默认流程预热，batch 2048、ubatch 512。每臂 3 个独立进程、同窗口交错，每进程生成 16 token。

单元格为 **Hyle / llama.cpp** 三进程中位数，每行采用该场景最近的有效同窗结果，不跨窗口推算差距。prefill 从请求开始算到首 token，不含模型加载及 ready 前准备；decode 为之后 15 个 token 的完整区间，包含首用开销；退出墙时包含进程启动到退出。

| 模式 | 输入 token | prefill ms | decode tok/s | ready s | 退出墙时 s |
|---|---:|---:|---:|---:|---:|
| 普通 | 1024 | 103.25 / 112.69 | 99.43 / 97.19 | 1.779 / 1.769 | 2.563 / 2.575 |
| 普通 | 4096 | 400.19 / 427.49 | 99.65 / 92.95 | 1.780 / 1.768 | 2.881 / 2.882 |
| 普通 | 16384 | 1884.39 / 1741.64 | 85.62 / 84.51 | 1.825 / 1.825 | 4.410 / 4.825 |
| 普通 | 32736 | 3977.78 / 3688.26 | 87.66 / 85.08 | 1.790 / 1.795 | 6.476 / 6.202 |
| MTP | 1024 | 112.73 / 122.54 | 120.79 / 121.73 | 2.947 / 1.844 | 3.775 / 2.641 |
| MTP | 4096 | 443.22 / 474.68 | 118.50 / 115.63 | 2.967 / 1.848 | 4.102 / 3.006 |
| MTP | 16384 | 1828.25 / 1944.95 | 110.30 / 106.78 | 2.973 / 1.823 | 5.518 / 4.497 |
| MTP | 32736 | 4123.03 / 4303.84 | 102.84 / 98.98 | 2.961 / 1.853 | 7.818 / 6.901 |

16K 独立同窗复测，旧版 `5b91c0cd7`／已提交版／参考的普通 decode 分别为 5.87／85.62／84.51 tok/s，MTP 为 13.60／110.30／106.78 tok/s；每臂3个进程，输出 ID 与旧版一致。普通新版范围为 73.20–88.88 tok/s，有较大波动，但新版和旧版范围不重叠。

普通 32K prefill 为参考吞吐的 92.7%，未达到 97% 验收。最新窗口的普通16K及MTP 1K／32K decode 两组范围均重叠，不能将中位数差距解释成稳定领先或退化。各场景双方输出 ID 一致。

主机 decode 准备中位数从1K到32K：普通 0.70→2.08ms，MTP 1.50→5.54ms，均超过1.3倍。末段 prefill 分块准备分别为普通 2.81→3.44ms、MTP 2.88→5.58ms。采样是同一进程内的准备阶段，不能把重叠步长相加当成完整 decode 耗时。

已检查的四档普通及 MTP teacher-forcing argmax 均匹配原版；自然 MTP 的16个 ID、已观察接力行及末步 argmax 匹配。同一数学路径下，原版与当前版第13、15、16项的已初始化状态载荷及所记录边界字段逐字节一致，包含 draft。Round / Single 的已检查 teacher-forcing argmax 一致；分别从两种状态继续生成16 token 尚未验证，不能宣称本卡完整正确性通过。

C2048 短负载使用 8 个不同 prompt、每轮 64 token。原版 / 当前版合并 decode：普通 101.57 / 102.17，MTP 125.57 / 125.99 tok/s；逐轮均达到原版的 98%，生成 ID 一致。

独立显存采样的 C32768、32736 输入场景：普通原版 / 当前 / 参考为 4593.08 / 4593.08 / 4297.16 MiB，MTP 为 5025.08 / 5025.08 / 4639.29 MiB。每臂 3 个独立进程，当前版相对原版的采样峰增量均为 0 MiB，通过 64 MiB 限制；采样峰是下界。

测量版本：Hyle `cf56700c2`；参考为冻结 llama.cpp `d9e03f1074dbd2979126d91dce1b5d304ec8394e` 的自建 consumer，非上游最新版标准 llama-bench。

复现长负载：`python .scratch/inference-long-context-20261101/acceptance.py timing --output <新目录> --hyle-version <实际版本> --lengths 1024 4096 16384 32736 --modes both`。

原始数据：[普通1K／32K](../../../.scratch/inference-long-context-20261101/relation-window1/m15-summary.txt)、[16K](../../../.scratch/inference-long-context-20261101/precommit-head-16k-window1/m15-summary.txt)、[MTP 1K／32K](../../../.scratch/inference-long-context-20261101/choice-only-window1/m15-summary.txt)、[4K](../../../.scratch/inference-long-context-20261101/full-window-poolbulk-current1/m15-summary.txt)、[短负载](../../../.scratch/inference-long-context-20261101/short-window-lazypool-bulk1/summary.txt)、[独立显存](../../../.scratch/inference-long-context-20261101/memory-lazypool-bulk1/m15-memory-summary.txt)、[输入与 tokenizer](../../../.scratch/inference-long-context-20261101/workload/sources.txt)。
