# CUDA 推理性能

更新：2026-11-02。**M1-5 尚未完成；以下为已提交版本的数据。**

RTX 4070 12GB，Qwen3.5-2B BF16，greedy、纯 CUDA、默认参数。真实文本由原生 tokenizer 生成；双方 capacity 与参考实际 target/draft `n_ctx` 均为 32768。参考按默认流程预热，batch 2048、ubatch 512。每臂 3 个独立进程、同窗口交错，每进程生成 16 token。

单元格为 **Hyle / llama.cpp** 三进程中位数，每行采用该场景最近的有效同窗结果，不跨窗口推算差距。prefill 从请求开始算到首 token，不含模型加载及 ready 前准备；decode 为之后 15 个 token 的完整区间，包含首用开销；退出墙时包含进程启动到退出。

| 模式 | 输入 token | prefill ms | decode tok/s | ready s | 退出墙时 s |
|---|---:|---:|---:|---:|---:|
| 普通 | 1024 | 102.75 / 107.10 | 100.41 / 93.94 | 1.811 / 1.778 | 2.601 / 2.589 |
| 普通 | 4096 | 412.91 / 397.80 | 100.70 / 96.85 | 1.799 / 1.786 | 2.908 / 2.889 |
| 普通 | 16384 | 1782.40 / 1710.01 | 94.21 / 88.87 | 1.802 / 1.795 | 4.278 / 4.206 |
| 普通 | 32736 | 3996.75 / 3717.79 | 85.85 / 86.60 | 1.815 / 1.779 | 6.532 / 6.231 |
| MTP | 1024 | 113.32 / 125.97 | 121.05 / 117.64 | 2.969 / 1.852 | 3.804 / 2.665 |
| MTP | 4096 | 440.22 / 480.80 | 117.75 / 117.44 | 3.014 / 1.864 | 4.178 / 3.021 |
| MTP | 16384 | 1831.93 / 1937.35 | 111.09 / 110.62 | 2.991 / 1.860 | 5.527 / 4.491 |
| MTP | 32736 | 4111.78 / 4291.09 | 101.41 / 99.69 | 2.972 / 1.880 | 7.826 / 6.868 |

普通 32K prefill 为参考吞吐的 93.0%，未达到 97% 验收。32K 的普通和 MTP decode，当前版与基线的三进程范围重叠；不能将中位数差距解释成稳定退化或提升。四档、两种模式的 16 个输出 ID 与基线及参考一致。

减少重复排序后，同一窗口中 32K MTP decode 的主机准备时间从基线 5.84ms 降到 4.42ms；完整 decode 为 101.71 / 101.41 tok/s（基线 / 当前），范围重叠，尚未测出吞吐提升。同一窗口从 1K 到 32K，当前版的主机准备中位数为普通 0.70→1.81ms、MTP 1.52→4.42ms，均超过 1.3 倍。逐步准备时间可能与执行重叠，不能相加当作完整 decode 耗时。

C2048 短负载使用 8 个不同 prompt、每轮 64 token。基线 / 当前版合并 decode：普通 102.41 / 102.20，MTP 125.91 / 126.01 tok/s；逐轮均达到基线的 98%，512 个输出 ID 一致。普通 4K 单轮生成 128 token 的诊断中，当前 / 基线 / 参考分别为 102.26 / 102.18 / 99.09 tok/s；当前与基线 ID 一致，参考从第 108 个 token 开始分叉，原因未验证。

基线 `cf56700c2` 已检查四档普通及 MTP teacher-forcing argmax；同一路径的部分状态载荷和边界字段逐字节一致。当前版通过原有页读权限、候选选择测试和实际 MTP 输出对照；Round / Single 分别继续生成 16 token 尚未验证，不能宣称本卡完整正确性通过。

独立显存采样的 C32768、32736 输入场景：普通基线 / 当前 / 参考为 4593.08 / 4593.08 / 4297.16 MiB，MTP 为 5025.08 / 5025.08 / 4639.29 MiB。每臂 3 个独立进程，当前版相对基线的采样峰增量均为 0 MiB，通过 64 MiB 限制；采样峰是下界。

测量版本：Hyle `9e7fb61ad`；参考为冻结 llama.cpp `d9e03f1074dbd2979126d91dce1b5d304ec8394e` 的自建 consumer，非上游最新版标准 llama-bench。

复现长负载：`python .scratch/inference-long-context-20261101/acceptance.py timing --output <新目录> --hyle-version <实际版本> --lengths 1024 4096 16384 32736 --modes both`。

原始数据：[普通1K／4K／16K](../../../.scratch/inference-long-context-20261101/reads-merge-ordinary-repeat1/m15-summary.txt)、[普通32K及MTP 32K](../../../.scratch/inference-long-context-20261101/reads-merge-window1/m15-summary.txt)、[MTP 1K](../../../.scratch/inference-long-context-20261101/reads-merge-1k16-repeat1/m15-summary.txt)、[MTP 4K／16K](../../../.scratch/inference-long-context-20261101/reads-merge-4k16k-window1/m15-summary.txt)、[普通4K 128输出](../../../.scratch/inference-long-context-20261101/reads-merge-4k128-window2/summary.txt)、[短负载](../../../.scratch/inference-long-context-20261101/reads-merge-short-window1/summary.txt)、[独立显存](../../../.scratch/inference-long-context-20261101/reads-merge-memory-window1/m15-memory-summary.txt)、[输入与 tokenizer](../../../.scratch/inference-long-context-20261101/workload/sources.txt)。
