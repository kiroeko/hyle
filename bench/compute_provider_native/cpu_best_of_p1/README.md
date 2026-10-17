# CPU Provider 比较消费者

这个开发消费者比较独立 Native/oneDNN/ggml/ATen factory；普通用户仍选择 CPU 路线。
消费者显式链接 SDK 导出的 ggml implementation target 以取得 factory feature definition；普通
CPU 路线通过私有依赖包含实现，不要求用户选择 Provider。本工程不添加 operator 数学。

使用包含 compute/graph 的 SDK 配置本目录，构建 `hyle_cpu_best_of_bench`。直接运行：

- `--probe`：逐项 prepare、记录实际拒绝与资源信息，运行接受的程序并做数值比较。
- `--pair <case> <candidate> <peer>`：用现有 nanobench adapter 测量选定 pair。
  具体 case 和 family 名称以源码及 probe 输出为准。

拒绝不等于性能落后；只有一个实际承载方时检查正确性和资源，不捏造速度比。
同一 pair 保留双执行顺序、双分配顺序和适用 A/A，保存完整原始输出并使用
[统计与公平比较方法](../../../docs/developer/performance/README.md)分析。按实际调用控制 OMP/MKL
线程数与调用线程落点，并记录配置。

normal/holdout 使用 inner widths 2048/4096、token/small prefill 1/32、F32 projection
K16384/N4 或 24、Qwen BF16 projection K2048/N6144。dense 与物理转置权重分开请求；
转换、copy、临时空间、检查和真实完成均在相应测量边界内。

若本地有 `qualified_pairs.h`，还会构建 `hyle_cpu_best_of_batch`；它使用固定 pair 表，
`HYLE_CPU_BESTOF_PAIR_FILE` 可选择已有条目，不是在线 Provider 搜索。旧 sealed adapter 是特定
Windows artifact 的传输方式，不能把其启动成本当作同进程调用成本。

原实验及实际 revision 见[实现历史](../../../docs/implementation/tasks/compute/cpu-final-f4-best-of.md)。
