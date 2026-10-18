# R4 CUDA：完整公共边界与原生调用的成本

产品候选755fb7c7，采集 checkout f01e9237（只有文档/状态/证据差异）。W1 RTX4070/SM89、
driver610.62、Toolkit13.3.73、MSVC19.50、Release/static；caller CPU2、process mask0x14、
AboveNormal、实际 CUDA/CRT DLL 邻置。自有 scratch 驱动调用既有统一采集/统计 API，未修改公共仪器。

**NATURAL_ROUTE / NOT_COMPARABLE**。两侧数学同为 Marlin Q4 gate/up → SGLang precise SwiGLU →
cuBLASLt down，使用相同精度、权重、两槽变化输入与冷配置；native 只有一个最终 event，缺少 HYLE
的 admission、owner 保活和失败槽退役服务。Direct HYLE 三段各有真实 Completion，最后逐一检查；
Graph 两侧都是真实 captured 完整程序。此表不能签同合同 NI、WIN/LOSS 或算法 SOTA。

最初的 A/A 启动因 scratch build 未登记 pinned-peer 路径在计时前拒绝；补齐 CMake cache 后，
六个 exe 的前后哈希完全相同，原拒绝日志和纠正命令保留。没有丢弃性能样本或改产品/benchmark。
各模式先取得同 build 的16进程 A/A receipt，通过完整 double oracle，再执行正反执行/分配顺序
四层、每层15个独立进程（每进程 M1/M16）。只有一次诊断、未扩样；raw/identity 保留 nonce、
artifact/DLL、机器策略与 GPU 状态。区间和逐层数据见 summary.json，下面只是各层中位数范围。

| 模式 / workload | 公共路径 μs | 原生路径 μs | 逐进程差值中位数 μs |
|---|---:|---:|---:|
| compute / decode-q4-ffn-m1 | 311.408–312.039 | 310.816–311.415 | 0.371–1.639 |
| compute / short-prefill-q4-ffn-m16 | 315.958–317.792 | 315.935–316.746 | -0.339–1.458 |
| graph / decode-q4-ffn-m1 | 307.300–308.675 | 308.031–308.786 | -0.376–0.080 |
| graph / short-prefill-q4-ffn-m16 | 312.264–313.435 | 312.429–313.216 | -1.481–0.851 |

## 成本解释及限度

源码账单：普通 Direct 在每个 operation 保留资源/绑定校验、独立受理/真实完成状态，native 将
三段置于一次 primary scope、一个 event。Graph 的 prepare_replay 用同一 sequence 录制片段，
完成事件在结束 capture 后处理；热路径是固定 cudaGraphLaunch 加整体受理、终态与保活处理。
没有用额外空操作对齐数值，也未将 Graph 与逐 kernel host launch 进行排名。

上述差值是整个边界的合计成本，**没有阶段插桩证明每项分别占多少**。不能仅凭差值判定某个
atomic、锁或 driver call 的责任。独立精确 reference 的同生命周期 NI 在 v3 P1 中裁决。

冷 pair_prepare_ms 只含两侧 Context/程序/槽准备，含适用 capture，不包含之前 DataSet 的权重
生成/packing，也不是单侧完整启动时间。raw 同时保留 Q4 resident bytes 与 native 每槽 workspace；
driver/cuBLAS opaque 内存仍 unknown，没有宣称已测完整峰值。稳态计时外做真实终态后析构，
析构时延没有单独计时。代码和这些边界均在计时前固定，未据结果调整公开 API。

[完整原始包](raw-bundle.zip)、[逐层汇总](summary.json)、[文件哈希](sha256.json)。原 build
`C:/hv4s2` 与 scratch 仍保留。这是当前两个 FFN 原型的工程诊断，不是生产模型吞吐。
