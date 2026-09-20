# M2-B1 CUDA 成熟原版批量接入证据

当前卡：[M2-B1 CUDA](../../../docs/implementation/tasks/compute/llm-moe-m2-b1-cuda.md)。

- [source-audit.json](source-audit.json)：固定 ggml pin、原版承载与有限 C1–C4 域、资源/capacity
  和 slice 对齐审计；不包含冻结自研算法的研究或比较。
- [development-windows.json](development-windows.json)：Windows SM89 首批开发检查点，
  8/8 CTest、42次 GoogleTest 执行和实现 hashes。原始配置、失败、修正、测试、selector 及
  LOC 保存在同目录 `development-windows-raw.zip`。
- [softmax-windows.json](softmax-windows.json)：第二批原版 Softmax 任意轴、浮点mask、精确原位及
  Graph 多slot发布/重放；8/8 CTest、47次 GoogleTest 执行。原始失败及修正保存于
  `softmax-windows-raw.zip`；旧开发 archive 不改写。
- [cccl-qualification.json](cccl-qualification.json)：原版CCCL数值资格，不代签产品覆盖。
- [collectives-windows.json](collectives-windows.json)：原版CCCL Scan/TopK的Provider、普通Direct/Graph、
  workspace/失败/生命周期开发检查点；Windows10/10 CTest和100次GoogleTest执行，原始记录在
  `collectives-windows-raw.zip`。其它平台、交付、插桩及P1仍未运行。

本目录不签整卡完成或完整模型覆盖。未绑定域及其它适用平台、交付、插桩、不可变候选性能仍待验；
完整需求覆盖维持0/256。新程序只接原版 ggml CUDA/CCCL，所有旧自研数学保持冻结。
