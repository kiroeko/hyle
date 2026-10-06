# C3 Windows 混合格式与 cold constant 消费验证

固定候选 `ba4e430f2717edd7904e10f1ec7a4ccf6e1593a8`。
本目录保存实际命令、原始日志、factor JUnit 与统一 LOC；不签整个 C3 或 goal 完成。

- 11 个新增 GGUF 格式的普通 Direct/Graph MatMul、GroupedMatMul、GetRows 复验通过。
- NF4/nested NF4 默认 BF16、AWQ 默认 F16、CT 默认 BF16，以及显式 F32 解码通过。
  解码结果作为普通 Direct MatMul 输入或 Graph borrowed constant 消费，改变 activation 后重放通过。
  解码是显式 cold 操作，没有把全权重解包放入重复 MatMul 的热路径。
- MLX affine4/mxfp4/mxfp8 的 linear 与选中专家在 F32/BF16 下通过，重复专家、改变 activation、
  越界索引写保护通过；affine BF16 embedding 只选择所需行，重复行通过。
- 编译并行 2、测试串行，复用 Windows producer，DLL 邻置。没有运行性能赢家比较。

数值测试使用原 packing/scale/计算次序的非零夹具；现有 MLX 夹具使用可精确表示的值，
不能替代低精度极值、故障/OOM、并发独立 slot、完整实际宽度与所有准入画像的最终资格。
后续 Linux factor/普通入口有限资格已通过，精确候选见 [Linux 收据](linux/summary.json)；
F16、真实分配失败、并发独立槽和额度回收，以及 Windows 普通源码消费者见
[Windows 后续收据](windows-followup/summary.json)。C3 整包的精度极值、完整画像、交付及
C4 新主干/MTP/DSpark 完整程序仍待验，不能由这些有限测试外推。

首批候选统一 29 路径 Compute+Graph 实现 61,471 行；后续可移植元数据候选为
**61,475 行 / 250 文件**，不含测试、vendor 或 benchmark，见 [LOC](windows-followup/loc.json)。
