# C3 原精度边界与公共头验证

固定候选 `2d1d787396f1f2798fe14c7ff3cece9bbebbbbda`。
Windows/Linux 普通 CPU Direct/Graph 新增格式与精度边界批次通过；两个平台的八个公共
Compute/Graph CPU header suite 通过。实际命令与原始日志位于平台子目录。

- MX scale 在 F16 中保持原下溢、最小次正规数及溢出；BF16/F32 覆盖最小 E8M0 scale。
- AWQ 的 F16、CT 的 BF16 scale 乘法先在原 dtype 舍入，再转显式 F32 输出。
  同批 cold-result MatMul 与 Graph constant 消费通过，没有改数学、扩大容差或插入热解码。
- 复用上一批两平台 source/static/shared、SDK 搬移与资源证据；本批产品源码没有变化。

C3 基础格式接线与有限资源/交付已取得资格；全部制品角色/shape 的最终矩阵随 C4/C5 归并。
本目录不签完整模型、C4 或整个 goal 完成，不解除旧 GPU 状态。
