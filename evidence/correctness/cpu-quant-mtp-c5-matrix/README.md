# C5 实际角色与格式的普通 CPU 批次

固定候选 `46a55325a465a94da3d79d2e5e92b0cb6b2cfb62`。
Windows/Linux 普通 Direct、Graph 各594项通过，合计2,376项：
[Windows](windows/summary.json)、[Linux](linux/summary.json) 保存原始命令、JUnit与准确范围。

- 559类真实GGUF权重/查表合同的实际K/N、非expert head groups、词表跨度及全部head行。
  21种压缩编码保留原payload和activation codec/vec-dot参考；plain F32/BF16/F16及I32端口分开。
  head驻留分片，expert驻留容量按计划缩小；不据此签整模型权重驻留或完整实权重推理。
- 104个精确GGUF制品身份各执行一个真实的最小projection，原软件/模型身份进入prepare/compile。
  2,413个制品到合同的关联是需求映射，不能称为2,413次完整混合模型运行。
- 34类真实非GGUFpacking/geometry：nested NF4、AWQ、CT与MLX混合三格式。
  原BF16/F16输出精度、cold decode→普通MatMul、改变activation后重放；MLX原BF16
  linear/选中group/embedding与原K-group次序。不会每轮展开完整专家库。
- 全部本批indexed消费检查越界、未修改发布输出、failed slot退役及新slot恢复。
  无新产品数学核或Provider升级；唯一产品修复为CPU-only新core的大表Gather总字节准入。

首次批次两处失败的原因分别为测试误加I32→F32 Cast，以及HYLE旧2GB总字节限制；
故障注入曾因测试只写4字节、违反Buffer完整视图写入合同而失败。
这些事实不覆盖历史失败或改变数值门；本目录通过收据对应修正后的不可变候选。

**尚未完成**：完整required-region/30类非线性与卷积参数对账、受产品修复影响的更新SDK、
最终全域适用检查、成本/文档/状态/红队、LOC和清理。本批不签C5或goal完成。
