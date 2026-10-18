# C5 原低精度卷积与raw text的P0

候选 `c740abc74a5a62ffacf60b26798228ca484317df`，新增产品只有现有Torch CPU program的
窄F16/BF16卷积接缝，追加kind/implementation ID9；原ID1–8及ggml F32程序未改变。

- prepare验证原公开推断结果，原input/output字节和F32 widening temporary int64容量；
  ndarray layout、shape和dtype在既有prepared Program中固定，submit不匹配模型、查硬件或重新选路。
- 原ATen permute/contiguous、输入/权重.to(Float)、groups=C的conv1d、一次回原output storage。
  无HYLE卷积数值循环。布局/转换复杂度O(B*W*C+C*K)，原卷积O(B*T*C*K)，真实热分配与结果copy保留。
- managed、host阻塞、可能私有线程、unknown私有内存、capture不兼容沿原Runtime声明；
  不把private upper bound写0，不承诺prepared-only或低dtypezero-copy。
- 输出只在原库程序完成后copy；真实alloc失败转ResourceExhausted，失败invocation退役，
  新slot恢复/独立并发/metadata回收已在两系统原factor实际测试。
- 本dtype的既有ggml carrier拒绝native低精度conv，没有第二个已资格同合同提案；
  使用唯一原ATen承载，无新winner/P1、自研竞争或GPU重签。

raw text保持原storage。Norm显式F32累加后回原storage；Qwen1+weight与普通gamma分开。
全vocab/head真实运行所需权重与native调用均计入正确性旅程；周期行只减少测试构造/编码和独立oracle
重复劳动，不降低K/N、payload长度、模型几何或原库计算。未将日志秒数作为性能结论。

旧F1/F2/C4产品资源/交付证据只在未变程序范围复用；新conv安装SDK必须重装并取得自己的普通入口资格。
