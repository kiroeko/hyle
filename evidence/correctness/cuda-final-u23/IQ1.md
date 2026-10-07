# 原 narrow IQ1 CUDA 程序接线

产品 `49f10948d`，最后测试接缝／两入口用例候选 `a14e3d90c8779dc6e55468ed40b6e9e1f9883119`。
固定 Unsloth `46cbf0e9` 三种IQ1_XS/XXS/XXXS，原数学/表/分发保持，ABI仅补sizeof traits及include路径。
11个有界源补丁、3个原template编入同一ggml核；已锁CPU源码的5个CPU实现文件hash未变。
CPU-only0.26与其它单GPU路线的source选择保持，不因此继承新的性能资格。

## 实际验证

- Windows SM89 / Toolkit13.3 / MSVC19.50，原全部CUDA源程序编译完成；重型j2串行批次。
  两次测试编译接缝失败（显式Operation overload、原ABI匿名union warning）保留，增量纠正未重跑全部CUDA编译。
- 三种格式原MMVQ/MMQ小形状和两轮activation重放通过；prepare的矩阵/分组专家绑定通过。
  原stock MatMul/grouped/Graph回归通过。这是私有carrier与原schema接缝，不签完整模型。
- 当前共同目录的8个精确expert K/N合同通过普通CUDA Direct及Graph，原制品repository/revision/profile保持；
  只减驻留expert/token容量，原inner widths不改，非零原block字节、重放与重复IDs保留。
  两个测试各明确迭代8项，无SKIP；source/native三交付与Linux仍未验。

## 原失败的原因与修正边界

第一版参考把原dequant weights直接乘F32，忽略原CUDA的Q8激活codec。原IQ1 dot为
`d1q * (ds.x * sumi + ds.y * delta)`；scale和原始32值部分和分别round到half。
对于常量0.25，scale重建值为0.2499847412，而partial sum仍为8；部分和校正使原解码点积为0的行
得到约-1.33514e-5（d=1/256），实测完全对应源码项。不能把两种不同执行合同当错误接线或性能胜负。

测试参考按原公开源码codec修正，未修改产品数学、原输入输出精度或扩大门；原失败保留。
8个模型shape测试另外使用127/256与127/128激活，使两种原codec的scale都精确可表示，
保留共同8e-5数值门。它证明该原格式/形状/接线边界，不外推任意payload/full region的误差资格；
完整模型/状态/困难payload仍由共同完整consumer继续验收。

完整记录见 [IQ1归档](iq1-windows.zip)，其中Graph无匹配用例的那次filtered CTest成功不算Graph IQ1资格；
真正Graph IQ1证据来自后续8项普通Graph用例。没有P1/WIN或三交付PASS。
