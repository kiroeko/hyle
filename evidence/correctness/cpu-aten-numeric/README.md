# CPU ATen 数值程序：原版准入证据

[当前卡](../../../docs/implementation/tasks/compute/cpu-aten-numeric-admission.md)，scope
`ec466a379cd8d0467f99fa57fbe7889191bed013`，baseline `7c263f8631902b01b20b06007fc59db4fd6a2506`。
目前仅原版程序diagnostic PASS，产品correctness/普通入口/资源/三交付均UNPROVEN。

## 实际原程序

已锁LibTorch2.8.0 CPU，Windows native；原DLL与源码/命令/日志hash见original-manifest.json。
最终原程序 **48,780断言** 通过，不是算子/模型数量，不代签HYLE普通入口或性能WIN。
三dtype（F32/F16/BF16）、1625个特殊值/−100..100尾部和−10..10稠密格；三activation、packed/
separate gate及大value/Inf放大、原Softplus，number Min/Max和Reduce的NaN与正负零，odd-byte
packing原程序。gate的activation输出明确F32，乘法F32，最终一次原低精度舍入。

原SiLU/GELU/GELU-tanh在−Inf输出NaN。SiLU原Double kernel后原where修复−Inf，再显式F32。
直接GELU及1+erf/tanh参考在负尾部有消减；早期日志−8.36250019073显示Double候选与消减参考
不一致，不能把消减后的0当正确oracle。最终erf-GELU借原erfc Tensor程序，tanh-GELU借原
Double sigmoid/pow/add/mul等价公式；独立标量参考使用稳定erfc/branch exp，不放宽误差门。
这些都是原成熟Tensor kernel组合，没有HYLE payload数学循环、近似多项式或SIMD改动；
原Double转换、多节点temporary与F32中间materialization成本必须包含，不拿单个raw gelu计时比较。

Min/Max借原fmin/fmax，单侧NaN取另一数；两零用原signbit/where修复IEEE number零号。
Reduce借原sum/mean/amin/amax，Min/Max传播NaN；零结果按原signbit/any/all/where修复，
没有自研reduction。空identity、完整axes/rank/资源等仍须产品测试，未凭这个probe签资格。
odd-byte numeric输入先原Byte copy到原对齐typed Tensor，原位view不截断公共byte stride。

## 固定裁决

MN1–4 BORROW/FIX：相同程序接到existing独立ATen factory、冷private native packing槽，
默认先原oneDNN/ggml再ATen；保持managed/private-thread/unknown成本属性。唯一数学不由adapter拒绝
推断；自研优化/性能淘汰冻结。已有GGML Sigmoid/SiLU用exp(−x)的有限尾部另有源码疑点，需后续
具名完整consumer反例，不在本卡内越权改已关闭的ggml owner，最终CPU收口前必须完成该核对。
目前没有产品完成声明或GPU/math/performance新资格。
