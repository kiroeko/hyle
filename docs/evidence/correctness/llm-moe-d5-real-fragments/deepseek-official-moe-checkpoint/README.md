# F43：DeepSeek 官方 MoE 接合检查点

候选 `a4b70bc28f1b7f24d93201f926cde1eced5ed2dc`，Windows RTX4070 / SM89 /
CUDA Driver API13040。**完整MoE数值资格FAIL；group128原子及资源资格PASS。**
完整画像ledger保持144/176，不能将本检查点当成两模型CUDA已完成。

## 已实现和已验证

- 原版group128 E4M3量化还原已接入默认CUDA Direct/Graph；两种实际专家宽度7168/4096、
  M1/3/33及两轮变化输入精确通过。原group64/group32检查继续通过。
- 新格式的workspace、对齐/别名拒绝、元数据OOM回滚、双队列和Completion保活通过。
  四项独立头通过，13项模型输入工具通过。四个CTest factor中8项GTest通过、2项MoE测试失败，零SKIP。
- group128原生12案例包含全部有限BF16编码、舍入中点、极小值、尾组与guards，结果/scales精确。
  三份量化程序离线重建hash全部一致；没有新增HYLE数学核。
- 两模型各连接score/hash两种MoE，保留A8/group128、原packed FP4专家、pre-down路由权重、
  BF16投影输出、动态FP8共享专家和最终BF16输出；每个case三轮真实变化输入。

## 失败边界与有界后续

首轮暴露诊断输出为alias-only View，测试runner没有独立可读buffer。消费者增加普通Copy后
绑定/读回问题消失；保留初次失败日志。产品执行器未修改。

随后Direct/Graph一致暴露数值失败：按既有参考最大值归一化，部分最终输出最大误差达
**0.0712669641**，原门为**0.04**；没有调整容差、换输入或隐藏中间失败。
FP4 down之前的激活量化输入也出现离散编码差异。固定ggml源码的CUDA `mmvq.cu`
将MXFP4映射到 `vec_dot_mxfp4_q8_1`，因此官方E4M3边界之后还存在Q8_1激活转换。
这提供了具体原因方向，尚未单独证明它是全部误差的唯一来源。

同一官方revision的原版 `fp4_gemm_kernel` 明确以FP8激活×FP4权重执行：子块将FP4转为FP8，
使用原成熟FP8 GEMM，再应用group128/group32缩放。按一次具名候选预算，N4096/K2048的
原程序已在SM89编译并运行；M1/3/33×两组代码/scales的6案例与独立参考逐BF16精确相同，
输出guards完整。它**尚未接入HYLE普通入口**，动态专家ID、其他shape与成本亦未资格化。
下一步据此定界原版接线，不转向HYLE自写核，不循环尝试更宽误差门。

初次构建51.55秒，诊断Copy修正的增量构建9.42秒；两次测试各约2.6秒。其余selected gates、
完整attention/compressor/MTP、其他路线、三交付及P1仍待完成；没有启动性能实验。
统一实现 **59,904行／238文件**，不含测试和第三方程序。

[机器收据](receipt.json)记录两个候选、原始日志hash、资格范围和未完成项。
`roundtrip-native-probe.py --artifacts <原子产物目录>`复核新增格式；
`fp4-native-probe.py --artifacts <原版fp4_gemm_kernel.cubin目录>`复核上述一个原生签名。
它们只复核native资格，不能替代普通入口验收。
