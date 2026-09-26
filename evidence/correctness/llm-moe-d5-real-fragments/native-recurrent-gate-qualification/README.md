# D5 原版 recurrent gate 的数值边界

## 裁决

**独立 gate API：REJECT。完整 raw-gate + GDN region：进入普通入口资格验证。**
原始探针整体 `verdict=FAIL` 原样保留；不能把其中通过的输出改写成整个探针 PASS。
用户选择按完整片段误差资格化，最终公共边界只发布 attended output 和 next state。
普通 `Unary::kSoftplus` 的精度与 Provider 准入没有改变。

预声明协议在 `bbcd3648` 冻结，结果和最终有限接线在 `2b66dd8f` 审定，先于产品候选。
当前施工卡：[MG1–MG5](../../../../docs/implementation/tasks/compute/llm-moe-d5-model-gate-qualification.md)。

## 输入与独立参考

- 固定 ggml CUDA 原版 Add/Softplus/Mul/Sigmoid/GatedDeltaNet/RMSNorm/SwiGLU；RTX 4070，SM89。
- 六组：本地 Qwen3.5-2B 官方参数，合成16头 GGUF，48/128头各一组 official/GGUF；D128、T1、B1。
  官方 Q/K 已按模型 grouped head 关系展开，GGUF 保留转换后的 cyclic head 对应。
- 本地 A_log 与 dt_bias 从已校验的 checkpoint fixture 读取；其它巨大模型使用真实形状和合成参数，
  不声称读取了它们的权重。Q/K 归一化，负 decay coefficient 为显式输入。
- 每组64次变化输入，native 与 F64 oracle 分别反馈各自的 F32 state；包含 softplus 负尾、正饱和、
  beta 两侧饱和。原始实现、固定输入、参考和编译命令见 [probe.cc](probe.cc)、[probe.py](probe.py)。
- 全部观察使用同一门：`abs_error <= 2e-6 + 4e-5 * abs(reference)`，逐元素判定且拒绝 NaN/Inf。
  没有运行后放宽门限、删掉失败输入或以平均误差代替最大误差。

## 实测

完整逐组计数见 [native-cuda.log](native-cuda.log)。下表为六组中的最大门限比，1为通过边界。

| 观察 | 最大门限比 | 结果 |
|---|---:|---|
| 内部 g | 1.00884305 | FAIL，不能单独发布此 gate 精度合同 |
| 内部 beta | 0.00143 | PASS |
| GDN attended output | 0.00335 | PASS |
| 完整反馈 state | 0.02940 | PASS |
| 后续 RMSNorm/输出 gate | 0.30907 | PASS |

本地真实16头参数的所有观察（包括 g）通过。独立 gate 的失败来自合成反偏置组，仍属于预声明输入，
因此原拟四输入、两 gate 输出的接口被否决。

完整 raw-gate + recurrent update 的边界对应固定成熟实现已有的模型入口，例如
[SGLang cutedsl_gdn.py](https://github.com/sgl-project/sglang/blob/06992c92ad4494fe9f9636e75be3e35040512447/python/sglang/kernels/ops/attention/cutedsl_gdn.py)。
这不是把普通 Softplus 换成低精度实现；完整片段仍需在四路线普通 Direct/Graph 中逐项通过同一输出门。

## 限制

这是有限形状与输入族的数值资格，不是整模型精度、任意权重或任意上下文长度证明。
数值探针不签普通入口、资源生命周期、三交付、P1 或整个 profile 的 `RUN_PASS`。
旧 CPU 预计算 gate 的 R05 收据仍不能单独证明完整动态 recurrent 路径。
