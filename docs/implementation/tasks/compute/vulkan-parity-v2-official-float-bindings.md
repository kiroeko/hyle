# Vulkan V2：官方 FP4／FP8 与 QAT 程序

<!-- MODULE REVIEW: SCOPED -->

baseline `890cf9f490aae747747520fa81df1f135668f6b9`。父轮
[Vulkan parity](vulkan-cpu-cuda-apple-parity-closure.md)；owner IREE Vulkan 与 Compute/
Graph Vulkan。上批artifact/MLX微型Windows资格继承，不签full catalogue/三交付/P1。
本卡SCOPED阶段只做有界原源码/接口审计，不授权产品改动。

## 需求与原来源

共同目录中的当前Pro0813/Flash0731 main/MTP/DSpark和其它已定模型需要：
`Float8WeightDecode`、`Float8BlockLinear`、`GroupedFloat8Linear`、
`GroupedFloat4Linear`、`BlockQuantizationRoundtrip`；现有encoded block128 MatMul
还须核对完整N尾部与真正dot-before-scale原程序，不能用旧微型样本代签。

原来源为已登记DeepSeek weight_dequant/act_quant/fp4_gemm、SGLang per-group quant/
block GEMM、ds4 `9139e2ae58a41503968a500f36f75895c1ba63fc` 原roundtrip；原 source/
许可在`third_party/deepseek_fp8`、`sglang_activation`、`ds4_quantization`。
CPU/CUDA/Apple现有原序列只作语义审计，不重开其CLOSED owner。

主纵切：BF16/F32 activation动态量化→原FP8/FP4 selected experts→正确BF16/F32输出，
含共享A=1与分立A=S、duplicate IDs和main/sidecar消费者。正交纵切：explicit weight
decode与BF16 QAT roundtrip的精确舍入、finite检查和普通prepared资源/失败语义。

## 有限待审 FIX

| ID | 具名缺口／拟接入 | 要保留的合同 |
|---|---|---|
| F1 | 三种QAT roundtrip原标准tensor程序 | FP4 group32、FP8 group64/128，原floor/ceil-power2/saturated RNE、精确BF16边界 |
| F2 | block128 FP8 weight decode与动态linear | 原bytes、ceil(N/128)尺度、显式cold typed结果；normal/power2和floor1e-10/1e-4 |
| F3 | 原grouped FP8/FP4完整片段 | A=1/S、selected-only一K group、dot→A scale→W scale→顺序F32累计、最终存储dtype |
| F4 | 普通Direct/Graph和named rank3编码承载 | canonical descriptor/索引/feature/grid/workspace；GPU finite/checked失败不发布 |
| F5 | 私有VM/native inputs/scalar floor/slots、完成及offline交付 | 沿原IREE/Vulkan生命周期、常量cold copy、mutable复制和全部成本；无runtime compiler或额外用户标记 |
| F6 | 既有encoded block128 MatMul的实际N尾部/算法接线 | 只接同原分组程序，不用已decode全矩阵dot代替官方精度次序；旧group32保持 |

原HYLE数学/GLSL继续冻结；只接原源码的标准tensor/arith/linalg/scf序列。
若原native转换不能给出精度，先具名源/设备/合同调查，不以新量化算法或更宽误差偷换。
finite检测只作GPU控制；不下载巨型模型、不增加loader、联合CPU+GPU模型执行或算子大全。

源码/接口/实现预算90分钟，资格纠正45分钟；AUDITED冻结参数/临时/状态后批接，一个
不可变候选、一producer build2/test1、重型串行。每具名source/call错误一次定向纠正；
不反复全集或P1。适用未验不能PASS；原模型完整geometry/链、Air和三交付仍由父轮完成。
