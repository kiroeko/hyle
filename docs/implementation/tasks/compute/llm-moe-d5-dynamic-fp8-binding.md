# D5-F38：真实动态 FP8 linear 接线

<!-- MODULE REVIEW: AUDITED -->

## SCOPED → AUDITED

baseline `cd12ca504`，Windows canonical；当前 D5 `compute_ops`、`compute_provider_cuda`
owner继续AUDITED。F37已验的预编码MatMul保持；本轮是四个官方画像的实际浮点激活入口。
禁止HYLE量化数学或修改成熟kernel正文。批量固定下面的有限FIX，再一次编译/验收普通入口。

## 固定源码与语义证据

- SGLang `06992c92ad4494fe9f9636e75be3e35040512447`：
  `fp8_utils.py::triton_w8a8_block_fp8_linear` 在没有input_scale时先逐token/group128量化，
  然后block128 GEMM，输出保留输入dtype。SM89 auto分支选择此成熟Triton回退。
- 同pin `fp8_kernel.py::_per_token_group_quant_8bit` 保持F32 scale；
  `_per_token_group_quant_8bit_colmajor` 的 `SCALE_UE8M0=True` 将scale向上舍入到二次幂，
  以F32列主序存储。这是无损scale载体，与UE8M0 group32编码是不同合同。
- 已登记DeepSeek官方固定文件重新取回并核对原SHA256：
  `inference/kernel.py` 为 `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2`，
  `inference/model.py` 为 `ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71`。
  官方 `act_quant` 使用BF16激活、group128、absmax floor **1e-4**，`scale_fmt` 开启幂次scale；
  SGLang普通回退的floor为 **1e-10**，不能混为同一个数值合同。该差异必须显式表达。

## 有限 FIX F38.1–F38.4

1. `model_ops::Float8BlockLinear` 表达完整动态量化+FP8 GEMM：plain F32/BF16 `[M,K]` 与
   已有E4M3FN/F32 block128 weight `[K,N]`，输出与激活同dtype。描述符保留scale舍入模式
   和absmax floor；共同推导检查finite positive floor、完整K组及布局。普通MatMul不暗中量化。
2. 固定原版Triton3.8 AOT追加BF16普通量化、BF16幂次量化、BF16输出GEMM三个程序；
   算法与默认launch不调参，原F32程序复用。幂次scale先只资格化官方BF16域。
   首次GPU执行前固定独立编码/scale及输出oracle，包含零组、floor域、正负值、scale跳变与N尾块。
3. 同SGLang Provider持有原生modules；每invocation独占caller workspace（FP8 codes+F32 scales），
   两次native launch，同队列顺序，无host同步、权重展开或运行搜索。scale列主序以GEMM stride接合，
   不另写transpose kernel。准备失败回滚、部分发射失败和Completion保活沿现有CUDA合同。
4. 普通Direct/Graph遍历实际dense角色，含BF16输入/输出及变化输入；组合完整共享FFN的
   gate/up → 成熟激活/乘法 → down动态量化。检查算子输入、资源OOM及并发工作区隔离。
   先完成批量接线，再统一构建和测试。完整官方MoE、其它路线、三交付/P1仍独立签署。

原生候选不通过时最多一次有明确源码依据的接缝纠正；不得修改误差门或改写kernel凑通过。
动态BF16幂次量化按官方round_scale数学逐code/scale检查，不能只以整片段容差掩盖编码差异。
不引入模型引擎/PyTorch。编译工具复用F37已下载目录；普通用户无需编译器runtime。

## 首次原生运行前的数值门

FP8 codes逐字节一致，幂次scale逐位一致；普通scale相对误差≤2e-6。
GEMM独立double block求和转F32，F32输出绝对2e-6+相对2e-6；BF16输出比较RNE舍入后的期望，
允许原累加误差最多1个BF16 ULP，并单独保留量化codes精确门。NaN/Inf不是本模型finite域资格。
九个实际投影签名，M1/M3，各两个输入；量化边界另用group的零值/floor与BF16相邻scale阈值。
本卡不增加完整画像格；所有适用未运行项保持未验。
