# D5-F39：GLM 官方 FP8 专家批量接线

<!-- MODULE REVIEW: AUDITED -->

## SCOPED → AUDITED

baseline `d8e4b4c1`，Windows canonical；owner为当前AUDITED的`compute_ops`和
`compute_provider_cuda`。F38局部通过和旧失败全部保留；本轮不重开D1–D4，不签完整画像。
真实consumer为GLM-5.3/Flash官方FP8 expert gate/up/down：H6144/4096、I2048、top8、
256/288个router打分。验收允许resident experts缩为top8+1，原hidden/intermediate不缩小。

## 已审计成熟边界

固定 SGLang `06992c92ad4494fe9f9636e75be3e35040512447`：

- `kernels/ops/moe/fused_moe_triton_kernels.py::fused_moe_kernel`直接支持E4M3FN专家权重
  `[E,N,K]`、F32 block128 scale、group128 activation scale和动态sorted token/expert IDs。
  `top_k`寻址可表达共享token输入和每slot独立激活；输出可直接保留BF16/F32。
- 同文件wrapper的`FUSE_SWIGLU`显式排除FP8；本次保留原FP8 GEMM路径与成熟独立激活，
  不是拆散一个该dtype已有的融合。不开SM90 TMA、PDL、LoRA、分布式或实验分支。
- `srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_config.py::get_default_config`的
  block128默认是64×128×128、GROUP_M32、4 warps、3 stages；固定此规则，不运行计时搜索。
- `kernels/ops/moe/moe_align_small_numel.py::_moe_align_small_numel_kernel`使用成对比较
  排列不超过64个token-slot，保留重复expert和block padding；大批次另有
  `kernels/aot/csrc/moe/moe_align_kernel.cu`的原CUDA align/count-and-sort程序。
- 原程序假定ID合法。公共接口不能把非法ID交给未检查的weight/cumsum寻址；必须有显式
  bounds适配和既有CUDA device-status/Completion失败，不能以性能为由省略。

## 有限 FIX F39.1–F39.4

1. 先资格化原版grouped GEMM和原版align程序，复用F38量化。编码权重物理布局为所有
   `[E,N,K]`codes后接`[E,ceil(N/128),K/128]`F32 scales；完整K组，不偷偷解码权重。
2. `float8_grouped_weight_description`与具名动态grouped linear表达weights、激活
   `[T,A,K]`、I32 IDs`[T,S]`→`[T,S,N]`，A为1或S。A=1只量化一次token，A=S保留
   down输入的每slot状态；普通GroupedMatMul不暗中量化，不要求用户排Provider。
3. CUDA同Provider准备时绑定量化/align/grouped程序和原生launch；每invocation独占
   codes/scales、排序/padding临时区，无host读回ID、逐expert虚调用、运行搜索或重编译。
   对原align入口只允许bounds与错误上报适配；有效ID的排序、矩阵运算和聚合数学不改。
   负ID/越界ID必须安全完成为错误，后续不得读取无界地址。固定并记录对齐、峰值scratch、
   accepted-prefix失败与Completion保活，复用现有错误通道。
4. Direct/Graph同一批次覆盖两个GLM expert形状、up/down、M1/M3和超过64 slots的prefill；
   两轮输入和ID变化，含重复ID/不同成员、尾块及非法ID；组合router→experts→成熟激活→
   down→加权聚合及共享分支。批量接线后统一构建，资源/schema测试同批完成。

## 验收与停止

原量化codes精确、普通scale相对2e-6；固定codes/scales的独立double sparse求和，
F32输出绝对2e-6+相对2e-6、BF16输出1 ULP。完整链复用已登记BF16链门，不能用整链
掩盖局部编码、ID或资源错误。初始资格必须同时包含实际hidden/intermediate和动态expert成员。
默认参数不调优；首次原生候选失败仅允许一次有明确源码依据的ABI/接缝纠正。真实算法
或精度问题停止该候选并保留失败，不无限重测或引入HYLE数学。

本轮先签CUDA局部能力；其它路线继续由D5固定目标约束，不将CUDA通过写成四路线通过。
新候选三交付和P1仍按D5-A5/A6的不可变批次完成；所选未运行门保留未验。目标仍是22画像
四路线两入口和D6，不能因为本卡更容易通过而删减目标。

## 当前接线检查点（2026-09-27）

F39.1原版grouped GEMM通过80个原生案例；带bounds适配的原CUDA small/generic align
通过24个案例，包括E9/65/256、8/24/136/1032 slots及负数、越界、I32两端ID。
产品采用原CUDA host的small/generic规则，不嵌入未采用的Triton小排序程序。
GEMM保持64×128×128默认tile，无逐专家host循环；复用F38原CUDA量化。

F39.2–4已批量接入`GroupedFloat8Linear`、独立grouped编码描述、调用者工作区和既有CUDA
错误通道。原生索引边界为top8、E<=1022、I32总偏移/网格，资格仍SM89/Driver13040。
BF16/F32、A1/A8、真实GLM宽度、prefill和两个GLM完整router→expert→shared组合均已
通过初次普通Direct/Graph批次；分配失败、并发、Completion保活和非法ID也已通过。
最终不可变候选、独立头与新增冷拒绝检查待下一条收据，不提前记为PASS。

资源测试初次误用固定绝对`1e-4`断言，在1152/3200等输出处误拒约1个F32舍入单位；
改回本卡事先固定的`2e-6+2e-6*abs(reference)`后定向通过。产品程序和原数值门均未修改。
五个离线程序由登记的Triton3.8/CUDA13.3重新生成，完整header字节一致。
完整画像ledger保持140/176；三交付、跨路线和P1仍待D5后续批次。
