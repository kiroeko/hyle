# Vulkan V1：ggml原8格式与实际embedding漏接

<!-- MODULE REVIEW: AUDITED -->

父轮[新Vulkan收口](vulkan-cpu-cuda-apple-parity-closure.md)，baseline
`a19391c7efa84d1f0c42919e8192e4ce78d1ee32`；owner `compute_provider_ggml_vulkan`。
现有AUDITED按真实共同目录重新定界；旧C3拒绝记录仍是历史事实，不限制本轮新增准入。
route/Graph原BLOCKED和其它CLOSED owner保持；本卡不授权修改其产品文件。

## source审计与有限FIX

锁定ggml `456172ec733a135778adcd32d00e576a58232e45`原Vulkan supports_op、
shader-generator与dequant/mul_mat_vec body已含Q4_0/Q4_1/Q2_K、IQ2_XXS/XS/S、IQ1_S/M，
对应119个共同目录合同。原GET_ROWS含Q2_K/Q3_K/Q4_0/Q4_1/Q5_K，但HYLE仅准入Q4_K/Q6_K/Q8_0，
对应另8个实际embedding合同。source存在不签实际device/shape/数值PASS。
未知IQ1_XS/XXS/XXXS仍不制造COUNT tensor，不由CPU枚举推定GPU实现。

| FIX | 唯一允许产品改动 | 集中验收 |
|---|---|---|
| G1 | `region.cc::native_quantized_type`只补上述原8种ggml_type | 原bytes/shape/stride/ggml_nbytes一致，实际MUL_MAT/MUL_MAT_ID与变输入 |
| G2 | `native_embedding_table`只补实际需要的Q2_K/Q3_K/Q4_0/Q4_1/Q5_K | 原GET_ROWS精确/encoding、native能力、重复/变IDs，public checked失败/恢复 |
| G3 | 原拒绝测试迁移为18类型静态准入与3无source扩展拒绝；复用普通Direct/Graph批量测试 | 两入口原FP32累加门、不解码/repack/换格式，不用prepare成功代签执行 |

canonical descriptors、buffer alignment/capacity/offset/overflow、cold原device supports_op、
native资源/workspace、完成/失败/旧lease都沿现有实现；不改上游kernel、公共API、编码、
runtime core、IREE程序或route框架。输入仍原F32/F16，grouped F32/I32，原输出精度不放宽。

P0：原packed bytes直接native调用；冷建立graph/workspace；hot无decode/repack/重选，
不开single-carrier P1竞赛。把库直接原调用与public入口的全成本比较留给父轮固定候选，
不把本批静态/source证明冒称速度或完整模型PASS。

## 预声明验证与停止

同批8格式MatMul/grouped、实际8种embedding、旧10格式回归的影响子集；动态activation/
weight/IDs、checked错误/恢复、旧结果/close、异常容量及native资源测试；Windows RTX4070
实际计算，Air按真实原capability另验。共同119/8合同的actual geometry在父轮矩阵统一
验证，微型源资格不冒称全部合同已完成。三交付/package按selector与父轮集中一次，
没有头/API/CMake变化时不新建三delivery producer。low parallel2/test1、批末immutable
candidate；新错误仅一次具名纠正，否则具体裁决。无mock/CPU计算fallback、无新self。

本次在产品前先提交本卡/owner record/source静态映射，随后才产品。当前runtime未运行，
G1/G2/G3尚未PASS；docs/closure和统一LOC不重建、不跑P1，旧perf blocker保持。

## 首批具名数值结果与一次原因捕获

`3b7d3ba9`统一编译PASS，`b2a66c57`原Provider25/25通过；普通Direct3/5、Graph2/4。
所有失败只涉及IQ1_S的MatMul/grouped，embedding八类型两入口通过。原其它类型没有
数值失败，不重跑它们。原门保持，当前只能签部分通过，不能签119合同完整runtime。
已查锁定backend：RTX整数dot路径会把F32 activation转为Q8_1（d/s也存F16），旧helper
只用decoded weights×未转换F32参考。它不是已证错误的末层kernel或precision FAIL。
固定一次Direct IQ1_S捕获：原same input/encoded weights；独立原Q8_1量化和IQ1_S整数
dot/rounded-s公式，与native结果和旧F32参考分别比较。只作原因证据，不改变模型输入/
输出门，不取更近结果签PASS；产品FIX零。若证实参考服务不同，再按原共同数值合同
定界；若无法证实，保留精确缺口停止同候选微调。只编译变更测试，不重配或全集/P1。

`c9419820`一次原因捕获：两replay的原Q8_1/IQ1_S整数dot/rounded-s独立方程与native
最大误差4.47e-7/8.94e-7，而原未转换F32参考最大差0.004785/0.009571。锁定source的
MMVQ在RTX整数dot、连续F32输入、该固定M2/K256下选择Q8_1，不是HYLE漏传F32 accumulator
或新增数学。旧FAIL保留，捕获不代签正式资格。
正式唯一测试修正：预先按原source/device事实固定IQ1_S程序参考，紧至1e-5+2e-6 relative；
同时按已经建立的共同GGUF归一化完整量化门比较独立未量化activation期望，不能只验证
路由到错代码也同错。原3e-3逐点门只对其原F32程序适用；不把它误当本库Q8_1程序合同，
其它17格式原门完全不变。输入/权重/output/state不改，QQ8转换/同步成本必须如实计入。
仅IQ1_S MatMul/grouped四个失败ID两入口delta验收，明确打印所测type，其它通过域复用。

`a9eced9d`IQ1_S定向正式结果Direct2/2、Graph2/2 PASS，紧原program门与共同模型量化门
同时通过；raw逐项打印IQ1_S，没有重跑其它类型/provider/embedding。仅一次测试私有
include纠正，两test增量重链，无产品改动/重配/P1。合并首批证据，18原格式MatMul、
9grouped样本、8embedding普通入口已有真实GPU资格；这是微型source链，不代签父轮
119/8全部actual geometry或559/34/45/新版MTP完整覆盖。该数值原因关闭，不继续探查。
下个包只按共同目录补actual矩阵和具名原成熟承载；原route/Graph性能BLOCKED仍保持。
