# D5：C10 / ds4 低精度成熟来源批量资格

<!-- MODULE REVIEW: AUDITED -->

## SCOPED → AUDITED

- baseline `5fabf548565a4598b8b7bf490b2eaabfc942cca1`，Windows canonical；初始工作树干净。
- owner：D5成熟来源资格维护者；产品FIX为零，不重开CPU CLOSED owner或解除Vulkan BLOCKED。
- 来源：[残余合同审计A1/A2](../../../../evidence/research/llm-moe-d5-residual-contract-audit/README.md)。
- 真实需求：CPU官方GLM/DeepSeek动态E4M3与状态roundtrip，Apple官方DeepSeek FP4状态。
  当前普通入口画像不能用单函数探针代签；四模型共12个CPU/Apple拒绝格仍保留。

## 固定候选及有限 Q1–Q3

| 编号 | 固定源码 | 本批资格与边界 |
|---|---|---|
| Q1 | 本机PyTorch 2.8.0+cu129，commit `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`，C10 E4M3 header | 调用原F32→FP8和FP8→F32函数，全部有限BF16值、全部有限code和中点两侧F32邻居；饱和输入clamp由既定合同规定。量化codes精确，不引入Torch运行库或GPU |
| Q2 | ds4 `9139e2ae58a41503968a500f36f75895c1ba63fc`，`ds4.c`原group64 FP8和group32 FP4 | 原函数原样摘录，只补include/失败stub；全部有限BF16分组、普通混合值、低scale和中点。按原公开roundtrip数值相等门，signed-zero位差另列，不擅自强化成新拒绝条件 |
| Q3 | 同pin的`metal/dsv4_kv.metal`原FP8 KV和Hadamard+FP4 kernel | MacBook Air单批原生执行；分别对照原F32融合边界及模型Hadamard后BF16边界。普通、低scale与边界输入；保持原kernel数学/dispatch，没有host执行替代GPU |

独立oracle只在验证目录，复用已登记表格nearest-even、F32 scale及BF16存储定义。
记录pin、源文件和摘录SHA、机器、编译选项、输出和错例；源码版本不是产品dependency准入。
不得删除Hadamard来伪造ds4本不存在的独立FP4 kernel，不因微小误差放宽已固定合同。
Q3同一批固定检查原默认模式与上游已提供的`DS4_METAL_MATH_SAFE`对应safe模式，
用于区分已有编译开关和算法边界；不遍历其它编译选项，不因失败增加第三种模式。

## 预算、工程成本和停止

一次源码摘录/身份核对、一个Windows CPU编译、一个Apple原生编译，重型串行；
每候选一次数值批验，只有明确include/ABI/布局错误允许一次修正。没有P1，不重建SDK，
不复跑旧F45/F46/F47，不下载模型。Mac仅验证复制品，源文件从Windows发送。

全部通过后，才另行绑定产品owner、确定完整group量化+成熟GEMM/状态承载和有限接线FIX；
转换原语通过不增加176格。失败则固定具体数值或合同反例并停止该候选，不自行改上游数学。
C10若只有标量转换可用，必须明确还缺少什么完整成熟组合，不能包装自写算子冒充Provider。

预声明检查：Q1–Q3与来源identity、selector docs/closure、D5 ledger结构和统一实现LOC。
本卡产品/公共API/资源模型未变，不需要重跑三交付、sanitizer或其它路线性能。

## 本批资格结果

Q1通过：原C10直接转换/解码66,290项，零编码差异。Q2通过：ds4 FP8与FP4各107,264个
输入，零数值差异；各有一处仅负零符号位变化，与当前roundtrip数值相等门一致，单列保留。
Q3仅FP8及普通F32融合域通过；原Metal FP4模型边界在32,768个普通输入中有107项差异，
9,216个混合尺度输入中有1,345项差异。原默认和safe模式相同，不接入该Apple FP4候选。
完整日志和源码身份见[本批证据](../../../../evidence/correctness/llm-moe-d5-real-fragments/mature-low-precision-batch/README.md)。

本资格批次完成并停止，没有重跑旧候选。C10尚缺完整分组量化/GEMM承载，不增加画像数；
ds4原CPU两个完整roundtrip已具备另定产品接线范围的依据。Apple仍需保留BF16中间边界且
保留极小BF16值的成熟程序，不能改变精度合同或把F32融合通过当模型通过。
