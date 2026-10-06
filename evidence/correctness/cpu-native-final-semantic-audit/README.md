# 冻结 CPU Native 的同合同数值诊断

2026-10-06；产品仍为CP candidate `a527dd73dcc21a5b4b89979d43339d67becabe86`，1337个产品/fixture/
构建输入hash相同；canonical审计HEAD `35114b83`。这是不计时的有界P0 witness，不是P1、SOTA或
全输入数学资格。没有修改产品、SIMD、Provider优先级或GPU。

最终**322条执行记录**：Native161、普通成熟默认161；schema/prepare拒绝0。Native31条不满足
与MN/TM现有成熟资格相同的误差门，成熟peer0条失败。每个peer都检查非Native identity：实际
oneDNN/优化ggml/ATen，双方同描述/资源/checked合同、变化不在计时中、submit到真实wait。
caller workspace只按provider归属提供；必要的packing和完整原程序没有被跳过。

| Native未资格域 | 实测边界 | 成熟对照 |
|---|---|---|
| GELU、GELU-tanh，F32/F16/BF16 | dense和element-strided；负尾部消减归零/误差，x=-5也有可见差距 | 稳定原ATen均通过 |
| Unary SiLU，F32 | x=-100的最终subnormal相对误差约1.7%，超同资格门；dense/strided相同 | ATen通过；不拿低精度结果与其竞速 |
| gate GELU/GELU-tanh，三floating | packed/separate；尾部误差在value放大后可见，部分归零而正确结果非零 | ATen完整F32 activation/mul后storage舍入通过 |
| gate SiLU，F32/BF16 | packed/separate；F32 x=-100/value1e38：-0.00037835058 vs -0.00037204474；BF16放大后也超门 | ATen通过 |
| RoPE，F32 data+长I64 positions | ±2147483651，Float位置/angle精度丢失，8个输出明显偏差 | 同端口ATen Double位置程序通过 |

这对应**16个op/kind×dtype/position域**、31条布局/packed-separate行，不是31个新算子。
没有签全域性能LOSS；不满足同门的候选不能签性能WIN。F16 SiLU gate与其它本批通过样例仍须
精确域/性能裁决，不因一次通过签KEEP_WIN或完整UNIQUE。

其余有界样例通过：Binary六kind/NaN/signed-zero，Reduce四kind/NaN与零号，两norm有限reference，
F32 row-major/transposed物理RHS MatMul，Softmax/log有限和全无效行，非零Bool Select，I64 Gather/
Assign-last-write/Add重复，Copy/Concat/Fill、I32小位置RoPE、equal-head SDPA与F32→Half Cast。
它们不是全部dtype/layout/mask/alias/模型的全域证明；原已有资格和当前源码还须分别匹配。

oracle采用既有独立RNE decoder/encoder与Double稳定公式，F32相对3e-6+4denorm、F16 .001、BF16
.008保持MN原门。首次witness fixture依赖GTest header和错误给provider-owned route传workspace；
一次具名fixture修正批处理后完成。扩展SDPA首次oracle误用了隐式1/sqrt(D)，public default scale
实际1；显式声明同参考scale后两方均通过，不归因任一库。

所有失败、实际命令/原输出、最终源码/独立reference、逐域summary及身份在
[diagnostic.zip](diagnostic.zip)，计数/hash见[receipt.json](receipt.json)。来源为固定Native源码与
已关闭MN/TC/TM/AS/CP接线；未重新研究历史全部任务，未优化自研或放宽数值门。

当前[有限裁决卡](../../../docs/implementation/tasks/compute/cpu-native-final-domain-verdict.md)将这些
未资格通用域纳入有限退场FIX；实施/回归尚未完成，不把诊断、文档或旧CLOSED算产品完成。
