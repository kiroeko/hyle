# CPU原库Transformer/矩阵接入：审计记录

产品baseline7a6d3535，原程序consumer HEAD392f3e91，LibTorch2.8 CPU DLL锁定。
同一witness最终212个独立断言通过：rank8广播MatMul与K0、Bool非零mask/log-softmax/无有效行NaN、
长I64正/负位置RoPE两种pair layout与未旋转tail、GQA/Dk-Dv/bottom-right causal与zero-key attention；
F32/F16/BF16原F32转换、matrix/Softmax及RMSNorm/LayerNorm原kernel亦有数值断言。

原fused SDPA全mask输出0，公共合同要求NaN；不得直接代签。候选采用原matmul→scale/mask→
softmax→value Tensor math程序，零keys用原full_like NaN；所有score/转换/GQA repeat/axis临时计入。
RoPE是原Double sin/cos/stack/cat/clone/slice_copy Tensor程序，I64先按原Double转换，tail原bits保持。
矩阵与norm不做HYLE payload数学，Bool storage先eq(0)按非零定义，不能reinterpret异常Bool位。

首次oracle把visible +Inf漏判为无效行，212最终witness纠正并保留失败日志，不记成原库数学失败。
[original.zip](original.zip)保存所有源码/命令/失败/DDL身份；这仅原程序实际witness，不签产品
stride/alias/OOM/普通入口/三交付、性能WIN或CPU整体完成。
