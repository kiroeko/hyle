# CPU Native域裁决前：成熟准入诊断

审计基线`155d1385c2b9a50eaa66a6c069b569f11daf4b03`；记录候选
`8f8cf1e41389afa9a2127c222b6b88b7dabef1fb`只更新scope/状态。
复用已经通过的Windows CPU静态SDK与DLL邻置，无产品改动、无性能样本、无算法优化。

## 实际结果

1. F32 separate SiLU gate相同输入含`-Inf, NaN, +Inf, -0, +0, -10000, -20, 2`，value全部1：
   Native在`-Inf`给0，锁定ggml的原版优化CPU程序给NaN；其它该组输入数值没有实质差异。
   这证明当前两实现不能未经数值域裁决就宣布整个GatedActivation同合同。
2. oneDNN普通F32 dense M1/K64/N64 prepare以“small serial … HYLE-native decode”拒绝；
   同版`create_brgemm_plan/execute_brgemm`原版carrier能实际计算，64个输出均精确为64。
   后者是private diagnostic，不能据此写普通oneDNN M1已接入；它证明资格被adapter条件挡住。

原始输出见`probe.log`，步骤/候选见`receipt.json`，可重现源码见`cpu_domain_probe.cc`。
原carrier没有替代public admission/check/Completion，此处没有使用其耗时签性能结论；receipt秒数仅为
命令运行耗时。它也不代表三平台、完整特殊值矩阵或所有CPU算子已审计通过。

下一步先固定完整数值与资源合同，完成具名成熟接缝修复/资格，再冻结有限Native比较与清退规则。
不能因库被错误拒绝就认定自研唯一，也不能把不等价实现的速度当胜负。
