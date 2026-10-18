# CPU resident SDPA 三形状独立适配器预检

CPU产品d91a不变，适配器901d926f4固定。[receipt](receipt.json)、[raw.zip](raw.zip)
保留两个placement与原control目标2路编译54步骤/160.39s，三个数学/readonly/双slots
oracle实跑3.42s通过；源文件SHA核对当前固定输入。增加Q17/KV257/D64 noncausal留出，
原Q1/128 KV1024 D256 causal与BF16 GQA8/2 BHSD不改。原double/finite/8e-3容差保留。

独立reference仍直接构造锁定原flash graph，经相同公开PreparedOperation/CPUQueue/
Completion/caller output；不调用candidate supports/lower。CPU原peer通过真实Git fetch
取得FFa4 pin并clean checkout，非伪造archive history。时序/费用账见
[预检说明](../../../../bench/inference_resident_sdpa/cpu_p1/README.md)。

selector因共享bench目录没有component归属返回BLOCKED，原计划保留，独立工具任务
已记录；没有改公共checker。这里仅签实际adapter compile/oracle，不签A/A、机器资格
或P1；native/private pool上界UNKNOWN不写0，整G01仍未收口。
