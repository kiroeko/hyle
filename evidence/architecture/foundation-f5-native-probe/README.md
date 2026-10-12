# F5 锁定native C++ lowering原型

SCOPED proof only，不是HYLE产品/Graph/capture/性能资格。FlashInfer0.6.18/69ff11fc、
WSL GCC14.2/NVCC13.3/RTX4070；直接原PrefillPlanWorkspaceSize/PrefillPlan和paged/ragged
原kernel模板，没有Python/Torch runtime或新math/copy kernel。

F16/BF16各paged/ragged四场景：head64/GQA2、两request Q长1/2、KV长19/10、
page16/物理页2,0,3/尾3,10，padding填64以反证误读；right-aligned causal对独立double
dense oracle。最大error F16约1.15e-4、BF16约1.01e-3，通过。

本fixture原plan选CTA16/no split，float workspace0、int workspace64；该小形状不能
外推其他batch/head/capture或私有allocation0。原planner临时vectors/occupancy与H2D
仍有成本；代码编译了原CTA16/64/128 dispatch，实际只执行本域。

build约16.807秒/run约0.365秒只记录预算，不是吞吐benchmark。原源码/输入hash、
精确机器/工具链、命令、日志和二进制hash在receipt/raw.zip，build1800/run120预算。
下一步依[活动卡](../../../docs/implementation/tasks/compute/foundation-f5-paged-ragged-batch.md)
固定capacity/metadata generation/native最后使用与Graph同slot后才AUDITED写产品。
