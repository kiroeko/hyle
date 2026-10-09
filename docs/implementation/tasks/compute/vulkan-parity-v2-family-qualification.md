# Vulkan V2：剩余模型族与参数合同验收

<!-- MODULE REVIEW: AUDITED -->

baseline `f42112fb2`；owners `compute_route_vulkan`、`graph_route_vulkan`、
`compute_provider_ggml_vulkan`。属于[父轮](vulkan-cpu-cuda-apple-parity-closure.md)，
延续其既定八族/80参数范围；已通过80/560/25两入口的证据复用，不重测。

## 有限范围

主纵切：共同 family-witness-index 的 R02–R13 与 parameter-witness-map 的80条
角色/shape记录。它们有30种去重存储形状、24个源码anchor；已资格当前 DeepSeek/Qwen
C5和sidecar条目复用，只运行余下16个已有 family/parameter test。
正交纵切：已有 Qwen3.5-2B 本地官方实际BF16 fixture、官方 mixed MoE/路由与
变输入、两状态、feedback/reset、全词表行。不是新模型、算子、性能竞赛或全模型推理。

| FIX | 有限动作 |
|---|---|
| F1 | 显式列出已有测试选择，映射80条参数/八族到实际当前Direct/Graph证据；按 dtype/epsilon/shape/state审计，不以名称即PASS |
| F2 | 复用当前producer和原fixtures，一候选串行运行剩余分支，两入口一次；local fixture不得SKIP |
| F3 | 三IQ1实际银行继续保留原E冷准备事实，按共同窗口合同检查可用银行实跑和资源边界，不把冷准备写成hot全银行资格 |

source审计只读既定consumer；无新产品FIX时直接使用已编译候选，元数据变更不重建。
如果出现具名原参考/接线错误，先记首失败、核对同输入和原精度；每个错误一次有限纠正，
不修改误差门、不新增数学/kernel。需要新public/API/算法时停止本包，明确另行scope。

预算：映射/source审计20分钟，串行集中资格20分钟，具名纠正最多20分钟一次。
本包结束给PASS/FAIL/未验映射，不无限调查，不因它重跑80/560/25全集。
P0为原成熟数学/资源/转换账；P1、默认、平台和三交付仍由V3/V4完成。
Windows RTX4070真实GPU，build2/test1，DLL邻置、隐藏child及validation层；只用本地fixture。
必需检查：selected工具/测试、closure（registry明确previous-ref）、docs/diff与统一LOC。

## AUDITED source 结论

[80条/八族当前源映射](../../../../evidence/correctness/vulkan-parity-v2-catalogue/family-parameter-source-audit.json)
固定每个角色、shape、F32存储、原consumer anchor/SHA及两入口case，不新造需求。
实际数学核对包括 Qwen/DeepSeek1e-6、Kimi/GLM1e-5、GLM indexer LayerNorm1e-6、
GDN folded A/bias与norm128、KDA逐key decay/L2、卷积单例轴/三参数拼接、
选择用router correction、latent/shared gate、AttnRes银行、mHC20轮、NextN原次序/carry。
24个参数anchor仍存在，另有family原consumer anchor；未运行项保持PENDING。

F3 8银行源metadata：E896的N/K3584/3072双方向与E512的8192/2048双方向；
packed字节仍小于I32，但逻辑元素超I32。复用既有原grid decoder作独立数值参考，
测试真正上传全部bank，访问E-1/E-2/0、改变activation和IDs，同slot重放、越界拒绝。
只增test，不增产品数学；容量/feature事实按实际device报告，不把OOM当算子覆盖PASS。
有限F1–F3已审计固定，现阶段不需要产品改变或公开API重开。
