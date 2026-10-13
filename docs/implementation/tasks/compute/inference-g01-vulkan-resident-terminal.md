# G01 Vulkan 单请求驻留子批终审

状态：CLOSED，2026-10-13。只关闭RS-03/04、NP-01..03、RR-01..03、VP-01..03的
有限Vulkan单请求域；整个G01仍未完成，CPU与多请求按总卡继续。
Provider/Route candidate `3d92eee97a70f7788dfb9bbcda73f6e0f801959f`；runtime candidate
`a40003f7edb54a3da3113d0cd401b46211e0b0f0`。原F5/其它数学与全部历史证据保留。
定界/授权见[resident](inference-g01-resident-sdpa.md)、[精度](inference-g01-vulkan-native-precision.md)、
[端口](inference-g01-vulkan-resident-ports.md)；不代签新paged/varlen、多GPU或完整引擎。

## 七问与固定裁决

1. 借原锁定ggml全F32 scalar flash与原command/descriptor pools，NOTICE/patch/锁SHA认证；
   没有新库或数学shader。normal和strict cache身份分开，native不支持域保原显式路线。
2. 主consumer为GQA8/2 D256 prefill/decode容量cache；D64/128、两batch、F16/F32、非causal、
   padding/readonly/slots、旧output与origin/owner重绑反证领域特化。只准BSHD同D64/128/256。
3. 原Graph/Compute、F1 ready权限、native queue/fence、copy、allocator与真实退休均复用；
   private弱graph身份防ABA、端口变化先失效，失败/设备丢失沿原accepted-prefix/drain。
4. 借原flash避免scores/KV按Q/head展开；原冷mask/query cast/private UNKNOWN如实保留。
   真正vec4合格port direct import消除warm Copy3；其他pitch显式Copy、坏origin提交前拒绝。
   Work记录只固定无host staging/event/semaphore图，其他图冷评估一次后保原managed成本。
5. 只有私有C++17 native seam，没有新增公开FFI、Provider策略或公共capture承诺。
6. [实现阶段](../../../../evidence/correctness/inference-g01-vulkan-resident-ports-stage/README.md)与
   [候选交付/风险](../../../../evidence/correctness/inference-g01-vulkan-resident-ports-candidate/README.md)
   实跑Air source9 tests、改名搬迁static/shared各9 tests；宿主ASan/UBSan/TSan各9公开+4
   native风险tests，六个TU插桩核对。Windows只更新相关TU/消费者编译；原WinSDK/ASan
   仍绑定旧candidate，不伪造新GPU资格。宿主风险不签shader/driver或LSan。
7. 普通SDPA+TensorDesc/view即可用；[guide](../../../user/vulkan-compute.md)说明pitch16、
   实际native origin对齐、copy/拒绝边界。旧结果仍阻止slot重用，两个独立slots不共享mutable池。

## 验证与费用结论

公开72个原数学episode和64步重新绑定保持原oracle/容差；成本反证原3Copy实际FAIL后
修成合格0Copy，65-element非vec4保持2KV Copy与正确数学。坏origin相同descriptor绑定
提前失败，后续合法绑定可正常运行；原normal/strict交错、D8/72/512拒绝与zeroquota回滚
均保。旧4个managed/mixed-provider回归实际通过，未重跑无影响全算子/全库矩阵。

[最终P1](../../../../evidence/performance/inference-g01-resident-vulkan-direct-ports-m1-p1/README.md)
资格PASS、performance PASS、原verify退出0，沿原[15,31]规则自动追加而不改变阈值。
decode比0.9941–1.0017，prefill1.0010–1.0032，留出1.0035–1.0092，全部5%非劣PASS。
方向分别INCONCLUSIVE/WIN/WIN，全部仍在5%band内；[分账](../../evidence/performance/inference-resident-g01-2026-10.md)
保两次原LOSS、失败原目录/日志及累计耗时。不是全模型吞吐或模块全局SOTA。

冷port对齐metadata为O(ports)，State增加私有vector对象，只有tagged FA分配元素；
warm只是原O(ports)验证/绑定与原native submit/wait/output归还。driver/private pool
上界UNKNOWN仍明确，G10另处理；无newmetadata字段冒充跨平台总预算。

到达有限FIX与本卡适用门停止条件，三个owner此有限轮CLOSED。reopen trigger仅为本
准入域可复现数学、owner/actual-retirement错误，明确成本回归或新真实消费者需当前
合同无法表达的机制。多请求/G02的新机制另起精确卡，不能以重新检查原矩阵伪造reopen。
