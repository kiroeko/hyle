# G03 原结果槽的来源聚合观察接缝

状态：SCOPED，2026-10-14；owner compute、compute_route_apple、compute_route_vulkan；
baseline `16d677b38db855482ada49ed260a3611f5c17147`。这是[主卡](inference-g03-pending-page-consumption.md)
的实际来源/后继结果合同，不修改已闭数学、原取消资格检查、公共构建或工具。

触发：原 attention 的成功 completion 不包含 source 的取消/错误。
[SCOPED 实际原型](../../../../evidence/architecture/inference-g03-owned-outcome-owner/README.md)已在
Apple/Vulkan 实测原 source owner + 原 OwnedResult/ResultSlot 的 move-only 组合：
成功保同一 payload/旧结果禁止 reuse；后继接纳之后取消 source，聚合拒绝发布结果；
abandon 原 completion 仍保 source 至实际退休。Vulkan 还实测 GetFenceStatus OOM
时 guard/source 保留，实际最终 wait 保第一 OOM，并验证该未知状态下 abandon。
没有产品修改、没有第二 executor/queue/result slot。原型只是 shape/lifetime 反证，
没有独立数学 oracle/64 步/完整七路线/公共 pending frame 证书。

真实消费形状另有硬约束：原 CPU+Apple/Vulkan graph invocation 的 native fragment
先 await_completion(submitted.completion()) 再 submitted.wait()。只在新 façade 的
result()/wait() 上合并来源，仍让 completion() 报成功，会丢失该真实消费者的合同。
因此产品须把聚合接到原 Completion/OwnedResult 观察入口，保原 storage/payload，
而不是让 consumer 再写一个 wrapper 或复制 Graph controller。

普通数值 producer→range copy→readonly reduction 是正交任务；与 KV/page/request
政策无关。内部接缝准入为 SELF，复用原 CompletionRetention/RetainedCompletion/
ResultSlot/F2 accepted phase/native actual-retirement callbacks。原 update controls 仍
唯一观察；borrowed prefix 不得在 suffix 实际退休前由新 observer 失效。

范围：一个 opt-in 冷观察槽；原 OwnedResult 的 trusted completion 替换/retention
接缝；原 F2 已接纳各 phase 的实际退休订阅。普通对象 layout/队列/allocator/锁/原子
写和数学不变。所有来源错误与 suffix 错误须保第一原状态；unknown 不伪造退休。
实现前固定有限 FIX/API/冷热费用，SCOPED 不授权产品修改。

预声明：四原 source family、实际 source→suffix 的成功/接纳后失败/重复观察/old result/
两 slots、unknown+actual wait、取消/abandon/close、原 prefix 寿命；无 warm 新分配；
逐 cold allocation OOM；相同 completion()/result()/wait() verdict。适用 source/static/
shared、相关独立头、宿主 sanitizer 与实际直接 Graph 消费入口。CPU/CUDA 只用 tiny
有限载荷，不重建 nvcc/kernel/全模型；Apple only 实查 Air；Windows 定向编译。

P0 先记已有 native 生命周期、冷槽、retention fan-out 与 phase scan 的真实费用；
没有新的数学/normal algorithm，不重复旧 P1。检查点调查/实现/验证各 2h，沿用户
常设授权有限追加，原 FAIL/TIMEOUT/累计不重置；heavy 串行/最多四 jobs。所有门
按实际影响裁决，未运行不写 PASS。本接缝完成也不代签主卡 pending page/late native
output 或整个 G03/G01–G10。
