# G01 Vulkan 原生 page16 数学阶段证据

产品草稿 `2500128ed`，五 owner 仍 AUDITED；这是有限 source 数学证据，不签
CANDIDATE/CLOSED、Apple 页数学、全 G01/G02 或其它 G 项。活动范围见
[原生精确页卡](../../../docs/implementation/tasks/compute/inference-g01-native-paged-read.md)。

actual MacBook Air Mac16,12/M4/arm64/macOS27.0.1 26A434/AppleClang21.0.0；原消费者
报告 Apple M4/MoltenVK1.4.1，新公开 Context 拒绝 software device。Windows canonical
SHA 输入单向同步验证副本，未从远端覆盖产品。原 pinned ggml456172ec733a 源和每项
patch 前后身份已重放；实际被编译的 ggml-vulkan.cpp 与 scalar shader 的 after SHA
分别核验，见 raw.zip 中 compiled-inputs/native-patch-identities。

新公开 PagedBatchInvocation 的 F16 D64/128/256、causal/noncausal 六组全部 PASS。
每组 64 次两请求 topology 演化及独立 slot，独立 double 数学 oracle 容差 0.002；真实
乱序/重复/共享/尾页/空 request，logical79>physical64，未引用 sentinel 页可写，foreign
schema/mixed generation 拒绝，旧 output 阻止更新并活过 controller，metadata allocator
最终零用量。报告每次仅上传 32byte 单 I32 表，staging32/device64/两 banks；未签 G06
低成本 planning/capture。物理 K/V roots 保真，私有 logical rows/table offset 进入原
FullF32 scalar flash，所有原 K/V 地址按 page16 映射，无全 KV gather 或每 Q/head mask。
普通 128byte push layout、原 math/reduction、normal coopmat 身份保留。

第一次 build83.783s FAIL 是测试错误使用 half bit carrier 的 float conversion。修为显式
encoder 与独立数学 decoder；同时在首次执行前修正 fixture 的 logical capacity。原源码/
日志/输入包和 FAIL 保留。第二次 build0.876s PASS，实际新 GPU 数学7.144s PASS，原
materialized direct/Graph/fault/append 回归5.345s PASS，累计97.149s，未重置原预算。
结果与精确输入见 receipt.json，全部 raw/scripts/原 selector selected/skipped reasons
在 raw.zip。中央 patch lock 更新导致 selector 保守全选，实际只构建改变的 native owner
及其两个真实 consumer；未重开其它 CLOSED owners或运行全库数学。

新 page 的 accepted-prefix/unknown retirement、宿主风险、三交付/逐头和不可变同合同
P1 尚未完成；原 reference fault PASS 不代签新 page 风险。Apple 原生分页未实现。
