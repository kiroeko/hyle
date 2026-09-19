# C1-Y34：成熟归约/概率的原生与三交付资格

范围见[当前施工卡](../../../../docs/implementation/tasks/compute/llm-moe-c1-vulkan-reduction.md)。
产品候选`05581df80e3b6290d4e7c5fc7a66ecf26821eb09`；
测试编译修正候选`11804be7e8661cafb535d2b372544e474bc9b1a6`。
两者只差新增测试的const-reference range-for及任务说明，产品输入不变。

## 已完成

| 环境/链接 | C++通过 | 明确跳过 | 证据 |
|---|---:|---:|---|
| W1 static | 136 | 0 | [结果与原始归档身份](windows-native.json) |
| W1 shared | 135 | 0 | 同上；Graph的static-only私有用例不进入shared |
| Air static / MoltenVK | 136 | 0 | [结果](y34-air-focused.json) |
| Mini static / MoltenVK | 135 | 1 | [结果及具名排除](y34-mini-focused.json) |

每套覆盖provider、Direct、Graph三个受影响factor；Mini唯一跳过项是未安装验证层对应的
`GgmlVulkanNative.ExplicitValidationUsesItsOwnInstance`，由W1/Air覆盖该风险。
Air/Mini各5项source contract检查另行通过。W1的测试编译修正按selector只重跑受影响
Direct因子，[91项通过](windows-test-correction.json)，不重复计成新能力。

新反例覆盖Sum/Mean尾行511/512/513/262145、三浮点、长Softmax分界及末块极值、Boolean
mask缩放顺序、additive的F32中间舍入和设备grid准入；既有任意轴/stride/空域、特殊值、
Recorder邻字、双槽/旧结果/close、OOM/drain回归一起运行。原容差未改变，无VUID/同步风险
诊断。构建parallel2，各机重型步骤串行；Windows DLL邻exe、runtime PATH隔离、禁弹窗。

## 首败与修正

- W1最初误用PowerShell5.1，配置前因缺少`Path.IsPathFullyQualified`停止；改用PowerShell7。
  原始失败保存在Windows归档。
- [Air首编译失败](air-first-build-failure.json)：测试中复制`Operation`触发Clang WERROR。
  仅把循环变量改为const引用，没有改输入、断言或产品；随后增量编译与完整受影响测试通过。
- 产品/算法纠正0；测试编译纠正1；独立consumer编译纠正0。没有性能采样。

所有ZIP在归档时逐成员核对原始字节SHA和CRC，回传后验证归档SHA/CRC。Windows编译日志保留
原始编码。结果文件记录命令、编译器/设备、测试数量和artifact身份；
[首批原生记录](native-progress.json)绑定当时的局部证据。

## 实际张量三交付与公开头

源码包候选`923895c5`仅比上述候选增加文档/证据，产品相同。
[Windows](windows-delivery.json)和[Air](air-delivery.json)各完成source/static/shared三条消费，
每条72例/216次执行，合计432例/1296次。均隐藏原producer/源码路径后消费搬迁SDK，记录
实际链接依赖。使用同一份本地2B embedding fixture、F32/BF16、三种形状；Sum/Mean、无mask/
Boolean/additive Softmax和L2，独立FP64公式、约定F32边界及原容差。双槽、变输入/mask、
published平方和、旧结果及close/quota通过；没有整模型或性能验收声明。

消费端修正一次显式Operation构造编译错误，另一次处理Direct多步链的结果保活：七个Invocation
串接后，下游输入仍保活上游结果，重放应拒绝覆盖。最终Direct L2使用现有Recorder与独立的
caller buffers，单算子和Graph使用Invocation；未改产品/公式/门限，也未比较两者的性能。
两次首败及其当时consumer封存在Windows原始归档，冻结合同和验证脚本另有
[输入与公开头归档](delivery-inputs-and-headers.json)。W1/Air/Mini各两个公开头门通过。

[本阶段记录](delivery-progress.json)明确保留工程余门与C1未完成。归档后清理重复producer和
解包源码共1,868,242,674逻辑字节：[Windows](windows-delivery-cleanup.json)、
[Air](air-producer-cleanup.json)，保留SDK、安装消费者与模块活动缓存。

## 仍未完成

selected工程余门尚未完成。[选择计划归档](native-plans-and-identity.json)记录112项selected、
8项未选理由，不能由本目录已完成的原生/真实片段三交付/公开头结果推导112项PASS。

C1其余Provider覆盖核对、完整成本比较、条件默认规则和存量实现淘汰保持未签收；CPU Q2
测量资格BLOCKED不变。没有C1完成、共同矩阵或性能赢家声明，也未进入C2。

统一脚本[实现统计](all-implementation-loc.json)：Compute+Graph共48,737代码行/197文件，
不含测试/第三方，绑定产品候选。归档后清理过期build-local源码及Mini上一验证源码副本共
191,054,277逻辑字节：[Windows](windows-cleanup.json)、[Air](y34-air-cleanup.json)、
[Mini](y34-mini-cleanup.json)；保留活动构建、当前源码、SDK和共享缓存。
