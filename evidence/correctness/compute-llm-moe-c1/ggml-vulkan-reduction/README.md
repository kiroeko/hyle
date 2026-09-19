# C1-Y34：成熟归约/概率的原生模块资格

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
[总记录](native-progress.json)绑定本目录证据。

## 仍未完成

本目录只签原生模块资格。Y34本地2B张量的L2/概率片段source/static/shared、适用公开头及
selected工程余门尚未完成。[选择计划归档](native-plans-and-identity.json)记录112项selected、
8项未选理由，不能由本目录局部结果推导112项PASS。先前LayerNorm或状态片段不代签新接线。

C1其余Provider覆盖核对、完整成本比较、条件默认规则和存量实现淘汰保持未签收；CPU Q2
测量资格BLOCKED不变。没有C1完成、共同矩阵或性能赢家声明，也未进入C2。

统一脚本[实现统计](all-implementation-loc.json)：Compute+Graph共48,737代码行/197文件，
不含测试/第三方，绑定产品候选。归档后清理过期build-local源码及Mini上一验证源码副本共
191,054,277逻辑字节：[Windows](windows-cleanup.json)、[Air](y34-air-cleanup.json)、
[Mini](y34-mini-cleanup.json)；保留活动构建、当前源码、SDK和共享缓存。
