# M2-B1 oneDNN：存储操作开发检查点

产品`02fe54151d00b34c09de6d7a452c62097c14182f`，tree`e2db09aa625912d45f6a7004db7d2c7d8b6d381f`。按[当前任务卡](../../../docs/implementation/tasks/compute/llm-moe-m2-b1-onednn.md)接入原版Copy/Select，共用既有执行资源；未新增HYLE数学kernel。

- [开发收据](storage-development.json)：源码/制品身份、11/11 Windows CTest、声明的验证范围与未运行门。
- [原始记录](storage-development-raw.zip)：原版资格源码/输出、所有构建/测试纠正记录、最后CTest完整日志、selector及LOC；CRC与逐项hash已验证。

原版浮点reorder不满足按位Copy，合格映射使用u8字节描述；s8条件的原版Select保留全部branch bits。原版softplus探针使用诊断相对门，不能把小尾值差直接判成模型失败。Select的张量condition选择两个标量还发现了公共输出推导缺陷，先按M2-S S4定界再修复。

本收据只签Windows开发检查点。其余新能力、跨平台/插桩/三交付、完整selected及P1未完成，owner仍AUDITED，完整覆盖新增0。Compute+Graph统一统计为56,908实现行/249文件，排除测试/vendor。

清理收据：临时探针EXE/OBJ/CRT的删除被自动审批拒绝（只给出blocked by policy），18,756,425字节暂留；没有执行删除。活动`C:/hv/m2od`构建约1.62GiB，供本批后续接入复用；缓存、模型与其它工作区保持。
