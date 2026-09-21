# M2-B1 Vulkan：原版张量映射开发检查点

按[当前任务卡](../../../docs/implementation/tasks/compute/llm-moe-m2-b1-vulkan.md)扩展原版
ggml Vulkan region，已接 dense 非空 F32 Copy/Concat、高rank双向广播、既有norm/unary布局域、
任意选轴Sum/Mean及公开中间结果的producer保护。只增加原版调用和资源适配，没有新增数学kernel。

- [开发收据](tensor-development.json)：候选、机器、通过与跳过项、成本及未完成验收。
- [原始记录](tensor-development-raw.zip)：Windows/Air configure/build/CTest、失败与纠正、
  artifact hash、selector、统一LOC和复现脚本；含逐项SHA256 manifest，CRC与hash均已核对。

产品提交`9a49fad478a1181d068a6a7ec8d924c82ca8bbf9`；Windows Release/shared/WERROR
9/9 CTest通过，166项C++测试通过、1项可选validation-layer测试跳过。Air Clang首次编译发现
新测试复制Operation的警告；Windows真相修正为const引用并提交`4a7058979d1fe7593de4a8bf8eaae6421d4a6a01`。
这一提交只改测试，Windows对应109项复验通过，其余原有制品hash不变；Air Release/static/
WERROR真GPU的8/8 CTest通过，167项C++测试通过、同一可选项跳过。没有降低WERROR。

新用例实际断言使用成熟region，覆盖独立广播/归约oracle、rank6/8、非相邻归约轴、513行尾部、
Copy特殊值原始位、所有Concat轴、两slot、变更输入、保留旧结果及close后资源生命周期。
原版repeat/cont、managed分配编码、边界copy与同步仍有成本；本检查点无P1或SOTA声明。

完整21行B1、其余precision/stride/empty/index/mask域、适用host插桩/OFF及新增域三交付尚未完成，
最终聚合INCONCLUSIVE，三个owner保持AUDITED，完整覆盖新增0。统一脚本统计Compute+Graph为
**57,086实现行/249文件**，排除tests/vendor，self-check通过。

Windows约0.73GiB的单一build和Air约0.77GiB的验证目录保留供本批后续复用；没有新建废弃build，
没有模型副本或二进制加入证据压缩包。
