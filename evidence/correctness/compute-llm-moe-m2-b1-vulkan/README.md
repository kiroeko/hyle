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

## V1-P：半精度与同设备 Provider 组合

当前产品候选`2191c6160`。dense非空F16/BF16 Copy、浮点Cast的合格方向以及已接算术族的
半精度输入已进入普通Direct/Graph。ggml在F32求值、F16结果在region内舍入；BF16结果使用
原版IREE标准转换，保留每个公开节点的舍入边界。未修改第三方或冻结自研的算术实现。

- [精度开发收据](precision-development.json)：候选、两机结果、失败与纠正、成本和未完成项。
- [精度原始记录](precision-development-raw.zip)：原厂资格探针、独立oracle合成输入、编译阶段、
  全部失败/成功日志、候选和制品身份、selector、统一LOC与清理收据；CRC/SHA256核验通过。

Windows RTX4070 Release/shared/WERROR的9/9 CTest通过，171项C++通过；Air Mac16,12/
MoltenVK 1.4.1 Release/static/WERROR的8/8通过，172项C++通过。两机均有同一可选validation
layer用例跳过。测试包含全部65,536种16位模式加奇数尾部、F32舍入中点/邻值、NaN/Inf/负零、
高rank半精度计算与F32 gamma、混合Provider、发布中间结果、两slot/重放/结果保活/close及quota。

首次MSVC编译纠正了测试把optional当指针的问题。首轮普通入口回归发现压缩权重被普通浮点域
误接纳、BF16提升精度绕过累加声明检查，已恢复公开拒绝条件；资源失败测试改为实际发生分配的
首次prepare，旧Graph资源计数跟随新region组合更新，数值oracle没有放宽。

原pin ggml的F32→BF16在451,837样本中有4个NaN分类错误，该方向未准入。原版IREE转换在
Windows通过；Air默认fast-math会消掉生成程序的`FUnordNotEqual(x,x)`检查，4个NaN边界失败。
仅关闭诊断进程fast-math可复现地消除失败；产品使用原版IREE公开编译阶段，给两个转换入口补充
标准ABI元数据，由原版MLIR生成`SignedZeroInfNanPreserve 32`，冷期核对物理设备支持。最终
两机默认环境均通过。没有修改compiler数学、runtime补丁，也没有全局关闭fast-math。
`compilation.json`的`precision`保存最初Windows资格，`precision_preserve`对应当前制品；
`compile_precision.py`是离线复现入口，不进入正常用户的CMake构建。

新增IREE制品12,251字节。F32暂存、必要布局转换、Provider边界copy/sync与原版运行时分配均
是成本；BF16公开节点分界可能增加提交，不声称原生BF16融合或SOTA。完整21行B1、适用host/
跨环境、新域三交付、锁文件变更触发的最终selected综合门与P1仍待完成，聚合INCONCLUSIVE。
源码/设计保守95/120分钟，未另开预算；四owner保持AUDITED，完整矩阵新增0。

统一LOC为**57,333实现行/249文件**，排除tests/vendor，self-check通过。已清理本批探针EXE/
OBJ及重复wheel包76,048,582字节；保留Windows约0.73GiB单一build和约0.22GiB活动编译工具，
Air复用原目录。当前Windows C盘约650GiB可用。未删除其它任务或用户文件。
