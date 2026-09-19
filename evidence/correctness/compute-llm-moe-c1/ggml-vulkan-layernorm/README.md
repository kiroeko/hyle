# C1-Y33 Vulkan LayerNorm 接入验收

候选 `4d255a5174825dc669de1684affd7729d3b430d7`，基线 `5f61546c`。
[汇总与文件身份](summary.json)仅签本轮 Y33-A–D，原版 ggml GroupNorm 承载中心化 LayerNorm，
包括必要的布局/类型/对齐适配和输出所有权；没有性能赢家或整包完成声明。

## 已运行验证

- [Windows 收据](windows-validation.json)及[原始记录](windows-raw.zip)：static132、shared131项
  C++回归，source contract、MSVC组合与适用头检查通过；9个selected门通过，111项未选理由
  保留。Compute/Graph source/static/shared六条通用消费按Vulkan实际路线隔离执行。
- [Air 回归](y33-air-focused.json)及[原始记录](y33-air-focused.zip)：132项C++及source contract
  通过；Vulkan验证层开启，无错误。
- [Mini 回归](y33-mini-focused.json)及[原始记录](y33-mini-focused.zip)：131项C++及source contract
  通过；未安装验证层的专用用例明确跳过1项，W1/Air覆盖该风险，Mini仅签正确性/可移植性。
- 本实际LayerNorm消费：Windows收据和[Air三交付](air-delivery.json)、[Air原始记录](air-delivery.zip)
  共同证明源码/静态/动态六条公开消费，每条60例/180次执行，合计360例/1080次执行；
  相同例集在不同交付重复验证，360不是互不相同的数学形状数量。

[消费合同](consumer-contract.json)和[冻结输入](inputs.json)绑定本地2B输入、独立两遍FP64参考、
最终F32/BF16 RNE、原数值门限、0/1/2个affine参数、尾宽/多行/重绑定、两个槽与关闭后的结果。
这是正交LayerNorm基础能力消费，不是Qwen原始forward或模型加载能力。
Windows原始包内含脚本、输入身份、工具链、CMake缓存、SDK清单和consumer源码；Air包保留
该官方源码包manifest、SDK清单及实际消费者。SDK重定位后原producer/source路径隐藏。

## 首败与预算

原始失败均保留：产品变量遮蔽WERROR、测试include/Operation调用编译错误、任务验证层环境
缺失导致Context创建失败、无关默认CPU/oneDNN构建在测试前中止、任务Vulkan SDK查找及
强制prefix隐藏已安装Hyle、外部consumer隐式缩窄WERROR。产品/测试纠正2次、算法0次，
独立consumer显式转换纠正1次；没有改变数值门限或删掉失败输入。

Mini最初夹具误要求未安装的验证层；18项Provider已通过，仅恢复未执行的Direct/Graph。
恢复脚本一次gtest JSON字段假设失败发生在新增测试前，修正读取明确的嵌套SKIPPED结果，
没有重复测试。[最终脚本归档](final-commands.zip)保留恢复和清理脚本。
Windows收据的待办与输入文件的FROZEN状态为对应时间点事实；本汇总及后续Air/Mini收据
补齐这些验证，不改写首轮记录。

## 统计、清理和剩余工作

[统一实现LOC](all-implementation-loc.json)：全部Compute/Graph实现48,424行、196文件，
排除测试/bench/第三方；不可变候选统计，自检通过。

证据ZIP hash/CRC及原始字节核验后，清理[误启用构建](aborted-delivery-cleanup.json)、
[Windows重复producer](windows-producer-cleanup.json)和[Air重复producer/源码](air-producer-cleanup.json)，
共4,207,211,641逻辑字节。SDK、独立已安装消费者、当前模块构建缓存和证据保留。
Windows首次收据汇总字段为空，[原收据](windows-producer-cleanup-initial.json)保留；
按未变逐目录账单纠正汇总，未重复删除。

本批停止重复验证，未运行P1。归约/Softmax等成熟接入、CUDA/Apple覆盖核对、完整成本比较、
默认规则与自有实现取舍仍待C1继续；CPU Q2测量资格保持BLOCKED。
C1整包及64行/256格矩阵未签收，C1完成通知未触发，不能进入C2。
