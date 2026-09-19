# Y26-F2：Gather 平台回归与真实 BF16 三交付

产品与测试候选：`d54ddc5afb8f9df3a3b5b6444e0b8e2bf03cb18e`。本记录仅签本节实际运行的
Gather 回归与局部消费；不签完整 C1、全局 selector 聚合门、模型推理或性能。

## 已运行

| 环境 | Provider / Direct / Graph | 结果 |
|---|---|---|
| Windows strict static | 19 / 62 / 21 | 102 PASS |
| Windows strict shared | 19 / 62 / 20 | 101 PASS |
| MacBook Air Mac16,12 static | 19 / 62 / 21 | 102 PASS |
| Mac mini Mac16,10 static | 18 / 62 / 21 | 101 PASS |

共406项；新增宽行用例内部另覆盖六种storage、511/512/513/1024/1025字、checked/unchecked共60种
组合，不将内部组合再次加进用例数。Windows两种构建的原版source合同及MASM parent组合各2项通过。
Windows/Air启用Vulkan validation，无VUID或同步错误；Mini未安装该层，其一项validation专属用例
明确不运行，不计PASS。Mini只签正确性与可移植性。

Windows与Air用同一source archive，分别完成source、relocated static和relocated shared三个
独立C++ consumer。输入为用户本地Qwen3.5-2B的34行、2048列BF16 embedding子表；每条交付执行
1/17/128行的Direct和Graph，共6种情况，各5次变化索引重放。Gather逐位校验；Gemma RMS使用
独立FP64 oracle及预先冻结的BF16容差0.008相对值加2e-7绝对值。没有下载或提交模型载荷。

每种情况验证旧结果保留、越界错误、失败槽退役、原表不变及全部owner释放后quota归零。
Direct冷建3个Gather槽与2个Norm槽：一个保存旧快照，另外两个Gather交替供同一个工作Norm消费，
使上一次输入的owner已释放才写回该槽。Graph使用两个完整片段槽，一个保留快照，一个重复执行。
热期不增建槽。最初consumer试图覆盖仍被下游Norm持有的Gather结果，被产品正确拒绝；该夹具失败
及源码保留在Windows交付归档，纠正仅改变consumer的冷槽安排。

SDK移动至独立前缀，producer路径在消费前隐藏；Windows DLL邻置、PATH unset、错误模式0x8003。
原始完整命令、toolchain/机器、日志、EXE/SDK摘要与source manifest均在下列归档。

## 证据和限制

- [聚合记录](result.json)绑定406项用例、平台receipt、六条实际交付及归档摘要。
- [Windows实际交付](windows-delivery.json)、[Air实际交付](y26-wide-air-delivery.json)分别绑定独立EXE与日志。
- `w1-and-provenance.zip`保存Windows回归、生成输入的脚本与前置receipt、冻结token case、
  原始selector计划；`y26-wide-*-focused.zip`保存Mac回归；两个delivery zip保存实际消费。
- source archive SHA256：`2499a6a3b9df755987c2d7b1f027a330175b9b5d66480eb2b87d9bbca235a8b5`。
  先前宽行漏写反例继续见[失败与纠正](../wide-row-counterexample/README.md)，不删除失败历史。
- 归档逐文件核对后，Windows/Air具名producer与展开源码分别清理999,469,301/877,197,471逻辑字节；
  cleanup receipt保留，SDK、consumer与依赖cache保留。归档没有模型载荷或编译二进制。
- 本节停止重复运行上述已通过集合。dependency lock触发的全局selected gates仍须在C1固定候选
  上完成；旧候选证据保持原身份，不自动改签到本候选。Scatter/Scan、完整C1真实片段、默认优选、
  等责P1与自有重复实现淘汰仍未完成；CPU oneDNN性能资格BLOCKED不变。
