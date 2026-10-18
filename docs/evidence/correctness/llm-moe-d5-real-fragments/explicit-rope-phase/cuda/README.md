# CUDA 显式相位与 hash 失败恢复

RTX 4070 / SM89、CUDA 13.3、Windows MSVC shared owner 构建；候选、命令、耗时及原始日志
hash 见[收据](receipt.json)。日志以 gzip 保存，构建失败同样保留，不算执行通过。

| 批次 | 候选 | 结果 |
|---|---|---|
| 首轮完整 owner | `87f2cb03` | 384个GTest中379通过；Direct两个DeepSeek片段、Graph相同两片段及hash恢复失败 |
| precise unary 首次配置 | `32fbe07c` | CUDA语言只在ggml子目录启用，新object缺编译规则；无测试运行 |
| 语言作用域修正 | `e55b0506` | 新object未继承ggml的C++17，NVCC拒绝C++23并退回旧语言模式；无测试运行 |
| 精度与受影响回归 | `889e0c4c` | 56个GTest中55通过；仅hash测试的Direct失败槽复用仍错 |
| hash恢复定向 | `27f37130` | 1/1通过：12轮坏/好输入、双Graph槽、Direct、旧槽拒绝和新槽恢复 |

精度批次包含全部30个ggml CUDA Provider、8个ggml Graph Provider，以及Direct8项、Graph10项。
新DeepSeek完整attention与压缩状态在两个入口通过；全部原数值门、长位置65536、状态反馈保留。
旧完整批次与定向修正分别留存，不能将它们写成最终候选重新跑过全部384项。

数值修正只对原版`unary.cu`禁用fast-math；[生成的编译规则](compile-rules.txt)确认该单元使用
C++17且不带该选项，其余原版`rope.cu`仍保留原选项。没有复制或改写数学kernel。
同文件Unary的性能仍需A6；本次测试耗时不作性能结论。

hash失败源于新消费者违反公开终态合同：Graph和Direct的Invocation均在异步失败后退役。
测试现断言旧槽拒绝，再从原Executable/PreparedOperation创建新槽；第二个Graph槽、Queue、
输入输出和不可变制品继续使用。没有改执行器或取消索引检查。

CPU先前完整回归可复用；Apple/Vulkan尚待本批结果。没有增加完整画像格，仍为128/176。
A5源码/静态/动态交付及A6性能尚未完成。统一实现统计58,965行/233文件，排除测试和第三方源码。
