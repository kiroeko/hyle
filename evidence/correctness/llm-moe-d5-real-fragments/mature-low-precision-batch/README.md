# C10 / ds4 原生低精度批次

baseline `5fabf548565a4598b8b7bf490b2eaabfc942cca1`；
[资格卡](../../../../docs/implementation/tasks/compute/llm-moe-d5-mature-low-precision-batch.md)。
Windows只编译一次CPU原函数调用；MacBook Air M4只编译一次原生调用，原Metal源码在
同一批次验证默认和上游safe开关。没有产品SDK构建、P1或重复旧F45/F46/F47。

| 候选/合同 | 数量 | 数值差异 | 裁决 |
|---|---:|---:|---|
| C10直接E4M3转换/解码 | 66,290 | 0 | 原语资格通过；不代表完整group128线性程序 |
| ds4 CPU FP8 group64 roundtrip | 107,264 | 0 | 原函数资格通过 |
| ds4 CPU FP4 group32 roundtrip | 107,264 | 0 | 原函数资格通过 |
| ds4 Metal FP8 group64（每模式） | 107,264 | 0 | 原函数资格通过；Apple已有合格FP8，不重复引入 |
| ds4 Metal Hadamard+FP4，普通F32融合边界（每模式） | 32,768 | 0 | 仅该融合合同通过 |
| 同一程序，模型Hadamard后BF16边界（每模式） | 32,768 | 107 | 不满足当前模型合同 |
| 同一程序，混合尺度/低域（每模式） | 9,216 | 1,345 | F32融合与模型边界均不满足 |

CPU roundtrip分别覆盖65,280个有限BF16、32,768个普通混合值、9,216个尺度边界值。
CPU各一项、Metal FP8若干项的差异只在数值相等的零符号位，原门用浮点相等，故单独记录，
没有凭新加的bitwise门拒绝候选。C10是encoded code精确比较，包含全部有限code以及中点
F32邻居；其66,290项不应与roundtrip输入量合并。

Apple普通例：index0的融合结果-1.5，保留BF16中间边界的参考为-2；这不是尾数微误差。
低域例：期望负`2^-127`，GPU为零。`MTLMathModeSafe`结果相同；不修改kernel、删除
Hadamard、添加host数学或放宽模型门来签署支持。

## 来源和复现

ds4固定`9139e2ae58a41503968a500f36f75895c1ba63fc`，原MIT许可随证据保留。
C10来自本机PyTorch 2.8.0+cu129，commit `a1cb3cc05d46d198467bebbb6e8fba50a325d4e7`；
CPU只include其原头，无Torch DLL/运行库或GPU初始化。`sources.json`保存完整源与摘录SHA。
`cpu-receipt.json`、`apple-receipt.json`保存命令、退出码、耗时与输入身份；零进程退出码仅表示
批次完成，候选verdict以日志中各项mismatch为准，不把Apple失败写成PASS。

`source/*.gz`包含本批只用于验证的oracle、原函数摘录、Objective-C++宿主和三个驱动脚本；
从Windows同步源码与CPU生成输入到Mac，没有从Mac反向修改产品。默认Metal模式与safe
对应上游`ds4_metal.m`的既有开关，两种之外没有继续搜索编译参数。

## 下一步边界

只有CPU原roundtrip具备完整操作资格；按产品owner另定同批接线、Direct/Graph、资源和交付。
C10不能冒充完整Provider，Apple FP4候选在本预算内终止。176格仍154通过/20拒绝/2失败。
本批产品实现未变，统一统计仍60,373行/241文件，不含测试和第三方。
