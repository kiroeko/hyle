# C4 双 K 预取：正确，但未证实完整调用收益

C4/v13 未单独采用，完成本实验时 canonical 仍为 C2/v11。它是对[计数器热点](../hardware-counters/README.md)
的实测检验，不改写前轮 C1 的显存或普通性能结果。

后续加入共享页索引的 [C5/v14](../shared-page-index/README.md) 已测得约1%完整调用收益并接入；
不能把它的收益与本轮跨进程相减，归为页索引操作的精确单项成本。

## 改动与实际机器码

仅在 SM89、私有 BF16-partial marker、paged/causal、D256、CTA64、KV-MMA1/Q-KVwarps4/1
域，将下一 K tile 的两段16B raw global load 移到 online softmax 前；原 vec_cast、shared
swizzle、有效行/NoFill 判定、初始 K/V、同步点和 scale 位置保留。与 C3 首段预取和旧的
producer 内两段预读不同，本次两段 payload 都真实跨过 softmax。

完整 native_cache TU 仅 SM89 编译通过，583.57秒；其余 archive members 保持逐字相同，
没有重编 C2 baseline kernel。实际两次 LDG 在 `0x3040/0x3050`，EX2 在 `0x3150..0x35a0`，
payload `R136..143` 直到 `0x3b30/0x3bc0` 首次消费都未被覆盖。解码/共享发布仍在原
同步点后。目标2024条指令、250个寄存器、STACK0/LOCAL0，无 spill；默认 F32 attention、
F32 merge、BF16 merge 的完整128bit指令及控制字与 C2 逐字相等。

公共路径、17项原生缓存测试、精度对照、memcheck、synccheck、feature memcheck全部通过。
24条精度数值记录与 C2 逐字段相同；8组各393216word 的 opt-in 输出与同源 BF16 peer
逐位相等，默认 F32 仍与 double 最终 BF16-RNE 逐位相等。这些是有限算子用例，不是
完整模型质量评估。

## 唯一普通采样

从实际 C2/C4 provider、native_batch、native_cache 三个 owner 构造独立符号闭包，核验
全部 weak/COMDAT、内部引用链、fatbin逐字不变、linkmap未误取原始三个 FI members。
运行时实际寄存器246/250、不同 hoststub、同 geometry 和 workspace caps均验证。

两 slot/每 slot11MiB、normal/public-replay、两分配顺序、两调用顺序，固定8矩阵各
32blocks×128pairs，共32768pairs；所有长尾保留，没有补样、裁剪或漂移扣除。

| 完整调用 | C2 µs | C4 µs | 改变量 | 描述性 block 区间 |
|---|---:|---:|---:|---:|
| normal | 78.182284 | 78.166043 | −0.0208% | [−0.5824%, +0.5409%] |
| replay | 73.681490 | 73.551782 | −0.1760% | [−0.8777%, +0.5256%] |

normal 四矩阵变化为 +0.2407/−0.2621/−0.2760/+0.1987%；replay 为
+0.9884/−1.3602/−0.2810/−0.0295%。区间跨零、方向混合，不支持稳定收益，也不能把
一个进程的 block 区间当跨进程置信保证。因此没有应用准备好的 v13 接入补丁。

## 机制复查与边界

普通采样结束后，用同一份 C2 peer-trace host object 仅改链接实际 C4 archive，进行一次
最小 NCU 诊断：同 normal NVTX 范围、每臂两个 attention，不采 merge，cache/clock none，
application strict replay。3次 application 完整 exit0，report 每 logical launch7次 replay。

原 main K 两次首解码消费者对应的 `0x3b30/0x3bc0` long-scoreboard 样本两 slot 均为零。
完整调用仍无稳定收益，因此不能把 C2 原热点占比当可节省的总耗时。

本次 FP8 总动态指令为3,453,104，仍比同 binary plain BF16 对照2,466,704多39.99%，
仅比 C2 少23,280条。总 long-scoreboard 样本为673/711，对照293/318。新的主要采样
位置为 `0x3270` 的 V 页地址乘法，248/280个样本；其 R169 来源为 `0x26c0` 的页索引
load，乘数为独立的 `v_stride_page`。K/V 对相同受保护页索引的重复 global load 在
C2 和 C4 均实际存在，不是本轮 asm 创造出来的。

这些数据确证指令归属和重复读取；单凭 operand def-use 还不能独占归因该 PC 的全部
硬件等待，scoreboard 可能还包含其他在途 load。解码前 BAR 的全部状态样本为0，后继
位置样本没有大增；未采 barrier 专项原因，不能把此写成 barrier 等待数或严格排除。
本报告不对各成本作精确微秒分摊。后续将以共享一次 page-index 读取、分别保留 K/V
stride 的方案检验这个真实重复工作，仍以实际 codegen/数值/普通采样判断。

另有一项有来源的静态推断：[CuAssembler固定版本的原始控制字实现](https://github.com/cloudcores/CuAssembler/blob/96a9f72baf00f40b9b299653fcef8d3e2b4a3d49/CuAsm/CuControlCode.py)
把 C4 的页索引 load 和两次 K load 均解码为 W5，`0x3270` 解码为 wait-mask0x20；
中间没有 wait5。该原实现的 split/merge/text roundtrip 对 C2 的2032条、C4的2024条
完整128bit均相等。结合 [Ampere依赖计数器实验研究](https://arxiv.org/html/2503.20481v1#S4)，
它支持“提前的组合等待可能截短 K 预取窗口”这一候选机制，而非“全部等待只来自页索引”。
这只是逆向格式核验和静态推断，不是 NVIDIA 公开 ABI 或新的 GPU 语义实验。官方
nvdisasm 的 JSON/hex 本身未暴露解码后的 dependency-barrier 字段；原件保留该边界。

[raw.zip](raw.zip) 和 [archive.json](archive.json) 包含源码、编译/隔离/机器码核验、全部
数值和普通测量、NCU/独立复算；自包含大 `.ncu-rep` 留在本机，清单保留其路径与 SHA256。
