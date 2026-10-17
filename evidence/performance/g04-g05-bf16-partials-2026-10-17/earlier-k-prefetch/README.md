# C6：QK 后、logits/mask 前预取 K

这是对 [C5 剩余 K 等待](../shared-page-index/README.md) 的一次定向检验，已接入
`hyle-flashinfer-bf16-early-k-prefetch-v15`。相对 C5，replay 完整调用快约1%；普通调用
平均快0.41%但描述区间跨零，不能称已证实稳定收益。只把同一段
双 K raw 预取从 mask 后移到所有 `compute_qk` 分支之后、`logits_transform` 之前，
没有移动到 QK 之前。地址、有效行判据、私有 SM89/BF16 gate、解码/发布、V 和同步
全部保留；独立逆移该代码块后，整个 header 与已验证的 C5 逐字节相同。

C6 header SHA256 为 `4859f99469ee3bf0d5c7445701fb424dcaaba5672216cf76d94b8ce630c0c2a5`，
完整2576文件 shadow inventory 为
`744430acf8f7a998ba4960f85a69e05fcdb8a652fb76c9c1828a66ae4817f7a4`。
产品 C5 的 header 为 `8655514c44249af1c52bfb019447c09eca0617f8475c2097c0e490fcabca3763`。

## 为什么选择此位置

C5 实际 SASS 中，next-page 索引在 `0x2bb0` 等完，K offset 在 `0x2c10` 已计算完成，
边界判断的输入也已经就绪，两个 raw load 直到 `0x3030/0x3040` 才执行。把读取放在
QK 后，旨在利用之后的 logits/mask 和 softmax 运算等待内存，同时避免 payload
跨过 QK fragment 的寄存器峰。源码顺序不能保证机器顺序或资源占用，必须核对 actual
codegen 和普通调用。

机会窗口有明确限制：C5 的 mask 分支执行4848次，但 mask 主体每条只执行96次，
约1.98%。不能把 `0x2c10..0x3030` 的全部静态长度当成每次都有的延迟隐藏窗口。
此次没有重复早期已测慢的 native FP8→half→BF16 转换链，也没有开展参数扫描。

## 实际构建

完整 `native_cache.cu` TU 以原 SM89 参数编译通过，583.167秒，无寄存器上限选项。
actual archive SHA256 为
`bbf3ec34a9aa0d7916ffc3ebb4a340583946ec3ce2bc33a6c0dc2711256ecec2`；
实际 C5 基线未重编，原 archive SHA256
`4b3e081080f3ec5401656793b4d101915f0c43277dfcdc18a4c75552a31cc473` 未变。
除 native_cache 对象外，其余四个 archive members 逐字保持相同。

实际机器码将两次 K load 提前到 `0x29b0/0x29c0`，最后的 QK HMMA 在 `0x2d30`，
即与 QK 尾部计算交叠；不能把 C++ 源码的 post-QK 位置说成机器严格在 QK 后执行。
payload 为 `R136..143`，EX2 为 `0x3230..0x35d0`。固定 CuAssembler 控制字模型下，
首次等待 K 写组在所有 EX2 后的 `0x3b00`。目标 REG250/STACK0/LOCAL0，无 LDL/STL。
默认 F32 attention、F32 merge 和 BF16 merge 的完整128bit指令及控制字与 C5 相同。
上述控制字解释保留逆向格式边界，不单独构成 GPU 耗时证明。

编译器也把发布前 BAR 从 C5 的 `0x3ac0`（EX2后）调至 C6 的 `0x3060`（EX2前）；
源码没有增删 barrier，四次 K shared store 仍位于该 barrier 之后。因此实际变化是
由这次源码移动引起的整体调度变化，不能称机器码只移动了两条 load，或把收益完全
归给单一指令的延迟。

## 正确性和唯一完整调用测量

公共路径、17项原生缓存测试、精度、memcheck、synccheck、feature memcheck全部通过。
24条精度数值记录与 C5 逐字段相同；8组各393216word 的 BF16-partial 输出与同源
plain BF16 peer 逐位相等，默认 F32 全部匹配 double 最终 BF16-RNE。
没有完整模型质量评估，也没有因为调度正确而省略实际 sanitizer。

实际 C5/C6 三 owner 符号闭包、fatbin、undefined 引用及 linkmap 均已验证；实际 binary
SHA256 为 `9381dd85053dde93d7be72694139bb3cebe5b9d19ae67e4b0da355627bc5c3fb`。
运行时两臂均250REG/LOCAL0，同 geometry、每 slot11MiB cap，不同 hoststub。

固定 normal/replay×两分配顺序×两调用顺序，8矩阵各32blocks×128pairs，共32768pairs，
保留全部长尾，未补样、裁剪或扣除漂移。

| 完整调用 | C5 µs | C6 µs | 改变量 | 描述性 block 区间 |
|---|---:|---:|---:|---:|
| normal | 78.105788 | 77.786106 | −0.4093% | [−0.9698%, +0.1512%] |
| replay | 75.749815 | 74.992445 | −0.9998% | [−1.6236%, −0.3761%] |

normal 四组均值变化为 −0.3149/−0.3737/−0.3369/−0.6021%；replay 为
−0.7729/−0.5083/−1.3652/−1.3522%。八组全部同向、replay 区间排零，支持这次
局部接入；normal 区间跨零，不为它追加样本或宣称稳定提速。这些同进程 block 区间
也不是跨进程、跨设备保证。C5 的收益和本轮收益来自不同进程，不相加或相乘宣称
一份新的 C2→C6 精确实测收益。

## 当前直接 peer 对照

之后复用固定 peer host object，仅改链接 actual C6 archive，再执行一次固定32768pair
ordinary；对照仍是冷精确展开实际 FP8 字节并吸收 scale 的同源 plain BF16 KV/BF16-partial。
冷展开不计入 warm 计时，不是外部推理引擎完整模型性能比较。双方 oracle、输入
postverify 和 drain 全部完成 exit0。actual binary SHA256 为
`f16c8baa1becab40e1e93e3571804b66ee047c2f869446ac7026fb62a5b7cf46`。

| 完整调用 | plain BF16 peer µs | C6 FP8 µs | C6 仍慢 | 描述性 block 区间 |
|---|---:|---:|---:|---:|
| normal | 78.072060 | 80.569536 | +3.1989% | [+2.4255%, +3.9723%] |
| replay | 70.934622 | 72.769808 | +2.5872% | [+1.8854%, +3.2889%] |

八个矩阵全部仍为 peer 更快，差额分别约2.497/1.835µs。没有把 C5 的 peer 比率与这里
跨进程相减，归因成 C6 单项收益。G04/G05 的剩余 peer 差距尚未补齐。

## 最后一次硬件计数器定位

ordinary 后复用同一 trace host object，仅链接 C6，再采一次相同最小六指标 NCU：
NVTX/kernel 双过滤、4 logical attention、每臂两个，cache/clock none、application strict
replay，3次完整 application exit0，每 logical launch报告7次 profiler replay。
profile binary 为 `01b83740e2845dd68d2a437cd3381fc74d8441d848fac2046e21dd3f3a5a364d`。

主循环 K 组首次等待 `0x3b00` 的 long-scoreboard 样本仅0/2，第二 payload 首消费
`0x3b90` 为0/0；与 C5 同语义消费点133/124的形状相比，支持等待得到了更多计算覆盖。
跨 profile 的样本数不换算独立节省微秒。初始页索引消费 `0x0d70` 仍142/150，
初始 K 三个解码消费点合计仍163/163，因此不能说所有 K 或 metadata 读取延迟都消失。

FP8 总 long-scoreboard 样本485/493，对照300/322。动态 warp 指令总数3,438,560，
对照2,466,704，仍多39.40%；相对 C5 还增加9696条，但 replay 更快，进一步说明
指令总数和 sample 数都不能直接当总时长。当前仍有额外 FP8 解码/搬运和启动阶段依赖，
并不支持“消掉主循环热点即可完全追平”的结论。
646144条精确定位的四类软件解码指纹，以及 HMMA/MUFU/LDSM/FMUL 数学工作量都未改变；
初始等待和这批执行工作依然存在，尚不能把剩余2.497/1.835µs精确拆成独立单项成本。

## 产品依赖核对

Windows canonical 三个文件按准备前及最终候选的精确 SHA256 应用；随后直接用
canonical manifest 重新材料化2576文件，逐项等于本次 actual compiled shadow。
inventory 仍为 `744430acf8f7a998ba4960f85a69e05fcdb8a652fb76c9c1828a66ae4817f7a4`。
CMake 权威 vendor tree 为
`5f3be4fef46cd725a37032d0a9e6b898fec9bdf81c0516207dfff124fe37e3f3`，与 lock 相同。
NOTICE 明确区分 C++ 源码位置和编译器实际调度，未修改 API 或默认精度。

## 原件

[raw.zip](raw.zip) 保存164个文件：源码/编译、完整机器码与控制字、隔离和运行身份、
六项验证、两项 fixed ordinary、最小 NCU raw、独立复算及 canonical 材料化核验。
清单与每文件 SHA256 见 [archive.json](archive.json)。zip SHA256 为
`a2abb753606eed75bf438b43912e845266269bf1f6b8d6aed48b977250900aec`。
大可执行文件、对象和自包含 `.ncu-rep` 留在本机，清单保存本机路径/字节数/SHA256。

scratch exact-patch 准备最初因3行 context 在 paged/ragged 两处重复而被 guard 拒绝，
改为8行唯一 context 后材料化通过；不是产品执行失败。NOTICE 的最终文案由保存的
generator 与原 v14 NOTICE 在内存重构后逐字相等，source-result 记录了此复现和实际
编译 header 绑定，不用准备中的状态替代实际构建或测试。
