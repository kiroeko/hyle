# C5：共享页索引与双 K 预取

本次在 [C4 双 K 预取](../k-pair-prefetch/README.md) 上，去除 main-loop K/V 对同一
不可变页索引的重复读取，检验 C4 提前出现的组合等待。C5 已接入
`hyle-flashinfer-bf16-kv-page-index-v14`：实际完整调用相对 C2 约快1%，八个矩阵全部同向。
该收益属于 C5 整体，不能从跨进程 C4 的无稳定收益结果中相减，宣称为页索引的纯因子收益。

后续将预取继续提前的 [C6/v15](../earlier-k-prefetch/README.md) 已接入；本页保留 C5
这份源码和测量，C6 的新结果单列，不改写或拼接以下原始数值。

## 源码与机器码

范围沿用 C4：私有 BF16-partial marker、SM89、paged/causal、D256、CTA64、
KV-MMA1/Q-KVwarps4/1。有效页只执行一次 `__ldg(indices + guarded_page_iter)`，
再用 K 和 V 各自的 stride 计算 offset；保留原 IdType 转换、边界判断和无效页零 offset。
初始页索引、有效行/NoFill、同步、数值运算和 scale 位置未改变。该局部改动并不假定
K/V stride 相等，也不修改调用方不得并发改写页索引的要求。

完整 `native_cache.cu` TU 使用原 SM89 编译参数通过，585.154秒，无寄存器上限选项。
实际 archive SHA256 为 `4b3e081080f3ec5401656793b4d101915f0c43277dfcdc18a4c75552a31cc473`，
header 为 `8655514c44249af1c52bfb019447c09eca0617f8475c2097c0e490fcabca3763`。
其余 archive member 与 C2 保持逐字相同，C2 对照未重编。

actual SASS 的 main-loop 页索引只在 `0x2680` 读取一次；两段 K raw load 位于
`0x3030/0x3040`，早于全部 EX2 `0x3150..0x3580`，payload 保留到原发布阶段。
目标2024条指令、250个寄存器、STACK0/LOCAL0，无 local spill；默认 F32 attention、
F32 merge 和 BF16 merge 的完整128bit指令与控制字均与 C2 逐字相等。

沿用 C4 已固定来源并做过完整 roundtrip 的 CuAssembler 控制字模型，页索引写组为 W3，
两段 K 写组为 W5，K 组首次 wait 位于全部 EX2 之后的 `0x3ae0`。C4 中途 `0x3270`
等同一组的静态形状已经消失。这是实际机器码上的格式核验和依赖推断，保留逆向模型
的边界；它不能替代 GPU 硬件等待采样或普通完整调用结果。

## 正确性和完整调用

实际公共路径、17项原生缓存测试、精度对照、memcheck、synccheck、feature memcheck
全部通过。24条精度数值记录与 C2 逐字段相同；8组各393216word 的 BF16 partial 输出
与同源 plain BF16 peer 逐位相等，默认 F32 全部匹配 double 的最终 BF16-RNE。
这些仍是有限算子测试，未宣称完整模型质量相等。

从 C2/C5 实际三个 owner 构造独立符号闭包，核对 weak/COMDAT/内部引用、fatbin逐字不变、
linkmap未误取原始三个 FI members。实际 binary SHA256 为
`c01aacb9cc362778b1d280879d048095825115aa442a8a10118f15516db61476`。
运行时确认不同 hoststub、寄存器246/250、同 geometry 和每 slot11MiB workspace cap。

唯一固定 ordinary：normal/public-replay×两分配顺序×两调用顺序，8矩阵各
32blocks×128pairs，总32768pairs，保留全部长尾，无追加样本、裁剪或漂移扣除。

| 完整调用 | C2 µs | C5 µs | 改变量 | 描述性 block 区间 |
|---|---:|---:|---:|---:|
| normal | 77.510923 | 76.782557 | −0.9397% | [−1.6030%, −0.2764%] |
| replay | 75.275248 | 74.465220 | −1.0761% | [−1.6567%, −0.4955%] |

normal 四矩阵变化为 −0.8988/−1.1167/−0.6403/−1.1018%；replay 为
−0.7483/−2.1470/−0.4628/−0.9359%。各 mode128blocks 中 C5 较快的 block 为84/78。
本批全部矩阵同向、描述区间未跨零，支持采用该局部优化；一个进程的区间不构成
跨进程或跨设备保证。没有重编 C2 基线，也没有将 profile 时长当普通调用收益。

Windows canonical 的 manifest、NOTICE 和依赖 lock 三个文件已按准备前及候选的精确
SHA256 核对后应用。候选实际 materialize 的2576文件逐项等于本次编译 shadow，只有
prefill header 相对 v11 变化；CMake 权威 vendor tree hash 为
`7b4e58dc09affe492d2d7db8d9975510a6a9d464678e4570b5d27612bf7ab808`。
未修改 API、默认精度或非 SM89 调度路径。

canonical 接入后再次直接运行其 manifest 材料化与 CMake hash 函数：2576个文件逐项
等于 actual compiled shadow，inventory 为
`e946873e3a64cf4957fc708bdaee7997ea078c95a7e7538a117bce9d1f812841`，lock 匹配。
只验证本次依赖，不运行已停用的治理门禁。

## 剩余 peer 差距

随后复用固定产品对照 host object，只把产品 archive 改链接 C5，另做一次独立的
32768pair 固定 ordinary，直接测量同源 plain BF16 KV/BF16-partial 对照与 C5。
对照冷精确展开实际 FP8 字节并吸收 scale；冷展开成本不进入 warm 计时，仍不是
外部三个推理引擎的完整端到端比较。双方 oracle、readonly-input/postverify 和 drain
全部完成 exit0；binary 为
`76dcaa60e609cd121b9998fb6957230489e70b92335db72b671f6c65204b2ce0`。

| 完整调用 | plain BF16 peer µs | C5 FP8 µs | C5 仍慢 | 描述性 block 区间 |
|---|---:|---:|---:|---:|
| normal | 77.487239 | 80.417502 | +3.7816% | [+3.2499%, +4.3133%] |
| replay | 68.214435 | 71.226120 | +4.4150% | [+3.8545%, +4.9755%] |

八个矩阵均为 peer 更快。因此不能宣称 G04/G05 已追平。也不能用这份与 C1/C2 不同时段
测量的比率相减，构造所谓纯解码节省；C5 的约1%收益由上一节同进程 C2/C5 直接比较证明。

## 剩余执行工作和等待

在上述 ordinary 完成后，用保留的同一份小型 trace host object 只链接 C5，进行一次
最小六指标 NCU 采样：相同 NVTX 范围、4个 logical attention、每臂两个，不采 merge，
cache/clock control 均为 none，application strict replay。实际3次 application 完整
exit0，每 logical launch报告7个 profiler replay passes。profile仅诊断，不替代普通计时。

C5 的每个 FP8 attention 动态 warp 指令为3,428,864，对照为2,466,704，仍多39.01%。
相较 C2 的3,476,384减少47,520条；指令计数是实际动态计数，不能按比例换算耗时。
HMMA、MUFU、FMUL 和 LDSM 动态计数未变，646144条精确定位的四类解码指纹也全部保留；
这次优化调度和页索引工作，没有把 FP8 解码指令本身消掉。
总 long-scoreboard 样本为579/533，对照310/327。

新的 main K 组首次消费 `0x3ae0 PRMT R136,R136,...` 仍有133/124个 long-scoreboard 样本，
第二段 payload 消费点为0；组合等待已经在第一消费点执行，不能将第二点为0理解为
第二次 load 完全无延迟。初始页索引消费 `0x0d70` 还有128/140个样本，初始 K 首消费
`0x1140` 还有73/65个。新共享页索引和 V 地址消费已不是 C4 的中途热点。

这支持“重复索引和过早组合等待已改善，尚有未隐藏 K 读取和额外解码/搬运工作”，
不支持“只消除一个热点就能按样本占比提速”或“所有剩余差距都已精确归给某个指令”。
不同 profile 的原始样本数也不直接构成可相减的微秒账。

## 原件

[raw.zip](raw.zip) 保存161个文件，包括实际源码、编译、全部机器码、隔离与运行身份、
测试/完整调用/NCU raw、独立复算和 canonical 材料化核验；逐文件 SHA256 见
[archive.json](archive.json)。大可执行文件、对象、archive 和自包含 `.ncu-rep` 继续
保存在本机，清单记录其路径、字节数和 SHA256。zip SHA256 为
`29df4419b51fedd56c104597f792956b3ac1f5391f8e92213005b674a6b35d9d`。

scratch vendor 准备脚本第一次因沿用旧 header hash 的标签替换错误被 guard 拒绝；
改为读取 actual C5 编译身份后重新准备通过，失败原件也保留。canonical 接入的
`git apply --check` 因换行 context 拒绝，未发生部分应用；随后仅写入严格核对前后
SHA256 的三个 Windows 候选文件，并完成上面的 actual canonical 全文件核验。
