# C2 的硬件计数器与逐指令等待定位

**比较边界更正：** plain BF16 与 FP8 在计时内承担不同表示转换工作，不能把以下
动态计数差直接解释为同工实现劣势。双方都实际加载数据，见[更正](../comparison-boundary-correction/README.md)。

Windows 管理员令牌本身未解除 WSL 的 `ERR_NVGPUCTRPERM`。用户在 NVIDIA App 中手动
允许 GPU 性能计数器访问后，实际 metrics query 和两次有限 NCU 采样均成功。原拒绝记录
保留；退出码为零但 stdout 含权限错误的旧 query 不计为成功。

本次测量直接使用已验证的 C2 产品消费者，没有重新编译基线或引入新内核变体。
设备为 RTX 4070 / SM89 / driver617.14，工具为 Nsight Compute2026.2.1 / CUDA13.3。
binary SHA256 为 `efce3f63a8e6ccda7ca476091774c13bb4582e2cd5a31e4a236b81e8c1e2ba48`。

两次都仅选择 `capture=0 alloc=0 order=0 shape=prefill` 的前四个 warm attention：
plain BF16 KV slot0、FP8/BF16-partial slot0、plain slot1、FP8 slot1；grid40×1×2，
block32×4×1，dynamic shared49152B。NVTX 祖先范围和 kernel-name 双过滤排除了 merge。
application replay 使用 strict/matchall，cache-control/clock-control 均为 none。
七个总体 section 采样用时157.892秒，实际18次 application 启动，四个 logical result
各报告22次 profiler replay；两种计数口径不同。随后四项逐PC采样用时17.629秒，实际两次
application 启动，各 logical result 报告6次 profiler replay。全部 application 完整执行
oracle 和 drain 并 exit0。profile 的干预时长不代替前轮 ordinary 完整调用的性能主判。

## 已确认的执行差异

| 每个 attention 的动态 warp 指令 | plain BF16 KV | C2 FP8 KV | 差额 |
|---|---:|---:|---:|
| 全部执行指令 | 2,466,704 | 3,476,384 | +1,009,680（+40.93%） |
| HMMA | 315,120 | 315,120 | 0 |
| MUFU | 90,400 | 90,400 | 0 |
| LDSM | 232,704 | 232,704 | 0 |
| LOP3 | 234,576 | 677,968 | +443,392 |
| IMAD | 228,592 | 470,432 | +241,840 |
| HFMA2 | 0 | 161,536 | +161,536 |
| PRMT | 0 | 80,768 | +80,768 |

两 slot 各自的上述动态计数完全重复，逐PC求和与硬件导出总数一致。额外整数操作包含
解码位域拼装，不能全称为重复地址计算。BF16 对照使用 async global→shared 搬运，FP8
则需要 global→register、解码、显式 shared store；不只是读取字节宽度不同。

寄存器238/246分别实际分配240/248，两边寄存器和 shared 限制均为2CTA/SM，wave均0.87，
local/shared spill均为零。没有新增 occupancy 降档证据。实际 active occupancy 约14%，
FP8 eligible warp 和 issue utilization 反而略高；不能说整个 scheduler 更差。

L2 hit rate均约80–83%，FP8 DRAM 流量更低，未见带宽饱和。FP8 long-scoreboard/issued
ratio约2.08，对照约1.22；math-pipe throttle 则从4.63降为约2.81。不能只看一个 stall
百分比，也不能把这些不同归一口径拼成一份精确微秒成本账。

## 逐PC定位

第二次采样提供真实、非零的 PC-correlated warp samples，逐表求和均与 raw 完全一致。
四个结果的总样本数依次3495、4544、3514、4408，无 drop/overflow。

FP8 main-loop 两段 K 的依赖链为相对函数起点的：

- `0x3af0 LDG.E.128 R4,[R72]` → `0x3b20 PRMT R4,R4,...`。
- `0x3e60 LDG.E.128 R4,[R72+0x80]` → `0x3ef0 PRMT R4,R4,...`。

两个首消费点在 slot0 有304+312=616个 long-scoreboard-not-issued 样本，占1152的
53.47%；slot1为258+291=549，占1004的54.68%。这是等待 K global load 返回值，不能
称为 PRMT 指令本身的执行延迟。定义明确的 main V load→decode 窗口 `0x4460..0x4cb0`
两 slot 此类样本均为零。初始 K 和页索引消费还存在其他等待，未被忽略。

由此可以确认当前差距包含额外解码/搬运执行工作，以及未隐藏的两段 K load→decode
依赖。样本占比不是总时长占比，不承诺消除这些样本就有同等加速。C2 数据也不能独自
证明 C3 未稳定提速的具体原因。下一项验证据此将两段 K 读取都提前到 softmax 前；
是否采用仍取决于实际 codegen、正确性和 ordinary 完整调用结果。

该检验已完成：[C4双K预取结果](../k-pair-prefetch/README.md)。两个原解码消费热点在
新样本中归零，但完整调用没有稳定收益，C4/v13未采用；不能将上述C2热点等同全部根因。

## 原件

[raw.zip](raw.zip) 保存两次采样的 CSV、逐PC SASS、命令、身份、完整日志与独立复算；
[archive.json](archive.json) 列出66个文件的逐项 SHA256。两个自包含 `.ncu-rep` 大文件
继续保存在 `.scratch/g04-bf16-ncu-{enabled,source}/`，其本机路径、字节数和 SHA256 也在
清单中，未将重复内核映像加入 Git。前轮 C1/C2/C3 的普通测量原件没有改写。

计数器权限与 WSL 说明见 [NVIDIA 官方说明](https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters)
和 [Nsight Compute 系统要求](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)。
