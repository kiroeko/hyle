# G04/G05：BF16 中间结果与剩余成本

用户要求继续定位 CUDA prefill 的差距，并明确允许以与对手相当的中间精度换取速度和显存。
治理保持停用。本轮没有采用此前双 K 加载或后续 K/V 预取实验。

## 定位结果

原比较不是两份相同精度、相同驻留的实现：产品直接读 FP8 KV，split-KV 保留 F32
partial；原 FlashInfer 对手使用额外冷展开的 BF16 KV、BF16 partial。旧约 4–5% 的暖态
完整调用差距不能直接解释为 Hyle 的 FP8 解码器慢。

本轮从原完整 `native_cache.cu` TU 构造四角，保留同一 planner、scale 位置与 direct
loader，排除了旧裁剪 bridge 和 BF16 partial 意外回退旧 loader 的混因：

| 路径 | KV 驻留 | partial / merge |
|---|---|---|
| 00 | FP8 | F32 / 两级流水 |
| 01 | FP8 | BF16 / 四级流水 |
| 10 | 额外精确、未缩放 BF16 snapshot | F32 / 两级流水 |
| 11 | 同一 BF16 snapshot | BF16 / 四级流水 |

原样 00 与修改前产品 prefill 的 normalized SASS 逐条相同。每次只预热和交替执行一对
路径，避免六路轮换把活跃数据挤出 L2。原样桥仍显示地址/时段波动，因此又用相同
plan、BatchSpec、Q、metadata、output、workspace、stream 的 native CUDA Graph 对照。
每 graph 串行重复 64 次 attention/merge，整组入队后才等待末 event。

这组同 buffer GPU replay 中，00→01 平均减少 **8.52%**，10→11 减少 **10.29%**。
00→10 仅减少 **0.48%**，同 SASS/同参数的 00→原样桥则相差约 **1.27%**；不能据此把
F32 路径下的小载体差额归给解码。原真实对手的 BF16 中间流水是一个已证实的成本差异。
partial 因子包含存储、attention codegen、merge 类型和 2/4 级流水，不能称为纯字节带宽。
不同 pair 的交互不能强行拼成一份精确加法成本账。

该 prefill 实际有 640 个 partial row states：F32 value 写入 5,242,880 bytes，merge
再读取同量；BF16 各为 2,621,440 bytes，往返减少 5 MiB。21,053,440 bytes 是含 Q/GQA
padding 的工作区需求，不是每次实际流量。双方每次有相同八项资源查询；无 pending
数学工作的 CPU 微测组合约 0.464µs，不是真实 critical path 节省的严格上界，也不支持
把所有 host wait 归因给资源查询。

## 精度与对手

把原抵消残差构造搬到实际 D256/CTA64/paged 域，在两种输入幅值（Vscale 均为 2）、
每种 393,216 个输出中，
01 与真实原对手的 BF16 输出逐位相同。单位尺度的最终真值为 0.001953125，两者均为
0.000644683837891；F32 路径保留正确最终 BF16 舍入。这说明原实现确实保留了对手没有
保留的精度；该有限比较没有发现相对对手的精度下降。它不是完整模型质量评估。

## 产品实现

`allow_bf16_partials` 是 program/Graph descriptor 的显式许可，默认关闭。当前仅在
BF16 Q/output、FP8 KV、page16、D256、causal 无界、实际 CTA64/split-KV 时启用，
由 `BatchPlanReport::bf16_partials` 报告。局部累积仍为 F32，最终 scale 仍在 merge
之后；FP8 KV 不展开。其他域保持原路径。Graph 两端策略必须匹配，改变策略须重新
创建 program/plan/capture。

planner query 与 materialization 共享实际 tile 判断，把本例需求由 21,053,440 降为
10,567,680 bytes，score 仍为 F32。slot 按 caller cap 预留；验证使用 F32 22 MiB 与
BF16 11 MiB，而非只减少有效数据却继续保留大工作区。

## C1 真实产品验证

本节是本轮第一份完整产品候选 C1，不能与上面的实验内核混用。实际完整产品编译后，
默认 F32 prefill 的寄存器由旧对象 248 变为 246，BF16 prefill 也从实验 01 的
2040 条指令变为 2032 条；都无 spill。两种 merge 与各自旧对象的完整指令/控制字相同。
因此本节采用重新链接的公共消费者实测，不继承旧实验的 8.52% 提速。

RTX 4070 SM89 / WSL2 / driver 617.14 / CUDA 13.3 实测：

- Direct、Graph、public prepare_replay、双 slot、旧 output 租约、tail 和实际 CTA16
  fallback 测试通过。17 项 cache/encode/replay 原生测试通过。公共消费者 memcheck、
  synccheck 和新增 feature oracle memcheck 均为零错误。
- 两 slot 的真实 Hyle CUDA resource 增量占用为 47,746,048 → 24,677,376 bytes，
  精确减少 **22 MiB**；不含 driver/private 分配。有效临时工作区需求为
  21,053,440 → 10,567,680 bytes，caller cap 每 slot 为 22 → 11 MiB。
- 非均匀特征 oracle 的最大绝对误差为默认 F32 0.00099498、BF16 0.00109081。
  另外在两种抵消输入幅值 × normal/replay × 双 slot 的每组 393,216 个输出中，
  C1 BF16 与同源原生 BF16 KV 路径逐位相等；默认 F32 全字匹配独立 double 的
  最终 BF16-RNE。该对照使用实际 FP8 字节冷精确展开并吸收 .5/2 scale 的 BF16 KV，
  不是三个外部引擎，也没有做完整模型质量评估。BF16 确实丢失了默认路径保留的残差。

两个普通进程测量均在所有编译、设备检查和 objdump 结束后执行。每矩阵固定
32 blocks × 128 pairs，全部样本和长尾保留；两种分配顺序、两种调用顺序分别记录。
下表为同一 mode 内等样本数合并的完整 public episode 算术均值，不扣负对照漂移：

| 对照 → C1 FP8 KV/BF16 partial | 普通调用 µs | public replay µs |
|---|---:|---:|
| C1 FP8 KV/F32 partial | 86.617936 → 83.961339（−3.07%） | 85.677450 → 82.318662（−3.92%） |
| 同源原生 BF16 KV/BF16 partial | 76.678519 → 80.923404（+5.54%） | 76.057061 → 79.852927（+4.99%） |

第一项 98,304 pairs 包含 prefill/window/holdout，第二项 32,768 pairs 仅目标 prefill。
原始矩阵、p99/max、描述性区间和限制见 [汇总](new-product-results-summary.json) 与
[全部矩阵](new-product-all-matrices.csv)，精度见 [独立复算](product-precision-review.json)。
同进程 block 区间不代表跨进程置信保证。所有八个 prefill 矩阵均显示相对 F32 的改善，
但普通模式有一格仅 0.21%；同源 BF16 KV 对照在全部八格仍更快约 5%。
**显存和自身速度收益已证实，G04/G05 的剩余 peer 差距尚未解决**，继续针对它归因。

首次独立 precision harness 按 logical 而非 capacity 大小读取 output，触发严格
view read-size 拒绝；修正 harness 后 C1 产品无需修改即通过，失败原件仍保留。
GGML 完整目标意外展开无关 vendor 重编，停止后只重编受新 options/report ABI 影响的
host TU 并原子更新归档，其余 member hash 保持。构建、测试和原始身份见 [结果](results.json)。

## 原件

- `direct-factor-diagnostics.zip`：完整 TU 构造、符号隔离、SASS 原样桥、计划、全部
  ordinary / Nsight raw 与独立复算；SHA256
  `677d8d12cf1b123de2a6877540c7881dca08d557712840aba12f33a7c4113066`。
- `supporting-diagnostics.zip`：其余实验原件、独立复算、vendor planner smoke。
- `product-validation.zip`：C1 完整产品构建、公共/原生测试、sanitizer、精度、两项性能及
  codegen 的源码/命令/身份/raw。完整文件 SHA 清单在每份 archive 内的 `FILES.json`。
- 两份 archive 的实际 SHA 和大小见 [archive 清单](archives.json)。
