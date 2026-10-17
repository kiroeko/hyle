# 比较口径更正：异格式 warm 差额不能判定实现落后

2026-10-17，用户指出双方计时区间内没有完成同一件事。复核确认该质疑成立。
撤回此前把 C6 对 plain BF16 的 +3.20%/+2.59% 称为 Hyle 同工性能落后、以及据此判定
G04/G05 未追平的结论。原始时间、计数器及失败/成功原件全部保留，不改写历史样本。

同 FP8 驻留输入的[公平重测已完成](../same-fp8/README.md)：在两个已通过数值及
并发检查的有限 case 中，C6 prefill/window 普通调用均值分别降低8.22%/6.30%。
参考的一处必要同步修正与失败的辅助 case 单独披露，不外推全域结论。

| 计时边界 | Hyle FP8 路径 | 现有 plain BF16 对照 |
|---|---|---|
| 计时开始前的 KV | 原始 FP8 device-resident pages | 已冷展开、吸收 scale 并上传的 BF16 pages |
| 计时内 | 读 FP8、tile 内解码、attention/merge | 读 BF16、attention/merge |
| 冷展开与上传 | 没有完整 BF16 副本 | 在 warm 计时外完成 |

参考同样要加载数据，确有 page-index、K/V load 和自身等待；它提前完成了表示转换。
39.4% 额外指令包含不同输入表示所需的工作，不能据此证明同输入约束下 Hyle 实现更差。
该对照只说明两种缓存表示的 warm 时间/驻留空间取舍，不能冒充原生 FP8 实现竞速。
当前 C6 对照是同一 Hyle FlashInfer provider 的 plain BF16 分支，也不是外部引擎基线。

当前 C6 peer 的展开由 CPU 完成，随后在冷阶段分别上传 BF16/FP8 页；更早 CUDA 原始
peer 使用 GPU 冷展开并保留 BF16 快照。两者的 warm 区间都不含展开/刷新。不能为弥补
这个问题，又给合法持久 BF16 缓存的每次 warm 调用人为加一次全历史转换。
原[消费者说明](../../../../bench/inference_resident_sdpa/cuda_encoded_p1/consumer-provenance.md)
和[既有适用性更正](../../../../docs/implementation/tasks/compute/inference-g04-g05-cache-attention-domain.md)
已经披露该边界；本轮错误是再次扩大了这些时间差的适用范围。

## 哪些结果仍成立

- C2→C5、C5→C6：相同 FP8 驻留、BF16 partial、输入/输出、完整调用和计时边界；这两项
  自身版本比较及其测试仍成立。C6 普通调用的描述区间跨零仍照实保留。
- C1 的 FP8/F32-partial→FP8/BF16-partial：用户允许的显式精度策略取舍及实际显存变化；
  不能混称为严格相同数值策略的实现竞速。
- 跨表示的计数器、时间和内存原始数值：仍可分析对应方案，不能单边列出加载等待就
  把它认定为全部时间差的根因。

## 重测方式

实现效率比较统一为：相同 device-resident FP8 K/V 字节、Q/metadata、输出类型、缩放和
允许的中间精度，从调用开始到 BF16 输出完成。原生 FP8 参考的读取、tile 转换、attention
及 merge 都在计时内，不能在前置准备中保留完整 BF16 快照。各实现可以使用各自合法的
tiling/planner，不强迫未修改上游使用 Hyle 专用 KV tile。

缓存方案比较另行记录冷启动和真实 append/dirty 刷新，再记录明确次数的复用、驻留显存
与 workspace；不漏转换成本，也不每次虚构全量转换。它回答方案取舍，不冒充前一种比较。

锁定 FlashInfer 上游模板原生支持 BF16 Q / FP8 KV / BF16 output，应优先构建该真实
FP8 参考。已保存的 BF16 peer、裁剪桥和 Hyle patched FP8 四角都不能换标签冒充它。

## 初始加载 PC 标签更正

旧 C5/C6 报告笼统称为初始 K 的部分 PC 实为 V：`0x0d70` 是 V 页地址链，
`0x1140` 消费 `0x0e00` 的 V payload。所列 `0x1180/0x14c0` 才是 K 消费点。
C6 原来的“三个初始 K PC 共163/163”应读为所列 K/V 消费点合计；其中 V 为101/92，
K 为62/71。这不改变原始样本，但撤回单边 K 独占这些等待的归因。

比较适用性的[独立复核](comparability-audit.json)及[逐寄存器标签更正](initial-loading-label-correction.json)
保存在本目录，旧 archive 不重写。后续实际同 FP8 参考构建、数学与计时另列，不将源码
支持类型组合当成已经取得公平性能结果。

[双方加载审计原件](loading-audit.zip)并列了两端实际 metadata/K/V 读取、异步等待和
shared 发布；文件清单与 SHA256 见[清单](archive.json)。该审计只使用现有原件，没有
新增 GPU 样本或编译。它还核验了真实上游 FP8 模板的原始 tar/头身份，未将可编译性
推测当成已经验证的性能结果。
