# G04/G05/G09 性能恢复：有界机制诊断

状态：**CLOSED，仅本轮有界机制诊断，2026-10-17；产品修复及原性能资格未验收**。
SCOPED 的两条单变体预算、AUDITED 的实际入口审查、全部执行与独立复核均已完成，已到停止条件。
文档合同与模块状态检查通过；本卡不是产品模块的 CLOSED 裁决。用户本轮明确要求处理三项性能问题；baseline
`8e04eb0c0`。本卡承接当前 G09 工具阻塞及 G04/G05 旧计时限制，不把历史尝试重新授权，
不创建 goal、不覆盖原失败、不修改正式门限或产品状态。

当前已有 G04/G05 CUDA warm LOSS 是 FP8 驻留/F32 partial 与额外完整 BF16 常驻对照的方案
差异；G09 CUDA 原正式 P1 已通过，Windows Vulkan/Air Apple/Air Vulkan 是 workload A/A
未资格，不能混记为三个产品回归。原完整证据继续有效。

## 有限诊断

1. 离线重算已有完整 individual/direct-debug 日志，量化 replay、poll、park、QueueSubmit 及
   同实现实例差；绑定原件 SHA，不将相关性改成 GPU readiness 或可消除时间。
2. G09 仅在可丢弃 Windows Vulkan TU 中，原 native frontier 已无 ready native job 且只有
   一个 pending prefix 时，释放 pipeline mutex，对同一原 Completion 调用 wait，再回原
   poll/finish/read/退休路径。记录 QPC 前后和结果；不增加线程，不变更数学/槽/4M 输入。
   原 shape0/1 是干预域，shape2/3 路径保持作为负控制；原两个 workers 和 native wait
   释放 runtime mutex 的事实已只读复核。此实验不设计通用产品等待算法。
3. G04/G05 沿当前 D256 direct-stage 的原 NoFill/tail 语义，先检查 K 两次 packed load
   是否被转换依赖串行化。仅准备可丢弃变体与 codegen，寄存器/spill/occupancy 和 load 顺序
   合格后再单独冻结有限 GPU 对照；不先改变 shipping patch 或 old BF16 comparator。

## 冻结成本、验证与停止

- 产品源码/API/layout/registry/公共仪器 FIX 集为空；诊断源码在 scratch，分析保存在
  `evidence/architecture/inference-performance-recovery-2026-10-17/`。
- G09 重用已验静态 SDK、原 oracle/结果租约/drain 与 instrument 的 8 warmups、12 paired
  rounds/exact1；最多四个短进程覆盖 execution/allocation 两因子的全部组合，不取有利子集。
  各进程前后完整 oracle、8 次 State 退休、正常退出、无记录溢出、真实 caller/mask/priority
  与加载模块身份须通过。缺一停止；结果全部为 DIAGNOSTIC，不恢复 A/A/P1。
- G09 先独立审查实际干预与映像归属再运行；不用 Nsight/debugger、不改 overlay/registry，
  只保留原 OBS 禁用环境。Completion.wait 返回只是 native prefix 完成的宿主上界。
- 编译最多 2 jobs，父进程最多 4 核/BelowNormal；重型任务串行。原同机采样 mask/priority
  只用于上述短诊断；不更改用户应用、设备/电源设置或驱动。
- 两条机制各只做上述一个有限变体。若无效果，保反证并停止该方法；不靠扩样、改 workload、
  改精度/估计量/阈值或追加 profiler 组合求通过。
- 有证据的产品修改须另按当前 owner 的合法 SCOPED/AUDITED 与有限 FIX 施工；本卡的
  工具干预不会自动进入产品。先形成机制结论，再决定适用回归和原正式门的恢复入口。
- 本轮运行 selector 的 selected gates，以及 docs/closure/diff。没有产品变更不构建全库。

## 结果

[离线等待分析](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/host-waits-analysis.json)
保留有/无 Nsight 两组原件；同实现实例的 ABBA 组 replay 差与 park 并集差关联很强，
但没有独立 native readiness，不能直接把 park 记成浪费。
[独立源码/产物/入口审查](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g09-native-wait-review.json)
已通过。审查纠正了进入等待前需在当前锁下再次检查 frontier predicate，以及实际目标身份/
八个记录块/负控的验证；首个未审构建未执行，修正后同树增量构建结果另存。
两个构建累计 41.749 秒，仅四个小 TU 及一 TU 的审查修正，无产品/SDK/native 重编译。
G09 新 launcher 先封存实际源码/产物/原映像/复用 helper 与 profile，再执行固定四进程。
G04/G05 的单一 load-ILP 假说与编译边界见
[源码及实际 SASS 分析](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g04-g05-load-ilp-analysis.md)。

### G09：观察策略对完整 replay 的影响已定位

四个短进程累计 11.173 秒全部完成，数学、实际 PID/nonce/策略、原 4+4 oracle、8 次 State
退休、8 块完整记录、无 OBS/Nsight 与正常退出检查通过；没有更换工作量、统计或完成边界。
[独立离线分析](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/native-wait-analysis.json)
及[可重算脚本](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/analysis_native_wait.py)
覆盖全部顺序和负控，不剔除反向组。

| 形状 | 四顺序原路径 replay 中位 ms | 四顺序 native-wait 中位 ms |
|---|---|---|
| 独立 CPU-first | 6.962 / 6.770 / 6.791 / 7.753 | 4.747 / 4.489 / 4.421 / 5.488 |
| 独立 GPU-first | 6.513 / 6.320 / 6.515 / 6.306 | 4.719 / 4.650 / 4.438 / 4.547 |

八个形状/顺序的平均 replay 均减少 1.770–2.257ms，原 ABBA 48 组有 47 组减少；
另 1 组反向结果完整保留。352 组 single-pending guard/wait 全部成功，处理域每 replay
两次等待、frontier timer park 为零；旧路径及两个负控全部零干预记录。负控的差异方向
不一致，没有被包装为稳定提速。

由此可以确认：当前周期观察/唤醒策略对具名完整消费者具有实质成本，不再只有相关性。
该干预会占用一个 Sched worker，不能直接作为通用产品修法；必须保单 worker、多在途、
并发 invocation、错误/取消和真实退休，不能为获得计时收益隐藏资源成本。
正确产品方向是 native 完成通知或借既有有界 BlockingPool 的冷准备 waiter，完成后唤醒
原 fiber，继续原 poll/finish/read/退休。现有 move-only Completion 不允许未经同步的
pool wait/root poll 并发；pool job 被取消也不证明 GPU 已退休。

这是 Windows Vulkan 两个有限形状的因果诊断，不签原 A/A/P1，也不外推 Air 性能；
原三个未资格 profile 和 CUDA 原合格 P1 均不变。

### G04/G05：K-load 假说通过 codegen

单一 full-TU 变体实际编译 291.811 秒。目标 D256 原两次 K 加载之间相隔 55 条指令，
变体两次 `LDG.E.128` 相邻，随后才进入原转换；寄存器 248→250，stack/local/spill 均为零。
转换/矩阵/共享存储指令数量不变，动态 shared 不变；没有借裁剪 bridge 的指令变化作证。
上述是执行前 codegen 检查点；运行结果另在下文记录，不能由生成代码预签产品修复。

实际入口的[独立审查](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g04-load-ilp-execution-review.json)
已通过，协调 owner 据此冻结一次运行：两个 VERIFY_ONLY（各含原三形状和一个仅验证用的
D256 partial tail）、两个 memcheck、一个 treatment racecheck，以及四组换序的八个短
CUDA/NVTX timeline 进程。每个验证进程须有 32 个 oracle、八次 teardown；计时仍仅原三形状、
64 warmups、八组各 64 episodes/context，不增样、不跑正式 P1。两侧完全使用 FP8/F32，
各进程内部两个实例只作自身负控，实施对照在进程之间。入口、实际对象/映像/链接归属、
尾部/只读数值与原 window/holdout/merge 同 SASS 均已复核；任何失败停止并保留。
获审 launcher SHA256 为 `07be8eb852a3b6b5d00be5ebe49b92925f65e4cc3c4d9bb7dcb9aecd3cab1acf`。

固定的 13 个执行步骤全部完成，累计 89.906 秒，无失败或重跑；两个版本的正常验证和
memcheck、处理版 racecheck 全部通过。每个检查含 D256 partial tail、32 个 oracle、
只读 KV 字节检查及八次完整 teardown。八个计时进程每个正常退出、完整前后 oracle/退休，
所有冻结源码/对象/映像前后 SHA 一致。实际 GPU/driver 为 RTX 4070 / 617.14，CUDA13.3/SM89。
没有构建全库或运行正式 P1。

原三个计时形状实际映射了 24,576 个完整 episode、32,768 个计量 GPU kernel；全部 raw
kernel 为 37,184 个，另 4,416 个属于 warmup/oracle。实际目标 prefill 的原 mangled name、grid/block/shared
及 merge 均匹配，目标寄存器如预期为 248→250、local 为零。

| 全部换序组 | prefill attention 原始延迟变化 | 同 SASS window 原始变化 | 同 SASS holdout 原始变化 |
|---|---:|---:|---:|
| 0 | −1.88% | +1.89% | +1.90% |
| 1 | −4.03% | −0.64% | −0.65% |
| 2 | −6.95% | −3.36% | −3.36% |
| 3 | −2.37% | +1.40% | +1.40% |

目标四组均减少 0.799–2.949µs，原始结果是有价值的正向线索；但两个未变 kernel 也共同
漂移，跨进程扰动仍存在，不能将原始百分点全归 load 顺序。未采集运行时频率/功率，
也不把共同漂移签成已证实的时钟原因。完整 host episode 的扰动更大，不以 kernel 差
替代公开消费者性能结论。不扣除负控求“纯收益”，不取有利形状、实例或换序，也不追加
采样。该有限变体到此停止，尚不满足产品采纳或原 attempt14 重判条件。

[独立原件复核](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g04-load-ilp-independent-review.json)
只读重算全部八份 SQLite，要求 API correlation 与 episode 同 host thread 且唯一归属，
逐一核对全部实例的 512 样本、八组均值、中位/均值、四组差值、资源身份、13 个日志与
全部冻结输入。结果与提取器一致；此复核也不签因果纯收益或正式 P1。

## 处置与后续准入

本轮将长时间未收敛的问题拆成具体机制、比较约束与测量资格分别裁决，没有修改阈值、
精度、样本或失败预算。G09 已取得可操作的产品问题依据；G04/G05 取得原 full-TU 的
加载顺序线索及完整运行反证，尚无可签发的同约束工程性能收益。两者不能共用一个
“再跑正式门”的入口。

G09 的最小[产品修复设计](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g09-production-fix-design.md)
已分别审查真实完成 owner 与 Sched/BlockingPool。Apple 可桥接已有主动完成者的窄内部通知；
普通 Vulkan 没有后台 observer，借现有有界 BlockingPool 需要明确的冷注入边界。既有
`subscribe_retirement`/lifetime attachment 不会凭空提供主动完成观察。设计列出单 worker、
多在途、已完成注册竞态、queued/running cancel、shutdown、OOM 和 accepted prefix 真实退休，
不另建隐藏线程池或重写通用 Scheduler。本页与设计均不是产品 AUDITED 施工卡。

当前产品、公开 API、SDK 与 registry 全部不变。G04/G05 原有限 CLOSED、attempt14 的
LOSS/5% NI 未决，以及 G09 原三个未资格 profile 均保留；CUDA 原正式 PASS 保留。
本轮没有恢复正式 P1，也没有将诊断 26%–35% 的有限形状降延迟冒充产品提速交付。

文档合同及相对链接检查通过，模块收口检查通过（64 个 components / schema4），
`git diff --check` 通过。selector 仅选择 docs/closure，保留
其余 146 项 skipped/reused 原因；没有因本轮文档/证据改动重建产品或重跑原 P1。
[完整 selector 结果](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/validation.json)
保留 selected gate 的实际 PASS 和全部跳过/复用原因。

## 可复核原件

- G09 的[源码与原始日志归档](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g09-native-wait-source-and-raw.zip)
  为 4,755,253 bytes，SHA256 `14f911427cabfc0991d7a75ef9ad8978a4b523e7d983955a597a0ae0800272e1`。
- G04/G05 的[完整 receipt](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/g04-load-ilp-receipt.json)
  记录全部有限成本/原件清单；[源码与 raw 归档](../../../../evidence/architecture/inference-performance-recovery-2026-10-17/source-and-raw.zip)
  为 14,699,442 bytes / 603 payload files，SHA256
  `537c50a63c9e9627057f0300a19dacebe4541def84c2f384254e12d7a1a8ae43`。
  含八份 SQLite/原 nsys report、源码/headers/对象归属/SASS、全部日志/脚本与两份独立审查；
  原件逐成员 SHA/CRC 读回通过，归档时 owned-target 进程扫描为空。

G04/G05 编译 291.811 秒、configure 30.934 秒、三次必要 host link 共 11.079 秒、
13 步运行 89.906 秒；G09 两次局部构建共 41.749 秒、四个进程共 11.173 秒。
离线分析/审阅时间没有单独测量，不写成零，也不将上述子步骤相加冒充本轮完整人工耗时。
