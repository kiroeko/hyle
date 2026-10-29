# 性能测量与慢项调查

当前 Qwen CUDA 的基线、热点假设与推进顺序见[decode 性能提升计划](decode.md)，
基础模块与外部库的实测对照见[模块性能数据](modules.md)。
本页维护各模块共用的测量和归因方法。

先保证数值、所有权、并发和资源限制正确，再比较速度。结论限定于实际版本、输入、硬件、
工具链与调用方式；微基准用于归因，真实消费者用于确认收益。稳定的小幅落后也要保留并调查，
实用容忍值不能把已测清的落后改名为持平。

## 推理对比固定报告

每轮优化比较固定给出下列四项，双方使用相同工作负载和完成边界；未测项写明未测，不能借用不同镜像或历史轮次的数据补齐。

| 指标 | 报告内容 |
| --- | --- |
| Prefill | 首轮输入和缓存后新增输入分别报token数、完整阶段耗时及tokens/s；多轮逐轮列出，不将新增输入数写成累计上下文数 |
| Decode | 生成数、完整decode耗时及tokens/s，保留首次实际capture／准备；稳态作为补充，不能替代完整值 |
| 显存变化 | 加载后、各轮prefill／decode结束、释放后的占用及全程峰值；给相对加载后／阶段开始的增量。分别列GPU dedicated／shared及采样间隔；采样峰是下界，缺失值保留null，不将reserved／预算当物理占用 |
| 综合速度 | 固定整段多轮会话，沿同一KV／recurrent缓存增长，报告总耗时和有效生成吞吐：`各轮实际交付的生成token总数 / 会话总耗时`。会话包含全部新增输入prefill、decode、实际状态更新、传输、prepare、等待及结果可见；不含人工用户思考间隔 |

综合速度按实际连续wall clock测量，不平均各阶段tokens/s，也不把累计上下文重复计入处理token数。
可另报`(新增输入token＋生成token) / 会话耗时`，但必须标为总处理吞吐，不能替代有效生成吞吐。
工作负载锁定每轮新增输入／输出长度、输入带、上下文容量及生成／MTP策略；逐轮列累计可见上下文。
固定输入带的teacher-forced性能回放须明确标注，不冒充自由生成或模型质量验收。

加载、冷图编译与ready耗时另列；同时保留进程启动到最后token可见的总耗时，以便观察冷启动。
驻留会话从接收首轮输入到最后token可见；首次实际capture仍在会话内，不另做未计时的虚假预热。
仅修内部机制的有限诊断可以不重跑全套，但须标为诊断；正式速度对比缺少任一项时不称完整对比。
显存旁采可与无profiler测速分开运行同一镜像和负载，标明是独立进程，避免把采样成本混入速度。
每进程先计算会话吞吐，再汇总独立进程的中位数、范围和样本数；`sum(step_seconds)`只称计算阶段耗时之和，不能冒充包含阶段间开销的会话wall time。
GPU dedicated＋shared仅可在同一采样时刻相加，不能相加两项各自峰值；CPU驻留集和private commit另列。

## 对齐工作量

先确定比较的是用户自然使用的完整路线、相同语义下的实现，还是一个内部机制。
沿双方源码确认操作发生在计时前、计时内还是计时后：

| 边界 | 需要对齐或分别报告 |
|---|---|
| 输入与结果 | 权重版本、形状、分布、dtype、量化与 scale、容差 |
| 执行 | backend、ISA、batch、线程、并发、布局、别名、buffer 与状态复用 |
| 准备 | 加载、解码、repack、上传、prepare/compile、分配与 warm-up |
| 完成 | 结果可见、wait、错误、状态提交、释放与延迟回收 |
| 资源 | 实际分配、驻留、峰值、workspace、辅助线程、搬运与能耗 |

“各调用一次 run”不保证同工。GPU 主机提交时间不能当作推理完成时间；设备 kernel 时间与
主机提交到同步完成的整次调用可以分别测。补齐语义所需 adapter 的成本也要计入。
把 poll/yield 改成 native wait 可能减少唤醒延迟，同时占用一个 scheduler worker；这不是免费 overlap。
请求取消或对外完成也不等于设备工作已退休，计时和资源释放都要到达真实的终结边界。

推理报告的`draft_rows`保留发布账簿语义；`actual_draft_rows`的范围为`admitted_active_input_rows`：
原Graph成功返回Completion时计入活跃draft输入，含后续丢弃／失败与私有proposal、repair及恢复。
`actual_restored_draft_rows`是其中的恢复分类，不再加到总量上。padding、provider内部准备、未返回Completion的部分失败及硬件FLOPs不在此数中。

例如一方提前将 FP8 KV 展开成 BF16，另一方在计时内解码 FP8，这回答的是两种缓存方案的成本。
比较同输入实现时，应从相同驻留 FP8 数据开始，冷准备另列。不能只把计时器放在同名函数两侧。

## 采样与统计

1. 先检查输出正确性，再测 Release 产物。记录参考修补，数值失败的用例不参与速度排名。
2. 冷启动、热调用和完整生命周期分别报告。用被测 workload 本身做同路径 A/A，交换 A/B、B/A 顺序，
   检查设备、allocator、频率与缓存的热状态是否已经偏向某一侧。
3. 优先成对交错采样，保存所有样本与失败。独立进程、独立 block 和进程内重复是不同层次，
   不能把相关内层重复当成独立样本来缩窄区间。
4. 分别报告顺序、分配位置、布局与规模的结果，再按明确方法汇总。不要选最快 replica 或剔除不利尾部。
5. 同时给绝对值、比值、区间、样本数和必要的尾延迟；按需要增加样本，先修偏差再处理噪声。

可以使用 `scripts/bench-stat.py` 和 `scripts/hyle_perf/` 的配对统计、日志解析函数；参数见工具 `--help`。
使用 nanobench 同进程双路线时，`Bench::compare()` 交错 alternatives；先完整 `run()` 一侧再运行另一侧
不能称为配对测量。解析按表头找列，避免 relative/CI 新列改变含义。

建议令比值大于 1 始终有利于 HYLE：吞吐用 `Hyle/peer`，耗时用 `peer/Hyle`。
区间全高于 1 支持更快，全低于 1 支持更慢；跨 1 或换顺序/位置后反转时，当前还不能裁决。
实用容忍度 `m` 的等价带可写成 `[1/(1+m), 1+m]`；区间落入带内才支持等价，
“未发现显著差异”本身不证明等价。单边非劣也不等于持平。

## 分辨率与极短循环

用相同 batch、完成边界测 timer floor 或空 harness，再以可控 load、依赖、shuffle、store 等已知增量
检查仪器能否响应目标差异。极短 CPU 操作检查最终指令与代码放置；GPU 说明事件、stream、同步及 profiler 开销。
过短的操作改成保真的聚合任务，不能用更多样本掩盖计时器或代码布局偏差。

SIMD 特别注意：

- 对齐 lane type、vector width/scalable tag、mask/tail、alignment、alias/in-place、NaN/Inf、subnormal、饱和与舍入。
- 双方使用相同输入、load/store、tail、防 DCE、初始化与 checksum 依赖，不让一侧多计 setup 或 dispatch。
- 在相同 executable/TU/ISA 内设置最终指令相同的 A/A anchor，交换顺序和位置；检查多个等价布局的完整分布。
- 保存相关 release binary/hash 与 loop 反汇编，核对 DCE、常量折叠、spill、inlining、dispatch 和链接跳转。
- alias/retag/construction 主要验证语义与零额外代码，不为它们强造纳秒循环。模拟器时序不能代表真实 ISA。

已有 op-parity 输出可用 `python scripts/op-parity-report.py --check-orders` 检查输入与顺序。
placement harness 的 `stagger/mark` 只调整代码放置，不单独证明校准或公平。

### HTTP/2 配对测量

Linux 的 `hyle_io_http2_p1_bench` 默认保留手工单 loop 参考；`--paired` 让双方共用同一
Scheduler、单 worker、双 endpoint Task、系统分配器、DATA 的 NO_COPY→owned wire 路线及 TCP
immediate 接口。Task 参考也执行 TCP 有效性/owner 检查和 typed 结果处理，避免裸 libc 路径省略这层成本。
双方均计入新建 socket/session、完整 payload 校验和销毁；固定 trace 参考仍不承担公共 API 的
session-owner、通用借用、取消、期限和 flush 合同，不能将差值叫作纯封装或调度成本。

已有该 benchmark 的 Release 构建后，在 Linux 运行（结果目录须不存在）：

```sh
python3 bench/io_http2/run_paired.py /absolute/path/hyle_io_http2_p1_bench \
  --output /local/path/http2-results --cpu 2
```

脚本只用 Python 标准库与 `taskset`，CPU 须在当前允许的 affinity 中；省略 `--cpu` 时优先选 2，
否则选最小允许编号。固定执行 A/A→三轮配对→A/A。每进程每臂预热 16 次，随后 12 个
ABBA/BAAB 区块，每格精确执行 64 个完整的 4×256 KiB 请求/响应 episode。
日志输出、socket 容量读回和 rusage 观察位于计时格外；所有原始格与失败均保留，不筛慢点。

每区块比较两臂的几何均值，汇总 12 个区块的中位数、几何均值、范围与描述性 IQR。
本脚本使用 `100×(Hyle耗时/参考耗时−1)`，**正数为 Hyle 较慢**。64 次内层重复不是独立样本；
IQR、A/A 范围和最小时钟刻度都不是置信区间或校准误差。A/A 用于观察整套方法的波动，
不能直接从 A/B 中扣除，也不证明双方等价。WSL pin 的是 guest CPU，不能据此声称宿主核心隔离。

### HTTP2 排查案例

2026-10-18，这次排查定位了双 Task 固定负载中约 1–2% 差距的重要来源：参考直接调用 libc，
Hyle 则经过 TCP 层的 socket/worker 检查及 typed 结果处理。**主要来源已定位，端到端优化未完成。**

排查中的关键转折：

1. 先拆比较边界。单 loop 的历史 8–11% 差距与双 Task 的 1–2% 差距属于不同对照；
   对齐 Scheduler、分配器与 DATA 拷贝后，底层 TCP 职责仍可能不同。
2. 删掉确认的冗余不等于证明提速。通知、分配、resume 等工作减少后，总耗时仍不稳定；
   owner 查询消融也没有稳定收益，不能因此删产品检查或认定 TLS 查询是主因。
3. 用阶段 CPU 与工作计数缩小范围。创建和最终销毁净差很小；复制、内容校验、ACK、yield、
   native 收发和 libc 返回大小分布相同。相同次数仍不代表调用链的职责、指令成本相同。
   libc 计数不含 io_uring 直接完成的 I/O；内联拷贝与 libc memcpy 的样本归属也不同。
4. 机器码只提供假设。私有 read_body 确有 64 B 栈传参区，但同 ELF 的按值/引用对照未证提速，
   候选撤回。功能测试通过也不能替代性能证据。
5. 用单变量干预确认来源。同一 raw Task 仅换成现成 TCP 接口，工作计数不变，成本增加约 2.4%；
   永久对齐后，Hyle 三轮区块中位差为 −0.92%/−1.49%/−0.43%。仍有 14/36 个区块为正，
   不能称稳定胜出、严格等价，或把不同进程的中位数相减作精确分摊。

实际使用与归因实验回答的是两个问题：

| 比较 | 回答的问题 |
|---|---|
| 双方共用 TCP 层 | HTTP2 层在相同底层服务上增加多少成本；用于归因 |
| 双方各用自己的完整网络栈 | 实际吞吐、延迟、CPU 与资源成本；用于产品性能排名 |

实际推理服务会走 `Hyle HTTP2 → Hyle TCP → 操作系统`，TCP 成本确实要支付。
最终排名应让对手使用其自然网络栈，完整计入双方准备、收发、错误处理与完成边界；
不能强迫对手接 Hyle TCP，再把对齐后的结果算作 Hyle 已提速。
本轮只修正基准与归因，库实现没有因此变快；也未证明 TCP 层的全部成本不可优化。
若继续消除裸 libc 对照下的真实差距，仍需在保留语义的前提下定位并优化 TCP 路径。

约 1–2% 是 AF_UNIX、小 socket buffer、4 流/2 MiB 强背压固定点的结果，不能直接套用到真实推理服务。
真实比较还要明确长连接复用、并发、TCP/TLS、流式响应粒度及推理计算占比，分别观察吞吐、延迟和 CPU，
不由本次微基准推导模型 tokens/s。

[阶段归因与未成立的假设](../../evidence/performance/library-review-2026-10-18/http2/phase-costs/summary.json)、
[受控干预、最终结果与原始数据入口](../../evidence/performance/library-review-2026-10-18/http2/tcp-transport/summary.json)。

## 环境与解释限制

运行前读取实际 CPU/GPU、内存、OS/kernel、driver、编译器/SDK、ISA、优化与链接配置。
当前验证机器与连接方式只在 [AGENTS.md](../../../AGENTS.md) 维护，避免在多份说明中复制过期机器表。

- 使用本地文件系统，WSL 的 `/mnt/c` 用于搬运，不用于重构建、压力或性能测量。
- 同机 Windows/WSL 共享 CPU、GPU、内存、电源和磁盘，采样错开运行；WSL 不是第二台物理机或原生 Linux。
- 单线程比较使用同类核心与优先级，并发比较记录各角色的实际落点、工作量与完成守恒；宽 affinity mask 不等于逐线程固定。
- WSL guest affinity 不保证 Hyper-V 的物理 P/E 核落点。macOS QoS 不等于固定核心，不能假称硬 pin。
- 记录供电、电源模式、温度、频率及后台负载。跨系统结果分别报告，不合成一个“全平台”速度比。

速度、尾延迟、实际驻留/峰值内存、搬运、错误率、公平性与能耗分别说明。
双方都预留一个大容量不能证明实际占用相同或减半；能耗注明整机/设备范围和采集区间。

## 定位慢项与保留结果

排除工作量、数值、完成边界和仪器问题后，依次检查复杂度、搬运、分配、同步、争用、
生成代码与真实消费者影响。多余布局转换、重复准备、隐藏同步和不自然状态管理也属于集成成本。
修复后检查受影响的类型、ISA、规模和数值边界，再用未参与调参的用例或真实调用确认。

结果留在最相关的实验目录，记录实现/参考 revision 和补丁、构建与运行命令、输入、硬件/工具链、
计时边界、原始输出和分析方法；需要时附 trace、counter 或反汇编。保留失败和未测项。
已有原始数据不倒写，结论修正时说明原因与适用范围。
