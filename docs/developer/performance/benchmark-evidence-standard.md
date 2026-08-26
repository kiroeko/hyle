# HYLE 性能证据与仪器资格标准

> 本页规定何时一个性能结果有资格影响产品。没有通过仪器资格的测量仍可用于探索，但只能标为
> `DIAGNOSTIC`，不得写入当前证书、SOTA 结论或驱动公共形状。

## 1. 先限定命题

性能结论的最小主语是：

```text
(capability, semantic contract, workload distribution,
 hardware/OS/toolchain, delivery shape, candidate commit)
```

“HYLE mem SOTA”或“总体差不多”都不可证。允许的表述例如：“在该 candidate、两台机器和已登记
size distribution 上，操作 X 相对 peer Y 的同合同路径通过 5% non-inferiority gate”。

算法复杂度与纳秒点值分开审查。若强 peer 在正常规模拥有更好的渐进复杂度、进展保证或显著
更少的搬运/分配/syscall，不能用小 N 的一次领先掩盖。

## 2. 三层证据

| 层 | 回答的问题 | 例子 | 能否单独签产品结论 |
|---|---|---|---|
| 消费者 workload | 用户是否真正获益 | 模型加载、prefill/decode、请求调度、工程管线 | 可以，仍需合同与资源门 |
| 组件 replay | 哪个模块贡献了差异 | 真实 allocation trace、queue contention、解析/容器分布 | 只能签该组件合同 |
| 机制 microbench | 为什么会差 | 一次 atomic、dispatch、探测 probe、单 syscall | 不能；只用于归因 |

从真实消费者抽取 replay 时，必须保留数据分布、并发、生命周期、冷热状态和计费边界。人为重复
单指令会把固定开销无限放大，不能冒充消费影响。

## 3. 测量前的合同卡

在看结果前冻结以下内容：

- candidate commit、依赖 lock、编译器、flags、linkage、LTO/ISA/backend；
- 能力、公开/内部入口、ownership、failure 与线程合同；
- 正常输入分布与至少一个边界规模，不能只选最有利 size；
- 强 peer、版本、配置以及为什么语义等价；
- setup、steady-state、teardown 哪些计费；allocation/copy/cache 是否包含；
- latency、throughput、峰值/常驻内存、fragmentation、tail、CPU/energy 等适用指标；
- 预声明 practical/material margin 和 non-inferiority margin；
- 样本数、随机化、统计方法、停止条件、开发集与 holdout；
- 预计最小差异和它对真实消费者的影响预算。

跑后更换 peer、规模、计费边界、margin 或剔除规则会使结果失去签署资格；可以建立新的实验，
不能回写原假设。

## 4. 仪器资格门

参考代码存在并不保证能测出实现差异。地址分布、链接顺序、分支预测、DVFS、大小核迁移、ASLR、
allocator 状态和 timer floor 都可能产生稳定但无产品意义的偏移。每类极短测量必须先过以下门。

### 4.1 A/A negative control

把同一实现以两个等价名字、相同编译和相同调用路径放进 harness。交换标签和顺序后，仪器必须
主要给出 `PARITY`，且不能稳定偏爱第一个、第二个或某个链接位置。若 A/A 能“证明”0.5 ns 输赢，
本轮分辨率低于噪声地板。

### 4.2 Known-delta positive control

在不改变 API/输入的前提下加入已知机制差异，例如每 N 次多一次不可优化掉的 load、branch、
atomic 或固定字节 copy，并设置多个 delta。仪器必须随 delta 单调响应，并在预期 material 区间
可靠判出差异。只能通过“大锤”而分不出产品关心差异的仪器没有资格。

### 4.3 Timer floor 与 MDE

测空 harness、timer/read-counter 成本和极短 aggregate 的量化台阶。用 controls 与重复运行估算
minimum detectable effect（MDE）。预计差异小于 MDE 时，先扩大真实工作量、改用 hardware counter
或消费者 replay；不得靠增加 loop 次数把一个不真实调用形状包装成高分辨率。

### 4.4 最终 artifact 与 codegen

保存被执行 binary/library hash、符号反汇编和关键 counter。确认双方都没有被常量折叠、死码删除、
内联差异、PLT/IAT、sanitizer、debug iterator 或意外不同 ISA 改变。只看源码或 benchmark 可执行
的名字不算确认。

### 4.5 Order 与 placement

- 运行完整的 A/B 与 B/A strata，或使用记录 seed 的进程级随机顺序；预声明时也可用 ABBA block
  抵消漂移。不得只保留一个固定顺序。完整固定 strata 与随机化是二选一的实验设计，不重复强求。
- 分别报告 order strata；方向不一致不得池化成有利结论。
- 对亚纳秒/少指令路径先写 placement 风险矩阵：**代码位置**至少做 link order、padding 或 function
  alignment 中一种；**数据位置**至少做 allocation order、对象地址 stratum 或 heap preheat 中一种。
  若 allocation 本身在计费合同内，再额外做 cold/preheated heap；caller-owned 且 allocation 明确在
  epoch 外时，不为凑门重复增加无消费含义的 heap 维度。结论若跟任一适用 placement 翻转，标
  `INCONCLUSIVE`，除非 placement 本身是产品合同。
- ASLR 开/关、same-process 与 separate-process 的选择写入 manifest，不择优。

### 4.6 Host 与调度

登记 CPU 型号/microcode、OS/kernel、电源模式、温度、频率、SMT、NUMA、内存速度和后台负载。
固定 affinity；混合大小核平台分别在明确的 P/E core 或 Apple QoS 上验证，不能让线程自由迁移后
把差异归因实现。GPU 记录 driver、clock/power 和同步边界。

只有全部适用 controls 通过，当前 `(harness, machine, scale)` 才是 `QUALIFIED`。资格不能跨机器、
完全不同工作量或新 harness 自动继承。任一 control 失败即 fail closed：所有产品比较降级为
`DIAGNOSTIC`，先修仪器。

## 5. 同合同对拍

双方必须完成同一语义任务。逐项核对：

- 输入/输出、精度、排序/稳定性和错误处理；
- allocation、初始化、hash、copy、同步和销毁边界；
- thread safety、progress、capacity、OOM 和 cancellation；
- cache cold/warm、batch、NUMA/affinity、ISA/backend；
- 安全检查、统计与诊断是否一方承担而另一方免除。

合同不能对齐就用 `NOT_COMPARABLE`，并分别报告绝对成本。不要把“功能更少所以更快”写成 SOTA，
也不要强迫 HYLE 模拟 peer 的怪异 API；可以建立机制 microbench 解释必要语义税。

强 peer 应是相同问题域中成熟、当前且配置合理的实现。`std` 可作为常识基线，但当 oneTBB、folly、
mimalloc、Highway、absl 等才是用户真实替代品时，赢 std 不能完成证据门。

## 6. 运行与统计

- 优先使用 release、最终 linkage 和真实 target；先 warm-up，再测 steady state。
- 默认至少 15 个独立进程级 paired samples；更少或更多必须由预声明 power/成本解释。
- 每个进程内保持相同输入与环境；进程间使用记录 seed 的随机 order，或分别运行完整的固定
  A/B、B/A strata。保存每个 raw sample，不只保存均值。
- 主统计使用 paired ratio/delta 的 median 与分布无关或 bootstrap confidence interval；sign test
  可作诊断，不能替 practical margin。
- latency distribution 报 median 与适用 tail；吞吐不能抵偿 tail、内存、错误率或公平性退化。
- 不从多个平台、规模、order 或 shape 中挑最好项。每个必需 stratum 先判，再按预声明规则合成。

五态 verdict 固定：

| Verdict | 含义 |
|---|---|
| `WIN` | CI 与 practical margin 支持 HYLE 有 material 优势 |
| `PARITY` | 通过预声明双边 equivalence 区间 |
| `LOSE` | CI 与 practical margin 支持 material 劣势 |
| `INCONCLUSIVE` | 数据、power、order/placement 或资格不足，不能判方向 |
| `NOT_COMPARABLE` | 语义合同无法合理对齐 |

SOTA/non-inferiority 是独立单边门：所有必需平台、规模、order 和资源 shape 均通过才 `PASS`；
任一 `FAIL` 即失败；资格不足为 `INCONCLUSIVE`，不得把统计平局自动写成 non-inferior。

## 7. 哪些结果可以改产品

API、对象布局、常态 metadata、分支或依赖的性能改动需要两把钥匙同时存在：

1. **机制钥匙**：复杂度、指令、allocation、copy、syscall、contention、cache miss 或 codegen 有
   可解释差异，并能由 counter/反汇编/trace 复核；
2. **消费钥匙**：差异在真实 consumer/trace replay 中可观察，或能基于已测调用频率与资源预算
   防御地放大到预声明 material 影响。

只有 microbench 的稳定 0.5 ns、换顺序后翻转、只在特定地址布局获胜，或总体 profile 看不到的
差异，都不满足第二把钥匙。可以保留调查记录，不得为了它扭曲类型。

反之，真实 consumer 出现 material regression 时必须下钻归因；不能因微基准“多数平”就解释掉。
优先修复杂度、搬运、分配、syscall 与争用，再处理无法传播到消费者的微小指令差。

## 8. 证据包与有效期

有效报告至少链接：

- contract card 与 workload manifest；
- HYLE/peer commit、dependency/toolchain/build manifest；
- host/affinity/QoS/NUMA/backend 信息；
- binary hash、适用 codegen/counter；
- controls 与资格 verdict；
- 所有 raw samples、分析命令和五态/单边 gate 输出；
- consumer correlation、holdout 和资源结果；
- 已知限制与 reopen trigger。

人读结论放 `docs/implementation/evidence/performance/`，原始数据放根级 `evidence/performance/`。
依赖、工具链、算法、合同、正常 workload、硬件或 artifact 变化会使相应结论过期；未绑定当前
candidate 的旧表统一是 `HISTORICAL`。

模块签署后，不再为无消费者的亚纳秒波动反复优化。只有真实 material 缺口、正确性问题、
复杂度更优的强 peer 或合同/平台变化才重开。
