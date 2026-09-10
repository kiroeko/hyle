# HYLE 性能证据与仪器资格标准

> 本页规定何时一个性能结果有资格影响产品。没有通过仪器资格的测量仍可用于探索，但只能标为
> `DIAGNOSTIC`，不得写入当前证书、SOTA 结论或驱动公共形状。

统一 schema、ratio、统计、controls、运行模式与停止规则以
[统一性能仪器协议 v1](instrument-v1.md)为唯一权威。本页补充如何选择命题、同合同 peer 和
Formal 发布证据，不允许模块 runner 重新定义这些公共语义。

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

## 2. 证据来源与比较类型是两个维度

| 层 | 回答的问题 | 例子 | 能否单独签产品结论 |
|---|---|---|---|
| 消费者 workload | 用户是否真正获益 | 模型加载、prefill/decode、请求调度、工程管线 | 可以，仍需合同与资源门 |
| 组件 replay | 哪个模块贡献了差异 | 真实 allocation trace、queue contention、解析/容器分布 | 只能签该组件合同 |
| 机制 microbench | 为什么会差 | 一次 atomic、dispatch、探测 probe、单 syscall | 不能；只用于归因 |

从真实消费者抽取 replay 时，必须保留数据分布、并发、生命周期、冷热状态和计费边界。人为重复
单指令会把固定开销无限放大，不能冒充消费影响。

统一合同中的证据 `tier` 与 `gate_role` 正交：mechanism 永远是 diagnostic；consumer/replay 也可以
显式作为 diagnostic，用于保真度或归因而不抬高为产品门。只有 `consumer/replay + product` 才可进入
产品聚合，并须具名 consumer、哈希绑定 retained provenance、逐侧陈述 candidate/peer ownership 与
equivalence，以及用 frequency/value/unit/budget/denominator/derivation 冻结 material contribution。
这些是 structured declarations 和审查输入，不是机器生成的真实性证明；独立 reviewer 仍须核对引用
确实对应自然 consumer、相同 value category/生命周期和可防御的调用频率或 trace weight。
声明必须成组出现；当前 NI 产品 adapter 只接受 `COMPARABLE` product 行。runner validators 是 bundle
全局完整性门，某条 diagnostic/mechanism 特有的风险必须归到该行 control 或独立 diagnostic contract，
不得借全局 validator 间接取得产品否决权。

模块的工程性能级别与证据形状相关但不相同：`P0` 以复杂度、资源和适用 codegen/size 结构门
收口；`P1` 必须在锁定性能机上以正常 workload/replay 通过预声明工程门，存在同合同 peer 时还须
通过 non-inferiority，默认 material margin 为 5%；`P2` 必须由真实 consumer/trace KPI 和贡献预算
守门。没有同合同 peer 的 P1 以预冻结的 consumer latency/throughput/resource 预算裁决，不能产生
peer/SOTA 声明。P1/P2 工程 `PASS` 是产品进入 `CLOSED` 的硬门。Formal 发布证书是独立状态轴，
不应成为普通模块每次重构的重复三平台门。

上表描述证据从哪里来，不描述两侧是否在做同一件事。每个有 peer 的 workload 还必须在采样前声明：

| Comparison class | Contract | 合法结论 |
|---|---|---|
| `NATURAL_ROUTE` | 同一用户任务；各库采用自身自然、公开、正确的最佳路线，允许产品语义不同 | route-level 用户收益或产品形状缺口；不能签 same-contract peer/SOTA 结论 |
| `SAME_CONTRACT` | minimum semantics、lifecycle、progress 和计费边界全部等价，或由公开记录的 adapter 对齐；任一侧不得额外承担 material 工作或保证 | 同合同 non-inferiority/better 与资源取舍 |
| `MECHANISM_DIAGNOSTIC` | 刻意隔离一个实现机制 | 归因；不能单独形成产品 gate |

因此 consumer-shaped replay 不自动获得 `SAME_CONTRACT` 身份。`NATURAL_ROUTE` 的输赢仍然重要：它
回答用户采用当前自然路线会发生什么，并可能证明 HYLE 缺少一个常见语义档位；但不能把额外 result、
cancel、error、allocation 或 suspension 合同的成本全部归因给底层 queue/scheduler/allocator。

没有当前真实消费者时，workload 来源按当前真实消费者、与模块相关的具名产品纵切、至少两个独立成熟
实现共同任务、标准/定义性算法、diagnostic synthetic 的顺序选择。产品纵切从推理、渲染/资源流送、
批处理数值管线和低延迟服务中选择；主纵切可能造成特化时，增加一个正交 holdout。由成熟共识而非
生产 trace 准入的 normal shape 标为 `PROVISIONAL_ARCHETYPE`，冻结一个 primary 和至多一个 holdout；
不得把 API 数、benchmark case 或单库独有示例当作调用频率。

## 3. 测量前的合同卡

在看结果前冻结以下内容：

- candidate commit、依赖 lock、编译器、flags、linkage、LTO/ISA/backend；
- 能力、公开/内部入口、ownership、failure 与线程合同；
- 正常输入分布与至少一个边界规模，不能只选最有利 size；
- 强 peer、版本、配置以及为什么语义等价；没有同合同 peer 时记录查找范围、`NOT_COMPARABLE` 和
  预冻结的 consumer 预算；
- comparison class，以及 evidence provenance 属于 real consumer、named product journey、
  provisional archetype、standard 还是 diagnostic synthetic；
- `SAME_CONTRACT` 的逐项语义等价矩阵；`NATURAL_ROUTE` 的额外语义与 route-level 解释边界；
- setup、steady-state、teardown 哪些计费；allocation/copy/cache 是否包含；
- latency、throughput、峰值/常驻内存、fragmentation、tail、CPU/energy 等适用指标；
- 预声明 practical/material margin 和 non-inferiority margin；
- 样本数、随机化、统计方法、停止条件、开发集与 holdout；
- 预计最小差异和它对真实消费者的影响预算。

### 3.1 完成边界源码预审（计时前硬门）

任何带 peer 的 P1/P2 在配置、编译或采样前，必须根据**实际源码**填写下面的时序账单；只在合同里
写一句“full drain”或“teardown 相同”不构成证据：

```text
admission -> user work ends -> result/error becomes visible
          -> cancellation/wait links detach -> coroutine/driver/stack cleanup
          -> allocator-visible reclamation -> owner returns
```

对 candidate 和 peer 的每个阶段分别记录 `CHARGED`、`DEFERRED` 或 `NOT_APPLICABLE`，并给出实现文件、
符号和行号锚点。还要列出 owner 返回后仍可能执行的 callback、finish hook、析构、deallocation、wake 或
异步提交；“最终都会完成”不能替代“都在测量边界内完成”。至少一侧把 material 工作延迟到 owner 返回
以后时，`SAME_CONTRACT` preflight 立即失败，不得启动 A/A、样本升级或 qualified P1。只能：

1. 用公开且自然的 adapter 把两侧推进到同一完成边界，并把 adapter 全部计费；或
2. 降为 `NATURAL_ROUTE`，只报告用户可见路线成本，不签同合同非劣/SOTA。

同一 preflight 还须附一张静态资源账单：每个正常 operation/owner 的 allocation/deallocation、共享
atomic、锁、wake、syscall、copy 和常驻 metadata。能从源码、对象布局和生成代码解释的 material 差异，
先完成归因或修正合同；不要先花样本预算再猜。独立 reviewer 必须沿源码锚点复核，不能只复述合同作者
写的等价性文字。

完成边界通过后，先用同一 artifact 对自身做**与产品完全相同的 workload 和执行拓扑**短诊断：同进程
成对 row 就把两行都替换为同一个真实 workload route，保留相同 Scheduler/allocator/working set/warmup；
分离子进程 pair 就必须以同样的启动、亲和、优先级和先后顺序启动两个相同 product executable。通用
算术 A/A 只能验证 timer、parser 和统计管线，不能替代涉及 runtime、allocator、线程或设备状态的
workload-matched A/A。至少覆盖 A/B 与 B/A；若首个/第二个位置、进程启动、热状态或 placement 产生接近或超过
material margin 的固定影响，仪器 preflight 失败，不能启动完整样本。公共 control executable 内部的
A/A 只能证明它自己的同进程拓扑；它不能替代 separate-process product pair 的同拓扑 A/A。此时应选择
更贴近真实 delivery 的已有 adapter，或登记独立仪器 blocker，不能用更多 n 掩盖拓扑失配。

placement control 还必须只改变待研究的地址/layout 变量。给每个函数增加大段 hot-patch padding、切换
LTO/异常/安全开关、显著扩大 `.text`/数据 working set，或改变调用/内联/分配语义，都不是纯 placement
replica。preflight 要记录两份 artifact 的 `.text` 大小与实际扰动机制；若代码工作集发生 material 膨胀，
先修 control，不能把它造成的输赢归给产品。小型前置 section contribution、受控 link order 或等价的
非执行地址位移可以使用，但它们不得进入计时区间。

跑后更换 peer、规模、计费边界、margin 或剔除规则会使结果失去签署资格；可以建立新的实验，
不能回写原假设。

## 4. 仪器资格门

参考代码存在并不保证能测出实现差异。地址分布、链接顺序、分支预测、DVFS、大小核迁移、ASLR、
allocator 状态和 timer floor 都可能产生稳定但无产品意义的偏移。极短 CPU 路径的公共基础门是
A/A 双顺序、near/large known-delta 单调性和 timer/empty floor；其余 controls 必须由合同中的风险
选择，不能为了“看起来严格”机械堆满每种 workload。

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

当前 CPU nanobench v1 的机械门以[统一仪器协议 §6](instrument-v1.md)为准：floor 必须与产品
`measurement_batch` 相同，并要求 pair 较快一侧至少为四组保守 floor 中位数的 20 倍。20x 不是
产品 margin 或可估计的精确误差率。只有消费合同本身就是自然 batch 时才能改变 batch；单操作未过门
应保持 `DIAGNOSTIC`，不能为取得资格调低阈值。

### 4.4 最终 artifact 与 codegen

保存被执行 binary/library hash、符号反汇编和关键 counter。确认双方都没有被常量折叠、死码删除、
内联差异、PLT/IAT、sanitizer、debug iterator 或意外不同 ISA 改变。只看源码或 benchmark 可执行
的名字不算确认。

### 4.5 Order 与适用的 placement

- 运行完整的 A/B 与 B/A strata，或使用记录 seed 的进程级随机顺序；预声明时也可用 ABBA block
  抵消漂移。不得只保留一个固定顺序。完整固定 strata 与随机化是二选一的实验设计，不重复强求。
- 分别报告 order strata；方向不一致不得池化成有利结论。
- 极短代码且存在 code-placement 风险时，至少选择 link order、padding 或 function alignment 中
  一种；caller-owned 且地址敏感时，至少选择 allocation order、对象地址 stratum 或 heap preheat
  中一种。allocation 在计费合同时才加入 cold/preheated heap；没有对应风险就不为凑门增加维度。
  结论若跟任一适用 placement 翻转，标 `INCONCLUSIVE`，除非 placement 本身是产品合同。
- ASLR 开/关、same-process 与 separate-process 的选择写入 manifest，不择优。

### 4.6 Host 与调度

登记 CPU 型号/microcode、OS/kernel、电源模式、温度、频率、SMT、NUMA、内存速度和后台负载。
固定 affinity；混合大小核平台分别在明确的 P/E core 或 Apple QoS 上验证，不能让线程自由迁移后
把差异归因实现。GPU 记录 driver、clock/power 和同步边界。

只有全部适用 controls 通过，当前 `(harness, machine, scale)` 才是 `QUALIFIED`。资格不能跨机器、
完全不同工作量或新 harness 自动继承。任一 control 失败即 fail closed：所有产品比较降级为
`DIAGNOSTIC`，先修仪器。

## 5. 同合同对拍

双方必须完成同一语义任务。逐项核对并把每格裁决为 `EQUAL`、`ADAPTER_EQUALIZED` 或
`INTENTIONALLY_DIFFERENT`：

- 输入/输出、精度、排序/稳定性和错误处理；
- allocation、初始化、hash、copy、同步和销毁边界；
- thread safety、progress、capacity、OOM 和 cancellation；
- cache cold/warm、batch、NUMA/affinity、ISA/backend；
- 安全检查、统计与诊断是否一方承担而另一方免除。

还必须核对 result 是逐项还是批级、failure/cancellation 是逐项还是共享、caller 是否参与执行、任务能否
挂起，以及 wait/wake 与 teardown 的生命周期。allocation 数量和 atomic/wake 次数可以正是待比较的
实现结果，但 allocation ownership、memory-resource、创建/销毁阶段和计费边界必须一致。

任何 material `INTENTIONALLY_DIFFERENT` 都禁止 `SAME_CONTRACT` 和 non-inferiority。此时改为
`NATURAL_ROUTE` 并分别报告用户可见绝对成本，或使用 adapter 补齐后建立一份新的同合同实验。不要把
“功能更少所以更快”写成 SOTA，也不要强迫 HYLE 模拟 peer 的怪异 API；可以建立 HYLE 内部语义成本
阶梯和机制 microbench 解释必要语义税。adapter 的构造、分配、同步和 teardown 全部进入 peer 计费，
不能只模拟返回类型而漏掉产品语义。

**禁止“苹果对橘子”的硬门：**相同输入、输出数量或最终 checksum 只是必要条件，不是
`SAME_CONTRACT` 的充分条件。若一侧额外提供或实际执行 material 的线程安全、精确一次、失败传播、
取消、排序、生命周期、安全检查、allocation、同步、clock、metadata 或销毁保证，即使更强的一侧是
HYLE，也必须把对应矩阵格写成 `INTENTIONALLY_DIFFERENT` 并注明方向，整行降为 `NATURAL_ROUTE` 或
另建 adapter 对齐实验。候选更强且仍落在预声明预算内，只能表述为“自然产品路线的额外保证未超出用户
预算”；不得把统计带内结果称为底层 primitive 的 `PARITY`、peer non-inferiority 或 SOTA。候选更弱、
两侧各有不同保证，或额外工作无法防御地计量时同理，且不能用结果签 correctness 缺口。

该门在采样前由合同作者和独立 reviewer 分别签一次；若 retained bundle 之后才发现不等价，raw samples、
artifact identity 与仪器资格仍可作为历史事实，但必须立即撤销错误的 same-contract 解释并显式记录新的
route-level 边界。精密统计不能把错误命题变正确，也不能因为修正证据分类就自动触发产品代码改写。

强 peer 应是相同问题域中成熟、当前且配置合理的实现。`std` 可作为常识基线，但当 oneTBB、folly、
mimalloc、Highway、absl 等才是用户真实替代品时，赢 std 不能完成证据门。

## 6. 运行与统计

- 优先使用 release、最终 linkage 和真实 target；先 warm-up，再测 steady state。
- `qualify-platform` 默认 15 个独立进程级 paired samples。只有合同预声明、controls 已通过且唯一
  问题是 power 不足时，才允许从 n=15 升级到一轮全新的 n=31；不能追加、挑样或跑到刚好通过。
- 每个进程内保持相同输入与环境；进程间使用记录 seed 的随机 order，或分别运行完整的固定
  A/B、B/A strata。保存每个 raw sample，不只保存均值。
- 主统计使用 paired ratio/delta 的 median 与分布无关或 bootstrap confidence interval；sign test
  可作诊断，不能替 practical margin。
- latency distribution 报 median 与适用 tail；吞吐不能抵偿 tail、内存、错误率或公平性退化。
- 不从多个平台、规模、order 或 shape 中挑最好项。每个必需 stratum 先判，再按预声明规则合成。

结果是三条独立轴，不是一张混合“五态”表：

| 轴 | 值 | 含义 |
|---|---|---|
| comparability | `COMPARABLE | NOT_COMPARABLE` | 两侧语义合同能否合理对齐 |
| qualification | `QUALIFIED | DIAGNOSTIC` | 当前机器、harness、controls 是否有资格影响产品 |
| exact performance | `WIN | PARITY | LOSE | INCONCLUSIVE` | 统计区间相对预声明 material band 的位置 |

comparison class 是这三轴之前的命题分类；只有 `SAME_CONTRACT + COMPARABLE + QUALIFIED` 才能产生
peer non-inferiority/SOTA verdict。`NATURAL_ROUTE` 即使统计清晰也只形成 route-level 结论。

统一使用 `advantage_ratio`，且大于 1 永远有利于 HYLE candidate：higher-is-better 为
`candidate/peer`，lower-is-better 为 `peer/candidate`。给定 margin `m`，等价带固定为
`[1/(1+m), 1+m]`：

| Verdict | 含义 |
|---|---|
| `WIN` | CI 完全高于 `1+m` |
| `PARITY` | CI 完全位于 reciprocal band 内 |
| `LOSE` | CI 完全低于 `1/(1+m)` |
| `INCONCLUSIVE` | CI 跨越 exact verdict 边界，不能判方向 |

non-inferiority 是独立单边门，固定为 `PASS | FAIL | UNRESOLVED | N/A`。所有必需 strata 均通过
才 `PASS`；任一 material `FAIL` 即失败；功效不足为 `UNRESOLVED`。`NOT_COMPARABLE`、
`DIAGNOSTIC` 和 `INCONCLUSIVE` 各留在自己的轴上，不能被平局或单边门覆盖。

### 6.1 固定方向点值复核

若一个 HYLE/peer 行在全部必需 order/placement/allocation strata 的中位 `advantage_ratio` 都小于 1，
人读报告必须复核并解释该固定方向，**不论幅度多小、exact verdict 是否仍为 `PARITY`、单边门是否为
`PASS`**。这是一项防止 margin 掩盖系统性问题的审计触发器，不是新的统计 verdict，也不能把“全部点值
小于 1”直接改写成 `LOSE`。

复核严格有界为：先重验 comparison class 与语义等价矩阵，再对照 A/A、order/placement、历史或
holdout 方向，最后做一次复杂度、资源账单或相关热路 codegen/trace 机制检查。报告必须把结论归入以下
一种可读类别：同合同且找到产品机制、自然路线承担额外语义、仪器/代码或数据布局分辨率内未找到机制、
或保留的未归因 diagnostic。只有[第 7 节](#7-哪些结果可以改产品)的机制钥匙和消费钥匙同时成立，
才触发产品修改或重开；否则保留点值与调查事实，不通过追加样本、第五种姿势或 benchmark 专用 API
追逐方向。

旧工具的 `RIVAL/BASELINE` throughput ratio 小于 1 才有利于 HYLE；它属于 legacy raw 格式，必须
由 adapter 取倒数并标注来源，不能与当前 `advantage_ratio` 混列。

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

对于存在多个真正不同语义档位的基础能力，还应报告 HYLE 内部成本阶梯，例如 fire-and-forget、批级
完成、逐任务 result/control、可挂起任务。该阶梯用来解释用户为哪些语义付费，不要求每一档都有 peer；
若两个档位没有 material 语义差异却只有跑分差异，应合并实现，而不是暴露 `fast/light/heavy` API。

## 8. 证据包与有效期

`probe` 永远只产生诊断；`qualify-platform` 产生模块 P1/P2 工程证据；`certify` 只校验和合成同一
contract hash/candidate 的平台 manifests，且从不池化跨平台样本。Formal 可增加平台矩阵、归档、
holdout 与发布声明要求，但不能改写统一统计或反向替代模块工程门。

有效报告至少链接：

- contract card 与 workload manifest；
- HYLE/peer commit、dependency/toolchain/build manifest；
- host/affinity/QoS/NUMA/backend 信息；
- binary hash、适用 codegen/counter；
- controls 与资格 verdict；
- 所有 raw samples、分析命令、三轴结果和单边 gate 输出；
- consumer correlation、holdout 和资源结果；
- 已知限制与 reopen trigger。

人读结论放 `docs/implementation/evidence/performance/`，原始数据放根级 `evidence/performance/`。
依赖、工具链、算法、合同、正常 workload、硬件或 artifact 变化会使相应结论过期；未绑定当前
candidate 的旧表统一是 `HISTORICAL`。

当前统一仪器 v1 的 hash/replay 证明 bundle 内部机械一致且引用文件未漂移，不是密码学真实性证明。
engineering 证据可在 live artifact 保留期内使用；Formal 发布必须先把所有受许可约束允许保存的
artifact、raw stage 与 build provenance 放进受信的内容寻址不可变存储，并按发布策略签名。缺少该
retention 层时应 fail closed 或标 `PENDING`，不得把“当前还能 verify”写成长期可审计。

模块签署后，不再为无消费者的亚纳秒波动反复优化。只有真实 material 缺口、正确性问题、
复杂度更优的强 peer 或合同/平台变化才重开。
