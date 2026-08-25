# 性能证据与判定标准

> 本页规定 HYLE 性能数字取得产品结论的资格。模块具体点值仍归各 `benchmarks-*.md`；模块
> 是否收口归 milestone/readiness。没有通过本页的结果可以用于诊断，不能进入 SOTA 总账。

## 1. 先写合同卡，再运行

每个 claim-bearing workload 必须在采样前登记：

| 字段 | 必须回答 |
|---|---|
| 用户/内部任务 | 谁在正常使用中完成什么，而非 benchmark 想测什么 |
| candidate 与 peer | exact commit/tree；为什么 peer 是该合同的成熟强实现 |
| 语义 | 返回值、检查、失败、ordering、精度与 QoS 是否一致 |
| ownership | 谁分配/释放，setup/teardown、cache/TLS、容量和 lifetime 是否一致 |
| 数据形状 | 类型、布局、分布、batch、小/中/大 N、预热与 residency |
| 并发/硬件 | 线程、亲和/QoS、NUMA、ISA/backend、OS 与编译/链接形状 |
| 计费边界 | 分配、dispatch、validation、转换、syscall、同步哪些计入 |
| 等价预算 | `m%`；由用户影响预先指定，不能看到结果后扩大 |

任一必要合同不同，结论为 `NOT_COMPARABLE`；可以同时报告两种产品选择的绝对成本，但不得
输出谁胜谁负。std 下界、raw syscall、unchecked kernel 和较弱 ownership peer 不会因为数值
接近就自动成为同合同 SOTA。

## 2. 证据分层

1. **算法层**：正常/最坏复杂度、常态元数据、进展保证和规模增长曲线；
2. **机制层**：codegen、内存访问、分配、锁/原子、syscall、dispatch 与 cache footprint；
3. **产品层**：公开入口或正常内部消费者的端到端同合同对拍。

复杂度落档不能用纳秒小 N 掩盖；机制微基准不能独立签产品 SOTA。产品层异常必须下钻前两层
定位，而不是先调整 API、对象字段或 benchmark 排列让数字变好。

## 3. 五种且仅五种精确结论

脚本的 `BASELINE` 是本轮 candidate，`RIVAL` 是 peer；配对吞吐比
`r = RIVAL op/s / BASELINE op/s`，因此 `r > 1` 表示 candidate 较慢。

| 结论 | 含义 |
|---|---|
| `WIN` | candidate 在预声明的 material threshold 外稳定更快 |
| `EQUIVALENT` | 已正向证明 candidate 与 peer 的差距落在预声明等价预算内 |
| `LOSE` | candidate 在同合同正常用法中稳定、实质落后；模块保持黄色直到处理 |
| `INCONCLUSIVE` | 样本/置信区间、执行顺序、平台或等价 code-shape 不能确认方向或等价 |
| `NOT_COMPARABLE` | 合同不同；不允许统计脚本制造胜负 |

“没有显著差异”不等于 `EQUIVALENT`。旧 `TIE` 必须按原始 paired ratios 重分类；没有原始数据
就只能写 `INCONCLUSIVE`/历史输入。

### 3.1 独立的 SOTA non-inferiority 闸

模块的“是否至少同档”与“精确属于 WIN/EQUIVALENT/LOSE 哪一类”不是同一个统计问题。为此，
在完整保留上述五态精确结论的同时，另行计算一个**单边、非劣**的 SOTA gate。它不是第六种
性能结论，也不得把 `INCONCLUSIVE` 改写为 `EQUIVALENT`。

仍使用采样前为该 workload 声明的同一个 `m` 和同一个中位数 CI，令 `U=1+m`：

| SOTA gate | 单个有效执行形状的条件 | 只允许表达 |
|---|---|---|
| `PASS` | CI 上界 `<= U` | 已排除 candidate 比该强 peer 实质慢超过预算 |
| `FAIL` | CI 下界 `> U` | candidate 相对该 peer 存在已证实的实质落后 |
| `UNRESOLVED` | CI 仍跨过 `U`，或预算/样本/区间不足 | 当前证据不能排除实质落后 |
| `N/A` | `NOT_COMPARABLE` | 合同不等价，SOTA gate 不适用 |

例如 CI 只跨越 `WIN` 与 `EQUIVALENT` 的边界、但其上界仍 `<=U` 时，精确结论必须诚实保留
`INCONCLUSIVE`，而 non-inferiority gate 可以为 `PASS`。这只证明“没有实质落后”，不证明两者
等价，也不证明 candidate 获胜。

non-inferiority **不得使用第二套或更宽的 margin**；其 `U` 必须机械派生自合同卡里采样前登记
的同一个 `m`。看完数据后新增/放宽 `m`、只给不利 workload 换阈值，或把单边 PASS 写成
`WIN/EQUIVALENT`，都属于无效证据。

## 4. 默认统计合同

- Release、无 sanitizer；同一进程内 candidate/peer 配对，至少 15 个独立进程；
- 每个进程保存两边 op/s 与 ratio，不比较两个独立绝对均值；
- paired ratios 排序后，取最窄且每侧 exact binomial tail 不超过 `.01` 的总体中位数区间；
  `n=15` 时为第 3 至第 13 个 order statistic，实际 coverage 约 99.26%、不低于 98%；
- runner 必须为每个进程、每个 requested pair 保存 baseline op/s、rival op/s 与
  `rival/baseline` ratio 的机器可读 raw JSON；只有中位数、spread 或区间汇总的日志不够资格
  复算证书。`bench-stat.py` 以 `# HYLE_BENCH_PAIR_RAW ` 开头输出 schema 1 JSON payload；
- 令预声明乘法预算为 `[L,U]=[1/(1+m),1+m]`。区间上界 `<L` 为 candidate `WIN`，区间下界
  `>U` 为 candidate `LOSE`，整个区间落在 `[L,U]` 才为 `EQUIVALENT`；其余都是
  `INCONCLUSIVE`。边界相等按保守侧处理，不签 material WIN/LOSE；
- 同一区间另按 3.1 节做单边 gate：上界 `<=U` 才 `PASS`，下界 `>U` 为 `FAIL`，其余
  `UNRESOLVED`；五态精确 verdict 与单边 gate 必须同时记录，不能互相覆盖；
- `m` 必须在采样前按 workload 声明；未给预算不得输出等价或胜负。预算没有全库默认值：
  hot primitive 通常可预声明 3%，确需 1% 就明确写 1%；OS/并发端到端按真实 QoS 声明；
- 默认序和 peer-first 的五态结论必须兼容；一边 WIN、另一边不是 WIN，或方向翻转，精确结论
  最终均为 `INCONCLUSIVE`，不能择优；SOTA gate 则要求默认序和 peer-first **各自** `PASS`
  才能合成 `PASS`；
- sub-cache、小 batch 或百纳秒 aggregate 还必须经过一种合同等价的 code-shape/placement
  扰动；精确方向或资格改变即为 `INCONCLUSIVE/placement-sensitive`；适用的每个 shape gate
  也必须单独 `PASS`，否则不得合成 SOTA `PASS`；
- Windows/Linux/macOS 分平台裁决。不可用 backend 明写未认证，不用模拟器时序替代。

非参数区间不假定正态分布；sign test 方向/p 值只保留作诊断，不再单独产生产品判定。15 个样本通常只能
给出较宽区间；证明不了等价就是 `INCONCLUSIVE`，应增加独立进程样本或改善 workload 隔离，
不能把等价预算事后放宽。并发/OS workload 可以有不同预算，但必须来自用户 QoS/容量目标。

## 5. 结果合成

- 一个平台/规模存在同合同 `LOSE`，就不能把模块性能写成全局 SOTA；
- `WIN` 与 `EQUIVALENT` 可以共同构成“同档或更快”，但每格仍保留原判定；
- 精确关系 claim 仍要求五态判定本身成立；精确 `INCONCLUSIVE` 不能被单边 gate 改写为
  `WIN/EQUIVALENT`。如果证书声明的只是“相对已登记强 peer 非劣/同档或更快”，则该正常路径
  的默认序、peer-first 和所有适用 shape gate 全部 `PASS` 后，可以通过这一项 SOTA 门，同时
  在账本中原样保留精确 `INCONCLUSIVE`；
- SOTA gate 合成规则固定：所有必需 gate 均 `PASS` 才是 `PASS`；任一 `FAIL` 就是 `FAIL`；没有
  `FAIL` 但至少一项 `UNRESOLVED` 就是 `UNRESOLVED`；`NOT_COMPARABLE` 对应 `N/A`。不得只选择
  有利的执行序或 shape；
- 任一 claim-bearing 正常路径的目标 claim 仍为 `UNRESOLVED`，模块相应性能门保持黄色；机制
  诊断未知可以不阻塞，前提是已有真实消费者覆盖同一风险；
- throughput 不能代偿错误率、tail latency、峰值内存、footprint、能耗或公平性。合同需要的
  维度必须分别守门；
- 所有结论都绑定 source/tree、可执行文件 hash、harness/runner revision、compiler flags、
  target/feature、peer revision、环境/QoS/affinity、原始日志和 fresh build log。

## 6. 禁止事项

- 不为 benchmark 增加公开 API、常态元数据、隐藏 TLS/cache 或不同失败语义；
- 不挑有利执行顺序、allocator 地址、函数排列、CPU 核、QoS 或一次性结果；
- 不把较弱合同 peer 的成本参照写成同合同胜负；
- 不以 absolute spread 代替 paired ratio，也不以 paired sign 显著掩盖 code placement 翻转；
- 不用“误差”“硬件墙”“by design”结束 LOSE；必须给算法/codegen/consumer 根因和裁决。

`scripts/bench-stat.py` 负责单次执行顺序的配对统计与 provenance；`--pair BASELINE RIVAL`
约定 BASELINE 是 HYLE candidate，并要求 `--equivalence-margin-pct` 才能在 clean evidence 中
产生 claim。合同不同但绝对成本仍有参考价值时，用 `--not-comparable-pair` 机械输出
`NOT_COMPARABLE`。脚本同时输出 `[order_verdict]`、`[order_sota_gate]` 和每进程 raw JSON；review
owner 负责合同资格、默认/peer-first、code-shape 与三平台合成。脚本结果永远是证据输入，不是
自动模块证书。
