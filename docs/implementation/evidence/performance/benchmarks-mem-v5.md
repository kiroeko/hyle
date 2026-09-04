# Mem v5 host-allocation 工程性能证据

> 当前机械状态只见 [`module-status.json`](../../module-status.json)。本页只绑定 Mem v5 候选
> `90ffc8499e199fb675c9c704c3cd7e1b2b91768d` / tree
> `d5385c47214ab71cb29f14f2b5e47aa48c8dae85` 与一个冻结的 host-allocation 正常任务；它不声明
> 整个 Mem、所有尺寸/线程/平台或端到端推理性能“全面 SOTA”。

能力、算法和所有权边界见 [Mem architecture](../../../architecture/mem-design.md)，用户选型见
[Mem guide](../../../user/mem.md)，完整收口事实见
[Mem v5 closure card](../../tasks/mem/mem-v5-multichunk-mimalloc-351.md)。

## 本轮性能命题

Mem v5 有两个不同的性能问题，不能混成一个微基准排名：

- `LifetimePlanner` 是推理 graph 建图冷路径，以复杂度和资源结构验收：build 为
  `O((N+E) log N)`、scratch/常态 metadata 为 `O(N)`，query/view 为 `O(1)`，instantiate 恰为每个
  physical backing 一次 provider allocation；没有 byte/page 表、常态锁、atomic、syscall 或
  allocation-request side metadata；
- shipping host allocator 升级到 mimalloc 3.5.1 并改变 sized-free code path，因此属于 P1。冻结任务是
  一个隔离 subsystem heap 通过 runtime-erased allocator 执行 64 个 24 B--3 KiB、16/32/64-byte
  alignment 的 allocate、端点 touch 和 reverse sized-free lifecycle。

P1 peer 不是另一种 allocator 产品：它是在同一静态二进制、同一 mimalloc 3.5.1 上拥有独立 heap 的
`std::pmr::memory_resource` runtime-erased route。两边 owner 构造、warmup、销毁都在计时外，所有
allocate/touch/free 都计入。完整语义矩阵见
[`consumer_shapes_v5.json`](../../../../bench/mem/consumer_shapes_v5.json)，合同见
[`mem-v5-host-allocation-v1.json`](../../../../bench/contracts/mem-v5-host-allocation-v1.json)。

## W1-WIN 结果

W1-WIN、MSVC 19.50、portable ISA、static、LTO off；15 个独立进程样本，双 execution order、双 heap
allocation order、primary/perturbed placement 共八个 strata。`advantage ratio > 1` 有利于 HYLE：

| Stratum | median advantage ratio | 99.26% CI | exact verdict | 5% NI |
|---|---:|---:|---|---|
| primary / candidate-first / candidate-allocation-first | 1.0447 | [1.0140, 1.0650] | `INCONCLUSIVE` | `PASS` |
| primary / candidate-first / peer-allocation-first | 1.0685 | [1.0569, 1.1259] | `WIN` | `PASS` |
| primary / peer-first / candidate-allocation-first | 1.0710 | [1.0212, 1.0985] | `INCONCLUSIVE` | `PASS` |
| primary / peer-first / peer-allocation-first | 1.0614 | [1.0382, 1.0995] | `INCONCLUSIVE` | `PASS` |
| perturbed / candidate-first / candidate-allocation-first | 1.0650 | [1.0460, 1.0974] | `INCONCLUSIVE` | `PASS` |
| perturbed / candidate-first / peer-allocation-first | 1.0647 | [1.0474, 1.0803] | `INCONCLUSIVE` | `PASS` |
| perturbed / peer-first / candidate-allocation-first | 1.0380 | [1.0076, 1.1012] | `INCONCLUSIVE` | `PASS` |
| perturbed / peer-first / peer-allocation-first | 1.0492 | [1.0338, 1.0988] | `INCONCLUSIVE` | `PASS` |

聚合 exact verdict 是 `INCONCLUSIVE`，因为七个 strata 的置信区间跨过 5% 实用胜负边界；聚合
non-inferiority gate 是 `PASS`。因此允许的结论是：**这个冻结的普通 isolated-heap lifecycle 没有
material regression，并通过 5% 工程非劣门；观察点多数偏快，但证据不支持把整个 Mem 宣称为普遍
获胜。**

## 仪器资格与身份

- A/A 双顺序均为 `PARITY`；5%/10% known delta、timer floor、placement、artifact identity 与 build
  contract 全部 `PASS`；timer-floor observed multiple 为 3578.6；
- contract SHA-256：
  `b98c1edd2e7aeaf71b4b9138887d80f3a4d719dce5b941789aaf23d4bc3a0999`；
- semantic matrix SHA-256：
  `905f4dbd99b93ffaedb7d73e9b8e1758d085212357c99abb76479848990bdf63`；
- manifest SHA-256：
  `bd45bff8097348cf4dff273b2f68df3db76a500d24852ea169e5873af0ffaf50`；
- summary SHA-256：
  `c7b8c8dae2c14746df45edd26f502195acfabaf966b8eb0a685e54c3d72f897c`；
- deep verifier：`QUALIFIED`、current certification eligible、environment `CURRENT`、paths verified、
  source/provenance `EXACT`；原始本地 artifact 位于
  `C:\public\project-hyle\_validation\mem-v5-perf-evidence-90ffc849`。

oneTBB `843162db8cf4b2531a4a6147b6ea3fd860d44668` 只是保留的 multi-row benchmark adapter 的 source-only
构建输入，不是本轮 workload 的 peer 或 gate。合同只采样并裁决 `isolated-heap-lifecycle-64`，不会把旧
Mem benchmark 的其他行冒充本轮证据。

## 声明边界

- multi-backing planner 没有统一同合同外部 API，因此只签复杂度、provider maximum、alias/lifetime、
  deterministic replay 和 allocation-count 合同，不宣称全局最少 backing 或最小 peak；
- P1 只覆盖 W1-WIN 的一个 provisional ordinary batch，不覆盖所有 size/thread/NUMA/OS、device
  allocation latency、RSS policy 或端到端 tokens/s；
- Formal 状态保持 `PENDING`。工程 `PASS` 不等于独立发布证书；
- 旧 [`benchmarks-mem.md`](benchmarks-mem.md) 的历史 route/diagnostic 结果继续保留自己的 comparison
  class，不与本页同合同 P1 合并成模块总分。
