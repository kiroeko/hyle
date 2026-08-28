# mem v1 当前工程性能证据

> 当前结论：`W1-WIN / closure-perf-v1 / P1_ENGINEERING / 5%` 为 **PASS**。
> 这只签署下列四个冻结的正常 consumer/replay 合同，不等于“mem 所有操作、平台和规模都最快”，
> 也不等于 Formal 发布证书。2026-07 至依赖升级前的 821 行旧点值表已删除；需要追溯时使用 Git
> 历史，不能把旧 harness、旧依赖或旧 candidate 的数字当成当前证据。

本页只解释机器证据。能力与边界见
[mem architecture](../../../architecture/mem-design.md)，用户选型见
[mem guide](../../../user/mem.md)，本轮施工事实见
[mem-v1 closure](../../tasks/mem/mem-v1-closure.md)。

## 1. 被签署的命题

候选产品是 commit `2c71dfae9f4fb43ff0d68e5c1a4c55099fc0f101`、tree
`fc02363512a0807f4d46bb57427eae4137ea39fd`。性能 harness 与候选状态说明随后加入，正式运行使用
clean source commit `094b6cd1525df3363e174c3db27014c40d34d152`、tree
`fdaadc3dc8320cc85a41aaf8848d6001aa4586a6`；两者之间没有 mem 产品实现变化。

| workload | 消费依据与计费边界 | 同合同 peer | 证据等级 |
|---|---|---|---|
| runtime allocator batch 64 | 64 次 64B allocate/deallocate；运行时 allocator policy 注入 | oneTBB `std::pmr::memory_resource` 运行时路由 | replay，P1 产品门 |
| request Region epoch 64 | HTTP request policy：4 KiB initial、64 KiB ordinary max、64 KiB large threshold；整 epoch 64 次分配 + clear | Boost monotonic resource，使用同一 HYLE runtime upstream | replay，P1 产品门 |
| scheduler fiber lifecycle 16 | Scheduler 实际 acquire/release 的 16 个 guarded stack；含缓存复用生命周期 | 相同 Scheduler 生命周期中的直接 guarded VM reserve/release | 当前直接 consumer，P1 产品门 |
| fixed-slot cross-core handoff 64 | 256 slots；producer/consumer 两颗物理核之间借出、归还 64 项；第 257 项必须非阻塞耗尽 | Marl `BoundedPool<T,256>` | north-star consumer，P1 产品门 |

`Allocator` 与 compile-time `SystemAllocator` 的比较只用于说明动态分派成本：两者提供不同的策略
绑定合同，因此明确为 `NOT_COMPARABLE / mechanism diagnostic`，不参与产品聚合门。当前 HYLE 产品内
没有 `ObjectPool` 直接消费者；handoff 行验证的是已冻结的基础库正常需求形状，不能冒充端到端消费证明。
完整来源分类与文件 hash 在 `bench/mem/consumer_shapes_v1.json`。

## 2. W1-WIN n=15 结果

统一 ratio 大于 1 表示 HYLE 更快。每个区间都是 15 个独立进程的 99.26% 分布无关区间；表中范围
合并该 workload 的全部 execution order、allocation order 和 primary/placement strata。所有单独
stratum 都通过预声明的 5% non-inferiority 门。

| workload | HYLE 优势比中位数范围 | 全部分层 CI 外包络 | exact | 5% NI |
|---|---:|---:|---|---|
| runtime allocator / oneTBB runtime resource | 1.452–1.461x | [1.392, 1.473] | WIN | PASS |
| request Region / Boost monotonic | 1.620–1.635x | [1.589, 1.651] | WIN | PASS |
| pooled fiber lifecycle / direct guarded VM | 8.043–8.115x | [7.723, 8.227] | WIN | PASS |
| cross-core ObjectPool / Marl BoundedPool | 2.131–2.209x | [2.070, 2.297] | WIN | PASS |

这证明的是这些语义和规模下的工程非劣与当前优势，不外推到任意 size distribution、thread count、
NUMA topology 或 allocator backend。runtime allocator 的 direct diagnostic 中，compile-time
`SystemAllocator` 吞吐比运行时 `Allocator` 高 11.1%–13.4%；这是一层间接调用换取 runtime policy
注入的正常成本。无需注入的用户应直接选 `SystemAllocator`、`SystemStdAllocator` 或
`system_memory_resource()`，不应为隐藏该成本给默认对象增加常态 metadata/cache。

## 3. 资源、容量与尾延迟门

owner 合同产生 60 份 product log，handoff 合同产生 120 份。模块 validator 对每份日志单独裁决，
不是只检查下列聚合值：

- Region 每次都为 `region_reserved=258111`、`region_peak_used=190656`，保留量低于冻结的 256 KiB
  上限；复杂度与资源预算均通过。
- FiberStackPool 每次都在 64 容量下保留 16 个 warmed stack。HYLE p95 为 7.1–13.1 us、p99 为
  7.4–37.3 us；直接 guarded VM p95 为 56.0–84.8 us、p99 为 57.7–124.9 us。每份日志的 HYLE
  p95/p99 都满足相对 peer 的 5% 门。
- ObjectPool 每次都证明 `capacity=256`、`batch=64` 和第 257 次非阻塞耗尽。HYLE p95 为
  1.9–2.2 us、p99 为 1.9–5.8 us；Marl p95 为 4.2–5.2 us、p99 为 4.3–24.6 us。每份日志的
  HYLE p95/p99 都满足相对 peer 的 5% 门。
- 单次 `max` 可受 OS interrupt 影响，因此保存但不拿来替代预冻结的 p95/p99 gate；原始 max 未被
  丢弃或 winsorize。

公共 A/A、人工 5%/10% known-delta、batch-matched timer floor、CPU placement 以及两个 mem
resource validator 全部 `PASS`。Windows child canary 证明 owner workload 固定到性能逻辑核 2；
handoff workload 固定到不同物理核 2/4，二者均为 above-normal priority。

## 4. 身份与可重放证据

构建为 Visual Studio 18 2026、MSVC `19.50.35728.0`、x64 Release/static、portable ISA、LTO off，
机器 profile 为 `W1-WIN`。强 peer 固定为：

- oneTBB `e349c15263bbb1d6b48f2b514c06201e1b90eea3`；
- Boost `df8eeb90846648464e3c8e378e312f0cec772e83`；
- Marl `b8406ab0a82524170ead6dd6c371b579bbc28b3e`。

Marl 该提交的 public `tryBorrow()` 带一行无法编译的 stale `item->pool = this`。合同只在 build tree
生成可哈希 overlay，删除这一行并把三个 sibling include 改成 `<marl/...>`；外部 checkout 保持 clean。
原头 SHA-256 为 `28b696dac0dcfd663db473df728760abf4e47f5f01ab04a2e97de7204764f6cb`，
overlay SHA-256 为 `612a0c7177732448372c90e66a592284426fc28547fda66cd1c9191b23d3c517`。

| bundle | contract SHA-256 | manifest SHA-256 | summary SHA-256 | verdict |
|---|---|---|---|---|
| owner | `911787ec1f4cc0b7c61d508cd2fea52a9baebbcb57ec29ebb19833e8df8e577e` | `be87b13ec7116e900b4415742e48a6aae27182fb1bf6f496d902904f6c282e8a` | `4cd891eae64846c358df8084cce90277fe20d4031a4a7c8b1ebb3512ab15d4c2` | QUALIFIED / PASS |
| handoff | `415619a7875f77744444cb306c3214726291496370de605dc3e688768b202dd8` | `b082c033061973b1a50d6d24122670d21798df5849e6b4b99cee51ce4ef194fb` | `9f18e4837f80a7bf766f5c076f9a29b390e91fb15b2c24e5988962e475ae211e` | QUALIFIED / PASS |

当前 engineering bundles 位于：

```text
C:\hyle-scratch\mem-v1-owner-engineering-output\manifest.json
C:\hyle-scratch\mem-v1-handoff-engineering-output\manifest.json
```

从 clean checkout 和不存在的 build/output 路径重跑：

```powershell
python scripts/hyle-perf.py qualify-platform --contract bench/contracts/mem-owner-consumer-v1.json --machine W1-WIN --source . --peer hyle-baseline=C:\hyle-scratch\peers\hyle-mem-baseline-2c71dfae --peer onetbb=C:\public\project-hyle\oneTBB --peer boost=C:\public\project-hyle\boost --build C:\hyle-scratch\mem-v1-owner-engineering-build --output C:\hyle-scratch\mem-v1-owner-engineering-output
python scripts/hyle-perf.py qualify-platform --contract bench/contracts/mem-handoff-consumer-v1.json --machine W1-WIN --source . --peer marl=C:\public\project-hyle\marl --build C:\hyle-scratch\mem-v1-handoff-engineering-build --output C:\hyle-scratch\mem-v1-handoff-engineering-output
python scripts/hyle-perf.py verify --contract bench/contracts/mem-owner-consumer-v1.json --manifest C:\hyle-scratch\mem-v1-owner-engineering-output\manifest.json
python scripts/hyle-perf.py verify --contract bench/contracts/mem-handoff-consumer-v1.json --manifest C:\hyle-scratch\mem-v1-handoff-engineering-output\manifest.json
```

重跑时必须换成全新的 build/output 路径；上面的现有路径用于核验当前 retained bundle，不应用于覆盖。

## 5. 没有被本表冒充签署的内容

- NUMA selected-node/chunk 的 placement mechanism 属于 platform；单 socket W1 不能证明多 socket
  bandwidth，HugeRegion 不复制该性能声明。
- FixedPool/GrowablePool/SharedPool、allocator adapters、snapshot/trim/reset 与异常路径仍由复杂度、
  allocation/metadata、correctness、sanitizer 和 consumer delivery 门约束；本轮没有为它们制造新的
  microbenchmark 姿势，也不发布数值 SOTA 声明。
- 当前证据是 Windows P1 engineering，不是跨平台或长期 Formal 证书。bundle 仍引用本机 live
  artifacts；丢失 bundle、合同/peer/toolchain/机器变化或产品实现变化会使相应结论过期。
- 只有真实 consumer 暴露 material 缺口、复杂度档落后、同合同 peer 给出可复现 material 优势，
  或上述证据身份失效，才重开性能；地址顺序翻转、单次点值和亚纳秒变化不触发产品改形。
