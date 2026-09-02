# Mem v2 当前工程性能证据

> 当前结论：`W1-WIN / closure-perf-v1 / P1_ENGINEERING / 5%` 为 **PASS**。
> 该结论只签署下列四个冻结的同合同 consumer workload，不等于 Mem 的所有操作、平台和规模都
> SOTA，也不是 Formal 发布证书。

能力和成本边界见 [Mem architecture](../../../architecture/mem-design.md)，用户选型见
[Mem guide](../../../user/mem.md)，完整功能、正确性、交付和红队事实见
[`mem-v2 task`](../../tasks/mem/mem-v2-rebuild.md)。旧
[`mem-v1 evidence`](benchmarks-mem.md)只负责 v1 历史命题，不能续签本页 verdict。

## 被签署的命题

产品候选为 commit `9eda54f9b36bfb0792ecf1544dc39d7ac9682265`、tree
`359c4832a91a498bb92e49b80766752a4ca9d9c6`。合同是
[`mem-v2-consumer-v1`](../../../../bench/contracts/mem-v2-consumer-v1.json)，其 SHA-256 为
`56083fe6817c74e010194aa96dae9f1f36d36698c4b5dd5206fd3d9ab31fd281`，语义矩阵 SHA-256 为
`85959bab834bf483f5b7b9506fcf6a7af79074819410be8ded78a1cd28edc4a8`。

所有 workload 均使用 n=15 独立进程样本、primary/placement 两种布局、candidate-first/peer-first 与
allocation order 的八个 strata，并运行 A/A、5%/10% known-delta、timer-floor 和 placement controls。
`advantage ratio > 1` 有利于 HYLE。

| normal workload | same-contract peer | 八个 strata verdict | advantage ratio | candidate throughput |
|---|---|---|---:|---:|
| 64 个 mixed-size/alignment allocation、touch、reverse-free 的独立 Heap lifecycle | 相同 vendored mimalloc binary 上的 non-devirtualized `std::pmr::memory_resource` owner | 8/8 `PARITY`，5% NI PASS | 0.999--1.029 | 91.3--94.0 M op/s |
| 128 KiB caller storage 上 reset + 64 个 scratch allocation 的 FixedArena epoch | 等大外部 storage 上按同一 epoch 重建的 `std::pmr::monotonic_buffer_resource` | 8/8 `WIN`，5% NI PASS | 1.196--1.223 | 546.9--552.3 M op/s |
| 一个 256 KiB Buffer 上 64 个非零 3--4 KiB checked subview fill | 相同 owner/provider 上执行相同 validity/range/overflow 检查和同一次 fill dispatch 的 fused lower bound | 8/8 `PARITY`，5% NI PASS | 0.999--1.007 | 22.83--22.95 M op/s |
| warmed、同时持有 64 个 variable-size block 的 thread-safe transient lifecycle | 当前 oneTBB `memory_pool` 加入计费内的 Result/RAII/view/exact-stats 等义 adapter | 8/8 `WIN`，5% NI PASS | 1.382--1.400 | 28.54--28.73 M op/s |

前两行回答 owner/arena 常用生命周期是否因 HYLE 形状产生 material 损失；第三行回答 Compute typed
shell 实际消费的 Mem view 是否增加 material dispatch/validation 成本；第四行回答完整的 fallible、
thread-safe、可观测 transient pool 语义，而不是把轻量 allocator 与重语义 pool 混比。所有 owner 和
working set 都在计时外创建并 warm；计时内的 allocation、touch、range operation、RAII return 和应有
stats 更新均完整计费。

## 资格与留存身份

- deep verifier：`QUALIFIED`；candidate certification eligible，environment `CURRENT`，artifact paths
  verified，provenance `EXACT`，product policy `consumer-evidence-v1`；
- A/A、5%/10% delta、timer floor、placement 与 `mem-v2-resource` oracle 全部 PASS；
- retained local bundle：
  `C:\public\project-hyle\_validation\mem-v2-perf-evidence-9eda54f9`；
- manifest SHA-256：`66919c54c9c61f696551c97c079e720eb6da8e89886bc6b8ecd3ad7a3588b983`；
- summary SHA-256：`aa43e2ec0be62563dbfea1fb98df1834b9c39783a4e8f5755b9cdf5dc535f7bc`。

retained bundle 是可重放工程证据，不是仓库内容或长期签名证书。candidate、合同、peer revision、
consumer provenance、机器、instrument 或 bundle 任一改变且现有声明需要续签时，必须重新 qualification。

## 声明边界

- mapping/residency、CompositeBuffer 与 LifetimePlanner 依其 syscall/copy/复杂度和资源合同验收；没有
  同合同正常 consumer timing 时不制造点值排名；
- CUDA/Vulkan provider 只签正确 native route、offset/range、无隐藏 allocation/copy 和真实硬件运行，
  不声明跨 backend throughput SOTA；
- 当前证据不外推多 socket NUMA、端到端模型加载、KV policy、token throughput、macOS 或 Linux 性能；
- Formal performance 仍独立为 `PENDING`。
