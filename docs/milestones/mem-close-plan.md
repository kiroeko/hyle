# mem 全面收口计划

> 状态：执行中（功能面已落地；性能、TSan、macOS 与 API freeze 尚未签字）
> 建立日期：2026-07-22
> 设计源：[`../architecture/mem-design.md`](../architecture/mem-design.md)

## 1. 收口定义

mem 收口不是减少功能数量，而是封闭一组完整、正交、可组合的 host-RAM
能力。每个公开类型必须有明确的 allocation shape、storage ownership、并发、容量、
生命周期和失败合同；benchmark 只验证已经由用户场景确定的形状，不能反向创造 API。

收口完成后，原则上只增加兼容策略或实现，不再反复推翻核心类型。

## 2. 模块边界

mem 拥有普通 CPU-addressable host RAM 的分配与生命周期原语：

| 能力族 | 目标公开能力 | 常用场景 |
|---|---|---|
| allocator core | `SystemAllocator`、`Allocator`、`PmrResource` | 默认分配、运行时注入、PMR 接入 |
| allocator adapters | `QuotaAllocator`、`StatsAllocator`、`LeakTracker` | byte budget、统计、诊断 |
| region | `Region`、`FixedRegion`、`PooledRegion`、`StackAllocator` | request/frame、parser scratch、graph phase、外部 workspace |
| fixed pools | `SharedPool`、`SharedPoolCached`、`FixedPool` | hard capacity、TLS churn、外部固定 storage 的变长分配 |
| object pools | `ObjectPool`、`ObjectPoolCached`、`GrowablePool`、`PoolLoan<T>`、Preserve policy | task/context/node 等对象复用 |
| topology | `HugeRegion`、NUMA/THP policy 与 capability/result | HPC、模型权重、长驻大块内存 |
| runtime specialization | `FiberStack`、`FiberStackPool` | scheduler stack |
| diagnostics | failpoint、liveness、poison、统计快照 | OOM、double return、泄漏与并发诊断 |

不属于 mem：GPU staging/completion（`compute/transfer`）、graph liveness planner
（`graph`）、io_uring registered-buffer lifecycle（`io`）和 blocking pool wait
（由 `sched` 同步原语组合）。

## 3. 统一公开合同

每个公开 pool/region/allocator 必须在头文件与设计文档中回答：

1. size model：fixed block、fixed object、variable-size 或 bump。
2. storage ownership：owned、external 或 borrowed。
3. concurrency：single-owner、shared 或 thread-cached。
4. capacity：hard bounded、soft cached、growable 或 byte-limited。
5. lifetime：individual free、bulk reset、reconstruct 或 preserve-return。
6. exhaustion：NULL、`Result`、throw adapter 或 caller-contract failure。
7. backing 与借用对象的存活顺序。
8. reset/destruction 时 outstanding allocation 的要求。
9. foreign pointer、double return 和重复 destroy 的诊断级别。
10. `used/reserved/peak/capacity` 是精确值、近似值还是稳定快照。

资源不足必须可恢复；只有调用方违反已经公开的合同才允许 assert/fail-fast。

## 4. 执行阶段

### M0：隔离与基线

- [x] mem 与 compute/Vulkan 改动分别审查，不让 GPU memory 语义回流到 mem。
- [ ] 保存当前公开 API、C 导出符号、测试和三平台 benchmark 基线。
- [x] 以 `mem-design.md` 为设计真源，旧 milestone 只保留历史证据。

退出：任意改动都能映射到一个合同与一个验证场景。

### M1：合同、命名与失败模型

- [x] 为现有公开类型补齐统一合同。
- [x] 当前单块 `Arena` 改名 `PooledRegion`，保留兼容 alias/C API。
- [x] `Region` 专指 owned growable bulk-reset region。
- [x] 固化 C ABI 范围与兼容策略。
- [x] 统一 invalid argument、exhaustion、backing OOM 的处理。

### M2：allocator core 与可组合 adapter

- [x] 全面使用 checked add/mul/align。
- [x] 增加确定性 `FailAllocator` 测试夹具，覆盖 Region backing allocation point。
- [x] 实现 `QuotaAllocator`：`used/peak/limit`、严格 reserve、wrap-safe realloc。
- [ ] 保证 Quota、Stats、LeakTracker 可叠加且不依赖隐式全局状态。

### M3：Region 能力族

- [x] `Region`：owned growable blocks、large block、hard size limit。
- [x] `clear()` 复用普通块，`release()` 归还所有容量。
- [x] `used/reserved/peak/limit` 精确可观察。
- [x] `FixedRegion(span<byte>)`：external storage，无隐藏分配和 fallback。
- [x] `PooledRegion`：从 hard `SharedPool` 借一个固定块。
- [x] `StackAllocator` 与 FixedRegion 共用 checked bump cursor。

### M4：Pool 与对象生命周期

- [x] 保持 `SharedPool` hard capacity 与 remote-free visibility。
- [x] 保持 `SharedPoolCached` soft visibility 与显式 publish。
- [x] 实现 `FixedPool(span<byte>)`：external fixed storage、variable-size allocate/free。
- [x] 实现 move-only `PoolLoan<T>`，自动 destroy/return。
- [x] 实现名称明确的 Preserve object pool/policy，不与 TLS Cached 混淆。
- [ ] diagnostics 构建支持 membership、liveness、double return、outstanding loan。

### M5：Topology 与诊断

- [x] HugeRegion 可观察 requested/accepted policy 与 placement result。
- [x] NUMA local/interleave capability query。
- [x] best-effort 与 strict policy 分开。
- [ ] allocation/free poison、failpoint 和统计快照覆盖新增类型。

### M6：C ABI、迁移与使用场景

- [x] 新增 untyped Region、FixedRegion、FixedPool、Quota C API。
- [x] options struct 使用 `struct_size` 预留 ABI 扩展。
- [x] `hyle_arena_*` 保留兼容并标记为 pooled single-block 语义。
- [x] 增加 request region、external workspace、hard pool、cached pool、RAII loan、quota、NUMA 示例。
- [x] 让 graph phase scratch 实际消费通用 `Region`，验证形状但不以内部消费者限制能力存在。

### M7：正确性认证

- [x] allocation point OOM、overflow、alignment、zero-size。
- [x] remote free、TLS thread exit、reset/destruction 与 outstanding allocation。
- [ ] foreign pointer、double return、duplicate destroy。
- [x] randomized reference-model 测试。
- [ ] ASan、UBSan、TSan。
- [ ] Windows MSVC、Linux GCC/Clang、macOS arm64。
- [x] C header self-contained、纯 C 编译和导出符号检查。

### M8：性能认证与 freeze

benchmark 必须按真实场景组织：request/frame bulk allocation、producer/consumer
handoff、same-thread batch reuse、hard admission quota、external workspace、expensive
context preserve、adapter overhead、NUMA/大页吞吐与 RSS。

- [x] 只和同语义实现比较；hard quota 不与 soft cache 混比。
- [ ] 报告 throughput、p50/p95/p99、峰值内存、fragmentation 与构造成本。
- [ ] 当前 HYLE 主场景无解释回退不得超过 5%。
- [ ] throughput-oriented 类型三平台不明显输同语义成熟实现，跨平台几何均值至少持平。
- [ ] 证书绑定 commit、编译器、参数和原始结果。
- [ ] 建立 API/ABI baseline，进入兼容扩展期。

## 5. 2026-07-22 执行记录

本轮已经完成：

- 新增 `QuotaAllocator`、owned `Region`、external `FixedRegion`、external
  variable-size `FixedPool`、`PoolLoan` 与 `PreservingObjectPool`。
- 单块 `Arena` 精确命名为 `PooledRegion`，保留 C++ alias 与 `hyle_arena_*`。
- C ABI 增加 Quota/Region/FixedRegion/FixedPool 和 HugeRegion policy result；Region
  options 带 `struct_size`。
- graph level-based parallel evaluation 的 phase scratch 改为实际消费 `Region`；异步
  coroutine frame 仍由原 allocator 管理，没有跨 phase 借用 scratch。
- 修正跨平台 `system_good_size` 合同：它是推荐 request size，不预测一次 allocation 的
  exact usable size。
- 修正旧 PMR benchmark：MSVC 上 `monotonic_buffer_resource::release()` 不能被当作
  external initial buffer 的可移植 rewind；现在每个 request 在同一 buffer 上重建 resource。

验证矩阵：

| 环境 | 结果 |
|---|---|
| Windows/MSVC Debug | mem 146/146；纯 C 通过；graph 47/47 |
| Windows/MSVC ASan | 非 death mem 139/139；纯 C 通过 |
| Ubuntu 24.04 / GCC 14.2 | mem 146/146；纯 C 通过 |
| Ubuntu 24.04 / GCC 14.2 UBSan | 非 death mem 139/139；纯 C 通过 |
| Ubuntu 24.04 / GCC 14.2 TSan | 编译通过；WSL 启动前因 `unexpected memory mapping` 无法运行，不计通过 |
| compute/Vulkan 隔离观察 | compute 78 pass/2 skip；Vulkan 14 pass/2 skip；GPU await 2/2 |

本机 F workload 首轮结果与解读写在 [`../guide/mem.md`](../guide/mem.md)。其中
FixedPool 约 20.45 ns/op、oneTBB fixed_pool 约 8.66 ns/op，约 2.36 倍差距；两次候选优化
（更窄 bins、lazy coalescing）均无改善，已撤回，不以增加复杂度换取零收益。因此
FixedPool 与 owned Region 的性能 freeze 尚未签字。

仍未完成：

1. FixedPool 的 profiling-driven 优化或真实 SLA 取舍；不能靠更换有利 workload 过门。
2. 独立 Linux host 上运行 TSan，补 macOS arm64 与 Linux Clang。
3. 完整 failpoint/foreign-pointer/poison diagnostics，而不只是 Region OOM 与 FixedPool
   double-return/reference-model。
4. p50/p95/p99、RSS/fragmentation、三平台原始结果与 commit-bound 证书。
5. compute/Vulkan staging/completion 架构与纳秒 benchmark 单独复审；现有测试通过不等于
   该设计已经收口。

## 6. 收口退出门槛

- [ ] 能力矩阵每项已实现，或明确归属其他模块。
- [ ] 不存在名称大于实际能力的公开类型。
- [ ] 所有类型具备完整 ownership/concurrency/capacity/lifetime/failure 合同。
- [ ] resource exhaustion 不因实现方便而 fatal。
- [ ] C/C++ API、文档和实现一致。
- [ ] sanitizer、故障注入、并发模型和支持平台测试全部通过。
- [ ] 性能来自真实 workload，并与同语义成熟实现比较。
- [ ] 连续两个功能迭代未因新场景推翻核心类型。

## 7. 提交组织

建议保持下列独立审查单元，不把功能、性能调优和 GPU 重构混成一次提交：

1. `docs(mem): freeze closure capability matrix and contracts`
2. `feat(mem): add quota allocator and failure-model tests`
3. `feat(mem): add owned and external regions`
4. `refactor(mem): name pooled single-block region explicitly`
5. `feat(mem): add external variable-size fixed pool`
6. `feat(mem): add object-pool loan and preserve policy`
7. `feat(mem): expose closure primitives through the C ABI`
8. `test/perf(mem): certify scenarios and freeze the public surface`
