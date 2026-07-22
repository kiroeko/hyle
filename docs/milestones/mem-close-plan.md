# mem 全面收口计划

> 状态：执行中（能力族与 C++ 公开形状已落地；C draft surface、Linux Clang 与独立主机 TSan 尚未签字）
> 建立日期：2026-07-22
> 设计源：[`../architecture/mem-design.md`](../architecture/mem-design.md)

## 1. 收口定义

mem 收口不是减少功能数量，而是封闭一组完整、正交、可组合的 host-RAM
能力。每个公开类型必须有明确的 allocation shape、storage ownership、并发、容量、
生命周期和失败合同；benchmark 只验证已经由用户场景确定的形状，不能反向创造 API。

首次发布前不承担兼容包袱；有真实场景、成熟库形状和性能证据时允许直接改 API/C surface。
收口完成后才要求核心类型稳定，避免再次由 synthetic benchmark 反推产品形状。

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
- [x] 保存当前公开 surface、C 导出符号、测试和 Win/Linux benchmark 当前证据；它们用于漂移检测，不是兼容承诺。
- [x] 以 `mem-design.md` 为设计真源，旧 milestone 只保留历史证据。

退出：任意改动都能映射到一个合同与一个验证场景。

### M1：合同、命名与失败模型

- [x] 为现有公开类型补齐统一合同。
- [x] 当前单块形状统一命名 `PooledRegion`；发布前删除 `Arena/reset` 兼容壳。
- [x] `Region` 专指 owned growable bulk-reset region。
- [ ] 审查并确定首次发布的 C surface；当前 manifest 仅检查草案同步。
- [x] 统一 invalid argument、exhaustion、backing OOM 的处理。

### M2：allocator core 与可组合 adapter

- [x] 全面使用 checked add/mul/align。
- [x] 增加确定性 `FailAllocator` 测试夹具，覆盖 Region backing allocation point。
- [x] 实现 `QuotaAllocator`：`used/peak/limit`、严格 reserve、wrap-safe realloc。
- [x] 保证 Quota、Stats、LeakTracker 可叠加且不依赖隐式全局状态。

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
- [x] diagnostics 构建支持 membership、liveness、double return/destroy、outstanding allocation。

### M5：Topology 与诊断

- [x] HugeRegion 可观察 requested/accepted policy 与 placement result。
- [x] NUMA local/interleave capability query。
- [x] best-effort 与 strict policy 分开。
- [ ] allocation/free poison、failpoint 和统计快照覆盖新增类型。

### M6：C ABI、迁移与使用场景

- [x] 新增 untyped Region、FixedRegion、FixedPool、Quota C API。
- [x] options struct 使用 `struct_size` 预留 ABI 扩展。
- [x] C/C++ 统一为 `hyle_pooled_region_*` / `PooledRegion` 和 `clear()`。
- [x] 增加 request region、external workspace、hard pool、cached pool、RAII loan、quota、NUMA 示例。
- [x] 让 graph phase scratch 实际消费通用 `Region`，验证形状但不以内部消费者限制能力存在。

### M7：正确性认证

- [x] allocation point OOM、overflow、alignment、zero-size。
- [x] remote free、TLS thread exit、reset/destruction 与 outstanding allocation。
- [x] foreign pointer、double return、duplicate destroy。
- [x] randomized reference-model 测试。
- [ ] ASan、UBSan、TSan。
- [ ] Windows MSVC、Linux GCC/Clang、macOS arm64。
- [x] C header self-contained、纯 C 编译和导出符号检查。

### M8：性能认证与发布面收口

benchmark 必须按真实场景组织：request/frame bulk allocation、producer/consumer
handoff、same-thread batch reuse、hard admission quota、external workspace、expensive
context preserve、adapter overhead、NUMA/大页吞吐与 RSS。

- [x] 只和同语义实现比较；hard quota 不与 soft cache 混比。
- [ ] 报告 throughput、p50/p95/p99、峰值内存、fragmentation 与构造成本。
- [ ] 当前 HYLE 主场景无解释回退不得超过 5%。
- [ ] throughput-oriented 类型三平台不明显输同语义成熟实现，跨平台几何均值至少持平。
- [ ] 证书绑定 commit、编译器、参数和原始结果。
- [x] 建立 C++/C current-surface 检查，捕获意外漂移；首次发布前不承诺兼容。

## 5. 2026-07-22 执行记录

本轮已经完成：

- 新增 `QuotaAllocator`、owned `Region`、external `FixedRegion`、external
  variable-size `FixedPool`、`PoolLoan` 与 `PreservingObjectPool`。
- 单块形状精确命名为 `PooledRegion`，C++ 与 C 都删除未发布的 `Arena/reset` 旧名。
- C ABI 增加 Quota/Region/FixedRegion/FixedPool 和 HugeRegion policy result；Region
  options 带 `struct_size`。
- graph level-based parallel evaluation 的 phase scratch 改为实际消费 `Region`；异步
  coroutine frame 仍由原 allocator 管理，没有跨 phase 借用 scratch。
- 修正跨平台 `system_good_size` 合同：它是推荐 request size，不预测一次 allocation 的
  exact usable size。
- 修正旧 PMR benchmark：MSVC 上 `monotonic_buffer_resource::release()` 不能被当作
  external initial buffer 的可移植 rewind；现在每个 request 在同一 buffer 上重建 resource。
- 建立 C++ current-surface 编译检查，覆盖全部公开 mem 能力族；C++ object layout 不承诺
  binary ABI。C manifest 同时校验 header 声明、纯 C 消费引用和 DLL/ELF 实际导出，但只是
  未发布草案的同步检查。审查已修：有 backing 的 opaque control 也由该 backing 分配；非法
  C layout/alignment 返回 NULL + last-error；adapter 统一 `*_as_allocator`。
- `HYLE_TESTING` 为 shared/cached/object/growable pool 增加 liveness、foreign/double
  return/destroy 和 outstanding allocation fail-fast；shipping 构建不含 per-slot 状态和
  热路径诊断指令。
- allocator composition 已认证 `Stats -> Quota -> SharedPool`、`Quota -> Stats ->
  SharedPool` 和 `PMR -> Stats -> Quota -> SharedPool`，没有隐式全局状态。

真实场景证据直接绑定功能测试，而不是另造有利 benchmark：

| 用户场景 | 合同证据 |
|---|---|
| request/frame bulk lifetime | `Region.GrowsHonorsLimitAndBulkReuses`、graph phase scratch 消费 |
| external workspace | `FixedRegion.ExternalStorageHasNoFallbackAndResets`、`FixedPool.RandomizedMixedSizeModelPreservesLiveAllocations` |
| producer/consumer hard handoff | `SharedPool.CrossThreadFreeMigratesAndReclaims`、`RemoteFreeImmediatelyVisibleWhileThreadLives` |
| same-thread cached reuse + handoff | `SharedPoolCached.SameThreadTlsCacheReusesPointer`、`RemoteTlsSoftThenPublishRestoresHardCapacity` |
| hard byte admission + observability | `QuotaAllocator.ConcurrentAdmissionNeverExceedsHardLimit`、三组 `AllocatorComposition` 测试 |
| expensive context preserve + RAII | `PreservingObjectPool.ConstructsOnceAndPreservesStateAcrossLoans`、`ObjectPool.ScopedLoanReturnsOnEarlyExit` |

验证矩阵：

| 环境 | 结果 |
|---|---|
| Windows/MSVC Release（`7f270bb`） | mem 167/167；纯 C 通过；C surface 93/93；全仓 CTest 50/50 |
| Windows/MSVC ASan（`7f270bb`） | 非 death mem 149/149；纯 C 通过；C surface 93/93 |
| Ubuntu 24.04 / GCC 14.2 UBSan（`7f270bb`） | mem 167/167（含 death）；纯 C 通过；C surface/纯 C 引用/ELF 导出均 93/93 |
| Apple M4 / AppleClang 21 Release（`7f270bb`） | mem 167/167；纯 C 与 C surface 通过 |
| Apple M4 / AppleClang 21 ASan（`7f270bb`） | 非 death mem 149/149；纯 C 通过 |
| Ubuntu 24.04 / GCC 14.2 TSan（前一基线） | 编译通过；WSL 启动前因 `unexpected memory mapping` 无法运行，不计通过 |
| compute/Vulkan 隔离观察 | compute 78 pass/2 skip；Vulkan 14 pass/2 skip；GPU await 2/2 |

F workload 的正式结果与解读集中写在 [`../performance/benchmarks-mem.md`](../performance/benchmarks-mem.md)。FixedPool 的首轮
20.45 ns/op 对 oneTBB 8.66 ns/op 暴露了真实差距；更窄 bins、lazy coalescing 和 metadata
overlay 候选均未改善并已撤回。路径分解定位到 mixed-size free 后的 coalesce→bin→下一次
allocate→split→bin 重复维护，最终加入立即合并后的 one-entry recent-free cache，并把 alignment
改成 block-relative checked arithmetic。Windows/MSVC Release `n=15` 新结果为 HYLE 10.314、
oneTBB 10.094 ns/op，HYLE/oneTBB throughput ratio 中位数 0.983、范围 0.912–1.023，spread/sign
均判 TIE。Linux/GCC 14.2 为 HYLE 7.555、oneTBB 7.774 ns/op，ratio 1.027、范围
0.948–1.088，spread TIE / sign WIN（13/15，p=7.4e-3）。两平台在同机错开运行。FixedPool
已通过 Windows/Linux 性能闸。Apple M4 / AppleClang 21 发现 4P+6E 调度审计缺口后，harness
显式请求 `USER_INTERACTIVE`，并以 `UTILITY` 复核；两组各 `n=31`，FixedRegion、owned Region、
FixedPool 均通过 sign 闸门。Darwin 无受支持的硬亲和/实际 core-class 证明，因此 M4 只签
application QoS 边界，不写成所有 P/E 核组合均已认证。

仍未完成：

1. 在 Linux Clang 重跑正确性与同一 F workload；GCC 14.2、AppleClang 21 已补。
2. 独立 Linux host 上运行 TSan；WSL 的启动前 mapping 失败不计通过也不计产品失败。
3. 完成 C surface 产品审查：独立 mem error channel、strict no-heap C 是否需要 in-place control、
   Stats/Leak 是否有真实 C 消费者；manifest 在此之前不冻结。allocation/free poison 与新增类型的完整 failpoint matrix；pool foreign/double/outstanding
   diagnostics 已完成。
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

1. `docs(mem): close capability matrix and contracts`
2. `feat(mem): add quota allocator and failure-model tests`
3. `feat(mem): add owned and external regions`
4. `refactor(mem): name pooled single-block region explicitly`
5. `feat(mem): add external variable-size fixed pool`
6. `feat(mem): add object-pool loan and preserve policy`
7. `feat(mem): expose closure primitives through the C ABI`
8. `test/perf(mem): certify scenarios and record the current public surface`
