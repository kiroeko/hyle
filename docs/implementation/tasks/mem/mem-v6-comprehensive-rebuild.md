# Mem v6：当前推理内存超集与产品形状重构

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮与 `mem_vm-v2` 协同，但 `mem` 与 `mem_vm` 仍是两个独立组件、两份候选和两项终局裁决。
> 当前状态为 `CANDIDATE`。下方能力账本、API freeze 与五项有限 `FIX` 已冻结并由候选实现；
> 后续只运行预声明验证，不得扩面或提前写成 `CLOSED`。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`mem` / Mem host-storage semantics /
  `mem-v6-comprehensive-rebuild` / `REOPEN`；baseline
  `5aaeaa770be72cd7a58465659b2abc7d98a2d89b`，tree
  `2035afb59e4a658526bc3f39e458b305a36f1a1d`；
- trigger ID：`MEM-V6-PLATFORM-V8-CONSUMER-REPLAY`；
- Mem v5 候选之后，其直接依赖 Platform 的 NUMA、VM range-error 与 CPU/resource mechanism 已进入
  `platform-v8` 候选 `c39df6f94c636ab8fa94cb700dd263763a42d680`。base Mem 的
  `ConcurrentBlockPool` shard policy、1 GiB reservation/NUMA validation 以及 Mem 产品中由
  `mem_vm` 承载的 placement/storage 路线都会消费这些真实产品语义；旧候选的 dependency、性能和
  delivery 证据不能跨该 identity 自动继承；
- owner 同时要求以当前量化推理旅程、当前 ggml 能力、成熟 allocator/pool 实现和完整 public-surface
  reverse trace 重新裁决功能超集、自然 C++ API、内部复用、系统交互、性能、红队、文档和三种交付。
  这不是按 benchmark symbol 扩 API；只有审计后冻结的 `MUST/FIX` 才能进入实现；
- 当前未提交工作只涉及 Mem/Container/Util 历史性能 evidence 的 comparison-class 重整和一个 Container
  queue harness，没有 `include/hyle/mem`、`src/mem` 或 `src/mem_vm` 产品改动。它保留为独立审计输入，
  不自动成为本轮性能 `PASS`、产品 trigger 或候选内容；
- 当前锁定参考身份：llama.cpp
  `74a7c897f049c17e7080423aa2111776eff6ebbf`、standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`、mimalloc upstream main
  `c683b7c60c91cb2809977e37d3f4d084d7497180`；shipping mechanism 仍固定为已审
  mimalloc v3.5.1 commit `34fbd7e7cd4627424490afe19b20f8066bfc537d`，升级与否必须另有 material
  产品事实，不能因 upstream main 更新而漂移；
- 初始受影响面只包含 base `mem` 的 allocator/resource/buffer/pool/arena/planner/API/documentation、
  它对已冻结低层组件的实际使用，以及 Mem 产品聚合/三交付。`mem_vm` 的 mapping/OS storage 由协同
  `mem-vm-v2` 独立处理；`mem_reclaim` 没有新事实时不重开；
- `CLOSED -> REOPENED` 前 registry 快照位于 canonical checkout 外：
  `C:\public\project-hyle\_validation\registry-before-mem-v6-mem-vm-v2-reopened-5aaeaa77.json`，
  SHA-256 `036760D948DFF33F9600A7B926F9F7E7CCB3304A328842EB2C7706E9383E9D6B`。

## 初步停止边界

- 先完成 capability-family map、全部 public entry reverse trace、dependency/reuse 与 OS/provider
  interaction audit；随后才进入 `SCOPED -> AUDITED` 并冻结有限修复集；
- 不重构 Compute/Graph、设备 backend、Tensor/KV/model policy、Native FFI/C API、稳定 C++ ABI、
  公共性能统计核心或无真实语义差异的 benchmark-only 入口；
- 不能在冻结的正常 consumer、comparison class、资源计费、5% material margin 与 holdout 以前计时；
  旧 natural-route/diagnostic 数字不冒充 same-contract SOTA；
- capability、correctness/security、API ownership、系统交互或 delivery 出现 material finding 时修复；
  只有亚纳秒、地址/执行顺序翻转或没有机制解释的非 material 差异时停止，不改产品形状。

## Closure card（`SCOPED`）

- component / owner / round / kind：`mem` / Mem host-storage semantics /
  `mem-v6-comprehensive-rebuild` / `REOPEN`；baseline
  `5aaeaa770be72cd7a58465659b2abc7d98a2d89b`；
- demand：`MEM-V6-PLATFORM-V8-CONSUMER-REPLAY`。真实 consumer 是量化模型加载及执行期 host
  storage：建立 allocator/resource owner，创建长期权重 buffer，复用短寿命 scratch，按已知 lifetime
  规划 backing，并在受限容量下报告失败；
- 范围：system/isolated allocator、standard/PMR adapter、arena/stack/fixed heap、fixed/growable/
  concurrent pool 与显式 local cache、quota/stats/pressure、resource-neutral Buffer/View/
  CompositeBuffer、transient BufferPool、tensor-free LifetimePlanner、进程级 1 GiB mimalloc reserve；
- 非目标：`mem_vm` 拥有的 OS mapping/residency，`mem_reclaim`，device/provider 实现、Tensor/KV/model
  policy、async completion、C/Native FFI、稳定 C++ ABI、公共性能仪器和 Compute/Graph；
- 直接依赖：`core`、`error`、`platform`，私有锁定机制为 mimalloc v3.5.1
  `34fbd7e7cd4627424490afe19b20f8066bfc537d`；直接消费者至少为 `mem_vm`、Container、Util-runtime、
  Sched、IO、Compute 与 Graph；Mem product 还聚合未重开的 `mem_reclaim`；
- 待复用设施：Core cache/alignment、Error `Status/Result`、Platform 的 CPU/NUMA/VM 事实和机制、标准
  allocator/PMR vocabulary；SIMD 不适用于 allocator control path，Container 不为内部小型固定 metadata
  反向制造依赖；每个保留的本地 primitive 必须在审计表说明合同差异；
- tier：`P1`。冻结正常 workload 为同一 vendored mimalloc owner 上的 runtime-erased isolated Heap
  mixed allocation lifecycle；若审计证明 BufferPool 或 pool handoff 是不同且同样常用的热合同，最多再
  加一个预冻结 holdout，不得按已有 benchmark 行枚举操作；
- 性能 class：Heap 为 `SAME_CONTRACT`，peer 是非去虚化 PMR owner over the same mimalloc binary；
  其他能力默认 P0 复杂度/资源检查。5% material/non-inferiority margin；计费 allocation/free、touch、
  owner lifetime、setup/teardown 与 delivery/codegen，不把更弱语义路线冒充同合同；
- 总调查预算：8 小时；其中性能最多 120 分钟。最多两轮产品修正；性能异常只允许一次反序复现、
  一次 codegen/profile 机制检查、一个产品修改和一次重跑；
- correctness machines：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 machine：`W1-WIN`；instrument：
  `closure-perf-v1`，P0 辅助为 `closure-correctness-v1`；
- strong inputs：当前 llama.cpp `74a7c897f049c17e7080423aa2111776eff6ebbf`、standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`；allocator/resource 以 mimalloc v3.5.1 与当前
  oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f` 为最多两个 peer，arena/pool 可用 Folly
  `65749da412d92fe27d87ae45a8ef670720712858` 替换 oneTBB，但每个 family 仍不超过两个；
- provenance：真实 loader/executor 路线为 `NORTH_STAR_JOURNEY`，已有 installed Mem sample 为
  `REAL_CONSUMER`，孤立 Heap batch 仅为 `PROVISIONAL_ARCHETYPE`；
- 适用门：全部 public surface reverse trace；正常、空/零、边界/溢出、OOM/invalid、move/destroy、
  并发、property/stress、ASan/UBSan/TSan；source embedding、installed static/shared、直接消费者、
  FFI 清洁、文档/易用性、内部复用、红队和 P1 工程性能；取消/close 与真实 ISA/backend 仅在语义
  确实不适用后才可记 `NOT_APPLICABLE`；
- 输出：能力 family map、七问、冻结 API、有限 `FIX` ledger、candidate、验证 manifest、终局 receipt
  和同候选 LOC；
- 停止：所有 `MUST` 为 `COMPLETE/ROUTED`、适用门通过且无 blocker 即 `CLOSED`；固定预算内无法
  裁决的 correctness/contract/P1 问题具名进入 `BLOCKED`，不得继续发明测试姿势。

## Capability-family audit 与有限 FIX（`AUDITED`）

### 当前超集与 consumer 反向追踪

审计覆盖全部 `include/hyle/mem` base headers、`src/mem`、Mem tests/sample、当前 llama.cpp/ggml 的
host allocator/scratch/buffer/pool/model-load 旅程、mimalloc v3.5.1、oneTBB 与 Folly。结果按任务而非
symbol 数量归并如下：

| family | 当前自然能力 | 覆盖/差距 | 裁决 |
|---|---|---|---|
| allocator / standard adapters | process system allocator、isolated `Heap`、runtime-erased ref、`std::allocator`/PMR、usable/good-size、collect | aligned/sized realloc、owner/move/OOM 和 private mimalloc mechanism 完整 | `COMPLETE` |
| arena / external variable heap | owned/fixed/stack/pooled bump owner；caller-storage TLSF-like variable-size individual free/realloc/reset | phase scratch、LIFO 与 hard external workspace 均有自然独立类型 | `COMPLETE` |
| fixed/typed/growable pools | concurrent hard block pool、explicit local cache、reconstruct/preserve typed pool、growable single-owner pool、RAII loan | 固定/未知容量、remote free 与 bulk release 路线完整 | `COMPLETE` |
| quota / instrumentation / pressure | strict byte quota、opt-in stats/leaks、allocator + whole-process snapshot | Linux whole-process fields 实为 mimalloc 子域；公开合同不实 | `PARTIAL -> FIX` |
| resource / byte ownership | property-bearing resource ref、move-only Buffer、host map、adopt、views、composite、sync copy/fill/coherence | 足以承载未来 host/device provider，Mem 不反向依赖 Compute | `COMPLETE` |
| transient resource pool | bounded retained bytes/count/slack/growth、lease/take/trim/OOM retry | 能力完整；`max_entries_scanned_per_bucket` 泄漏内部 bucket/benchmark 形状 | `COMPLETE + API FIX` |
| lifetime planning | tensor-free size/alignment/interval/alias、多 backing、deterministic placement 与 peak facts | static graph scratch/model planning 任务完整；dynamic graph/eviction 是上层策略 | `COMPLETE` |
| explicit allocator large-page arena | process-start GiB-sized mimalloc arena、可选 NUMA preference | 旧 API 丢失 partial side effect，并把 vendor 的 1 GiB→2 MiB fallback 误报成 1 GiB OS pages | `PARTIAL -> FIX` |
| VM/reclamation | anonymous/file/shared/large/guarded mapping；read-mostly reclamation | 分别由 `mem_vm`/`mem_reclaim` 独立因子拥有，Mem product 聚合 | `ROUTED` |

没有准入新的 base-Mem owner family：自己增长且 individual-free 的通用可变池由
`system_memory_resource()` + 标准 PMR pool 完整路由；Tensor/KV placement、eviction、async completion、
device provider 与 secure-secret policy 都属于已列非目标或尚无具名基础 consumer，不能因“可能有用”复制
上层系统。这个结论不取消未来由真实 consumer 触发的 reopen。

### 冻结公开形状

`SystemMemorySnapshot` 保留 allocator current/peak 四项；process current/peak resident 来自
`platform::process_memory_info()`。只有 OS 能证明时才给出：

```cpp
std::optional<std::uint64_t> process_private_committed_bytes;
std::optional<std::uint64_t> process_peak_private_committed_bytes;
```

旧的必填 `process_committed_bytes/process_peak_committed_bytes` 删除；它们不能再混入 mimalloc 统计。

large-page arena reserve 改为可观察 partial outcome，并把容量单位与 OS page granularity 分开：

```cpp
enum class LargePagePlacement : std::uint8_t {
  kUnconstrained,
  kPreferenceUnverified,
};

enum class LargePageGranularity : std::uint8_t { kUnverified };

struct LargePageArenaReservationReport {
  std::size_t requested_gib;
  std::size_t reserved_gib;
  int requested_numa_node;
  LargePagePlacement placement;
  LargePageGranularity page_granularity;
  Status outcome;
  bool complete() const noexcept;
};

Result<LargePageArenaReservationReport> reserve_large_page_arena(...);
```

参数/溢出是 `Result` error；调用 vendor 后即使只完成一部分也返回 report，`outcome` 仅在全部完成时 OK。
process-global arena 不能 rollback；selected-node 只是 mimalloc/OS preference 且明确 unverified。GiB
只是 request chunk/容量单位：vendor 会回退到 2 MiB/native large pages，公开 arena API 不能证明最终
物理页粒度，因此也明确 unverified。默认 `kNumaAny` 不再虚称 interleave。

`BufferPoolOptions` 删除 `max_entries_scanned_per_bucket`；固定有界 search 是实现复杂度而非消费策略。
retention、reuse slack、growth 和 allocation granularity 保留，因为它们直接决定真实资源预算与复用语义。

### 冻结 FIX ledger

| FIX | 修改 | 成本/复杂度 | 证据 |
|---|---|---|---|
| `MEM6-F01` | 组合 Platform whole-process facts，改 optional/private commit 字段和 sample/tests | 冷查询；默认 allocation/free 热路零变化 | 三平台 process allocation-domain fixture |
| `MEM6-F02` | 以 mimalloc `*_at_ex` + `mi_arena_area` 报告实际 arena 容量和 partial outcome；移除虚假 1 GiB page/interleave/placement guarantee | 启动期 O(1) vendor call；无常态 metadata | zero/invalid/partial-capable/granularity-honesty contract tests，base Mem 自己覆盖 |
| `MEM6-F03` | 删除 BufferPool bucket scan 公共 knob，内部固定 bounded scan；改掉 benchmark/proven 注释 | acquire 仍 O(word bits × fixed bound)，对象 layout 只删策略 byte/padding | existing pool property/stress + API surface |
| `MEM6-F04` | 修正 CMake 依赖注释、architecture/user/API docs 和 source/static/shared sample | 不新增依赖；继续 `mem PUBLIC core error platform` | selected factor + 三交付实调用 |
| `MEM6-F05` | 仅运行冻结的 isolated Heap same-contract P1；其它变更签 P0/资源事实 | 5% NI，same mimalloc binary/runtime erasure | `closure-perf-v1` W1-WIN |

以上五票是全部产品授权。allocator/pool/arena/buffer/planner 若验证失败可在同一合同内修 correctness；不得
新增 family、benchmark knob、device/Tensor policy 或公共仪器。红队已明确这些未发现 material mechanism
缺陷，验证不能反向变成无界重写许可。

### 七问预裁决

1. 超集：九族全部 `COMPLETE/ROUTED`，仅两处 false result 和一个 API 污染进入 FIX；
2. 复用：Core alignment/assert、Error Result/Status、Platform process/NUMA facts 已是正确边；不依赖上层；
3. C++ API：owner/lifetime/failure 显式；删除 bucket implementation knob；无 C/Native FFI；
4. 性能：默认 allocator 无统计税，pool/arena 保持既有复杂度；只对真实 Heap runtime-erased 合同计时；
5. build：mimalloc 仍 private locked mechanism，selected Mem 三交付不得导出 `mi_*` 或伪 C symbols；
6. 文档：architecture 讲 owner/route，guide 讲“何时用什么”及 partial/unknown/NUMA 陷阱；
7. 红队：五项 FIX 后复核系统事实、partial side effect、OOM/overflow/concurrency 和直接消费者。

`SCOPED -> AUDITED` 前 registry snapshot：
`C:\public\project-hyle\_validation\mem-v6-mem-vm-v2\module-status-before-audited.json`，SHA-256
`CB157D90B34C09F95BF62DE4F62821286E09E49321B15BBA3992FC14575F096B`。

## Candidate

- product commit：`9604bf4a33f13bd5f84ef78fedce55f305ac9d50`；tree：
  `57adb69f185cf02d42ab7f220ebeccbb91170b1b`；
- `SystemMemorySnapshot` 分离 allocator-domain 与 whole-process OS facts；不可证明字段为
  `std::optional`，不以零或 mimalloc committed 伪装；
- allocator arena API 改为 `LargePageArenaReservationReport`，如实保留 process-global partial side
  effect，并明确请求 GiB 数不证明物理页粒度或 NUMA placement；
- `BufferPoolOptions` 删除 bucket scan benchmark seam；内部 search 仍固定有界，正常对象没有新增
  metadata、allocation 或 dispatch；
- C++ sample 已覆盖 isolated Heap、snapshot、BufferPool 和 large-page report；没有恢复 C/Native FFI；
- `AUDITED -> CANDIDATE` 前 registry snapshot：
  `C:\public\project-hyle\_validation\mem-v6-mem-vm-v2\module-status-before-candidate.json`，SHA-256
  `173814FFC86DB38FC7587C43D64F2433E7CFC4AB6DC09A2E6C33B75CDB37553C`。
