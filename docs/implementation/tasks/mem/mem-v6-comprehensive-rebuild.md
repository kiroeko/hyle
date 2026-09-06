# Mem v6：当前推理内存超集与产品形状重构

<!-- MODULE CLOSURE RECORD: HISTORICAL -->
<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本轮与 `mem_vm-v2` 协同，但 `mem` 与 `mem_vm` 仍是两个独立组件、两份候选和两项终局裁决。
> 本轮终局裁决为 `CLOSED`。下方能力账本、API freeze、五项有限 `FIX`、全部适用工程门与接受限制
> 已绑定不可变候选；本记录只作历史证据，不再授权产品修改。

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

## Owner 七问终裁

1. **第三方产品路径：**唯一直接产品 mechanism 是 private、锁定的 mimalloc v3.5.1
   `34fbd7e7cd4627424490afe19b20f8066bfc537d`。它拥有 system/isolated heap 的底层实现，HYLE 拥有
   allocator/resource/owner 语义、API 与交付决策；public type 不泄漏 mimalloc。标准 PMR 是公开互操作
   vocabulary；oneTBB/Folly/llama.cpp/ggml 只作 capability/consumer peer，不进入产品链接。升级必须按
   dependency trigger 定向重开 allocator family，不能自动继承本轮性能证据。
2. **推理引擎能力完整度：**allocator/adapter、arena/external heap、fixed/typed/growable pool、quota/
   diagnostics/pressure、resource/Buffer、transient BufferPool、lifetime planning、large-page arena 与
   routed VM/reclamation 九族全部 `COMPLETE/ROUTED`。当前 llama.cpp/ggml model-load、scratch、long-lived
   weight、resource-neutral future device provider 与 static lifetime packing 均有自然路线；Tensor/KV
   eviction、async completion、device backend 和 IPC protocol 是上层 owner，不是漏做 Mem family。当前
   evidence 下未处置 `MUST=0`。
3. **内部复用与加速：**Core 提供 alignment/assert/cache facts，Error 提供全部可恢复 `Status/Result`，
   Platform 提供 whole-process/NUMA/VM truth，标准 allocator/PMR 提供互操作；实际依赖与
   `hyle_mem PUBLIC core error platform` 一致。没有反向依赖 Container/Sched/Compute；FixedHeap 的局部
   bitmap/free-list、planner 的本地排序 metadata 与 pool 的 inline slots 均因 ownership、依赖方向或
   固定布局为 `LOCAL`，未重复已有同合同 HYLE facility。verdict `PASS`。
4. **拟真公平性能：**P1 只比较两个同一 mimalloc binary、相同 owner lifetime/alignment/failure 与
   64 次 allocation/touch/sized-free 的 runtime-erased isolated heaps。qualified r2 的 8/8 exact
   `PARITY`、8/8 5% NI 和全部 controls 为 `PASS`；点估计有利于 HYLE 0.68%--2.41%，不宣称 material
   win。其余 family 由复杂度、allocation/metadata/syscall 与资源事实作 P0；没有苹果对橘子排名。
5. **旧 C ABI / Native FFI：**没有 HYLE C header/schema/bridge、FFI option、undecorated HYLE export 或
   FFI-paid layout。static consumer header 的六个 `extern "C" mi_*` declaration 是 private vendored
   mimalloc archive 的链接机制；shared `hyle_mem.dll` 不导出任何 `mi_*`。verdict `CLEAN`。
6. **Source/static/shared：**Windows exact-candidate selected Mem smoke 的 source embedding、relocated
   installed shared 与 relocated installed static consumer 全部编译运行；shared runtime DLL staging 与
   static mimalloc link-only dependency 正确。安装集包含新 `large_mapping.h` / `large_page_reservation.h`
   且不含旧 hugepage headers。额外 `HYLE_COMPONENTS=sched` fresh build 与 `hyle_sched_test` 证明真实
   FiberStack consumer 通过。verdict `PASS`。
7. **C++ API / 文档：**所有 owner move-only，borrow/view、individual/bulk、hard/soft、thread-safe/
   single-owner、hint/guarantee、unknown/partial 都由类型或 Result 明示；删除唯一 bucket scan
   implementation knob。architecture 只讲 owner/route/成本，user guide 按任务说明选型、lifetime、OOM、
   NUMA/large-page 与 mapping 陷阱，performance 页只绑定冻结命题。verdict `PASS`。

## Verification manifest

### Correctness risk matrix

| Risk | Verdict | 证据 / 限制 |
|---|---|---|
| normal；empty / zero | `PASS` | 三平台 selected Mem suites；zero owner、allocation、plan、reserve、pool 与 view cases |
| capacity / boundary / overflow | `PASS` | alignment/count/range/provider maximum、quota、planner、large-page byte overflow |
| OOM / system failure / invalid input | `PASS` | injected backing OOM、constructor rollback、invalid allocator/resource/NUMA；process-global reserve 的不可回滚 partial 被 report 保存 |
| move / destroy / lifetime | `PASS` | Heap/Buffer/arena/pool/planner move/reset/destructor 与 borrowed-view contracts |
| concurrency / reentrancy | `PASS` | concurrent/remote-free/cache/stats/object/buffer-pool tests；exact-candidate TSan 11 cases |
| cancellation / close | `NOT_APPLICABLE` | base Mem 没有 async operation；显式 reset/trim/collect/bulk release 已在 lifetime 行验证 |
| property / differential / fuzz / stress | `PASS` | randomized FixedHeap、LifetimePlanner property、pool capacity/accounting stress |
| ASan / UBSan / TSan | `PASS` | WSL exact candidate：ASan 2/2 suites、UBSan 2/2 suites；TSan 11 个并发用例 |
| real ISA / backend | `NOT_APPLICABLE` | host control path 不含 SIMD/device backend；native OS/NUMA 由三平台验证 |

### Platform、correctness 与交付

| Gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | Verdict |
|---|---|---|---|---|
| exact-candidate Release/WERROR selected correctness | `35/35` | `34/34` | `33/33` | `PASS` |
| sanitizer | — | ASan `2/2`、UBSan `2/2`、TSan targeted `11/11` | — | `PASS` |
| source/static/shared selected Mem journeys | 三种均 `PASS` | correctness/source compile | correctness/source compile | `PASS` |
| direct Sched consumer | fresh selected Sched build + `1/1 PASS` | build closure covered | native compile closure covered | `PASS` |

macOS source 来自候选 archive
`hyle-mem-v6-candidate-9604bf4a.tar.gz`，Windows 与 remote SHA-256 均为
`9DA2830B1EF9C0A89AE409F73BB40212346517381EA6518FD5081EC5C4ED7AF4`；remote checkout 只作验证，
没有回流产品代码。TSan 在 WSL 6.18 / GCC 14 启动时要求 shadow address layout；验证命令仅在同一
shell 内把 `vm.mmap_rnd_bits` 从 32 临时置 28，并以 trap 恢复，最终值已复核为 32。

### Aggregate gates

| Gate | Verdict | 依据 |
|---|---|---|
| scope | `PASS` | 只实现 `MEM6-F01..F05`，候选后无产品改动 |
| architecture | `PASS` | 九族完整/路由；Platform/Error/Core authority 与依赖方向唯一 |
| correctness | `PASS` | 三平台、边界/OOM/property、ASan/UBSan/TSan |
| delivery | `PASS` | source/static/shared、relocation、selected product 与 Sched direct consumer |
| usability/docs | `PASS` | natural C++ owner/view/API；architecture/user/performance/task 职责分离 |
| red team | `PASS` | 独立 hostile review 的 material remaining finding 为 0 |
| engineering performance | `PASS` | P1 qualified 8/8 5% NI；P0 family 成本/复杂度通过 |

### 性能证据

冻结合同为 [`mem-v6-host-allocation-p1-v1.json`](../../../../bench/contracts/mem-v6-host-allocation-p1-v1.json)，
完整结果见 [`benchmarks-mem-v6.md`](../../evidence/performance/benchmarks-mem-v6.md)。qualified r2 位于
`C:\public\project-hyle\_validation\mem-v6-host-perf-evidence-r2`，contract-lock / manifest / summary
SHA-256 分别为
`297958D9E2F9F98D12370D99DAFA76B8D1E89D50D04F55F630F8563193093144` /
`A7749296A3A772DB71F2534D01C8E38F2506830FCCDCE61F4014F6151C3B8EC0` /
`5054BED366D60C2D7F1B8926D50FBD9BBEC7B2A6C4B7A6AFB3BD389144A2FA0F`；deep verify 为
`QUALIFIED/CURRENT/EXACT`。r1 因 A/A 一个顺序和 5% delta control 未收敛而只保留 `DIAGNOSTIC`，
没有被拼入 PASS。

### 独立红队

独立 reviewer `mem_vm_red_team` 于 2026-09-06 在冻结候选上复核 capability、contract、OOM/lifetime/
concurrency、系统事实、依赖/交付、性能 comparability 与 benchmark pollution。

| Finding | severity | 处置 | 证据 |
|---|---|---|---|
| fully released repeat 与 zero/subpage release 曾在判空前分配 metadata | high | `fixed` | 先 bounds/page pre-scan，幂等 0 无分配；tests |
| large-page reserve 曾把请求 GiB/fallback 容量冒充物理 1 GiB page | high | `fixed` | `LargePageArenaReservationReport` 分离容量、placement、granularity、outcome |
| `0 < reserved < requested` 很难以安全 1 GiB 环境稳定制造 | evidence gap | `accepted` | vendor `*_at_ex` + `mi_arena_area` mechanism/code review；API 保存 partial，禁止为测试分配多 GiB |
| 其它 allocator/pool/planner/public shape | medium+ | `disproved` | reverse trace、property/OOM/concurrency、P1 same-contract 与三交付未发现 material defect |

## Verification verdict

所有适用门均有明确 verdict，五项 FIX 完成，未处置 `MUST=0`、material red-team finding=0、blocker=0。
候选 LOC 由 `loc-report.py --commit 9604bf4a... --self-check` 复算：全库 454,713；自实现含测试
117,630；自实现不含测试 74,519；Mem boundary 实现 7,631。`CANDIDATE -> VERIFIED` 前 registry
snapshot 为
`C:\public\project-hyle\_validation\mem-v6-mem-vm-v2\module-status-before-verified.json`，SHA-256
`FDE8CBEAB0EF0BB63714E0C037EABF357879EC7E150445E27BEB25A45F2773DA`。

## 接受限制与重开条件

接受限制：

- 自增长、individual-free 的通用可变池走标准 PMR pool + HYLE resource，并无独立 HYLE owner；只有
  foreign-runtime 或不同成本/失败合同的具名 consumer 才能改变该路由；
- process-global large-page arena 的 partial side effect 可观察但不可 rollback；物理页粒度和 NUMA
  placement 在 vendor/OS 无可证事实时保持 unverified。测试不为强制 partial 分支冒险分配多 GiB；
- Tensor/KV eviction、async completion、device/provider implementation、IPC protocol 和 secure-secret
  policy 由上层拥有；本轮只保证开放 resource/buffer 组合点；
- P1 只签 W1-WIN 上冻结的 isolated runtime-erased heap lifecycle；其它 family 为结构/复杂度/资源门，
  Formal 仍 `PENDING`，不构成全平台、全尺寸、全线程或端到端推理 SOTA。

只有以下新事实重开 Mem：

1. 真实 consumer 复现 allocator/resource/Buffer/pool/arena/planner 的 correctness、OOM、lifetime、并发、
   ownership 或 C++ 易用性缺陷，或证明当前 `ROUTED` family 无法完成具名基础任务；
2. mimalloc 或直接 HYLE dependency 的源码/语义/默认策略/ABI/codegen/交付变化影响产品路径；
3. public API、layout、默认 metadata、复杂度、组件依赖、三交付或 system fact 语义发生变化；
4. same-contract 正常 consumer workload 证明可复现 material regression，或成熟 peer 证明更优复杂度档；
5. 新 OS/NUMA/large-page 事实使当前 unknown/partial/placement 合同不再诚实。

新 benchmark、peer 发布本身、LOC/API 数量、命名偏好、单次点值、地址/顺序翻转、亚纳秒差异或无新事实
的“再看一遍”不构成重开。

## Closure decision

本轮裁决 `CLOSED`：五项冻结 FIX 全部完成，九族能力全部 `COMPLETE/ROUTED`，scope、architecture、
correctness、delivery、usability/docs、red team 与 P1 engineering performance 七门均为 `PASS`，没有
blocker 或未处置 material finding。候选保持
`9604bf4a33f13bd5f84ef78fedce55f305ac9d50` /
`57adb69f185cf02d42ab7f220ebeccbb91170b1b`，terminal 文档不改变产品 identity。

同候选 LOC（cloc 2.08，`--self-check`）：全库 454,713；自实现含测试 117,630；自实现不含测试
74,519；Mem boundary 实现 7,631。raw performance evidence 与 hashes 已列在 verification manifest；
terminal end commit 由随后 registry 记录完整 ID。
