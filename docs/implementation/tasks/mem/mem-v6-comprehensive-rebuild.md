# Mem v6：当前推理内存超集与产品形状重构

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮与 `mem_vm-v2` 协同，但 `mem` 与 `mem_vm` 仍是两个独立组件、两份候选和两项终局裁决。
> Closure card 已冻结，当前状态为 `SCOPED`。能力账本、API freeze 和有限 `FIX` 集尚未完成，
> 因此仍不授权产品源码修改。

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
