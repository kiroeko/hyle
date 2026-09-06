# Mem v6：当前推理内存超集与产品形状重构

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮与 `mem_vm-v2` 协同，但 `mem` 与 `mem_vm` 仍是两个独立组件、两份候选和两项终局裁决。
> 当前只登记合法重开事实；完整 closure card、能力账本、API freeze 和有限 `FIX` 集尚未冻结，
> 因此还不授权产品源码修改。

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
