# Mem v5：multi-chunk lifetime 与 mimalloc 3.5.1 定向完成

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`mem` / Mem / `mem-v5-multichunk-mimalloc-351` / `REOPEN`；
- baseline：`ae01203b35d7f6d0b9c76117a1a50fc837c07772`；前轮为
  `mem-v4-reclamation-facade`；
- trigger IDs：`MEM-LIFETIME-MULTI-CHUNK-01`、`MEM-MIMALLOC-351-01`；
- trigger evidence：[`current product replay`](mem-current-product-replay-2026-09.md)；
- 具体新事实：
  1. `LifetimePlan::instantiate()` 对每个 resource 只创建一个 `peak_bytes` Buffer，无法表达已经由
     `MemoryResource::maximum_allocation` 声明、且当前 ggml graph allocator 实际支持的多 physical
     allocation placement；
  2. shipping host allocator 仍硬锁 mimalloc v3.5.0，而 stable v3.5.1 改变实际 allocator 源码、
     hot-path codegen、平台适配与 correctness，命中 direct-dependency trigger；
- 最小受影响面：base `mem` factor 的 lifetime-plan API/实现、mimalloc vendored/lock/custom build，及由
  两者改变的 tests、三种 C++ delivery、architecture/user/performance/task 事实；
- 明确不受影响：`mem_vm`、`mem_reclaim` 的算法与 API，普通 Buffer layout，Tensor/Graph/Compute 策略，
  backend usage/reset/host-pointer import、async completion、hazard pointer、Native FFI/C API 和公共性能仪器；
- 当前授权仅登记 trigger。进入 `SCOPED` 前须冻结完整 closure card；进入 `AUDITED` 前须冻结自然 API、
  capability/gap ledger 和有限 `FIX` 集。此前不得修改产品源码、公开头、依赖或正常 workload。

## Closure card（`SCOPED`）

- component / owner / round ID：`mem` / Mem / `mem-v5-multichunk-mimalloc-351`；
- task kind：`REOPEN`；baseline commit：
  `ae01203b35d7f6d0b9c76117a1a50fc837c07772`；
- 进入原因与 demand/defect：`MEM-LIFETIME-MULTI-CHUNK-01`、`MEM-MIMALLOC-351-01`；证据见
  [current product replay](mem-current-product-replay-2026-09.md)；
- 真实 consumer task / normal workload：
  1. 推理 graph 的静态 lifetime 集合需要先得到可检查的 physical-backing/offset placement，再在具有有限
     `maximum_allocation` 的 host/device-like `MemoryResource` 上实例化；同一 alias root 必须保持在一个
     backing 内，多个 resource slot、半开 lifetime、no-reuse/output 和零大小请求仍保持原合同；这是由
     ggml graph allocator 的真实 multi-chunk 路线和 HYLE package journey 共同准入的
     `NORTH_STAR_JOURNEY`，planner 是 cold P0 路径；
  2. source、static SDK 与 shared SDK 消费者通过 `SystemAllocator`、`AllocatorRef`、PMR 或独立 `Heap`
     完成 64 个 24 B--3 KiB、16/32/64-byte alignment 的成功 allocate/touch/reverse-free lifecycle；这是
     `samples/05_installed_sdk`、Mem package 与 Container/Sched 默认分配路线的普通 shape；
- 范围：base `mem` factor 的 lifetime-plan public C++ API、O((N+E) log N) planner/instantiate 实现与
  tests；private mimalloc vendored snapshot/lock/custom target/明确平台配置；只重放受影响的 host allocator
  P1、三种 C++ delivery、依赖/license/symbol、architecture/user/task 事实；
- 非目标：`mem_vm` 或 `mem_reclaim` 算法/API；普通 Buffer layout；Tensor/Graph/Compute backend policy；
  reset/host-pointer import、async completion、hazard pointer；Native FFI/C API、稳定 C++ ABI、公共性能仪器、
  全 Mem benchmark 重做或 Formal certification；
- 直接依赖：`PUBLIC core error platform`，private locked mimalloc；直接产品消费者：`mem_vm`、Container、
  Util-runtime、Sched、IO、Compute、Graph、Mem product façade 与 source/static/shared package journey；只有
  package sample 当前直接调用 `LifetimePlanner`，ggml 是 capability/shape peer 而非 HYLE 产品依赖；
- tier：聚合为 `P1`，因为 mimalloc v3.5.1 material allocator/codegen change 进入正常 host-allocation 热路；
  multi-backing planner 只签 P0 complexity/resource verdict，不为 cold plan 构造纳秒 benchmark；
- material margin / resource budget：host allocator 对同一 v3.5.1 binary 的 direct mimalloc same-contract
  route 为 5% non-inferiority；planner 保持 build O((N+E) log N)、O(N) scratch，query/view O(1)，不产生
  byte/page-proportional table，每个 physical backing 不超过 provider max，root/alias group 不跨 backing，
  backing/placement metadata O(N)；不承诺全局最优 bin packing；
- 总调查预算：一个工程日（上限 8 小时）；其中 API/算法/测试 4.5 小时，vendor/build/correctness 1.5 小时，
  P1 45 分钟，delivery/docs/red-team/receipt 1.25 小时；超时必须按协议裁决，不能以“继续研究”续期；
- 最多实验/修改：P1 一次资格化采样；若异常，只允许一次顺序反转复现、一次 codegen/profile 机制诊断、
  一个由性能证据触发的内部产品修复和一次候选重跑。上述次数不把已冻结的两个 correctness/capability
  FIX 冒充性能实验；
- machine profile：W1-WIN 是 P1 工程签名机；W1-WIN、W1-WSL、M1-MAC 都执行适用 correctness、
  sanitizer/build/delivery；编译用已登记工具链，WSL 使用 GCC 14、`-j2`；
- instrument profile：P1 用 `closure-perf-v1`；planner 的 P0 structural/resource 检查随 correctness 证据；
- strong peers：
  1. standalone ggml `d4716378882593333721eb33f153144b6885caf2` 的 graph allocator multi-chunk
     `{chunk, offset}` / backend maximum-buffer-size shape，只作为能力、边界与复杂度 peer；
  2. mimalloc v3.5.1 tag peeled commit `34fbd7e7cd4627424490afe19b20f8066bfc537d`（2026-09-01），
     作为 private mechanism 与 host-allocation lower bound；v3.5.0 commit
     `18b08671c9302247bfb682286e6bf3cc1773f801` 只作 upgrade provenance/方向性诊断，不与不同 binary
     placement 结果混成产品 SOTA gate；
- 无同合同 peer：multi-backing plan 没有统一 API 的外部 SOTA 排名；以 provider max、alias/lifetime
  correctness、上述复杂度/metadata 预算和 deterministic replay 为 gate。禁止声明最少 backing、最小 peak、
  device allocation latency、end-to-end token/s、所有 allocation size/thread/platform 或 blanket Mem SOTA；
- API 语义档位与选择前提：单一 `resource-aware static lifetime plan` 档；caller 在 build 时借用 resource
  properties，plan 不持有 provider，可先查询 placement/backing plan，再用兼容 resources 实例化；实时动态
  图、runtime grow/release 使用既有 Arena/Pool/Buffer/CompositeBuffer，不塞入本 planner；
- 常态成本合同：build 冷路径 O(N) metadata/scratch、无 provider allocation；instantiate 每个 planned
  backing 一次 provider allocation，失败用 `Result<...>` 回滚全部已建 owner；plan query 与 instantiated
  view/backing lookup O(1)、无 allocation/lock/atomic/syscall；host allocator 不增加 per-allocation metadata、
  copy、lock、queue、wake 或 syscall，只保留既有 compile-time/runtime erasure seam；
- correctness gates：ordinary/single/multiple backing、exact max/alignment、reuse、alias/no-reuse/output、dense
  multi-resource、zero/empty、overflow、oversized root、incompatible instantiate resource、partial OOM cleanup、
  move/reset/destroy/property replay；mimalloc version/cross-thread free/realloc/alignment/PMR/stats/collect/NUMA
  reserve；Release + ASan/UBSan/TSan 的适用 Mem tests；
- delivery/build gates：selected Mem source embedding、relocated static SDK、relocated shared SDK；dependency lock、
  vendored tree/archive identity、offline source bundle、license install、private symbol/visibility 与 only-selected
  C-language enablement；direct consumers 至少 package Mem、Container allocator compile/run 和 mem_vm API surface；
- performance gate：冻结的 `isolated-heap-lifecycle-64` normal aggregate，HYLE `Heap::allocator()` 对同一
  mimalloc v3.5.1 binary 上的 runtime-erased PMR heap route，双方相同 heap ownership、size/alignment/touch/
  reverse-free、setup/warmup/teardown charging；15/31 process samples、candidate/peer execution 与 allocation
  order、placement、A/A、known 5%/10% delta、timer floor 资格化，5% non-inferiority；
- 预期输出：自然 multi-backing API/实现/测试/recipe；精确 v3.5.1 vendor+lock+配置决策；候选与三平台、三
  delivery、P0/P1、红队证据；只更新当前事实，不重写历史 task/evidence；
- 停止条件 / 可接受限制：两个 FIX、任一 correctness/sanitizer/delivery/lock gate 或一次允许的性能修复后
  的 5% P1 仍失败即 `BLOCKED`；若 v3.5.1 在支持平台无法保持明确 baseline/THP/ISA 交付合同也
  `BLOCKED`，不静默回退 3.5.0 或 OS allocator。允许 deterministic next-fit backing boundary + best-fit
  lifetime-slot reuse，不要求 NP-hard 全局 bin packing；Formal 与端到端推理排名保持 `PENDING`。

## SCOPED verdict

两个 trigger 已压缩为上述有限范围、预算、normal workload、peer、适用 gate 与停止条件。允许进入只读
capability/API/实现审计；在 capability map、public-surface trace、API freeze 和有限 `FIX` 集完成前仍不改
产品代码、依赖锁或 benchmark。
