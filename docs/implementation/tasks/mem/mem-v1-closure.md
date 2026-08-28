# mem-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`SCOPED`。本文是 mem 当前轮唯一 closure card 与施工记录；既有 mem task、设计页、
> guide 和性能页只作审计输入，不继承旧 `SIGNED`、Native FFI 或 SOTA 叙事。权威机械状态见
> [`module-status.json`](../../module-status.json)。

## Closure card（`SCOPED`）

- component / owner / round ID：`mem` / mem owner / `mem-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`c04e12c7dd65ea5858314d9ebfbea841119225ae` /
  `472e0cd3838c0852d85a8e0e3e8d69e37348283a`
- 进入原因与 demand/defect ID：`MEM-CAP-01`（按基础库完整性标准重新枚举 host-memory 能力族）、
  `MEM-NORTHSTAR-01`（模型加载、prefill/decode 临时区与通用 KV-lifecycle backing）、
  `MEM-API-01`（C++23 owner/view、失败、默认值和可发现性）、`MEM-ADOPT-01`（下游是否绕过 mem，
  或错误使用 error/platform）、`MEM-PERF-01`（旧性能页没有绑定当前统一仪器和当前 candidate）、
  `MEM-DOC-01`（设计页明确为 `NOT_REVIEWED`，历史性能叙事仍待裁决）、`MEM-DELIVERY-01`
  （mimalloc 外部依赖及 source/static/shared 交付复签）。
- 真实 consumer task / normal workload：
  - 本地量化推理与 graph evaluation 的 phase-lifetime 临时区：多种尺寸和对齐的小对象成批取得，
    epoch 结束整批释放；普通 block 复用，大对象不会永久抬高 retained footprint；
  - scheduler 的 guarded fiber stack：长寿命 worker 在多个 fiber tier 上 acquire/run/drain/release，
    跨 worker 归还后在 owner-level shutdown/trim 回收；
  - sched/io/graph/container 的 runtime allocator 注入：typed、PMR 与显式 alignment 路线保持相同
    backing identity、OOM 与 sized-free 合同；
  - 异步 state/node 的固定槽 handoff：固定容量、耗尽可观察、分配线程与释放线程可不同；
    owner-local cache 是显式 soft visibility policy，不冒充 hard capacity；
  - 工程计算留出：调用方固定 buffer 上的 variable-size allocate/free/reallocate/reset，以及
    NUMA/大页上的大块 host workspace；不引入模型、token、KV 或 backend 专用类型。
- 范围：`include/hyle/mem/**`、`src/mem/**` 的 alignment、system/runtime allocator、标准 allocator/
  PMR adapter、quota/stats、owned/external/borrowed scratch、fixed/growable/shared/typed pool、fiber stack、
  大页与 NUMA owner；全部 public surface 反向追踪；直接依赖和代表性直接消费者；mem-only C++
  source/static/shared；当前 architecture/user/performance 文档；统一仪器所需的窄 consumer replay。
- 非目标：device/VRAM、pinned/registered host buffer、backend coherence 与 staging ring；KV page/cell、
  graph liveness/reserve、io registered-buffer 和 scheduler blocking policy；generic `MemorySpace`、自动
  NUMA placement policy、稳定 C++ ABI、Native FFI/C API、下游模块整体收口、Formal 发布证书，以及
  为 benchmark 点值加入常态 metadata、TLS registry、padding 或公共入口。
- 直接依赖 / 直接消费者：`PUBLIC core error platform util_core`；代表性直接消费者为 `container`、
  `util`、`sched`、`io`、`compute`、`graph`。本轮只修 mem 及 correctness/build 阻断所需的直接边，
  不替仍为 `QUEUED` 的消费者签署整体结论。
- tier 与理由：`P1`；allocator adapter、region bump、fixed/shared slot 与 fiber-stack reuse 都可能位于
  高频或高资源贡献路径。NUMA/大页、统计快照、collect/trim 和构造/销毁控制面按 P0 复杂度、
  syscall/allocation 与空间结构门审查，不能池化成一个“mem 总分”。
- material margin / resource budget：W1-WIN 同合同 consumer/replay 使用 5% material/non-inferiority
  margin；默认 allocator/typed adapter 不增加隐藏 allocation、lock 或周期线程；Region 热分配 O(1)，
  一个 epoch 对 retained blocks 总遍历 O(blocks)，无 O(blocks²) 重扫；FixedPool fit/free/coalesce O(1)，
  move realloc 仅复制 `O(min(old,new))`；固定槽 acquire/release expected O(1)，hard core 不把 free 隐藏
  在线程 cache；fiber reuse 不增加 guard/VM 语义差异。峰值 reserved、retained footprint、碎片、
  tail latency 与 hard capacity 分别守门，吞吐不能抵偿资源退化。
- 总调查预算 / 性能预算：能力/API/依赖/实现审计最多 300 分钟；统一性能取证最多 120 分钟，只允许
  一次 probe、一次双 order/适用 placement 正式采样，以及协议允许的一次 material 异常机制诊断、
  一个产品修改与一次重跑。`FIX` 集在 `AUDITED` 冻结；预计实现超过一个工作日则拆成具名 defect，
  当前轮进入 `BLOCKED` 或重新 scope，不默默扩面。
- 最多复现、机制诊断、产品修改次数：每个异常一次反序复现、一次 codegen/profile/counter 机制
  检查、最多一个同时拥有机制钥匙和消费钥匙的产品修改及重跑；不得修改中央统计语义或发明第五种
  benchmark 姿势。
- machine profile：`W1-WIN` 为 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 承担适用 correctness、
  sanitizer、OS/backend 与交付复核，不跨平台池化性能。当前没有登记的多节点 NUMA 真机，因此本轮
  只签 selected-node/chunk 的语义、rollback、复杂度与单节点 fallback，不签跨 socket 带宽。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用
  `closure-correctness-v1` 与当前 component/package 合同。
- strong peers（每项 capability 最多两个，均绑定当前源码）：
  - system allocator、remote free、size/pressure：HYLE 锁定的 mimalloc v3.5.0
    `18b08671c9302247bfb682286e6bf3cc1773f801`，以及 oneTBB
    `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 的 scalable/cache-aligned allocator/resource；
  - owned region：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的 Arena，与 C++23
    `std::pmr::monotonic_buffer_resource` 标准路线；
  - external variable-size / fixed and typed pools：oneTBB 同 commit 的 fixed_pool/memory_pool，
    与 marl `b8406ab0a82524170ead6dd6c371b579bbc28b3e` 的 BoundedPool/PoolPolicy/Loan；
  - NUMA/大页与 guarded stack：mimalloc 同锁定版本及 oneTBB 同 commit 的 NUMA/allocator 能力，
    marl 同 commit 的 fiber-stack lifecycle 只作形状输入。
  llama.cpp `7584430716ee229751771ed0d6bbcb780d105eeb` 与 ggml
  `d99724f2b141fed9e1a9f402213506c081433465` 的 host buffer、arena 与 workload 生命周期只用于北极星
  取证，不冒充已有 HYLE 生产消费者。
- 无同合同 peer 时的预算及禁止声明：Quota/Stats、显式 selected-node chunk、preserving object 与
  scheduler-owned guarded-stack cache 若不能对齐完整语义，只按预冻结的 allocation/syscall/copy、
  peak/retention、tail 和 correctness 预算裁决并标 `NOT_COMPARABLE`。不得声明“mem 全部操作 SOTA”、
  物理多节点 placement/bandwidth、device/unified-memory 管理或未测 workload 的性能。
- correctness gates：normal；null/empty/zero；size/count/alignment/address/offset overflow；hard capacity/
  fragmentation/realloc；OOM/metadata OOM/native VM/NUMA failure；foreign/misaligned/double return 与
  outstanding object；move/release/reset/clear/destructor/Loan 早退；remote free、cache publish、并发
  acquire/release 与 shutdown quiescence；property/differential/randomized fragmentation/stress；
  ASan/UBSan/TSan（按合同适用）；Windows/Linux/macOS 真 backend 与 public-header self-containment。
- delivery gates：中央 `mem PUBLIC core error platform util_core` 与 include-owner 对账；mimalloc 锁、
  vendored/license/offline 投影；mem-only source build；relocated static/shared C++23 SDK consumer；
  代表性 container/util/sched/io/compute/graph seam；产物/安装树无旧 C schema、bridge 或 Native FFI 符号。
- performance gates：冻结四个独立 family，不互相池化：
  1. `runtime-allocator-adapter`：正常 size/alignment 分布、allocate/touch/free 与 PMR/typed consumer，
     对同一 mimalloc backing/direct seam 和同合同 oneTBB route 做 A/A、known delta、floor、双 order及 5% NI；
  2. `phase-region-epoch`：graph/inference scratch 分布的 allocate/use/clear、long-lived block reuse、
     peak retained 与 large-allocation release，对 Folly/标准路线做相同语义与资源门；
  3. `fixed-slot-handoff`：hard-capacity local/remote producer-consumer handoff，同时报告 throughput、
     p95/p99 与容量可见性；不以 soft cache 结果替 hard core；
  4. `scheduler-fiber-stack-lifecycle`：相同 guarded stack 语义下的 scheduler batch create/run/drain/
     destroy，比较 pooled reuse 与 direct VM lifecycle，并独立守 tail/retention 预算。
  FixedPool fragmentation/realloc 作为资源与复杂度 replay；不能对齐 peer 时保持 `NOT_COMPARABLE`，
  不伪造同合同 SOTA。
- 预期输出：capability-family map、全部 public-surface trace、有限 FIX 集、owner/复杂度/资源表、
  correctness 与直接消费者证据、统一 P1 bundle、三种交付、更新后的 architecture/user/performance、
  独立红队与四项 LOC。
- 停止条件 / 可接受限制：所有 `MUST` 为 `COMPLETE` 或完整 `ROUTED`、四个 P1 family 的适用工程门
  均为 `PASS`、其余能力通过 P0 结构/资源门、无 correctness/delivery blocker 即停止；多节点 NUMA
  物理带宽、Formal、foreign runtime 与 device memory 可保持明确限制。任何 material `MUST`、仪器资格
  或 P1 NI 无法在冻结预算内裁决，进入具名 `BLOCKED`，不无限调查。

## Capability-family coverage map

`AUDITED` 阶段填写；在此之前不得修改产品实现。

## Capability / gap ledger

`AUDITED` 阶段填写，并冻结有限 `FIX` 集。

## Candidate

尚未冻结。

## Verification manifest

尚未执行；未运行项不得写成 `PASS`。

## 独立红队

尚未执行。

## Closure decision

尚未裁决。
