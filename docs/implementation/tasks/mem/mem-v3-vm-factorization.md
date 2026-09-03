# Mem v3：VM/storage owner 定向迁移

<!-- MODULE CLOSURE RECORD: ACTIVE -->

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`mem` / Mem / `mem-v3-vm-factorization` / `REOPEN`；
- baseline：`f077a34c73fba624cbcfaedd2b5cf8dd31655847`；前轮 `mem-v2`；
- trigger：`MEM-VM-FACTORIZATION-2026-09-03`；
- 具体新事实：Platform v3 已接受的因子图要求同步 fs 与 host facts 分开，但当前 Mem public mapping
  在 layout 中直接持有两个公开 Platform storage owner；这既是 Mem public dependency/layout，也是真实
  source/static/shared delivery shape 变化，命中既有 reopen trigger；
- 当前授权仅登记 trigger。完整 capability trace、API/owner freeze、有限 FIX、预算和验证矩阵必须在
  `SCOPED/AUDITED` 后才能修改产品。

## Closure card（`SCOPED`）

- component / owner / round / kind：`mem` / Mem / `mem-v3-vm-factorization` / `REOPEN`；baseline 与
  trigger 同上；
- 真实 consumer：model weight mapping/prefault、large NUMA arena、Sched guarded fiber stack、named
  shared-memory transport；北极星为锁定 llama.cpp/ggml model loader；
- 范围：只迁移 mapping/residency/huge/guarded storage 的 owner、public layout/source/header/target，补齐
  同步 populate 旅程，并复签被触及的 Mem product delivery；
- 非目标：allocator/heap/arena/pool/resource/buffer/planner 算法或 public shape、partial-hole mapping、
  async/direct IO、backend policy、Compute/Graph product、Native FFI/C API、stable ABI、公共仪器；
- 预期图：保留现有 `mem` ID 作为其余 Mem factor；新增 `mem_vm -> PUBLIC core,error,mem,platform,
  platform_fs`，普通 `hyle::mem` façade 同时选择 `mem` 与 `mem_vm`；
- tier / cost：`P1`，但只续签受影响的 model-mapping normal journey 与现有 Mem P1 regression；owner
  wrapper 不增加 heap/refcount/virtual dispatch/copy，mapping setup O(1)，populate O(pages)；
- 总调查/实现预算三小时，性能预算三十分钟；每个 family 最多一次机制诊断、两次产品修正；
- machines：W1-WIN correctness/P1，W1-WSL 与 M1-MAC correctness/sanitizer/delivery；instrument
  `closure-perf-v1`，5% non-inferiority；
- peers：Folly MemoryMapping 与 latest llama.cpp mapping journey；OS mapping ABI 是 correctness authority；
- gates：range/EOF/offset/overflow/mode/flush/advice/populate/lock/move/reset/system failure、anonymous/NUMA/
  shared/guarded paths、three-platform sanitizer、exact factor、source/static/shared、direct Mem/Sched consumers、
  docs/FFI scan/red team；
- 停止条件：只执行本卡和后续审计冻结 FIX；若无法建立唯一 owner 或同交付无额外 cost，进入
  `BLOCKED`，不能保留两套 public storage。

## 定向能力与 surface 审计（`AUDITED`）

审计复用了 mem-v2 的完整 allocator/pool/resource/buffer/planner 账本，只重放所有 mapping、huge、
fiber-stack public entries 和真实 consumers。最新 llama.cpp/ggml 增量需要同步 prefault 与可独立释放的
model fragments；前者属于通用 mapping residency，后者不应破坏连续 `bytes()` owner，可由现有
`open/map(offset,length)` 创建多个独立 mapping 表达。因此没有理由建立 holey view 或每页 metadata。

| ID | class | frozen shape / ownership | cost target | action / verification |
|---|---|---|---|---|
| `MEM3-VM01` | `MUST` | `mem::VirtualMapping/FileMapping/SharedMapping/HugeMapping/FiberStack` 是唯一 public storage owners；删除公开 Platform MappedFile/SharedMemoryRegion | inline native state，无 pimpl/refcount/heap/额外 virtual call | `FIX` native-state migration、layout/lifetime/range tests |
| `MEM3-VM02` | `MUST` | `FileMapping::populate(offset,length)` 同步触页；backing file 在 view lifetime 内不得 truncate | O(covered pages)，无 per-page metadata；不是 will-need hint | `FIX` range/empty/subrange/real page-fault journey |
| `MEM3-VM03` | `MUST` | fragment 通过多个 offset/length mapping owners；`DontNeed` 只释放 residency，不制造 hole | 每个 fragment O(1) owner；不添加 hidden segment table | `KEEP/ROUTE` loader policy，guide 说明 |
| `MEM3-VM04` | `MUST` | `mem_vm` 精确拥有 mapping/huge/guarded headers/source；Mem facade 聚合 `mem + mem_vm` | graph acyclic，旧普通 Mem consumer 不多取 fs/thread | `FIX` manifest、tests、direct Sched/IO edge、三交付 |
| `MEM3-VM05` | `MUST` | 受影响 mapping normal journey 与未触及 Mem-v2 P1 rows 回归 | 5% non-inferiority；setup/touch/reset、bytes 全计费 | `FIX` qualification/replay；无同合同则限于结构结论 |
| `MEM3-VM06` | `MUST` | architecture/user/performance/sample/FFI 清洁度跟随唯一 owner | 无 runtime cost | `FIX` docs/scans/red team/LOC |

上述六票是全部 FIX。API freeze 为 `FROZEN`（2026-09-03）：path 或 owned FileHandle 映射、显式
advise/populate/lock、named shared create/open、anonymous NUMA mapping 与 guarded stack 的自然 journeys
和公开头自含均已通过。失败继续使用共享 Status/Result，控制操作外部同步，byte access 的 data race
仍由 caller 负责。没有 C/Native FFI、stable C++ ABI 或兼容 alias；P1 不得反向改变 owner/layout。

六问当前结论：mimalloc 是未触及 Mem factor 的既有 direct product dependency，`mem_vm` 自身只用
OS/Platform mechanisms；推理 mapping 能力在 VM02 落地后完整，partial-hole 被自然分段 owner 路由；
性能已在 API freeze 后进入候选取证；旧 FFI 扫描干净；新 factor 的 source/static/shared 尚待
candidate；C++ taxonomy 保持 mem-v2，用户不需要知道 factor 名。`AUDITED` 只授权以上六票。

## 候选前实现检查点

- VM01--VM04/VM06 已完成，VM05 进入候选取证；Platform storage owners 与旧 convenience 已删除，
  Mem 类型直接持有 native state，`FileMapping::populate` 是唯一新增操作；
- W1-WIN 精确 mem_vm 根下 Mem `223/223`、VM `60/60`；GCC 14 `-Werror` 同为 `223/223` 与 `60/60`，
  两个 factor 的公开头自含通过；
- 这不续签未触及 Mem-v2 families 的新性能冠军结论，也不提前宣称 `CANDIDATE`。M1-MAC、sanitizer、
  source/static/shared、direct consumers、P1 与红队仍待执行。
