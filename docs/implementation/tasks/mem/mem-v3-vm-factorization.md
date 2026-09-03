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
