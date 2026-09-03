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

## Closure card（待 `SCOPED` 冻结）

- component / owner / round ID：`mem` / Mem / `mem-v5-multichunk-mimalloc-351`；
- task kind：`REOPEN`；
- baseline commit：`ae01203b35d7f6d0b9c76117a1a50fc837c07772`；
- tier：`P1`，只因 mimalloc material codegen change 影响正常 host allocation；multi-chunk planner 本身为
  P0 complexity/resource journey；
- machine / instrument：W1-WIN、W1-WSL、M1-MAC；`closure-perf-v1`；material margin 5%；
- provisional scope / non-goals：与 Reopen evidence 的最小影响面一致；完整预算、normal workload、peer、
  correctness/delivery gates 和停止条件在下一状态提交冻结。
