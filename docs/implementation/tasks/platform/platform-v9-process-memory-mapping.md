# Platform v9：进程内存事实与映射 range 机制

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由 Mem v6 / Mem-VM v2 的具名 consumer replay 触发。当前仅为 `REOPENED`：本页先保存
> 可复现事实与最小影响面；closure card、能力账本和有限 `FIX` 集必须在后续 `SCOPED/AUDITED`
> 分别冻结，之前不得修改 Platform 产品源码。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform` / OS process-memory and VM range mechanism /
  `platform-v9-process-memory-mapping` / `REOPEN`；baseline
  `93dce49922ac1721c45f155d66f3805e74faf7cc`，tree
  `068f7084b986310bdc6fef621282fba4104beb82`；
- trigger ID：`PLATFORM-MEM-PROCESS-MEMORY-TRUTH-2026-09-06`。Mem 的真实 pressure consumer
  承诺 `system_memory_snapshot()` 的 process fields 覆盖进程中的全部 allocator/mapping domain；但 Linux
  实现只读取 `mi_process_info`。锁定 mimalloc v3.5.1 的 Unix primitive 只用 OS 值覆盖 peak RSS，current
  RSS、current/peak committed 仍是 mimalloc 子系统统计，遗漏 file/anonymous mapping 和其它 allocator。
  这不是措辞误差，而是 Platform v8 reopen trigger 明列的“named consumer reproduces missing OS fact”；
- trigger ID：`PLATFORM-MEMVM-VM-RANGE-AUTHORITY-2026-09-06`。`mem_vm::FileMapping` 直接重复
  `madvise/PrefetchVirtualMemory`、`mlock/VirtualLock` 与 unlock/limit policy，而 Platform 已公开
  `vm_advise/vm_lock/vm_unlock`；`VirtualMapping` 还以 enum ordinal cast 连接两套类型。真实 model-loader
  partial-release 需要统一的 mapped-range mechanism/error translation，不能继续增加第二套 OS authority；
- 受影响范围只含：公开、冷路径、whole-process memory fact；Platform 已有 VM range control 的合同和
  对 file-backed ranges 的适用性；对应实现、三平台 fixture/系统测试、build/package projection、Platform
  architecture/user 文档及 Mem 直接 consumer replay；
- 明确不重开 `platform_thread` 或 `platform_fs`：文件句柄 owner 和 parker 没有新事实。也不重做 CPU/
  NUMA/cache/frequency、scheduler policy、Mem owner、IO、Native FFI/C API、公共性能仪器或 Formal；
- 初始性能判断：新 process snapshot 是显式冷查询，按 P0 检查公开 OS API、复杂度、无常态热路状态；
  现有 host-planning P1 路线作为未受影响 holdout 复签，不为新增 API 发明纳秒 benchmark；
- `CLOSED -> REOPENED` 前 registry snapshot：
  `C:\public\project-hyle\_validation\platform-v9-process-memory-mapping\module-status-before-reopened.json`，
  SHA-256 `DAEEFAC912D4B29103A0116DF43BEFDDCB61D973C3B81FE491EA444E82669E3A`。

## 初步停止边界

- OS 没有公开、稳定事实时返回 absent，不把 Linux `VmSize`、allocator committed 或型号/采样推断冒充
  process private commit；
- Windows/Linux/macOS 各自只用公开 OS 机制；Platform 返回 OS fact，Mem 决定如何与 allocator fact
  聚合；
- range control 若现有实现本来已支持任意映射，则只修合同/复用和测试；不得把 file owner、fragment
  metadata 或 model policy 下沉到 Platform；
- 审计若发现必须改变公开合同、依赖或正常 workload，先在 `AUDITED` ledger 具名，不得边施工边扩 scope。

## Closure card（`SCOPED`）

- component / owner / round / kind：`platform` / OS process-memory and VM range mechanism /
  `platform-v9-process-memory-mapping` / `REOPEN`；baseline
  `93dce49922ac1721c45f155d66f3805e74faf7cc`；
- demands：`PLATFORM-MEM-PROCESS-MEMORY-TRUTH-2026-09-06` 与
  `PLATFORM-MEMVM-VM-RANGE-AUTHORITY-2026-09-06`；真实 consumer 分别是长期推理进程的 host
  pressure snapshot，以及 model loader 对 file-backed ranges 的 advice/residency/partial release；
- 范围：新增稳定 POD/optional 形状的 whole-process OS memory fact；使既有
  `vm_advise/vm_lock/vm_unlock` 对合法 mapped page range 的合同不再错误限定为 anonymous region；对应
  Windows/Linux/macOS 实现、测试、文档、selected header、source/static/shared 和 Mem 直接消费；
- 非目标：CPU/NUMA/cache/frequency、memory-pressure policy、allocator facts、file/shared mapping owner、
  fragment metadata/unmap policy、Platform-fs/thread、IO、Native FFI/C API、稳定 C++ ABI、公共仪器与 Formal；
- 依赖：`core`、`error`；直接 consumer：`mem` 与 `mem_vm`。Platform 只返回 OS 可证事实和 range
  mechanism，不依赖 Mem，也不把 allocator 聚合或 loader policy 下沉；
- API 语义档：resident/current+peak 必须覆盖 whole process；private committed 仅在 OS 有公开对等量时
  `optional`，不得用 virtual address size 或 allocator committed 代替；range mechanism 接受由合法 owner
  提供的 page-aligned committed/mapped range，owner 仍负责生命周期和合法范围；
- tier：`P1`。新增 snapshot 为显式冷路径，按 P0 结构/系统事实验证；唯一数值 holdout 复用
  `bench/platform/host_bench.cc` 的既有 one-shot host-planning 正常 journey，W1-WIN、5% margin，不新增
  snapshot 微基准，也不重签 Platform 全模块 SOTA；
- 总调查预算 3 小时，其中性能最多 30 分钟；最多一轮 Platform 产品修改和一次 P1 replay。若现有 VM
  implementation 已符合通用范围，只改合同、测试和 Mem 复用，不为“统一”重写 syscall；
- correctness machines：`W1-WIN`、`W1-WSL`、`M1-MAC`；P1 machine：`W1-WIN`；instrument：
  `closure-perf-v1`，P0 为 `closure-correctness-v1`；
- authoritative inputs：Windows `GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS_EX`，Linux
  `/proc/self/status` 的 `VmRSS/VmHWM`，Apple public Mach task information/getrusage；range 语义取各 OS
  官方 mmap/VirtualMemory mechanism，Folly `65749da412d92fe27d87ae45a8ef670720712858` 仅作成熟
  mapped-range mechanism 对照；
- 适用门：公开 surface reverse trace；正常/字段缺失/parse overflow/OS failure；并发冷查询；三平台；
  source/static/shared；Mem 直接消费；FFI 清洁；文档/易用性；红队；新增能力 P0 和既有 host P1 holdout；
- 停止：冻结的 `MUST` 完成、适用门通过且 holdout 无 material regression 则关闭；OS 不提供字段时
  absent 是成功而非 blocker；无法在公开 OS 合同下证实 whole-process 语义则 `BLOCKED`，不得猜值。

`REOPENED -> SCOPED` 前 registry snapshot：
`C:\public\project-hyle\_validation\platform-v9-process-memory-mapping\module-status-before-scoped.json`，
SHA-256 `ADEB548C7A15275FFC8F51E67721E0E39DF6A871CE0C30786A97BC64ADE9530E`。
