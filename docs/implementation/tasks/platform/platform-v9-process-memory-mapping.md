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
