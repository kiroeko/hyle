# Platform v9：进程内存事实与映射 range 机制

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由 Mem v6 / Mem-VM v2 的具名 consumer replay 触发。当前为 `CANDIDATE`：四项冻结 `FIX`
> 已由下述候选实现，后续只运行预声明验证，不再扩大产品范围。

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

## Capability / surface audit 与有限 FIX（`AUDITED`）

### 能力与 owner

| ID | 真实任务与合同 | 当前实现 | 裁决 |
|---|---|---|---|
| `PLAT9-C01` | whole-process current/peak resident，以及 OS 确实公开时的 private committed；显式冷查询、失败可见、unknown 不用零伪装 | `HostMemoryInfo` 只有系统容量；Mem 错把 mimalloc 子系统量命名为 whole-process | `MUST / FIX`：Platform 新增唯一 OS fact，Mem 只聚合 |
| `PLAT9-C02` | 对 owner 已验证的 page-aligned committed/mapped range 做 advice/lock/unlock；错误与平台能力统一 | free function 机制已存在且实现无需知道 anonymous/file owner，但文档限定 anonymous，Mem-VM 又复制 syscall | `MUST / FIX`：扩正合同、补 file-backed 系统测试，Mem-VM 直接复用 |
| `PLAT9-C03` | file mapping owner、range 合法性、fragment metadata 与 partial unmap | Platform 没有且不应拥有 file owner | `ROUTED` 到 Mem-VM；不得下沉 |
| `PLAT9-C04` | process commit 在 Linux/macOS 无公开同语义量时仍给出数值 | `VmSize`/allocator committed/phys_footprint 都不是该量 | `REJECT`：字段 absent，文档列出平台含义 |

public surface reverse trace 覆盖 `host_memory.h`、`virtual_memory.h`、`host_memory.cc`、
`virtual_memory.cc`、Platform tests、selected-header/CMake、Mem `stats/system` 与 Mem-VM mapping。没有第二个
Platform product owner 或新依赖需要进入；`platform_fs` 的 FileHandle 不受影响。

### 冻结 C++ API

```cpp
struct ProcessMemoryInfo {
  std::uint64_t resident_bytes;
  std::uint64_t peak_resident_bytes;
  std::optional<std::uint64_t> private_committed_bytes;
  std::optional<std::uint64_t> peak_private_committed_bytes;
};

Result<ProcessMemoryInfo> process_memory_info() noexcept;
```

- Windows：working set current/peak；`PrivateUsage` 只在公开结构可得时作为 current private commit；
  不能由 `PeakPagefileUsage` 无证明地改名为 peak private commit；
- Linux：`/proc/self/status` 的 `VmRSS/VmHWM`，缺行、单位、溢出或 I/O 错误可见；private committed
  两项 absent；
- macOS：public Mach task resident current/max；private committed 两项 absent；
- peak 必须至少为同次 current；query 不缓存、不增加任何 allocation/VM 热路径状态。

`vm_advise/vm_lock/vm_unlock` 的函数签名不变；合同改为任意由 owner 保证合法的 mapped page range。
Platform 不校验或保存 owner identity，也不新增 fragment API。

### 冻结 FIX ledger

| FIX | 产品修改 | 复杂度/成本 | 验证 |
|---|---|---|---|
| `PLAT9-F01` | 实现上述 `ProcessMemoryInfo/process_memory_info` 和三平台 parser/system tests | Windows/macOS O(1) OS query；Linux O(`/proc/self/status` bytes) 冷读；热路径零状态 | fixture + native platform tests + failure/overflow |
| `PLAT9-F02` | 修正 mapped-range 文档与测试，证明 file-backed page range 可走既有 advice/lock/unlock；只在复现错误时改 syscall body | 同一 syscall 数；无 wrapper allocation/dispatch | Windows/WSL/macOS capability-aware tests |
| `PLAT9-F03` | selected headers、source/static/shared、user/architecture 文档及 Mem direct-consumer compile/run | 无新依赖；Platform 仍只依赖 Core/Error | exact component/package journeys |
| `PLAT9-F04` | 复跑既有 host-planning P1 holdout；新增 snapshot 仅签 P0 | 5% NI；不得添加 snapshot 纳秒 case | `closure-perf-v1` W1-WIN |

以上四票是本轮全部授权。CPU/NUMA/cache、Platform-fs/thread、explicit huge-page owner、fragment storage、
新的 public knob 或公共仪器改动均不在 ledger；出现这类需求必须另行裁决。

### 七问预裁决

1. 功能超集：新增两项正好填平具名 Mem consumer；其它候选均 `ROUTED/REJECT`；
2. 依赖/复用：Core/Error 足够，Platform 是 OS authority，禁止反向依赖 Mem；
3. API：optional 表达 unknown，range free functions 保持无 owner state；没有 benchmark seam；
4. 性能：新增冷查询按复杂度/系统调用裁决，原 host journey 只做一次 5% holdout；
5. FFI：无已准入 Native FFI/C API；扫描不得出现新的未装饰 export；
6. 构建：同一 `hyle_platform` selected component，三种 C++ 交付均须实调用新符号；
7. 文档/红队：必须删除“anonymous-only”错误边界，并由 Mem 红队 finding 逐项关闭。

`SCOPED -> AUDITED` 前 registry snapshot：
`C:\public\project-hyle\_validation\platform-v9-process-memory-mapping\module-status-before-audited.json`，
SHA-256 `B167906C62A60E03623745B4B8DB457BF7CF54F05B426522208946DD3C6B5BAC`。

## Candidate

- product commit：`9604bf4a33f13bd5f84ef78fedce55f305ac9d50`；tree：
  `57adb69f185cf02d42ab7f220ebeccbb91170b1b`；
- `ProcessMemoryInfo` 使用各平台公开 OS 事实；Linux/macOS 对不可证明的 private commit 保持 absent，
  Windows 只报告 `PrivateUsage` current；不缓存、不向内存热路径增加状态；
- `vm_advise/vm_lock/vm_unlock` 保持既有公开签名与 syscall 路线，只修正为通用合法 mapped range 合同；
  Mem-VM 不再复制匿名 VM wrapper 或 range syscall authority；
- 直接消费者 `mem::system_memory_snapshot()` 组合 Platform whole-process 与 mimalloc allocator-domain
  两种事实，不再把后者冒充前者；
- `AUDITED -> CANDIDATE` 前 registry snapshot：
  `C:\public\project-hyle\_validation\mem-v6-mem-vm-v2\module-status-before-candidate.json`，SHA-256
  `173814FFC86DB38FC7587C43D64F2433E7CFC4AB6DC09A2E6C33B75CDB37553C`。
