# Platform v8：操作系统资源真相与系统调用韧性

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮由 Platform v7 冻结后得到的可复现产品事实定向触发。当前状态只认
> [`module-status.json`](../../module-status.json)；本记录中的计划在进入相应状态以前不授权产品修改。

## Reopen evidence（`REOPENED`）

- component / owner / round / kind：`platform + platform_thread + platform_fs` / Platform /
  `platform-v8-os-resource-contract` / `REOPEN`；baseline：
  `f679026d3dc7fe3e8f68df005df90637dd29bebd`；
- base trigger `PLATFORM-OS-RESOURCE-TRUTH-2026-09-05`：Linux `host_memory_info()` 只读取硬编码
  cgroup mount 下的 leaf `memory.max/current`，没有按 `/proc/self/mountinfo` 解析 namespace root，也没有
  折叠祖先 hard limit；`available_cpu_count()` 只表达 affinity/cpuset placement，却被文档推荐为通用
  worker/shard count，遗漏 cgroup CPU quota/capacity；CPU frequency 没有公开能力；x86 的 coarse cache
  路径优先 CPUID 而不是受支持 OS 公布的 cache hierarchy；Linux NUMA 查询失败时退回系统 online nodes，
  可能把进程不可用节点报告为可用；
- thread trigger `PLATFORM-THREAD-PARKER-PROVENANCE-2026-09-05`：
  `src/platform_thread/wait.cc` 明示 parker 从 Rust std futex parker 移植，但当前
  `THIRD_PARTY_NOTICES.md`、依赖锁和安装包没有登记该 integrated source；这击中既有 delivery/reuse
  reopen trigger，不代表停车算法或其已签 P1 性能有产品退化；
- fs trigger `PLATFORM-FS-EINTR-2026-09-05`：POSIX read/write/pread/pwrite/flock/readlink 和递归目录
  路线会重试 `EINTR`，但 open/fstat/stat/ftruncate/fsync/fdatasync 与普通目录枚举等同类可安全重启的
  系统调用并不一致；真实信号负载可能把瞬时中断错误暴露给自然同步文件调用者；
- API/documentation trigger：`VirtualMemory::decommit` 文档声称 misalignment 触发 fatal assert，实际实现
  返回 `InvalidArgument`；当前 CPU placement 与 CPU capacity 的概念混在一起，不能靠修改措辞掩盖能力
  缺口；
- complexity trigger：Windows CPU Set logical-id 映射在每条记录中重复扫描 processor groups，当前为
  O(CPU×group)，而同文件已有 prefix-map 形状可以做到 O(CPU+group)；
- authoritative contracts：Linux cgroup v2 hierarchy、CPUFreq sysfs 与 CPU topology sysfs；Windows
  `GetLogicalProcessorInformationEx`、CPU Sets、Job Object CPU rate/nesting limits 与
  `CallNtPowerInformation(ProcessorInformation)`；Apple `sysctlbyname` 的 system capability/performance
  level keys。OS 没有公开的事实必须保持 unknown，不用型号表、计时或私有接口猜测；
- real consumers：通用线程池/分片器区分“能放在哪些 CPU”与“进程实际获准多少 CPU 带宽”；推理引擎
  在 worker 数、KV/cache blocking、NUMA placement 与容量规划前读取进程视图；同步配置、模型元数据和
  小文件路径在信号存在时仍应遵守 documented Status contract；
- 当前实现与 peer 定位：oneTBB 当前 revision 已有 Linux cgroup quota 解析；Folly 提供完整 no-interrupt
  syscall family；Highway 明确优先 OS sysfs cache facts；llama.cpp/ggml 证明 worker、hybrid-core、memory
  planning 与 opened-file 路径是真实消费，但其应用策略和部分较弱 host facts 不照搬进 Platform；
- 初步影响面：`cpu.{h,cc}`、`host_memory.cc`、`numa.cc`、`platform_fs/fs.cc`、Parker 来源/依赖登记、
  factor/package links、单元与三平台测试、Platform architecture/user/performance 文档和直接消费者回放；
- 暂定 performance：base 与 fs 为 `P1 / closure-perf-v1 / W1-WIN / 5%`，用于续签既有同合同 host/file
  normal workloads；新增 OS discovery 能力和 group/cgroup parser 以 P0 复杂度、无额外热路径工作和
  fixture correctness 裁决。thread 为 `P0 / closure-correctness-v1`，因为本轮只修 provenance/package，
  不改 parker bytes；
- 暂定 non-goals：scheduler/engine worker policy、SIMD target dispatch、device/GPU inventory、Mem/IO
  产品重构、私有 Apple per-core frequency、型号/微基准推断、C/Native FFI、稳定 C++ ABI、公共性能仪器
  重写、Compute/Graph 与 Formal certification。

本阶段只登记冻结后出现的具体产品证据。自然 API、失败语义、有限 FIX、比较合同和验证矩阵仍须在
`SCOPED` 固定，并在 `AUDITED` 冻结 FIX 后才允许修改产品源码。

`CLOSED -> REOPENED` 前的 registry snapshot 为
`C:\public\project-hyle\_validation\platform-v8-os-resource-contract\transitions\module-status-before-reopened.json`，
SHA-256 `04E557C22D8D4225C9E0B3449A2E7F7C40AB2B9948E1E9AD6326A1310C44046D`。

## Closure card（`SCOPED`）

- component / owner / round：`platform + platform_thread + platform_fs` / Platform /
  `platform-v8-os-resource-contract`；产品 baseline 仍为 `f679026d3dc7fe3e8f68df005df90637dd29bebd`；
- locked peer snapshots：oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f`、Folly
  `65749da412d92fe27d87ae45a8ef670720712858`、Highway
  `63f4b16cda024e2adff8a18dee81e067f8a87d61`；消费证据固定为 llama.cpp
  `4d9176092d00586775af140581bb0b558ddc4389` 与 ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`。这些本地 checkout 均 clean；最多只让 oneTBB/Folly
  进入数值 peer，Highway 只用于 OS-fact 实现审计；
- public delivery：C++23 source embedding、static SDK、shared SDK；没有 C/Native FFI 或稳定 C++ ABI。

### 冻结的公开形状与语义

1. `available_cpu_ids/count/topology` 只表示当前进程/调用线程的 **placement**，不再宣称是 worker 数。
   新增 `CpuQuota { capacity, period }` 与 `ProcessCpuCapacity`：包含 placement logical CPU 数和可选的
   ancestor-folded hard quota；`parallelism_ceiling()` 只做 `min(placement, ceil(capacity/period))` 的
   明确整数推导。Linux 读取 cgroup v2 `cpu.max` 或 v1 quota/period；weight/share 没有 hard ceiling。
   Windows documented query 只暴露 immediate Job，而嵌套 rate 相对父 Job，调用方没有 ancestor handles
   时无法证明最终有效 quota，因此 Windows 与 macOS 都为 absent。调度/线程数 policy 仍由
   Sched/engine 决定。
2. 新增 `CpuFrequencyInfo { id, hardware_max_hz, policy_max_hz, current_hz }` 与 caller-buffered
   `available_cpu_frequencies(dst, cap)`，并在 convenience header 提供 `vector` overload。记录与当前
   available CPU id 对齐，0 表示 OS 没有可信事实；Windows 使用 documented processor-power records，
   Linux 使用 CPUFreq sysfs（kHz 经 checked conversion），Apple 只使用公开且实际存在的 sysctl key，
   不按型号或计时猜测 Apple Silicon 频率。
3. `cache_info()/cache_line_size()` 保留 cached coarse convenience，但改为 OS-first：Windows processor
   relation、Linux cache sysfs、Apple documented sysctl/performance-level；coarse capacity 取 system-wide
   各 class 的保守非零下界，line size 取所见最大值并保留 64-byte unavailable fallback。不能用首次调用
   的动态 process affinity view 生成终身缓存；CPUID 不再抢在 OS cache hierarchy 前；动态
   `available_cpu_cache_classes()` 仍是调优入口。
4. `HostMemoryInfo` layout 不变；Linux process limit 由 mountinfo-aware cgroup resolver 找到真实 v1/v2
   controller path，并把从 leaf 到可见 mount root 的每一层 hard `limit - usage` 取最小。`memory.high`、
   shares 与 pressure 不是 hard capacity；`max`/v1 unlimited 不伪造成 0。
5. 现有稳定 `numa_nodes()` span 不改 ABI 形状；Linux 初始化时按 `get_mempolicy`、
   `/proc/self/status:Mems_allowed_list`、resolved cgroup `cpuset.mems.effective` 的可信进程视图收窄。
   查询失败时不得把 system-online nodes 宣称为 process-available；unsupported/unknown 仍诚实退化为 `{0}`。
6. Platform-fs 只对 POSIX 明确 restart-safe、且尚未产生可观察副作用或重复执行幂等的
   open/stat/lstat/fstat/fstatat/ftruncate/fsync/fdatasync/fcntl/posix_fallocate/opendir/readdir 等调用统一处理
   `EINTR`。`close/closedir` 不重试；可能已经完成部分
   transfer 的调用继续使用既有 short-transfer contract；不吞掉非 `EINTR` native status。
7. Parker 算法、layout、memory order 与对象 bytes 全部不改。Rust std 1.94.1 commit
   `e408947bfd200af42db322daf0fadfe7e26d3bd1` 的 origin/selected license 被登记为 integrated source，
   完整 license text、文件注释、dependency lock、package registry、installed notice 必须一致。
8. 修正文档与实现分歧：VM misalignment 是 recoverable `InvalidArgument`；OS fact 的 unknown、动态性、
   placement/capacity、P/E/cache/frequency 选择和容器陷阱都在 architecture 与 user guide 中直说。

### 有限 FIX budget

| ID | factor | frozen work |
|---|---|---|
| `P8-CG-01` | platform | 一个 mountinfo-aware、安全规范化的 Linux cgroup v1/v2 resolver；memory/quota/cpuset 共用 |
| `P8-CPU-01` | platform | `ProcessCpuCapacity`、hard quota 祖先折叠与 placement/capacity 文档边界 |
| `P8-CPU-02` | platform | OS-published per-available-CPU frequency snapshot 与 convenience overload |
| `P8-CPU-03` | platform | OS-first coarse cache；Windows group prefix map 从 O(CPU×group) 降为 O(CPU+group) |
| `P8-NUMA-01` | platform | Linux process-allowed NUMA fallback 链；禁止 system-online 假可用 |
| `P8-FS-01` | platform_fs | 可安全重启 POSIX controls 的 EINTR 一致性与故障测试 seam |
| `P8-TH-01` | platform_thread | Rust parker provenance/license/lock/package projection；产品算法零改动 |
| `P8-API-01` | platform | VM 注释、错误诊断与新增 API 的 null/truncation/overflow/unknown 合同 |
| `P8-T01` | all three | parser fixtures、边界/失败/复杂度、三平台与 sanitizer 测试 |
| `P8-B01` | all three | target links、逐头、exact factor、source/static/shared、relocated install 与 notice 验证 |
| `P8-P01` | platform/fs | 修正 direct-peer apples/oranges；续签 host planning 与 warm 4 MiB file 的 5% P1 门 |
| `P8-R01` | all three | Mem/Mem-vm/SIMD/Container/Sched/IO 直接消费者只读回放，不顺手重构冻结模块 |
| `P8-D01` | all three | architecture、guide、capability、performance、dependency/license 当前真相 |

FIX 外的新产品能力、公共仪器扩建或相邻 frozen 模块产品修改都退回 `AUDITED` 重新裁决；不能用“顺便”
扩大本轮。

### 比较合同、预算与停止条件

- `platform` 为 P1：W1-WIN host planning 的 candidate 与 direct OS peer 必须构造相同 topology、memory
  字段和错误路径；两边 common host 均不得在 timed call 自造 heap。Windows immediate Job CPU rate 不能
  证明嵌套 effective quota，不进入数值排名。5% margin 只裁决该正常 planning
  journey，不把 Linux parser、frequency getter 或冷 discovery 微基准塑造成 API；
- `platform_fs` 为 P1：沿用 4 MiB warm open/read/checksum/close，同 flags、buffer、bytes、durability、
  EOF/error accounting；EINTR synthetic fixture 只裁决 correctness，不和无信号 peer 排名；
- `platform_thread` 为 P0：integrated-source bytes 未变则以 source hash、memory-order/layout diff、原测试和
  三交付证明没有成本变化；只有产品 bytes 改变才需要重新资格化 Parker P1；
- P0 结构门：cgroup resolver 对 mount/ancestor 输入 O(lines+depth)，Windows CPU mapping O(CPU+group)，
  frequency discovery O(available CPU)，dynamic cache class 为 O(available CPU × exposed OS records)，
  cold coarse cache 为 O(system CPU × exposed OS records)，warm cached coarse cache O(1)，无常态 scheduler
  metadata 或热路径 syscall；
- apples-vs-oranges 禁令：placement 不与 quota、cached 不与 fresh discovery、完整 Status 不与 unchecked
  scalar、cursor IO 不与 explicit-offset、signal-free 不与 injected-EINTR 路线排名；不能用较强 HYLE 语义的
  劣势冒充同合同 FAIL，也不能用较弱 peer 的胜利冒充 SOTA；
- 若 supported OS 无法诚实表达某字段则返回 0/absent，不猜测；若实现要求 Sched/Mem/IO policy、私有 OS
  接口、常态 heap/lock/metadata、或无法在有限 FIX 内消除 false resource facts，则退回 `AUDITED` 或
  `BLOCKED`，不得继续研究代替状态。

### 预声明验证

- W1-WIN：MSVC Release/WERROR，三个 exact factor，Platform product，unit tests；CPU Sets/group mapping、
  Job quota absent honesty、processor power/cache；source/static/shared + relocated package；host/file P1 qualification；
- W1-WSL：GCC 14 Release + UBSan/WERROR，cgroup v1/v2/mountinfo/ancestor/cpuset fixtures、真实 process
  snapshot、NUMA fallback、EINTR seam；三个 exact factor与三交付；
- M1-MAC：AppleClang Release + UBSan/WERROR，public sysctl/cache/perflevel/frequency unknown 合同、thread/fs
  路线、三个 exact factor与三交付；远端只接收 exact candidate archive，不向 Windows 回传产品改动；
- all：header self-containment、no Native FFI/C symbols、component contract、dependency lock/license hash、
  installed `THIRD_PARTY_NOTICES` 与 Rust license、直接消费者 compile/test、`check-module-closure.py`、
  `check-docs.py`、candidate tree/LOC 自检和独立红队。

本卡只冻结边界。下一阶段必须完成实现/peer/consumer/source-license 审计，把每个 FIX 定为具体处置后进入
`AUDITED`；此前仍不修改产品源码。

`REOPENED -> SCOPED` 前的 registry snapshot 为
`C:\public\project-hyle\_validation\platform-v8-os-resource-contract\transitions\module-status-before-scoped.json`，
SHA-256 `F2CBBFAF6040DCA6B10436A91247206948CB2D3B012FEAE31A8E5216A3D7C281`。

## Audit 与冻结 FIX ledger（`AUDITED`）

### 实现与依赖追踪

- `cpu.h -> cpu.cc`：placement、per-CPU topology、cache class 与 feature facts 已有同一 id space；新增
  capacity/frequency 可以复用其 buffer-fill 和 unknown 合同，但不能把动态频率塞入 cached `CpuInfo`，也
  不能让 quota 过滤 placement ids。实现拆到普通 `cpu_resource.cc`，不扩大 LTO-off feature-probe TU；
- 当前 x86 `detect_cache_line()/detect_cache_info()` 的预处理顺序确实先命中 CPUID，绕过 Windows/Linux
  OS hierarchy。Windows/Linux 现有动态 cache-class 路线已经分别读取 RelationCache/sysfs；coarse facts
  应聚合这一路而非再建第三套探测。macOS 继续直接读取官方 sysctl/perflevel；
- Windows `query_available_cpu_ids()` 在每条 CPU Set 记录内累加前序 group；同文件 topology path 已有一次
  prefix-map。修复固定为先构造 group prefix，再 O(1) 映射每条记录；不改变公开 CPU id space；
- `host_memory.cc`、`cpu_resource.cc` 与 `numa.cc` 需要同一 cgroup resolver。固定一个 Platform-private
  `linux_cgroup_internal.{h,cc}`：逐行解析 `/proc/self/cgroup` 和 `/proc/self/mountinfo`，只接受 exact
  controller token，解码 mountinfo 的 `\\040/\\011/\\012/\\134`，拒绝 `..`/截断/越界，保存 leaf 与
  visible mount root，并提供不会越过 mount root 的 parent walk。生产路径不依赖 libnuma/hwloc/容器库；
- cgroup memory 每层配对读取 hard limit/current；CPU quota 比较用整数有理数而非浮点，选择最严格祖先；
  cpuset 优先 syscall 与 `/proc/self/status`，再使用 resolver 的 effective/继承值。解析器用纯 fixture
  覆盖 v1/v2、combined controllers、non-default mount、namespace root、escaped mount、ancestor、unlimited、
  malformed 与 traversal；
- `string_buf`/`shared_pool` 的 shard 是 CPU/cache placement partition，继续使用 placement count 合同合理；
  Sched affinity 使用 ids 也正确。未来推理 engine 的默认 worker 数应读取 `process_cpu_capacity()`；本轮不
  伪造下游修改。Mem/Mem-vm 只需回放 cgroup/NUMA 行为，无 public source change；
- Platform-fs inventory 显示 read/write/pread/pwrite/flock/readlink 与 recursive readdir 已重试，但
  open/stat/lstat/fstat/fstatat/ftruncate/fsync/fdatasync/ordinary readdir 等不一致。固定一个 private generic
  `retry_eintr` helper，使 synthetic lambda 能精确证明 retry count；现有每个 call site 显式使用该 helper。
  `close/closedir` 的结果后 descriptor/DIR 状态不确定，`rename/unlink/link/symlink/mkstemp` 可能已产生外部
  副作用，均明确 `KEEP` 为单次调用；
- `wait.cc` Parker 状态机与 Rust std 1.94.1 的 `EMPTY/PARKED/NOTIFIED`、fetch-sub、CAS、swap 和
  release/acquire 合同同源，HYLE 只增加 bool timeout 结果和三个 OS wait backend。本轮选择 Rust 的 MIT
  option；central lock 固定 repository `rust-lang/rust`、revision
  `e408947bfd200af42db322daf0fadfe7e26d3bd1`、tree
  `b0ddb99197bbc990f1abf224afe329f4d3197e08` 与原始 futex source SHA-256
  `13617a66b7684288ef3c81e582d17f83948d3016f8210b857cfe15b12fd7ce86`。adapted HYLE content hash 在
  candidate 上由现有 integrated-source 计算器签署；
- package owner 固定为 `platform_thread`，安装独立 `rust-parker-LICENSE-MIT` 并由锁哈希/registry
  multiset、source/static/shared package smoke 检查；`THIRD_PARTY_NOTICES.md` 与 `wait.cc` 保留 origin、
  revision、selected license，不把 HYLE 的 backend 代码误称为 Rust 原样文件；
- Windows frequency 需要 `PowrProf` 私有系统链接；static export 的 link-only 系统依赖和 shared library
  都由三交付验证。其余 component graph 不变：base 公开 Core/Error，Thread/FS 公开 Core/Error/Time 并
  私有 base；
- `secure_random()` 的 NTSTATUS 当前被丢为纯文本；不重开 Error enum，而用已冻结
  `NativeErrorDomain::Unknown` 保存 signed 32-bit native value与 canonical `Unavailable`。POSIX `dlerror`
  只有 loader text，无可移植数值域，保持现状。
- audit amendment `P8-WIN-JOB-SEMANTICS-2026-09-05`：Microsoft 的公开合同说明 nested Job CPU rate
  相对 immediate parent，且 `QueryInformationJobObject(nullptr, ...)` 只查询进程所在的 immediate Job。
  没有父 Job handles 就不能折叠有效 hard ceiling。原卡中“接受当前 Job hard/max rate”的假设会在嵌套
  Job 中高估容量，故在仍处于 `AUDITED` 且尚未候选化时修正为 Windows `hard_quota=absent`；不采用私有
  NT API，不把错误字段带入 host benchmark。processor-power record 也必须校验返回 `number` 与目标 id
  一致，否则该 CPU 的频率字段保持 unknown。
- 同一 documented immediate-only 限制也击中 Windows host memory：父 Job 可以施加更紧的 process/job
  commit limit，父级 job usage 还包含当前 immediate Job 看不到的同级进程。原实现把 immediate
  `ExtendedLimitInformation` 与 usage 当成 effective process budget，会遗漏父级约束；本轮一并修正为
  Windows `process_limit_bytes/process_available_bytes=absent`。Linux cgroup 的完整可见祖先链仍逐层合并；
  已读到 finite hard limit 但瞬时读不到 current 时，limit 本身仍进入 `effective_available_bytes()`。
- cache lifetime red-team：原审计方案让 cached `cache_info()` 从首次调用时的 process-available class
  snapshot 聚合；临时只绑 P-core 后再开放 E-core 会把 P-core 容量永久缓存为全局 coarse fact。处置改为
  Windows system `RelationCache`、Linux system CPU sysfs 与 Apple system/perflevel keys 的一次性保守聚合；
  动态 process view 仅留给 `available_cpu_cache_classes()`。这仍属于 `P8-CPU-03` 的 OS-first FIX。
- audit amendment `P8-CGROUP-MOUNT-ROOT-2026-09-06`：cgroup namespace 可把 membership `/`
  映射到 non-root mountinfo root，但 non-root membership 若不在该 mount root 下，两条记录就不是
  同一可见树。resolver 现在仅对 namespace `/` 取 mount point；其余路径必须是 root 的
  component-boundary descendant，否则拒绝。mountinfo 也只解码内核文档定义的
  `\\040/\\011/\\012/\\134`，不接受任意八进制字节绕过 canonical-path 检查。fixture
  已覆盖非交叉 root、namespace root、非法 escape 和 bounded parent walk。
- audit amendment `P8-WIN-NUMA-PROCESS-VIEW-2026-09-06`：Windows 原路线从
  `GetNumaHighestNodeNumber` 枚举全机 node，在 CPU Set/Job 限制进程中会广告不可达
  node。现在仅枚举 `available_cpu_ids()` 的当前 process placement，用
  `GetNumaProcessorNodeEx` 得到、去重并排序 node id。这与 Linux 的 mempolicy/status/cpuset
  fallback 是同一个语义：公开的是进程可用事实，不是 machine inventory。

### 成熟库与真实消费对照

- oneTBB `e00c68dcac0e83947c9b54f6d5304c0616c9639f` 已把 Linux cgroup CPU quota 用于
  default concurrency，证明 capacity 与 placement 分离是成熟需求；但它的当前路线仍固定
  `/sys/fs/cgroup` 根，没有 HYLE 本轮所需的 mountinfo namespace/ancestor 合并；
- Highway `63f4b16cda024e2adff8a18dee81e067f8a87d61` 在 Windows 使用
  `GetLogicalProcessorInformationEx(RelationCache)`，在 macOS 使用 `hw.perflevelN.*` 的
  cache/core facts，与 HYLE 的 OS-first topology/cache 路线一致。Highway 为其 thread-pool
  tuning 可偏向 perflevel0；HYLE 的 coarse API 则取所有已知 class 的保守值，并另提供
  class snapshot，两者是不同公开合同；
- llama.cpp `4d9176092d00586775af140581bb0b558ddc4389` 在 macOS 查
  `hw.perflevel0.physicalcpu`，Windows 查 processor-core relation；ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` 还以 `hw.nperflevels`/
  `cpusperl2` 选 kernel 配置，并用 NUMA 事实改变 chunking。这些是 HYLE
  topology/cache/NUMA 能力的实际推理消费证据；ggml 当前 Linux NUMA 初始化先扫全系统
  node/CPU 再保存 affinity，HYLE 不复制这个全局/进程事实混合。
- 结论是复用同一类公开 OS 事实，但不照抄 peer 中较弱的 namespace、restriction
  或 owner 语义。本轮没有用型号表、微基准猜测 P/E、cache、频率或 NUMA。

Rust 上游校验归档使用 `rustc 1.94.1` 对应 archive
`rust-e408947b.tar.gz`，archive SHA-256
`063f7b364a2f0dee6d8f6733486ea7b6510c79682129448b2e6bd4ffc3e2a8fe`；选用 MIT，
不改 Parker 产品算法。

### Capability / disposition ledger

| ID | baseline | audited disposition |
|---|---|---|
| `P8-CG-01` | hard-coded leaf-only mount | `FIX`：共享 resolver + bounded ancestor walk；三 consumer 不各写一套 |
| `P8-CPU-01` | placement 被误作文档 worker count、无 quota | `FIX`：独立 capacity value；Linux cgroup quota；Windows nested effective quota 不可证明则 absent；placement API/layout 不变 |
| `P8-CPU-02` | 无 OS frequency surface | `FIX`：动态 per-id snapshot；unknown=0；Mac 不猜 |
| `P8-CPU-03` | x86 cache 绕 OS；Windows O(CPU×group) | `FIX`：复用 OS class truth；prefix map；cached warm O(1) |
| `P8-NUMA-01` | syscall fail 后宣称 system online | `FIX`：status/cgroup process view；未知 `{0}` 且 native=false |
| `P8-FS-01` | restart-safe calls 不一致 | `FIX`：同一 tested retry primitive + audited call-site set；close-like/side-effect calls不重试 |
| `P8-TH-01` | source comment 有 origin，交付链无登记 | `FIX`：MIT text、integrated lock、package owner/notice/hash；algorithm bytes frozen |
| `P8-API-01` | VM 注释错误、NTSTATUS 丢失 | `FIX`：recoverable contract + machine-readable Unknown native code |
| `P8-T01/B01/P01/R01/D01` | 新合同无证据 | `FIX`：严格执行 scope 中预声明矩阵和同合同 P1；不扩建仪器 |
| scheduler policy、pressure budget、device inventory、async/direct IO | 相邻 owner | `ROUTED`：不以“基础库尽量多”破坏职责边界 |
| cache associativity、Apple Silicon per-core Hz/id mapping | public OS facts不足 | `UNAVAILABLE`：0/absent，不建型号表或微基准猜测 |

### API、性能与红队冻结结论

- `ProcessCpuCapacity` 是 value snapshot，不是 scheduler；quota rational 保留原始最严格 hard contract，
  `parallelism_ceiling()` O(1)、无浮点/溢出。`available_cpu_frequencies` 遵循既有 null=count/truncation
  buffer-fill；查询失败的单个频率字段为 0，placement 查询整体失败才返回 Status；
- coarse cache 的 conservative rule 为每个 level 对所有已知 class 取最小非零 capacity，line size 取最大
  可信值；没有任何已知值则 capacity 0/line 64。它不代表每核精确值，不能替代 class snapshot；
- cgroup 输入规模有硬路径/行长/node 上限，所有乘法、ratio、Hz 与 path join checked；
  resolver 的字符串路径运算不会越过 mount root，OS 视图/文件在查询间变化则让 optional
  constraint unknown，不把未完整验证值当硬上限；
- FS retry 不改变成功调用的 syscall 数；只有 `EINTR` 才重试，错误转换使用最后一次 errno。synthetic
  fault route 与正常 warm-file timing 分离；Parker product bytes 不变，避免为 provenance 重跑无关微计时；
- direct peer 必须构造 HYLE 同样的 host fields与 cgroup/unknown semantics；peer 若省略 error、ancestor、allocation
  或 process limit，只能是 diagnostic/natural-route，不得生成 non-inferiority verdict；
- 没有固定 benchmark N、地址、顺序、peer、fast/slow 开关、常态 metadata 或额外 dispatch。公开 shape 来自
  系统资源 consumer；复杂度优化不改变布局；unknown 与 error 不被“便于 benchmark”抹掉。

### 实现期 P1 诊断与有限修复

- Windows 首次 v2 host diagnostic（产品 `6b69d8d1`，15 独立进程）的 A/A、10%
  known-delta、timer-floor 与 placement 控制通过；5% known-delta 的 peer-first 置信区间
  越界，所以该 bundle 只是 diagnostic，不是资格证据。四个产品 strata 却一致显示
  HYLE 约慢 7.8--8.4%，不能当计时器幽灵忽略；
- same-contract 调用追踪先发现 HYLE aggregate API 物化 `ids+ranks` 后再转
  topology visibility table，raw aggregate 直接在 CPU-set stream 上聚合。去除这段可避免的 O(P)
  memory traffic 后，本地同形回放仍慢约 7--11%，因此没有把一次合理优化冒充
  根因修复；
- 唯一一次 phase profile 将差距定位为 processor-group prefix：CPU-set visitor 内已调用
  `GetActiveProcessorGroupCount/GetActiveProcessorCount` 构造该快照，topology relation 段又做
  一次。raw 路线复用了第一份 prefix。重复段约 1,760 cycles，与实测约 0.6 us
  差距一致；这也是两次 OS snapshot 可能不一致的 correctness 问题；
- 有限修复把 Windows CPU-set 解析收敛为一个 template private visitor：
  `available_cpu_ids/info` 仍写 caller buffer，`available_cpu_topology` 则直接聚合可见 mask
  和 rank counts，不产生中间 ids/ranks；visitor 同时回传本次快照的 group prefix，
  relation 段不再重读 OS。修复后三次本地完整回放为约 0.0--0.8% 差异；正式结论仍只由
  新合同独立进程资格证据给出。同时把 malformed CPU-set record 从“返回有效前缀”
  修正为 `DataLoss`，并保证 native `EfficiencyClass=255` 的 `+1` rank 不在 byte 中
  溢出。公开 API、OS 调用数、输入与结果合同不变。
- v2 合同原误用历史 task 的旧整文件 hash，会在采样后才将真实消费证据排除。
  候选化前已把 host/fs 各自更正为当前历史文件的实际 SHA-256，不使用
  renewal 绕过新合同的 pre-sampling gate。

审计没有发现需要扩大 scope 的相邻产品缺口；上述 FIX 集现已冻结，允许开始产品实现。若实现证明 API
语义在任一 supported OS 不成立、需要修改 frozen 相邻模块或 close-like retry，则退回本阶段重新裁决。

`SCOPED -> AUDITED` 前的 registry snapshot 为
`C:\public\project-hyle\_validation\platform-v8-os-resource-contract\transitions\module-status-before-audited.json`，
SHA-256 `9FEA589026C3DA8544927EE4CBECC279572FF833F2AFA85BB9AA01F9259A603D`。

## Product candidate（`CANDIDATE`）

- product commit/tree：`e2a0ce9ddeb39d319591e5231f88112c06c75ef2` /
  `f1c4bf59e432cd283b8318ca2eaaf4d12a6580f5`；公开合同、算法、依赖、license 与正常路径成本在该树冻结；
- exact source archive：
  `C:\public\project-hyle\_validation\platform-v8-os-resource-contract\candidate\hyle-e2a0ce9d.tar.gz`，
  SHA-256 `AAFBE2A7284F1C355655C457AD85C6E9D834225288989B33AC19B616F108E6DC`；
- W1-WIN MSVC static/shared 与 W1-WSL GCC 14 static/shared、UBSan 的 Platform 实现测试已通过；三个
  exact factor 的完整回放也已通过。这些是 candidate 前验证，不代替尚待完成的 macOS、
  relocated package、direct-consumer 与最终红队；
- W1-WIN host P1 bundle
  `C:\public\project-hyle\_validation\platform-v8-perf-host-evidence-r3` 为 `QUALIFIED/PASS`；31 个独立样本下四个
  strata 全部 `PARITY` 且 NI `PASS`，A/A、5%/10% known-delta、timer-floor、placement 全部
  `PASS`，contract SHA-256
  `9561919627816e603e2ae6c42a8a2ccf38b5366929a2eff485651e00ec5db3bf`；
- W1-WIN warm-file P1 bundle
  `C:\public\project-hyle\_validation\platform-v8-perf-fs-evidence-r2` 为 `QUALIFIED/PASS`；预先冻结的
  15 -> 31 样本升级后，8 个 execution/allocation/placement strata 全部 `PARITY` 且 NI
  `PASS`，五项 controls 全部 `PASS`，provenance `EXACT`，contract SHA-256
  `c72d5c879273dffcb805892bc0acc746af61b1841a45e57d332411bf82e83adc`。首个 r1 bundle 的产品行同样全为
  `PARITY/PASS`，但 5% control 的一个顺序未通过资格门，故仅保留为 diagnostic，不用来裁决产品；
- `AUDITED -> CANDIDATE` 前态 registry snapshot 为
  `C:\public\project-hyle\_validation\platform-v8-os-resource-contract\transitions\module-status-before-candidate.json`，
  SHA-256 `1442FE58A10203418F6D108DE93732C8DF5092EAC0D74891E24659EE49DC7CE9`。

本状态只冻结候选产品身份；工程性能、交付、文档和红队的 aggregate gate 要在完成全部预声明回放后才能写
`PASS`。候选后若修改产品 API、layout、算法、依赖或正常路径成本，必须退回 `AUDITED`。

### 首候选撤回（`CANDIDATE -> AUDITED`）

M1-MAC 从上述 exact archive 进行 AppleClang 21 Release/UBSan/WERROR 编译时，
`src/platform/linux_cgroup.cc` 的 `line_is_complete()` 在非 Linux 路径成为 unused private function，
`-Werror,-Wunused-function` 正确拒绝候选。该 helper 只为 `read_linux_cgroup_path()` 的 Linux
`fgets` 生产路径服务，纯 parser fixtures 和其他 OS 不消费它；处置固定为将它收入
`HYLE_OS_LINUX` 条件编译，不改公开 API、Linux 算法、layout、依赖或正常路径成本。修复后必须重新生成
candidate identity，并至少重放 M1-MAC WERROR/UBSan、Windows/WSL exact factors 及三交付。已完成的
P1 evidence 与未改产品路径的关系将在第二候选明示记录，不默认继承。

`CANDIDATE -> AUDITED` 前态 registry snapshot 为
`C:\public\project-hyle\_validation\platform-v8-os-resource-contract\transitions\module-status-before-candidate-withdrawal.json`，
SHA-256 `7DCD67D13E9EB55C4D8C53AC26DEC4CBA0715B5C273C8799839856B3C429046D`。

## 第二 Product candidate（`CANDIDATE`）

- product commit/tree：`c39df6f94c636ab8fa94cb700dd263763a42d680` /
  `37889346564f3a5cdc895f26c9d610883076ba02`；
- exact source archive：
  `C:\public\project-hyle\_validation\platform-v8-os-resource-contract\candidate\hyle-c39df6f9.tar.gz`，
  SHA-256 `4C0A0D30B75D97B020253765621F244FDFD111A1799E4E45159E8BEA04B70A32`；
- 首候选暴露的不是运行路径缺陷，而是 OS 私有 helper 的编译 ownership 缺陷。修复将
  `line_is_complete`、Linux quota/frequency 文件 helper 与 Linux-only memory-budget merge helper
  收入 `HYLE_OS_LINUX`；Linux 预处理输出与运行算法不变，Windows/macOS 删除从未被调用的 private
  definition，公开 API、layout、依赖和正常路径成本不变；
- 候选化前，W1-WIN MSVC Release、W1-WSL GCC 14 Release/UBSan/WERROR、M1-MAC AppleClang 21
  Release/UBSan/WERROR 的三个 exact factor 与逐头测试均已通过。候选后仍须从本 archive 重放三交付、
  direct consumers、性能证据关系和最终红队，不能把首候选结果静默继承；
- `AUDITED -> CANDIDATE` 前态 registry snapshot 为
  `C:\public\project-hyle\_validation\platform-v8-os-resource-contract\transitions\module-status-before-candidate-2.json`，
  SHA-256 `2B711855F1C7EE2B698D2563C72BF42A4445023AF022F9F15D46806C8AE08484`。

第二候选只冻结新的 product identity；aggregate gate 在全部候选精确回放完成前仍为 `PLANNED`。
