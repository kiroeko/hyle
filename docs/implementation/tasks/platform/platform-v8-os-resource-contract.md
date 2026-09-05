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
  `GetLogicalProcessorInformationEx`、CPU Sets、Job Object CPU rate 与
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
   明确整数推导。Linux 读取 cgroup v2 `cpu.max` 或 v1 quota/period；Windows 只接受 Job Object hard-cap
   或 max-rate；weight/share 没有 hard ceiling，macOS 无公开 hard quota 时为 absent。调度/线程数 policy
   仍由 Sched/engine 决定。
2. 新增 `CpuFrequencyInfo { id, hardware_max_hz, policy_max_hz, current_hz }` 与 caller-buffered
   `available_cpu_frequencies(dst, cap)`，并在 convenience header 提供 `vector` overload。记录与当前
   available CPU id 对齐，0 表示 OS 没有可信事实；Windows 使用 documented processor-power records，
   Linux 使用 CPUFreq sysfs（kHz 经 checked conversion），Apple 只使用公开且实际存在的 sysctl key，
   不按型号或计时猜测 Apple Silicon 频率。
3. `cache_info()/cache_line_size()` 保留 cached coarse convenience，但改为 OS-first：Windows processor
   relation、Linux cache sysfs、Apple documented sysctl/performance-level；coarse capacity 取 process-view
   各 class 的保守非零下界，line size 取所见最大值并保留 64-byte unavailable fallback。CPUID 不再抢在
   OS cache hierarchy 前；动态 `available_cpu_cache_classes()` 仍是调优入口。
4. `HostMemoryInfo` layout 不变；Linux process limit 由 mountinfo-aware cgroup resolver 找到真实 v1/v2
   controller path，并把从 leaf 到可见 mount root 的每一层 hard `limit - usage` 取最小。`memory.high`、
   shares 与 pressure 不是 hard capacity；`max`/v1 unlimited 不伪造成 0。
5. 现有稳定 `numa_nodes()` span 不改 ABI 形状；Linux 初始化时按 `get_mempolicy`、
   `/proc/self/status:Mems_allowed_list`、resolved cgroup `cpuset.mems.effective` 的可信进程视图收窄。
   查询失败时不得把 system-online nodes 宣称为 process-available；unsupported/unknown 仍诚实退化为 `{0}`。
6. Platform-fs 只对 POSIX 明确 restart-safe、且尚未产生可观察副作用的 open/stat/fstat/ftruncate/
   fsync/fdatasync/fcntl/opendir/readdir/closedir 等调用统一处理 `EINTR`。`close` 不重试；可能已经完成部分
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

- `platform` 为 P1：W1-WIN host planning 的 candidate 与 direct OS peer 必须构造相同字段、错误路径和
  process/job limit；两边 common host 均不得在 timed call 自造 heap。5% margin 只裁决该正常 planning
  journey，不把 Linux parser、frequency getter 或冷 discovery 微基准塑造成 API；
- `platform_fs` 为 P1：沿用 4 MiB warm open/read/checksum/close，同 flags、buffer、bytes、durability、
  EOF/error accounting；EINTR synthetic fixture 只裁决 correctness，不和无信号 peer 排名；
- `platform_thread` 为 P0：integrated-source bytes 未变则以 source hash、memory-order/layout diff、原测试和
  三交付证明没有成本变化；只有产品 bytes 改变才需要重新资格化 Parker P1；
- P0 结构门：cgroup resolver 对 mount/ancestor 输入 O(lines+depth)，Windows CPU mapping O(CPU+group)，
  frequency/cache discovery O(available CPU × exposed OS records)，warm cached coarse cache O(1)，无常态
  scheduler metadata 或热路径 syscall；
- apples-vs-oranges 禁令：placement 不与 quota、cached 不与 fresh discovery、完整 Status 不与 unchecked
  scalar、cursor IO 不与 explicit-offset、signal-free 不与 injected-EINTR 路线排名；不能用较强 HYLE 语义的
  劣势冒充同合同 FAIL，也不能用较弱 peer 的胜利冒充 SOTA；
- 若 supported OS 无法诚实表达某字段则返回 0/absent，不猜测；若实现要求 Sched/Mem/IO policy、私有 OS
  接口、常态 heap/lock/metadata、或无法在有限 FIX 内消除 false resource facts，则退回 `AUDITED` 或
  `BLOCKED`，不得继续研究代替状态。

### 预声明验证

- W1-WIN：MSVC Release/WERROR，三个 exact factor，Platform product，unit tests；CPU Sets/group mapping、
  Job quota、processor power/cache；source/static/shared + relocated package；host/file P1 qualification；
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
