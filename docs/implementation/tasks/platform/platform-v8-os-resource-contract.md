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
