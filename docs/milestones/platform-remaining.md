# platform 剩余事项收口

> **后续替代（2026-07-22）：**本文后文提到的旧 `MemoryArena` 当前由 `SharedPool` / 显式 `SharedPoolCache` 覆盖；旧头与 C ABI 已删除。

> **后续替代（2026-07-24）：**platform 已删除模块私有 TLS last-error，公开可恢复失败统一走
> error 的 `Status/Result`；CPU 聚合系统拓扑与动态进程可用 CPU 也已拆分。本文是历史任务单，
> 当前使用合同以 guide 为准。

> 2026-07-19。原清单 A1/A2/A3/B1 已实现并完成三平台验证。
> 本文保留实现边界与仍有意不做的内容，避免后来者把“未知”误读成
> “同构”，或再次引入启发式探测。

> **预发布冻结证书（2026-07-24）：**实际使用导向复审已完成。Windows MSVC、WSL GCC 14、
> Apple M4 AppleClang 21 均通过 platform C++/纯 C 功能测试、**118/118** C 声明↔纯 C
> 引用↔DLL/ELF/Mach-O 导出一致性，以及 **9 个 C++ + 1 个 C** 公开头独立编译。下游审计
> 确认 mem/container/sched 消费动态进程 CPU 视图，无自造 OS 拓扑/affinity/priority；唯一
> `hardware_concurrency` 是可用 CPU 查询失败后的保底。VM/fs/MappedFile 当前 `Result/Status`
> 形状已补齐 Linux/M4 统计证书，详见 performance 文档。platform 进入预发布冻结；只有真实
> 消费者缺口、支持平台变化、正确性 bug 或当前同合同性能回退才能重开。

> **NUMA 定向复开（2026-07-26）：**对照 oneTBB 后补 actual sparse node IDs、runtime capability、
> selected-node/chunk allocation 与 raw/RAII/C ABI；当前 surface 为 **120/120**。旧 118 项证书是
> 复开前历史基线，不再代表 HEAD 数量；本轮 Windows/WSL 验证与单节点 peer 数据见 guide/performance。

> **SOTA 性能回审（2026-07-24）：**`504d671` / `75f3a45` 把 zero-timeout 从 OS wait
> 收回单 consumer 用户态 load，Win/WSL/M4 n=15 为 **1.690/0.564/1.346 ns**，platform 与
> sched C/C++ 三平台回归全绿。`3997964` 将 MappedFile raw 对照补成同等 UTF/path、文件大小
> 与范围验证，并新增 FileHandle raw floor 和 16 MiB 持久随机页访问：mapping 同合同距 raw
> ≤1.8%，安全 FileHandle 距 raw ≤2.4%，持久随机访问赢 ifstream seek/read 168–2355x。
> 点值和选择边界只以 performance/guide 为准，本任务单仅记录重开原因与验收提交。

> **易用性回审（2026-07-24）：**C ABI 保持 allocator-neutral buffer-fill/callback/opaque-owner
> core；C++ 新增 capturing/move-only callable `Thread::spawn` 与 opt-in header-only owning
> string/CPU 查询。普通用户不再手写 count/allocate/retry，性能敏感和 FFI 调用者也不支付
> 隐藏分配。真实配方、截断/动态增长和 required 失败销毁路径纳入三平台测试。

> **全库下游消费回审（2026-07-24）：**生产实现（`src/`，不把 benchmark/test/tool 当产品
> 缺陷）已按“platform 通用机制 / 下游特化机制”逐项扫描。发现并修复两处真实绕过：compute
> 的 GPU poller 改用 `platform::Thread + Parker`，线程/注册资源失败返回共享 `Status`，完成扫描
> 零分配；Vulkan 管线缓存的环境变量与文件访问改走 platform 的 UTF-8 fs 合同，并固定为启动
> 时同一路径快照，缓存 fixture 直接使用非 ASCII 文件名验收。依赖可见性也按公开形状修正：
> compute/container 的直接实现消费为 private，公开 `SchedulerOptions` 命名 platform 类型的
> sched 为 public；io 只有编译期 `HYLE_OS_*` 宏而没有 platform 符号/类型，不再伪报二进制依赖。
> 其余原生调用的保留边界如下表；这张表是防止以后“为了统一而错误统一”的审计基线。

| 下游 | 结论 | 原因 |
|---|---|---|
| `mem` / `container` | 已正确消费 | VM、NUMA、页信息、动态进程 CPU 视图均走 platform；此前收口已回归。 |
| `sched` | 已正确消费 | worker 使用 `Thread/Parker/available_cpu_ids`；blocking pool 与停机屏障的 `std::condition_variable` 是多等待者谓词协调，不等价于单 token Parker。 |
| `compute` | 本轮修复 | poller 的线程/等待/错误与 Vulkan cache 的 env/fs 已收束；CUDA/Vulkan runtime 调用仍是 compute 后端语义。 |
| `io` | 有意保留原生后端 | IOCP/epoll/io_uring、OVERLAPPED 文件句柄、ring `mmap` 与 named shared memory 是异步 I/O/跨进程共享语义，platform 的同步 FileHandle、匿名 VM、MappedFile 不能替代。 |
| `time` | 有意保持零 HYLE 依赖 | 单调/墙钟与 sleep 是 time 自己的产品语义；按依赖图不反向依赖 platform。 |
| `obs` / `util` | 有意保留窄适配 | TTY/debug sink 与符号反解是模块私有外部适配，不形成用户需要的通用 platform 类型。 |
| `simd` / `math` / `graph` | 无重复 OS 抽象 | 扫描未发现应改走 platform 的生产调用。 |

判定规则：只有跨模块可复用的 OS 机制进入 platform；设备/runtime、异步完成模型、命名共享
对象、时钟语义和输出/符号适配仍由拥有其合同的模块实现。传递链接不算依赖声明，直接使用
platform 类型或符号的模块必须显式声明依赖。

## 已完成

### A1. cache capacity

- C++：`CacheInfo` / `cache_info()`，返回 L1D/L1I/L2/L3 字节数。
- C ABI：`hyle_cache_info` / `hyle_cpu_cache_info()`。
- x86 用 deterministic cache parameters（CPUID leaf 4）；macOS 用
  `hw.l1dcachesize` 等 sysctl；Linux 用 `_SC_LEVEL*_CACHE_SIZE`。
- 不可得的层级返回 0；结果由函数内静态对象缓存。

### A2. hybrid topology

- C++：`CpuTopology` / `cpu_topology()`。
- C ABI：`hyle_cpu_topology` / `hyle_cpu_topology_get()`。
- 计数统一为 OS online logical CPU：Windows 读取 CPU Set
  `EfficiencyClass`，macOS 读取 `hw.perflevelN.logicalcpu`，Linux 只在
  每个可见 CPU 都有 `cpu_capacity` 且容量确实分级时分类。
- 无可信数据时返回：
  `performance_cpus=system_logical_cpus, efficiency_cpus=0, hybrid=false`。
  这是“无法确认异构”，不是宣称物理上一定同构；不使用最大频率猜测。
- 聚合视图仍用于容量展示；逐 CPU 的可调度事实由 A7 的动态进程视图提供。

### A3. C wrapper 共享错误

后续 error 收敛已取代本任务当时的 TLS last-error 过渡形状：Parker、Thread、DynLib、
MappedFile、VM/NUMA、fs/env 等公开可恢复失败都直接返回共享 `hyle_status`，C 调用方按
error 的统一所有权释放；platform 不再保留私有 last-error。

### A4. 系统拓扑与进程 CPU 配额

- `cpu_topology()` / `hyle_cpu_topology_get()` 是缓存的系统在线 CPU 与可信 P/E 聚合。
- `available_cpu_ids()` / `hyle_cpu_available_ids()` 动态读取 affinity/cpuset/CPU Sets，
  返回可直接用于 thread affinity 的真实（可能稀疏）id，worker 数取其完整 count。

### A5. Thread 配置合同与下游 placement（2026-07-24）

- `ThreadOpts::config_policy` 默认 best-effort；required 在用户 entry 前完成设置，失败返回
  `Status` 且 entry 不运行。C ABI 具名 policy 与 setter 1:1 覆盖。
- `this_thread::set_name/set_affinity/set_priority` 返回共享 `Status`，不再只能静默吞错；
  Linux 大 CPU id 使用动态 mask，macOS priority 使用 QoS，硬 pin 明确不支持。
- sched worker 用 `available_cpu_ids` 返回的真实（可能稀疏）id，而非把 worker ordinal
  当 OS CPU id；placement 仍属 sched 策略，platform 只提供机制与错误。
- 默认 Thread 创建路径不做 required 握手、不增加 refcount RMW；required 的同步成本只在
  用户明确请求强保证时发生。

### A6. 单 CPU 与 CPU 集合 affinity（2026-07-24）

- `CpuAffinity` 统一表达无约束、单 CPU 与 CPU 列表；`ThreadOpts` 不暴露互斥的两个 C++
  option。列表是调用期 view，spawn 同步复制到同一次 `ThreadStart` 分配的尾随区。
- C ABI 保留常用单 CPU setter，并增加数组 setter / calling-thread 数组入口；空/null、同时
  设置单 CPU 与数组会在边界返回共享 `InvalidArgument`。
- Linux 动态 mask 精确允许集合内迁移；Windows 硬 mask 要求集合位于同一 processor group；
  macOS 继续诚实返回不支持。one-of/轮转/超订仍属于 sched placement 策略。

### A7. 逐 CPU 性能能力与强核优先（2026-07-24）

- `available_cpu_info()` 返回当前进程可用 CPU 的真实 id、单机内单调的
  `performance_rank` 与 `CpuKind`；C ABI 等价覆盖。rank 0 明确表示未知。
- Windows 只信任 CPU Set `EfficiencyClass`，Linux 只信任内核导出的
  `cpu_capacity`；不根据逻辑 id 或瞬时/最大频率猜测。macOS 无公开的逻辑 CPU 到
  perflevel 稳定映射，因此逐 CPU 保持 unknown，软偏好使用已有 QoS priority。
- `available_cpu_ids(CpuOrder, ...)` 保留所有可用 CPU，只改变已知能力的选择顺序；未知
  能力不会被猜成强核。默认重载继续保持 OS 顺序与原有零额外探测路径。
- sched 通过 `SchedulerOptions::worker_cpu_order` 显式消费该顺序。默认不替用户改变功耗/
  调度策略；用户选择 performance-first 时，少量 worker 会先占已知强核。需要“只用强核”
  的硬约束时，先过滤 `CpuInfo::kind`，再传给 `CpuAffinity`。
- `SchedulerOptions::worker_priority` / C ABI 对等字段把 worker 的 OS priority/QoS 公开为
  独立 opt-in；这是 macOS 无硬 affinity 时可执行的强核软偏好，不把 QoS 偷塞进 CPU 排序。

### B1. macOS parker timeout

`Parker.ParkTimeoutReturnsFalseOnTimeout` 直接测量 20 ms 无唤醒等待：
下限 10 ms（防立即超时/忙等），上限 5 s（防单位错误导致近似无限等待）。
macOS 真机通过，确认 `os_sync_wait_on_address_with_timeout` 当前参数语义与
实现一致。

## 本轮审计一并修复

- VM 非空 null range、page-rounding overflow、guarded reserve 加法溢出及
  `VmRegion::adopt` 错位/溢出不再伪成功。
- MemoryArena/StackAllocator 的空指针与外来指针运算不再触发 C++ UB。
- GrowablePool 在对象构造抛异常时归还槽位。
- StatsAllocator 开启 leak tracker 后，realloc/free 用记录尺寸做诊断计数。
- Windows 纳秒转毫秒上取整不再溢出；文件 append 改用原子
  `FILE_APPEND_DATA`；目录枚举会区分正常结束与系统错误。
- MappedFile Windows 路径转换不再有 1024 wchar 人工上限。
- 对照 Folly `MemoryMapping` 补齐显式 access advice、prefetch 与 page residency；默认 open
  不预取、不锁页，POSIX 走 madvise/mlock，Windows 只诚实实现 will-need + VirtualLock。
- 对照 libuv 的 portable file open 语义，Windows FileHandle/MappedFile 增加
  `FILE_SHARE_DELETE`，原子 rename/delete 不再被已打开 owner 阻塞，并有三平台回归测试。
- POSIX fs、MappedFile、VM、NUMA 不再各维护近似 errno code switch，统一消费 error 的
  canonical taxonomy；操作特定 override 仍留在 owner。
- C ABI 同步覆盖四个 mapped advice/residency 操作；platform surface 机械清单增至 118 个，
  C++/C 独立头与 Windows/WSL/M4 定向测试通过。
- Linux NUMA policy mask 覆盖实际稀疏 node id（固定 4096-bit 上限），非法 node/range 明确失败；
  topology 查询与 membership 分别缓存为 O(1) count 和 O(1) 位图 lookup。
- ErrorRecord 析构自动恢复、重复 install 幂等；空 format/无效 handler
  不再崩溃。

## 有意保留

- cache associativity、每级 line size：当前消费者只需要容量、false-sharing line、可信的
  系统 P/E 聚合和独立的进程可用 CPU 能力；不从聚合值猜 worker 数或单核类型。
- Linux 不依据 cpufreq 猜 P/E；若内核不导出 scheduler capacity，则保持
  unknown/homogeneous fallback。
- selected-node set + `bytes_per_chunk` 已在 2026-07-26 定向复开：公共合同是实际 node id +
  runtime capability + structured Status，不要求每个 OS 伪装相同 primitive；Linux/Windows
  多节点安装条带 policy，重复 id 表达 weighted placement，macOS 单节点返回 NotNeeded。Linux
  极细自定义布局使用固定批次 first-touch/move 限制 VMA policy 数。只有 Linux bind 是严格节点集合；
  Windows preferred commit 与 Linux interleave 都可能在内存压力下回退，`Applied` 不是逐页
  residency 证明。raw/RAII 与 C ABI 均覆盖，selected region 禁止 decommit/recommit，防止
  policy 被静默丢失。
- platform C ABI 已覆盖 wait/Parker、Thread、DynLib、VM/VmRegion、
  hugepage/NUMA、完整 fs/env/fd、MappedFile、CPU/cache/topology。2026-07-19
  复审又补了 thread opts `struct_size`、具名权限/open flags 与逐符号纯 C 验收；后续
  已删除过渡期 `hyle_platform_last_error` 并统一到共享 error；不再有“其余
  fs/mapped/NUMA 待更大范围 ABI”的缺口。
