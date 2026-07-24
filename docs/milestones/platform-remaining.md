# platform 剩余事项收口

> **后续替代（2026-07-22）：**本文后文提到的旧 `MemoryArena` 当前由 `SharedPool` / 显式 `SharedPoolCache` 覆盖；旧头与 C ABI 已删除。

> **后续替代（2026-07-24）：**platform 已删除模块私有 TLS last-error，公开可恢复失败统一走
> error 的 `Status/Result`；CPU 聚合系统拓扑与动态进程可用 CPU 也已拆分。本文是历史任务单，
> 当前使用合同以 guide 为准。

> 2026-07-19。原清单 A1/A2/A3/B1 已实现并完成三平台验证。
> 本文保留实现边界与仍有意不做的内容，避免后来者把“未知”误读成
> “同构”，或再次引入启发式探测。

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
- Linux NUMA policy mask 覆盖 256 nodes，非法 node/range 明确失败。
- ErrorRecord 析构自动恢复、重复 install 幂等；空 format/无效 handler
  不再崩溃。

## 有意保留

- cache associativity、每级 line size：当前消费者只需要容量、false-sharing line、可信的
  系统 P/E 聚合和独立的进程可用 CPU 能力；不从聚合值猜 worker 数或单核类型。
- Linux 不依据 cpufreq 猜 P/E；若内核不导出 scheduler capacity，则保持
  unknown/homogeneous fallback。
- platform C ABI 已覆盖 wait/Parker、Thread、DynLib、VM/VmRegion、
  hugepage/NUMA、完整 fs/env/fd、MappedFile、CPU/cache/topology。2026-07-19
  复审又补了 thread opts `struct_size`、具名权限/open flags 与逐符号纯 C 验收；后续
  已删除过渡期 `hyle_platform_last_error` 并统一到共享 error；不再有“其余
  fs/mapped/NUMA 待更大范围 ABI”的缺口。
