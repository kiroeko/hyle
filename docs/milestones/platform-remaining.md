# platform 剩余事项收口

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
  `performance_cpus=logical_cpus, efficiency_cpus=0, hybrid=false`。
  这是“无法确认异构”，不是宣称物理上一定同构；不使用最大频率猜测。
- 当前只提供聚合计数，不提供逐 CPU 类型数组。

### A3. C wrapper OOM last-error

Parker、DynLib、MappedFile、VmRegion 与 HugeRegion 的 C wrapper 分配失败
均写入对应 thread-local last-error；非法参数也不再静默返回空句柄。
error C ABI 另有 `hyle_error_last_error()` 覆盖 push-handler token OOM。

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

- cache associativity、每级 line size、逐 CPU P/E 类型数组：当前消费者
  只需要容量、false-sharing line 与聚合 worker 计数。
- Linux 不依据 cpufreq 猜 P/E；若内核不导出 scheduler capacity，则保持
  unknown/homogeneous fallback。
- platform C ABI 已覆盖 wait/Parker、Thread、DynLib、VM/VmRegion、
  hugepage/NUMA、完整 fs/env/fd、MappedFile、CPU/cache/topology。2026-07-19
  复审又补了统一的 `hyle_platform_last_error`、thread opts `struct_size`、
  具名权限/open flags 与逐符号纯 C 验收；不再有“其余
  fs/mapped/NUMA 待更大范围 ABI”的缺口。
