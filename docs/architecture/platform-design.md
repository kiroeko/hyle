# Platform：保留真实系统差异的机制层

Platform 为 Windows、Linux、macOS 提供可组合的 OS 事实与资源机制。它回答“系统能做什么、操作
是否成功”，不替调用方决定 worker 数、模型放置、内存淘汰或设备计算策略。示例与平台适用性见
[Platform 指南](../user/platform.md)。

## 一个产品，三类内部依赖

用户选择 `hyle::platform`；内部拆分用于避免不必要的依赖：

| 内部 factor | 拥有的内容 | 公开依赖 |
|---|---|---|
| `platform` | CPU/host facts、raw VM/NUMA、动态库、环境、OS entropy | Core、Error |
| `platform_thread` | OS thread、地址等待、Parker | Core、Error、Time |
| `platform_fs` | 同步文件与文件系统控制 | Core、Error、Time |

thread/fs 实现私有复用 host factor。Time 边来自 `Duration` 与 `UnixTimestamp` 等真实公开类型，
不是为构造顺序添加的伪依赖。普通用户无需知道拆分；维护者按实际消费声明直接依赖，具体图由
[HyleComponents](../../cmake/HyleComponents.cmake) 与[构建架构](build-system.md)维护。

Core 的 `HYLE_OS_*` 只描述编译当前 TU 的目标，不能代替运行期 capability 查询。

## CPU 信息为何不能合成一个“最高性能档”

| 信息 | 寿命与含义 | 不代表什么 |
|---|---|---|
| `cpu_features()` | 进程可安全执行的具名 ISA facts | 当前实现的完整编译要求、provider winner |
| `cache_info()` | 缓存的保守系统容量提示 | 当前 affinity 内每类 CPU 的实际 cache profile |
| `hardware_concurrency()` | OS-online 粗略数量 | 进程允许 placement 或持续 CPU quota |
| `available_cpu_ids/info/topology()` | 动态进程 placement 视图 | `[0,count)` 连续 ID、永久 topology |
| `available_cpu_cache_classes()` | 当前可见异构 class 的容量、共享宽度和 rank | 某个 logical ID 的硬绑映射 |
| `process_cpu_capacity()` | placement 数与可选 hard bandwidth quota | SMT 效率、超订或服务 latency 策略 |
| `available_cpu_frequencies()` | OS 公布的 hardware/policy/current Hz | 跨微架构强弱、ISA gate、可信 P/E 分类 |

ISA 探测同时检查硬件位与 OS register-state 支持，probe TU 保持 baseline ISA；不能因父工程 LTO
把探测本身编成尚未确认可执行的指令。kernel 的实际 owner 将完整 TU requirements 与这些 facts
匹配，Platform 不发布一个适合所有实现的“best target”。AMX 等硬件 feature 与调用线程许可分开，
线程的 vector width/permission 通过专门接口查询或申请。

动态 placement 尊重 affinity、cpuset/cgroup 与 Windows CPU Sets；ID 可稀疏，topology domain ID
只在一次 snapshot 内比较。查询放在初始化或重配点，不能为省查询成本把旧 cpuset 永久缓存，也不能
在每 task 热路径重复读取全部 OS topology。

`performance_rank` 只采用可信 OS 映射；未知保持 unknown，不从编号、频率或型号猜 P/E 核。
`CpuOrder` 排序完整可用集合，不偷偷过滤 CPU。macOS class-level cache facts 可用，但 per-logical-ID
硬绑映射不可信；强核软偏好通过 QoS 表达，不能伪造 affinity。

Linux CPU quota 按实际 cgroup namespace/mount 解析并合并可见祖先的 hard limit，shares/weight
不是 hard quota。Windows 只能查询 immediate Job，不能据此证明嵌套相对 rate 的最终上限，因此
对应字段当前 absent。`parallelism_ceiling()` 只是基于已知事实的保守推导，应用仍决定 worker 策略。

## Host/NUMA 预算不是资源预留

`host_memory_info()` 是动态、可失败的 planning snapshot。成功时物理量存在，commit/swap/process
约束只有能诚实取得时才提供 optional；合法 0 不表示 unknown。`effective_available_bytes()` 取
已知约束的保守下界，不保证下一次分配成功。UMA 的 host/device 可用量不能当两个独立池相加。

Linux 内存约束同样合并可见 cgroup 祖先，包括祖先组其他进程的占用。Windows immediate Job 不能
证明嵌套父组的最终有效预算，所以不把局部值冒充有效 process limit。

`numa_nodes()` 使用实际可用且可能稀疏的 node ID；不要由 node count 构造 `[0,count)`。该 span
是首次查询固定的进程寿命视图，不能在宿主后来改 placement 后假装仍是新快照。逐 node memory
查询则是动态事实：Linux node available 是 free/file-backed/reclaimable 的估计，不能冒充全局
MemAvailable 的精确同义值；多节点 Windows 能报告 available，却不能证明 per-node total。
全局 cgroup/Job/commit 约束只能组合一次，不能复制到每个 node。

NUMA preferred 是软策略；Linux bind 限制严格节点集合；interleave/selected interleave 表示系统
接受 placement policy，并非逐页永久驻留证明。selected nodes 的顺序与重复 ID 可表达加权条带；
能力不足时明确失败或报告 unavailable，不静默伪造同等机制。大页、prefault、NUMA 放置也分别计费。

## Thread 与等待不成为第二个 scheduler

`Thread` 是 move-only OS owner。原始 `entry + context` 服务 allocation-minimal runtime；owning
callable 重载才承担 capture storage。`ThreadOpts` 中短期 view 在 spawn 返回前复制。

默认 best-effort 允许附加配置失败而执行 entry；required 则在 entry 前完成握手，配置失败时 entry
不运行。未请求 priority 时不改宿主策略；显式 priority 仅是 OS hint，提升可能因权限失败。Linux
process-priority 形状作用于调用线程并由之后创建的线程继承，完整启动效果要求在建 worker 前调用。
Windows priority/QoS 是多次控制，不是事务；macOS QoS 不是指定某颗强核。

仍 joinable 的 Thread 析构是生命周期错误，不暗中 detach。线程池、任务取消、超订和结构化并发归
[Sched](../user/sched.md)。

裸地址等待允许伪唤醒，调用方循环检查自己的原子谓词。`Parker` 增加一个 token，解决先 unpark
后 park；重复 unpark 合并，不是计数 semaphore。只允许一个等待消费者，timed park 是 single-shot；
外层 deadline loop 自己计算剩余时间。它不是 fiber-aware wait，不能直接用于阻塞 scheduler worker。

## VM、mapping、同步文件和异步 I/O 的边界

`VmRegion` 与 `vm_*` 拥有匿名地址空间 reserve/commit/protect/decommit/release 机制。
普通用户优先 RAII；已有自定义 owner 才用裸函数。`allocate_rw` 满足一步分配，不能为统一实现强迫
所有用户支付 staged reserve/commit。guard page 是明确选择，不等于所有 allocation 都带保护。

file/shared mapping、residency、range flush 与统一 Buffer/View 归 [Mem](mem-design.md)。
`VirtualMapping` 是 `VmRegion` 的别名，避免第二套 anonymous owner；FileMapping 消费已打开的
FileHandle，避免再次 path lookup。设备内存、coherence 与设备 completion 归 Compute。

同步文件分成 cursor-owning `FileHandle` 和 caller-synchronized、显式 offset 的 `RandomAccessFile`。
duplicate 只复制 OS ownership，不保证复制 cursor/lock 状态，更不使别名自动线程安全。短读写保留
native 语义，exact/all helper 才继续剩余传输；异步、direct/vectored 与完成通知归 IO。

文件 API 保留这些区别：

- UTF-8 路径、opened-object identity 与纯词法路径不同；路径算法归 Container；
- 不存在可返回成功 false，权限/I/O 失败不能吞掉；
- rename 的原子目录项替换不等于文件内容持久化，sync 与 close 的结果需要显式观察；
- preallocation 与 EOF 调整不是跨系统统一事务；
- 文件锁是 cooperating handles/processes 的协调，不是线程 mutex 或分布式一致性；
- POSIX 只重试可安全 restart 的 EINTR；close 和可能已经产生外部副作用的 path mutation 不盲目重试。

## 其他资源与失败

`DynLib::get` 成功后也持有 loader reference，不是与 unload 竞争的裸借用。symbol view 不能活过
owner，类型签名由调用方负责；相对路径和搜索名仍属于宿主加载信任边界。析构只兜底，需要观察
卸载错误时调用显式 close。

environment 的 unset 与已设置为空不同；修改进程环境只适合同步启动/重配阶段。`secure_random`
使用 OS entropy，完整填满或失败，不维护隐式 PRNG；可复现采样使用 Util 的显式 RNG。

构造/有值查询使用 Result，变更使用 Status，原生 code 在调用边界及时获取。buffer-fill core 不
隐藏 owning 结果分配；`*_convenience.h` 才提供显式 vector/string 便利面。当前公开交付是 C++
source/static/shared，不以 opaque C handle 再造一套资源系统。

维护时关注动态事实是否过期、native side effect 与错误是否匹配、required 配置是否真的在 entry
前生效，以及各系统无法等价的路线是否诚实拒绝。公开合同见[头文件](../../include/hyle/platform/)。
