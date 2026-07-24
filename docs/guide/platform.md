# HYLE platform：什么情况选择什么

platform 是 HYLE 的跨 OS 原语层。它服务两类调用者：HYLE 内部模块需要稳定、可组合的
线程/等待/页内存地基；库用户需要在 Windows、Linux、macOS 上表达同一资源合同。它不把
每个系统调用都公开，也不因为 benchmark 有一行就制造一个类型。先按资源和生命周期选
形状，再到 [`c-platform.md`](./c-platform.md) 查纯 C / FFI 的精确签名。

## 一页选择表

| 需求 | 选择 | 不要选 | 关键合同 |
|---|---|---|---|
| 一块立即可读写的页内存 | `VmRegion::allocate_rw` | `reserve` 后立刻 `commit` | 一次 OS mapping；RAII 释放 |
| 先占地址、以后分段提交 | `VmRegion::reserve` + `commit/decommit` | 普通 heap 分配 | 页粒度；地址空间与物理提交分离 |
| 有越界保护的 fiber/线程栈 | `VmRegion::reserve_guarded` | 手写 magic padding | 低一页 no-access；`usable()` 才可写 |
| 大块页内存偏好某 NUMA node | `reserve_numa` / `vm_bind_numa` | 把 soft preferred 当硬保证 | 检查 `NumaPlacement` |
| 普通小对象、容器 backing | mem 的 `Allocator` / `Region` | platform VM | platform 不替代 malloc |
| 顺序读写普通文件 | `FileHandle` | `MappedFile` | move-only；短读/短写正常；显式 close 可观察错误 |
| 原子创建临时文件 | `create_temp_file` | 先生成名字再 open | 返回已打开 handle，避免 TOCTOU |
| 只需低成本枚举目录名和类型 | `for_each_entry` | 每项都 `stat` | entry 类型不跟随 symlink；常见路径零额外 stat |
| 需要完整单项元数据 | `stat` | 把目录遍历的 entry type 当完整 stat | 跟随 symlink，并另报 path 自身是否 symlink |
| 大文件随机访问/共享映射 | `MappedFile` | 小文件必然 mmap | RO/RW/COW；Linux 小文件顺扫未必赢 read |
| 查询 cwd/temp/home | 对应 `*_directory` buffer-fill | 猜环境变量/固定 `/tmp` | `Result<size_t>` 是完整长度；可先 size query |
| 读取/修改进程环境 | `get/set/unset_env_var` | 把 unset 和空值都当 0 | unset=`NotFound`；写操作仅应在启动期做 |
| 一次性启动 OS 线程 | `Thread::spawn` | sched task/fiber | move-only；join 回收；配置默认 best-effort，可选 required |
| 大量轻量任务 | sched | 每任务一个 `Thread` | platform 只提供线程地基 |
| token 不丢失的线程 park/unpark | `Parker` | 裸 condition variable 拼状态 | unpark-before-park 可消费 token |
| 等待调用方已有的原子字 | `wait_on_address` | `Parker` 再包一层 | 允许伪唤醒，必须循环检查谓词 |
| 插件/可选系统库 | `DynLib` | 裸 `dlopen`/`LoadLibrary` 分叉 | `Result` 保留原生 loader 诊断 |
| 运行时 ISA 派发 | `cpu_features` / `supports` | 编译机宏代替运行机探测 | 只调用当前进程可执行的 ISA |
| 决定当前 worker 数/绑核 | `available_cpu_count` / `available_cpu_ids` | `hardware_concurrency` / `cpu_topology` | 动态遵守 affinity、cgroup/cpuset、Windows CPU Sets；id 可稀疏 |
| cache 容量提示、P/E 总量 | `cache_info` / `cpu_topology` | 假定探测值永远存在 | 未知 cache 为 0；P/E 不做频率猜测 |

## 页内存

普通调用应直接请求立即可用的 mapping：

```cpp
#include "hyle/platform/virtual_memory.h"

HYLE_ASSIGN_OR_RETURN(
    hyle::platform::VmRegion pages,
    hyle::platform::VmRegion::allocate_rw(2 * 1024 * 1024));
auto* bytes = static_cast<std::byte*>(pages.data());
bytes[0] = std::byte{0x2a};
```

只有确实需要 staged commit 时才拆开：

```cpp
HYLE_ASSIGN_OR_RETURN(auto arena, VmRegion::reserve(1ull << 30));
HYLE_RETURN_IF_ERROR(arena.commit(0, 4ull << 20));
// 未提交部分不可访问；之后可 decommit 已不用的页而保留地址。
```

`vm_allocate_rw` 等自由函数用于实现自有 owner 的底层代码；普通用户优先 `VmRegion`，避免
忘记释放或把错误 size/base 传回。VM 大小向上取整到页，`size()==0` 不是有效分配。

## 文件：流式 I/O、映射和路径操作

`FileHandle` 对 POSIX fd 与 Windows HANDLE 使用同一所有权形状，但不假装两者是同一种
原生类型。析构会关闭；写入后若 close 失败必须可见，就显式调用 `close()`。

```cpp
#include "hyle/platform/fs.h"

HYLE_ASSIGN_OR_RETURN(
    auto file,
    hyle::platform::open_file("state.bin",
        hyle::platform::kOpenWrite | hyle::platform::kOpenCreate |
        hyle::platform::kOpenTruncate));
HYLE_ASSIGN_OR_RETURN(std::size_t moved, file.write(data, size));
if (moved != size) { /* 继续写剩余部分 */ }
HYLE_RETURN_IF_ERROR(file.close());
```

read 返回 0 才是 EOF；read/write 返回短计数仍是成功。`MappedFile` 适合映射本身带来价值的
场景，例如大模型权重、随机访问、共享页和 COW；不要假设 64 KiB 顺序全扫也会比缓冲 read
快。stat/size/权限/symlink 和所有 path 变更都返回 Result/Status：不存在、空文件/0000 权限、
visitor 提前停止不再共享一个 0。跨平台代码优先 `is_readonly/set_readonly`，不要把 Windows
合成的 POSIX mode 当真实 ACL。cwd/temp/home/env 同样返回 `Result<size_t>`：值是完整长度，
可先用 `dst=nullptr` 查询；环境变量未设置是 `NotFound`，已设置为空是成功的 0。
`temp_directory` 只发现 OS/环境配置的路径，不额外探测可写性；需要真正可用且无 TOCTOU
窗口的临时文件时直接调用 `create_temp_file`。

`stat(path)` 成功就表示对象存在，`FileInfo` 因此不再复制一份可能互相矛盾的 `exists`
状态；`type` 是跟随 symlink 后的 target 类型，而 `is_symlink` 描述 path 自身。批量目录
扫描则用 `for_each_entry(name, FileType)`：它返回 entry 自身类型（symlink 不跟随），优先使用
`dirent` / `FindFirstFile` 已有元数据，只有 POSIX 文件系统返回 `DT_UNKNOWN` 才补一次 `lstat`。
这是一条无分配快速原语；确实需要 size/mtime 时由调用方只对选中的 entry 调 `stat`，platform
不制造持有字符串和分配器策略的高层 directory object。

`create_temp_file` 返回 `CreatedTempFile{file, path_size}`。handle 始终是权威资源，`path_size`
遵循其它 buffer-fill API 的完整长度合同，因而调用方不会在 path 被截断时误拿半条路径删除。
Windows 创建 symlink 时必须编码 file/directory：已有 target 可用 `SymlinkKind::kAuto`；创建
dangling link 时显式传 `kFile` 或 `kDirectory`，才具有跨平台一致语义。

## 等待与线程

`Parker` 是一个 token mailbox：`unpark()` 先发生也不会丢，下一次 `park()` 会消费 token。
它不是计数 semaphore，多次 unpark 可以合并成一个 token。`wait_on_address` 更底层，适合
对象已经拥有原子状态字的内部同步原语；伪唤醒允许发生，因此始终循环检查业务谓词。

`Thread` 适合少量长寿命 OS 线程、runtime worker 或必须设置 stack/name/affinity 的边界；
任务级并发用 sched。`ThreadOpts` 默认 `kBestEffort`，适合“能设则设、失败仍可运行”的
可移植 runtime。确实依赖设置（隔离核、基准控制线程、实时服务线程）时改成 required：

```cpp
unsigned cpu = 0;
HYLE_ASSIGN_OR_RETURN(auto count, available_cpu_ids(&cpu, 1));
if (count == 0) return Status::FailedPrecondition("no available CPU");

ThreadOpts opts;
opts.name = "ingest";
opts.affinity = cpu;  // 单 worker 固定归属：单 CPU 便利形状
opts.config_policy = ThreadConfigPolicy::kRequired;
HYLE_ASSIGN_OR_RETURN(auto thread, Thread::spawn(&entry, ctx, opts));
// 返回成功前 name/affinity 已设置；设置失败时 entry 根本不会运行。
```

已在线程入口内、需要逐项处理错误时，直接用 `this_thread::set_name/set_affinity/set_priority`；
三者返回共享 `Status`，不另造 errno/TLS 通道。Linux priority 提升通常需要权限；macOS
priority 映射到系统 QoS，且没有公开的硬 CPU pin API，所以 required affinity 返回
`Unimplemented`。best-effort 会显式忽略这些失败并继续运行。绑核 id 必须取自
`available_cpu_ids()`；Linux cpuset 可能产生 `{2, 6}` 这种稀疏 id，不能拿 `[0, count)` 猜。

单 CPU 与 CPU 集合不是两套策略。`CpuAffinity{cpu}` 表示固定到一个 worker/core，适合
per-core shard、NUMA arena owner；`CpuAffinity{std::span{cpus}}` 表示线程可在给定集合内
迁移，适合把线程池限制在应用保留核、性能核或 NUMA-local 分区。集合是输入 view，但
`Thread::spawn` 返回前会复制，所以 backing array 只需活过该次调用：

```cpp
const unsigned service_cpus[] = {2, 6}; // 实际代码应来自 available_cpu_ids()
ThreadOpts opts;
opts.affinity = CpuAffinity{std::span<const unsigned>(service_cpus)};
opts.config_policy = ThreadConfigPolicy::kRequired;
HYLE_ASSIGN_OR_RETURN(auto thread, Thread::spawn(&entry, ctx, opts));
```

Windows 的硬 affinity mask 一次只能属于一个 processor group，跨组集合返回
`InvalidArgument`；这比把 CPU Sets 的软选择冒充硬约束更可靠。跨组并行应由 sched/线程池
按 group 拆成多个 worker。`this_thread::set_affinity(cpu_or_set)` 提供同样的单核/集合形状。
“从集合中轮转选一个”“超订 worker 如何放置”属于 sched placement，不放进 platform。

## CPU、cache 与 NUMA

`cpu_features()` 回答“本进程现在能安全执行什么指令”；`cache_info()` 的 L1/L2/L3 是容量
提示，未知项为 0。`cpu_topology()` 是缓存的系统在线硬件视图：
`system_logical_cpus` 与 OS 明确报告的 P/E 总数，适合诊断和粗粒度硬件策略；它不回答容器
或当前进程能开多少 worker。

`available_cpu_count()` / `available_cpu_ids(dst, cap)` 才是进程视图：前者是创建 worker、
选择 shard 数的常用形状；后者返回完整可用数，最多写 `cap` 个可直接交给
`ThreadOpts::affinity` / `CpuAffinity` 的 id；`dst=nullptr` 可只查询数量。查询刻意不缓存，因为 Linux
affinity/cgroup/cpuset 和 Windows CPU Sets 可在进程运行时改变；Linux 具体读取调用线程的
mask，新线程会继承它。常见 Windows / Linux
拓扑不分配；时间是读取 OS CPU 记录必需的 O(N)，用于 runtime 初始化/重配，不应塞进每个
task 的热路径。

需要优先使用强核时，不要假设“大编号/小编号就是 P-core”。`available_cpu_info()` 返回当前
进程可用 CPU 的 `id / performance_rank / kind`：rank 越大表示 OS 在本机报告的能力档越强，
0 表示无法建立可信的 CPU-id 映射。常用软偏好直接用有序重载：

```cpp
HYLE_ASSIGN_OR_RETURN(auto count, available_cpu_count());
std::vector<unsigned> cpus(count);
HYLE_ASSIGN_OR_RETURN(auto ordered_count,
    available_cpu_ids(CpuOrder::kPerformanceFirst,
                      cpus.data(), cpus.size()));
// cpus 仍包含全部可用 CPU，只是已知强核排前；未知平台保持 OS 顺序。

SchedulerOptions scheduler_opts;
scheduler_opts.worker_cpu_order = CpuOrder::kPerformanceFirst;
scheduler_opts.worker_priority = ThreadPriority::kHigh; // macOS: QoS 软偏好
const unsigned workers = static_cast<unsigned>(std::min<std::size_t>(4, count));
Scheduler scheduler(workers, allocator, 1024, scheduler_opts);
```

若业务必须**只**用强核，读取 `CpuInfo::kind == CpuKind::kPerformance` 后自行过滤，再交给
`CpuAffinity`；若全部为 `kUnknown`，HYLE 不会把猜测冒充硬保证。Windows 使用
`EfficiencyClass`，Linux 只信内核 `cpu_capacity`。macOS 能报告 P/E 数量却没有可靠的
logical-id 硬绑映射，所以 per-CPU rank 明确 unknown；低延迟/强核软倾向应使用
`SchedulerOptions::worker_priority = ThreadPriority::kHigh/kHighest` 对应的系统 QoS，而不是
猜 CPU id。该 priority 选项同样是显式 opt-in；默认不改变 OS 调度优先级。

`cache_info()` 仍只是容量提示，不猜 cache sharing。NUMA/cache-aware stealing 属于 sched
策略；platform 只提供可组合的 CPU 事实、排序与 affinity 机制。

NUMA preferred 是软策略：分配成功而 placement 为 `Unavailable` 仍是一块可用内存。只有
bind/interleave 的硬策略失败才应让依赖精确放置的工作失败。

## 错误与所有权

资源构造/查询返回 `Result<T>`，资源变更返回 `Status`；成功态不分配，完整原生诊断只在
失败路构造。`exists/is_directory/is_symlink` 把不存在表达为成功的 false，但不会吞掉权限或
I/O 错误。platform 的公开可恢复失败全部直接携带共享 error，不存在另一套 TLS last-error。

所有 RAII owner 都是 move-only。借出的 `data()`、symbol pointer、raw handle 或 unparker
视图不得活过 owner；`release()` 明确把回收责任转给调用方。

性能证据和仍存在的平台差异见
[`benchmarks-platform.md`](../performance/benchmarks-platform.md)，不要把单个平台点值外推到
另外两个 OS。
