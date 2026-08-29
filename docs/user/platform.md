# HYLE platform：什么情况选择什么

> platform 当前只提供 C++23 source/static/shared 产品面；没有 C API 或跨语言调用层。
> 本页按实际资源生命周期说明选择、错误与平台差异。

platform 是 HYLE 的跨 OS 原语层。它服务两类调用者：HYLE 内部模块需要稳定、可组合的
线程/等待/页内存地基；库用户需要在 Windows、Linux、macOS 上表达同一资源合同。它不把
每个系统调用都公开，也不因为 benchmark 有一行就制造一个类型。先按资源和生命周期选
形状。

macOS 使用 platform component 时部署目标必须是 14.4 或更新版本；原因是地址等待/Parker 使用
系统从 14.4 才公开的 `os_sync_wait_on_address*`。顶层构建未指定时 HYLE 默认 14.4；作为
`add_subdirectory()` 子工程时由父工程明确设置，低于 14.4 会在 configure 阶段失败。

## 一页选择表

| 需求 | 选择 | 不要选 | 关键合同 |
|---|---|---|---|
| 一块立即可读写的页内存 | `VmRegion::allocate_rw` | `reserve` 后立刻 `commit` | 一次 OS mapping；RAII 释放 |
| 先占地址、以后分段提交 | `VmRegion::reserve` + `commit/decommit` | 普通 heap 分配 | 页粒度；地址空间与物理提交分离 |
| 有越界保护的 fiber/线程栈 | `VmRegion::reserve_guarded` | 手写 magic padding | 低一页 no-access；`usable()` 才可写 |
| 大块页内存偏好某 NUMA node | `reserve_numa` / `vm_bind_numa` | 把 soft preferred 当硬保证 | 检查 `NumaPlacement` |
| 连续地址按选定 NUMA nodes/chunk 条带 | `VmRegion::allocate_numa_interleaved` | 调用方逐页手写迁移 | 已提交；node id 可稀疏/可重复加权；整段 release，不 decommit |
| 普通小对象、容器 backing | mem 的 `Allocator` / `Region` | platform VM | platform 不替代 malloc |
| 顺序读写普通文件 | `FileHandle`；完整传输用 `read_exact/write_all` | `MappedFile` | move-only、单 cursor；显式 `sync` 买 durability，显式 close 可观察错误 |
| 原子创建临时文件 | `create_temp_file` | 先生成名字再 open | 返回已打开 handle，避免 TOCTOU |
| 只需低成本枚举目录名和类型 | `for_each_entry` | 每项都 `stat` | entry 类型不跟随 symlink；常见路径零额外 stat |
| 需要完整单项元数据 | `stat` | 把目录遍历的 entry type 当完整 stat | 跟随 symlink，并另报 path 自身是否 symlink |
| 大文件随机访问/共享映射 | `MappedFile::open`；已有 handle 用 `map(std::move(file))` | 小文件必然 mmap | RO/RW/COW；const bytes / checked writable bytes；range flush；Linux 小文件顺扫未必赢 read |
| 映射权重/数据集的显式 warm-up | `MappedFile::prefetch` / `advise` | `open()` 自动触碰全文件 | hint 不影响正确性；Windows 仅 will-need |
| 整段映射必须驻留以压低 fault 尾延迟 | `MappedFile::lock`，检查 Status | 把 prefetch 当驻留保证 | 默认不改进程限额；用后 unlock |
| loader 逐步确认要驻留的映射前缀 | `lock_prefix(target)` + `locked_size()` | 每次重新锁 whole mapping | 单调增长、页粒度；失败保留旧前缀 |
| 大映射只在 first fault 时锁页 | `LockMode::kOnFault` | 悄悄把 unsupported 当 eager | Linux-only；其它平台返回 `Unimplemented` |
| 查询 cwd/temp/home | `convenience.h` owning 重载；自管存储用 buffer-fill | 猜环境变量/固定 `/tmp` | owning 返回 `std::string`；core 返回完整长度 |
| 读取/修改进程环境 | owning `get_env_var(name)` 或 buffer-fill；`set/unset` | 把 unset 和空值都当 0 | unset=`NotFound`；写操作仅应在启动期做 |
| 一次性启动 OS 线程 | `Thread::spawn(callable, opts)` | sched task/fiber | callable 状态由线程拥有；join 回收；可选 required |
| 大量轻量任务 | sched | 每任务一个 `Thread` | platform 只提供线程地基 |
| token 不丢失的线程 park/unpark | `Parker` / `park_for(Duration)` | 裸 condition variable 拼状态 | unpark-before-park 可消费 token；一个 Parker 只有一个等待者 |
| 等待调用方已有的原子字 | `wait_on_address` | `Parker` 再包一层 | 允许伪唤醒，必须循环检查谓词 |
| 插件/可选系统库 | `DynLib` | 裸 `dlopen`/`LoadLibrary` 分叉 | `get` 会 pin；symbol 借用不活过 owner；显式 close 可观察错误 |
| 运行时 ISA 派发 | `cpu_features` / `supports` | 编译机宏代替运行机探测 | 只调用当前进程可执行的 ISA |
| 决定当前 worker 数/绑核 | owning `available_cpu_ids()` 或 buffer-fill core | `hardware_concurrency` / `cpu_topology` | 动态遵守 affinity、cgroup/cpuset、Windows CPU Sets；id 可稀疏 |
| cache 容量提示、P/E 总量 | `cache_info` / `cpu_topology` | 假定探测值永远存在 | 未知 cache 为 0；P/E 不做频率猜测 |

## C++ 的 owning 与 caller-owned 输出两层

platform 对变长查询提供两层，不让一种形状勉强服务所有用户：

- 日常控制面包含 `hyle/platform/convenience.h`，直接取得 owning
  `std::string` / `std::vector`，自动处理 count、分配、变化后的重试和截断；分配失败仍是
  `ResourceExhausted`，不会从这个边界抛出。
- allocator/arena、需要控制**结果存储**，或正在实现真实 binding wrapper 的代码包含
  `fs.h` / `cpu.h`，使用
  buffer-fill core。它不返回 owning STL 对象，但长路径、UTF 转码和 OS 查询仍可能申请有检查的
  临时 scratch，OOM 会成为 `ResourceExhausted`；buffer-fill 不代表实现绝无内部 allocation。

```cpp
#include "hyle/platform/convenience.h"

HYLE_ASSIGN_OR_RETURN(std::string cache_dir, temp_directory());
HYLE_ASSIGN_OR_RETURN(std::vector<unsigned> cpus,
                      available_cpu_ids(CpuOrder::kPerformanceFirst));
```

owning 查询是初始化/重配路径的易用入口，不应放进每 task 热路径；需要复用存储时直接用
buffer-fill core。`convenience.h` 是显式 opt-in，原有 platform 头不会因此拉入 owning STL
类型或改变核心调用路径。

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
HYLE_RETURN_IF_ERROR(file.write_all(data, size));
// 只有崩溃/断电后的持久性属于合同，才支付这次 OS flush。
HYLE_RETURN_IF_ERROR(file.sync());
HYLE_RETURN_IF_ERROR(file.close());
```

read 返回 0 才是 EOF；read/write 返回短计数仍是成功。已知必须完整的普通文件传输直接使用
`read_exact/write_all`：前者提前 EOF 返回 `DataLoss`，后者处理短写，两者都不分配。`seek` 返回
移动后的绝对位置，`position()` 查询当前共享 cursor；同一 handle 的并发 positional I/O 不在该
合同内，应使用 `io::File` 或独立 handle。close 不等于 durability，只有显式 `sync` 才请求落到
OS storage boundary。

`MappedFile` 适合映射本身带来价值的
场景，例如大模型权重、随机访问、共享页和 COW；不要假设 64 KiB 顺序全扫也会比缓冲 read
快。默认优先使用返回 `Result<size_t>` 的安全 read/write；只有 profile 已确认这层成为瓶颈，
才借 `FileHandle::get()` 调 native I/O。`get()` 不转移所有权，不得关闭借出的 raw handle；
需要转移时使用 `release()`。stat/size/权限/symlink 和所有 path 变更都返回 Result/Status：不存在、空文件/0000 权限、
visitor 提前停止不再共享一个 0。跨平台代码优先 `is_readonly/set_readonly`，不要把 Windows
合成的 POSIX mode 当真实 ACL。普通 C++ 代码可从 `convenience.h` 直接取得 owning
cwd/temp/home/env/symlink 字符串；要自管存储时使用 `Result<size_t>` buffer-fill，值是完整
长度，可先用 `dst=nullptr` 查询。环境变量未设置是 `NotFound`，已设置为空是成功的 0。
`temp_directory` 只发现 OS/环境配置的路径，不额外探测可写性；需要真正可用且无 TOCTOU
窗口的临时文件时直接调用 `create_temp_file`。

`stat(path)` 成功就表示对象存在，`FileInfo` 因此不再复制一份可能互相矛盾的 `exists`
状态；`type` 是跟随 symlink 后的 target 类型，而 `is_symlink` 描述 path 自身。批量目录
扫描则用 `for_each_entry(name, FileType)`：它返回 entry 自身类型（symlink 不跟随），优先使用
`dirent` / `FindFirstFile` 已有元数据，只有 POSIX 文件系统返回 `DT_UNKNOWN` 才补一次 `lstat`。
这是一条不制造 owning entry 的快速原语；普通路径不做逐 entry heap allocation，长 Windows
search pattern 可能申请一次 checked scratch。确实需要 size/mtime 时由调用方只对选中的 entry
调 `stat`，platform 不制造持有字符串和分配器策略的高层 directory object。
`modified_time` 使用共享的 `time::UnixTimestamp{seconds,nanoseconds}`，保留 1970 年前的
有符号值和文件系统亚秒精度；它不是会 wrap 的 `uint64_t mtime_ns`。需要 chrono/time_t/
timespec 互转时包含 `hyle/time/unix_timestamp.h`，checked 转换不会静默饱和为另一个时刻。

`copy_file` 的成功面是单个文件。POSIX 对成功打开的对象验证 regular，再按 EOF 拷贝（因此
`st_size==0` 的 procfs/sysfs virtual regular file 仍可得到内容）；若 non-regular path 在 open
阶段先失败，则保留该 native error，例如无 reader 的 destination FIFO 会报 `ENXIO`，而不是
虚构成统一的 `FailedPrecondition`。Windows 的 reparse follow/replace 语义直接沿用 `CopyFileW`。

`create_temp_file` 返回 `CreatedTempFile{file, path_size}`。handle 始终是权威资源且不会跨
process exec 继承；prefix 必须是单个 basename 片段（不能含路径分隔符），因此返回路径始终
直接位于 `dir` 下。合法 prefix 原样保留，不会在 Windows 转换失败或较长时静默替换。
`path_size` 遵循其它 buffer-fill API 的完整长度合同，因而调用方不会在 path 被截断时误拿
半条路径删除。显式 `dir` 不能为空；实现用 checked dynamic scratch 拼接路径，不再额外施加
Windows `MAX_PATH` 或 POSIX 1024-byte ceiling（文件系统自身的 component/path 上限仍生效）。
POSIX `read_symlink` 在 `readlink` 恰好填满 scratch 时会扩容重试；`n==capacity` 不会被误报成
完整 target。
Windows 创建 symlink 时必须编码 file/directory：已有 target 可用 `SymlinkKind::kAuto`；创建
dangling link 时显式传 `kFile` 或 `kDirectory`，才具有跨平台一致语义。

`remove_directory_recursive` 在调用方拥有、遍历期间不被别人改名或替换 entry 的目录树中不会
跟随 symlink/junction。它不是对敌对并发目录的 sandbox primitive：跨平台 path API 无法把逐项
检查和删除做成一个原子事务。删除临时树或 cache 前应先停止写入者/把树原子移到私有名字；面对
不可信方可并发修改的目录，使用 OS 专用的 handle-relative 安全删除方案。实现用 resumable
native iterator frame 做后序 DFS，不让目录深度增长进程调用栈：每层一个 OS iterator，所有层
共享一个可增长 path buffer，普通文件/symlink 立即删除且从不缓存同层兄弟。辅助内存因此是
`O(depth + max_path)`，与目录宽度无关；极深树若命中 frame 分配或进程 handle/fd 上限则明确返回
`ResourceExhausted`。

模型/数据集需要 warm-up 时对已经打开的 mapping 显式调用，而不是让所有 open 承担 page
fault：

```cpp
HYLE_ASSIGN_OR_RETURN(auto file,
                      open_file("weights.bin", kOpenRead));
// map() 消费 owner；成功和失败都不会把 handle 留给调用方。
HYLE_ASSIGN_OR_RETURN(auto weights,
                      MappedFile::map(std::move(file),
                                      MappedFile::Mode::kReadOnly));
HYLE_RETURN_IF_ERROR(weights.prefetch());
// 只有尾延迟确实要求常驻时才购买这一资源合同。
HYLE_RETURN_IF_ERROR(weights.lock());
for (std::byte value : weights.bytes()) { /* read-only model access */ }
HYLE_RETURN_IF_ERROR(weights.unlock());
```

`Advice` 是性能 hint：POSIX 支持 normal/random/sequential/will-need/dont-need；Windows 只有
will-need 可对等实现，其余返回 `Unimplemented`，调用方可选择忽略 hint 继续使用 mapping。
`lock` 则是明确请求，失败通常表示权限或进程 quota，不应静默当成功。打开的 FileHandle/
MappedFile 在三平台都不会阻止 rename/delete，因此可先原子替换路径，让旧 owner 自然读完。
`bytes()` 始终只读；只有 `kReadWrite`/`kCopyOnWrite` mapping 的 `writable_bytes()` 成功。
`flush(offset,length)` 只对 read-write shared mapping 请求回写，`length==0` 表示直到末尾；COW/RO
没有需要落回原文件的 dirty data。offset/length 越界会显式失败，不会把错误 range 对齐成别处。

模型 loader 往往只在解析后才知道哪些权重留在 CPU。不要为了这个生命周期反复 whole-lock；按
已完成的 public byte prefix 增长：

```cpp
using LockMode = MappedFile::LockMode;
using LockLimitPolicy = MappedFile::LockLimitPolicy;

MappedFile::LockOptions residency;
residency.mode = LockMode::kEager;
// 只有应用明确接受进程级 soft limit / working-set 改变时才打开。
residency.limit_policy = LockLimitPolicy::kAdjustIfNeeded;

for (std::size_t loaded_end : cpu_weight_prefixes) {
  Status status = weights.lock_prefix(loaded_end, residency);
  if (!status.ok()) {
    // weights.locked_size() 仍是此前成功的前缀；应用可记录并降级。
    report_residency_fallback(status, weights.locked_size());
    break;
  }
}
HYLE_RETURN_IF_ERROR(weights.unlock());
```

`locked_size()` 是 OS page 向前取整后的实际覆盖，可能超过最后一个 target；它不是“已经读取了
多少模型数据”。锁定模式在非空前缀期间不能切换，先 `unlock()` 才能用另一模式重建合同。
Linux 上 `kOnFault` 避免调用本身同步 fault 全段；Windows/macOS 明确返回 `Unimplemented`。
`kAdjustIfNeeded` 只在 quota failure 后查询并尝试提升一次既有进程 soft limit，然后重试一次；
它不能越过 POSIX hard limit 或系统权限，且成功调整不会被 `unlock()`/析构恢复。默认
`kPreserve` 没有这项全局副作用。

释放 offload 后不用的物理页可组合 `advise(Advice::kDontNeed, offset, length)`；它是 hint，Windows
没有对等实现。`MappedFile` 不提供 partial unmap，因为打洞后 `bytes()` 将不再是有效连续 view，
而 Windows 也不能提供同一合同。需要真正缩小虚拟映射时，在 loader 设计阶段建立精确 subrange
或多个独立 mapping owner。

## 等待与线程

`Parker` 是一个 token mailbox：`unpark()` 先发生也不会丢，下一次 `park()` 会消费 token。
它不是计数 semaphore，多次 unpark 可以合并成一个 token。`wait_on_address` 更底层，适合
对象已经拥有原子状态字的内部同步原语；伪唤醒允许发生，因此始终循环检查业务谓词。
`park_for(time::Duration)` 是单次等待；非正 duration 只是无 syscall 的 token poll，正值返回前
仍可能因 deadline 或竞态醒来。业务 timeout 应在 deadline loop 中重查谓词和剩余时间。不要让
两个线程同时在同一个 Parker 上等待；需要多 waiter/取消/队列公平性时使用 sched 同步类型。

`Thread` 适合少量长寿命 OS 线程、runtime worker 或必须设置 stack/name/affinity 的边界；
任务级并发用 sched。`ThreadOpts` 默认 `kBestEffort`，适合“能设则设、失败仍可运行”的
可移植 runtime。普通 C++ 调用直接传 capturing/move-only lambda；线程拥有 callable，
spawn/config 失败时销毁而不调用。未捕获异常与 `std::thread` 相同会 fail-fast。内部 runtime、
或需要自己管理 context 的 C++ 代码仍可使用零额外 callable 分配的 `entry + void*` core；它不是
跨语言 ABI。
确实依赖设置（隔离核、基准控制线程、实时服务线程）时改成 required：

```cpp
unsigned cpu = 0;
HYLE_ASSIGN_OR_RETURN(auto count, available_cpu_ids(&cpu, 1));
if (count == 0) return Status::FailedPrecondition("no available CPU");

ThreadOpts opts;
opts.name = "ingest";
opts.affinity = cpu;  // 单 worker 固定归属：单 CPU 便利形状
opts.config_policy = ThreadConfigPolicy::kRequired;
HYLE_ASSIGN_OR_RETURN(auto thread, Thread::spawn([state = std::move(state)] {
  run_worker(*state);
}, opts));
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

`available_cpu_count()` / `available_cpu_ids(...)` 才是进程视图：前者是创建 worker、
选择 shard 数的常用形状；后者返回完整可用数，最多写 `cap` 个可直接交给
`ThreadOpts::affinity` / `CpuAffinity` 的 id；`dst=nullptr` 可只查询数量。查询刻意不缓存，因为 Linux
affinity/cgroup/cpuset 和 Windows CPU Sets 可在进程运行时改变；Linux 具体读取调用线程的
mask，新线程会继承它。常见 Windows / Linux
拓扑不分配；时间是读取 OS CPU 记录必需的 O(N)，用于 runtime 初始化/重配，不应塞进每个
task 的热路径。

需要优先使用强核时，不要假设“大编号/小编号就是 P-core”。`available_cpu_info()` 返回当前
进程可用 CPU 的 `id / performance_rank / kind`：rank 越大表示 OS 在本机报告的能力档越强，
0 表示无法建立可信的 CPU-id 映射。普通 C++ 初始化代码用 `convenience.h` 的 owning 有序
重载；复用调用方数组时使用三参数 buffer-fill 重载：

```cpp
HYLE_ASSIGN_OR_RETURN(
    std::vector<unsigned> cpus,
    available_cpu_ids(CpuOrder::kPerformanceFirst));
// cpus 仍包含全部可用 CPU，只是已知强核排前；未知平台保持 OS 顺序。

SchedulerOptions scheduler_opts;
scheduler_opts.worker_cpu_order = CpuOrder::kPerformanceFirst;
scheduler_opts.worker_priority = ThreadPriority::kHigh; // macOS: QoS 软偏好
const unsigned workers = static_cast<unsigned>(std::min<std::size_t>(4, cpus.size()));
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

`numa_current_node()` 与 `numa_node_of_cpu(cpu)` 返回 `Result<unsigned>`。查询失败、无效 CPU、
或 OS 不提供可信映射都不是节点 0；必须处理错误后再决定是否显式退化到 `kNumaAny`。如果 worker
已 required-pin 到某 CPU，先用 `numa_node_of_cpu` 选 arena node；没有硬 affinity 时 current-node
只是瞬时事实，不应缓存成线程永久归属。

NUMA preferred 是软策略：分配成功而 placement 为 `Unavailable` 仍是一块可用内存。Linux
bind 是严格节点集合；interleave/selected interleave 的 `Applied` 只说明 OS 接受 policy，
Windows 和 Linux 都可能在目标节点内存压力下回退，不能把它当成逐页 residency 证明。

先用 `numa_nodes()` 取得真实可用 id；它们可能稀疏，因此不要写
`for (unsigned n=0; n<numa_node_count(); ++n)`。`numa_capabilities()` 适合启动期预检或 UI
报告，实际操作返回的 `Status` 仍是最终结果。`HYLE_OS_*` 是 HYLE 实现层分派宏，不是用户
选择 NUMA 路径的公共合同：

```cpp
using namespace hyle::platform;

constexpr std::size_t kMiB = 1024u * 1024u;
const auto available = numa_nodes();                 // stable non-owning view
const auto capabilities = numa_capabilities();
if (available.size() > 1 &&
    has_numa_capability(capabilities,
                        NumaCapability::kSelectedInterleave)) {
  const unsigned chosen[] = {available.front(), available.back()};
  HYLE_ASSIGN_OR_RETURN(
      VmRegion weights,
      VmRegion::allocate_numa_interleaved(64u * kMiB, chosen, 2u * kMiB));
  // weights 已提交可写；析构/整段 release。不要 decommit 后再 commit。
}
```

空 node span 选择所有可用节点；显式列表必须是可用的实际 id，条带 policy 按输入顺序；重复 id
表达权重，例如 `{n0,n0,n1}` 每轮给 n0 两个 chunk、n1 一个。`bytes_per_chunk==0` 表示一页，否则
必须是页大小的整数倍。单节点机器返回正常 allocation 和 `NumaPlacement::kNotNeeded`。Linux 的
普通布局在 first-touch 前惰性设置 policy；极细自定义布局为避免每页一个 VMA policy，可能在
allocate 内完成 first-touch，因此大分配不要把“未传 prefault flag”当作惰性承诺。Windows 仅在
`VirtualAllocExNuma` 可用时报告 selected-interleave capability，并在一个普通连续 reserve 中按条带
preferred-node NUMA commit；只有 `VirtualAlloc2` 的环境仍可做 preferred reservation，但 selected
allocation 会返回 `Unimplemented`，不会用多 owner placeholder 条带伪装成可一次 release 的 span。
macOS 当前只有逻辑 node 0。policy 成功不是物理页
永久驻留证明；需要托管生命周期时优先 `VmRegion`；裸
`vm_allocate_numa_interleaved` 返回 page-rounded size，只与整段 `vm_release` 配对。selected
region 的 `decommit` 会以 `FailedPrecondition` 拒绝，防止 recommit 静默丢掉 placement。

## 错误与所有权

资源构造/查询返回 `Result<T>`，资源变更返回 `Status`；core 成功态不隐藏分配，owning
convenience 的分配是调用形状本身的显式选择，完整原生诊断只在失败路构造。
`exists/is_directory/is_symlink` 把不存在表达为成功的 false，但不会吞掉权限或
I/O 错误。platform 的公开可恢复失败全部直接携带共享 error，不存在另一套 TLS last-error。

所有 RAII owner 都是 move-only。借出的 `data()`、symbol pointer、raw handle 或 unparker
视图不得活过 owner；`release()` 明确把回收责任转给调用方。

`DynLib::open(path)` 加载新引用；`get(path)` 只附着到已经加载的 image，但成功后同样持有/pin
一份 loader reference，所以不会与另一个 owner 的 unload 竞争。`symbol_as<Pointer>` 只接受指针
类型，函数签名仍由调用方负责；指针在 `close`/析构后失效。相对路径和搜索名属于进程加载信任
边界，外部输入不得未经验证直接传入。需要观察 unload 诊断时显式检查 `close()`，析构只做兜底。

性能证据和仍存在的平台差异见
[`benchmarks-platform.md`](../implementation/evidence/performance/benchmarks-platform.md)，不要把单个平台点值外推到
另外两个 OS。
