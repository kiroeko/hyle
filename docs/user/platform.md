# HYLE platform：什么情况选择什么

> platform 当前只提供 C++23 source/static/shared 产品面；没有 C API 或跨语言调用层。
> 本页按实际资源生命周期说明选择、错误与平台差异。

platform 是 HYLE 的跨 OS 原语产品。它服务两类调用者：HYLE 内部模块需要稳定、可组合的
host、线程、等待和文件机制；库用户需要在 Windows、Linux、macOS 上表达同一资源合同。
它不把每个系统调用都公开，也不因为 benchmark 有一行就制造一个类型。先按资源和生命周期
选择形状。

用户只需选择 Platform 产品；构建系统内部把它分成 host、thread、filesystem 三个可独立构建的
质因数，使只需要 CPU facts 的模块不会被迫链接线程和文件代码。匿名地址空间仍由 Platform 的
`VmRegion` 拥有；模型权重、共享区和其它 storage mapping 则只有 Mem 的 `VirtualMapping`、
`FileMapping`、`SharedMapping` 一个公开 owner。两层通过低层机制组合，不复制 allocation 或
payload。

macOS 使用 platform component 时部署目标必须是 14.4 或更新版本；原因是地址等待/Parker 使用
系统从 14.4 才公开的 `os_sync_wait_on_address*`。顶层构建未指定时 HYLE 默认 14.4；作为
`add_subdirectory()` 子工程时由父工程明确设置，低于 14.4 会在 configure 阶段失败。

## 一页选择表

| 需求 | 选择 | 不要选 | 关键合同 |
|---|---|---|---|
| 自建低层 owner 的立即可写页内存 | `VmRegion::allocate_rw` | `reserve` 后立刻 `commit` | 一次 OS mapping；统一 storage view 则选 Mem `VirtualMapping` |
| 先占地址、以后分段提交 | `VmRegion::reserve` + `commit/decommit` | 普通 heap 分配 | 页粒度；地址空间与物理提交分离 |
| 有越界保护的 fiber/线程栈 | `VmRegion::reserve_guarded` | 手写 magic padding | 低一页 no-access；`usable()` 才可写 |
| 大块页内存偏好某 NUMA node | `reserve_numa` / `vm_bind_numa` | 把 soft preferred 当硬保证 | 检查 `NumaPlacement` |
| 连续地址按选定 NUMA nodes/chunk 条带 | `VmRegion::allocate_numa_interleaved` | 调用方逐页手写迁移 | 已提交；node id 可稀疏/可重复加权；整段 release，不 decommit |
| 普通小对象、容器 backing | mem 的 `AllocatorRef` / `Arena` | platform VM | platform 不替代 malloc |
| 顺序读写普通文件 | `FileHandle`；完整传输用 `read_exact/write_all` | 默认直接 mapping | move-only、单 cursor；显式 `sync` 买 durability，显式 close 可观察错误 |
| 同步读取固定 offset 的模型/header/index | `RandomAccessFile`；完整传输用 `read_exact_at/write_all_at` | `seek+read`、默认上 coroutine IO | blocking、caller-synchronized、无公开 cursor；append 被拒绝；并发/async/direct/vectored 才选 IO |
| 保留同一 opened object 的第二个 owner | `FileHandle::duplicate()` / `RandomAccessFile::duplicate()` | 再按 path open | 生命周期独立，但 duplicate aliases 共享 native cursor/lock state；不是独立 position |
| 查询/改变已打开文件的 identity/size | `FileHandle::info/size/resize` | 对原 path 再 `stat/resize` | 不二次解析名字；rename/unlink 后仍指原对象；扩容可能 sparse |
| 写大模型/cache 前保证增长空间 | `FileHandle::preallocate(minimum)` | 用 `resize` 冒充物理预留 | 小于 minimum 才预留并增长；不缩小；可能部分成功，不承诺连续 extent |
| cooperating processes 协调整文件 writer | `lock/try_lock/unlock` | 线程 mutex、lock-file 名字或分布式锁 | shared/exclusive；portable 承诺是 advisory coordination；close 释放 |
| 原子创建临时文件 | `create_temp_file` | 先生成名字再 open | 返回已打开 handle，避免 TOCTOU |
| 只需低成本枚举目录名和类型 | `for_each_entry` | 每项都 `stat` | entry 类型不跟随 symlink；常见路径零额外 stat |
| 需要完整单项元数据 | `stat` | 把目录遍历的 entry type 当完整 stat | 跟随 symlink，并另报 path 自身是否 symlink |
| 大文件共享页/COW/零拷贝 view 或已有 handle 映射 | Mem `FileMapping::open/map` | 把一次 fixed-offset read 当成必须 mmap | RO/RW/COW；`map` 消费 `FileHandle` owner；显式 offset blocking I/O 选 `RandomAccessFile` |
| 映射权重/数据集的异步提示或同步 warm-up | Mem `FileMapping::prefetch` / `populate` | `open()` 自动触碰全文件 | hint 不保证 page-in；`populate` 同步触页且与范围成正比 |
| 整段映射必须驻留以压低 fault 尾延迟 | Mem `FileMapping::lock`，检查 `Status` | 把 prefetch 当驻留保证 | 默认不改进程限额；用后 unlock |
| loader 逐步确认要驻留的映射前缀 | `lock_prefix(target)` + `locked_size()` | 每次重新锁 whole mapping | 单调增长、页粒度；失败保留旧前缀 |
| 大映射只在 first fault 时锁页 | Mem `ResidencyMode::kOnFault` | 悄悄把 unsupported 当 eager | Linux-only；其它平台返回 `Unimplemented` |
| 查询 cwd/temp/home | `fs_convenience.h` owning 重载；自管存储用 buffer-fill | 猜环境变量/固定 `/tmp` | owning 返回 `std::string`；core 返回完整长度 |
| 读取/修改进程环境 | owning `get_env_var(name)` 或 buffer-fill；`set/unset` | 把 unset 和空值都当 0 | unset=`NotFound`；写操作仅应在启动期做 |
| 推理装载前估算 host budget | `host_memory_info()` | 只看物理总量 | 返回 fallible 动态快照；Linux 合并 namespace 内全部祖先 hard cgroup limit；`effective_available_bytes` 保守合并已知约束 |
| 密钥、nonce 或随机种子 | `secure_random(span<byte>)` | 统计 RNG / 时间戳 | 使用 OS entropy；完整填满或返回错误；空 span 成功 |
| 一次性启动 OS 线程 | `Thread::spawn(callable, opts)` | sched task/fiber | callable 状态由线程拥有；join 回收；可选 required |
| 为整个应用选择启动期节能/性能倾向 | workers 创建前 `set_process_priority` | realtime 或拿 thread priority 冒充 process policy | 显式五档；提升可能失败；Linux 靠后续线程继承；normal 不是任意 native snapshot restore |
| 大量轻量任务 | sched | 每任务一个 `Thread` | platform 只提供线程地基 |
| token 不丢失的线程 park/unpark | `Parker` / `park_for(Duration)` | 裸 condition variable 拼状态 | unpark-before-park 可消费 token；一个 Parker 只有一个等待者 |
| 等待调用方已有的原子字 | `wait_on_address` | `Parker` 再包一层 | 允许伪唤醒，必须循环检查谓词 |
| 插件/可选系统库 | `DynLib` | 裸 `dlopen`/`LoadLibrary` 分叉 | `get` 会 pin；symbol 借用不活过 owner；显式 close 可观察错误 |
| 自定义 kernel 的运行时 feature gate | `cpu_features().has_all(requirements)` | 把 Platform facts 当“最佳 ISA”，或只检查一个营销级 feature | kernel owner 声明其生成 TU 的完整 requirements；Platform 不替它选择 target |
| 决定 work 可放在哪些 CPU/绑核 | owning `available_cpu_ids()` 或 buffer-fill core | 只看 `hardware_concurrency()` | 动态遵守 affinity、cgroup/cpuset、Windows CPU Sets；id 可稀疏；只表达 placement |
| 选择 CPU-bound 默认 worker 上限 | `process_cpu_capacity()` + `parallelism_ceiling()` | 把 placement 数直接当获准 CPU 带宽 | 独立保留 placement 和可选 hard quota；weight/share 不伪造成上限；最终策略仍由调用方决定 |
| 按 package/core/SMT/NUMA/cache domain 放置 worker | `available_cpu_topology()` | 假定 CPU id 连续或 cache 独占 | 返回当前进程可用 CPU 的动态快照；未知 domain 使用 sentinel |
| 按异构 CPU class 选择 tile/working set | `available_cpu_cache_classes()` | 在 Apple 上把 `cache_info()` 当 P-core cache | 动态 class snapshot；强 class 排前；同 rank 的不同 cache profile 不合并；未知为 0 |
| 诊断/调优 OS 公布的逐 CPU 频率 | owning `available_cpu_frequencies()` 或 buffer-fill core | 用频率猜 P/E、可运行 ISA 或未来吞吐 | Hz；hardware/policy/current 分开；0=unknown；current 只是瞬时报告；不按型号或计时猜测 |
| 保存并恢复调用线程 affinity | `this_thread::affinity()` + `set_affinity()` | 猜测原 mask | buffer-fill/owning 两层；macOS 无硬 pin 时明确不支持 |
| 查询线程可用的 SVE/SME/RVV width 或 AMX 状态 | `this_thread::isa_state()` | 只看系统级 feature bit | system capability 与 per-thread enabled state 分开 |
| 只需保守的整机 cache 容量提示 | `cache_info()` | 用它做异构 class tuning | cached/O(1)，未知为 0；Apple 返回最低性能 class 的全局值 |

## C++ 的 owning 与 caller-owned 输出两层

platform 对变长查询提供两层，不让一种形状勉强服务所有用户：

- 日常控制面按能力包含 `cpu_convenience.h`、`thread_convenience.h`、
  `fs_convenience.h` 或 `environment_convenience.h`，直接取得 owning `std::vector` /
  `std::string`。这些 header-only adapter 自动处理 count、分配、动态变化后的重试和截断；
  分配失败仍是 `ResourceExhausted`，不会从这个边界抛出。
- allocator/arena、需要控制**结果存储**，或正在实现真实 binding wrapper 的代码包含
  `fs.h` / `cpu.h` / `thread.h` / `environment.h`，使用
  buffer-fill core。它不返回 owning STL 对象，但长路径、UTF 转码和 OS 查询仍可能申请有检查的
  临时 scratch，OOM 会成为 `ResourceExhausted`；buffer-fill 不代表实现绝无内部 allocation。

```cpp
#include "hyle/platform/cpu_convenience.h"
#include "hyle/platform/fs_convenience.h"

HYLE_ASSIGN_OR_RETURN(std::string cache_dir, temp_directory());
HYLE_ASSIGN_OR_RETURN(std::vector<unsigned> cpus,
                      available_cpu_ids(CpuOrder::kPerformanceFirst));
HYLE_ASSIGN_OR_RETURN(std::vector<CpuCacheClassInfo> cache_classes,
                      available_cpu_cache_classes());
```

owning 查询是初始化/重配路径的易用入口，不应放进每 task 热路径；需要复用存储时直接用
buffer-fill core。各 convenience header 都是显式 opt-in；核心 header 不会因此拉入 owning STL
类型或改变调用路径，也不会因为需要一种便利查询而引入另一个 factor。

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
移动后的绝对位置，`position()` 查询当前共享 cursor。只需要 caller-synchronized、blocking 的显式
offset 读取（例如读取模型 header、tensor shard 或索引块）时，直接选择 `RandomAccessFile`，不要把
`seek+read` 拼成会受共享 cursor 干扰的序列：

```cpp
HYLE_ASSIGN_OR_RETURN(
    auto weights,
    hyle::platform::open_random_access_file(
        "model.bin", hyle::platform::kOpenRead));

std::byte header[4096]{};
HYLE_RETURN_IF_ERROR(weights.read_exact_at(0, header));
constexpr std::uint64_t tensor_offset = 64u * 1024u;
std::byte tensor_chunk[4096]{};
HYLE_ASSIGN_OR_RETURN(std::size_t got,
                      weights.read_at(tensor_offset, tensor_chunk));
// got 可能短于 buffer；0 是 EOF。需要完整 tensor 时用 read_exact_at。
```

`RandomAccessFile` 不公开 cursor，`read_at/write_at` 保留短传输，exact/all 版本检查 offset overflow；
`kOpenAppend` 会在 open 时返回 `InvalidArgument`。一个 owner 及其 duplicate aliases 仍由 caller
同步；需要多个线程同时提交、取消、direct/vectored I/O 或 completion 时使用 `io::File`。Windows
原生 offset call 可能移动其内部 file pointer，但这个不可移植状态不会通过该类型暴露。close 不等于
durability，只有显式 `sync` 才请求落到 OS storage boundary。

同一已打开对象的控制不要退回 path API：名字可能已经被另一个进程 rename/replace，`file.info()`、
`file.size()` 和 `file.resize(n)` 仍操作 caller 真正拥有的文件。`info()` 返回与 path `stat` 同值域的
type/size/mtime，但 `is_symlink=false`，因为 handle 已经指向解析后的对象；unlink 后查询仍有效。
需要把同一 opened object 的生命周期交给另一个 owner 时调用 `duplicate()`，不要按旧 path 再 open：

```cpp
HYLE_ASSIGN_OR_RETURN(auto mapping_owner, file.duplicate());
HYLE_ASSIGN_OR_RETURN(auto opened_identity, mapping_owner.info());
```

两个 owner 生命周期独立，但不是两个独立 file positions：POSIX 和 Windows duplicate 都共享 native
open-file cursor，并可能共享/关联锁状态。若 worker 需要互不影响的 cursor，应分别重新 open；若需要
explicit offset，则优先 `RandomAccessFile`。扩大的 `resize` 只承诺 logical EOF，可能产生
sparse holes；真正需要在写入前请求磁盘空间时用 `preallocate(minimum)`：

```cpp
HYLE_ASSIGN_OR_RETURN(auto current, file.size());
if (current < model_bytes) {
  HYLE_RETURN_IF_ERROR(file.preallocate(model_bytes));
}
```

`preallocate` 不缩小较大文件，也不把既有 sparse 文件重新物理化。allocation 与 EOF update 不是
跨平台事务；后一步失败时可能已有空间被预留，所以错误后应重新查询文件并按应用协议决定 cleanup。
不要在仍有 Mem `FileMapping` view 时 resize 同一文件：截短会使已有 view 的访问语义依 OS 而异，
应先结束 mapping 生命周期。`resize/preallocate` 需要 writable handle；权限或 filesystem 不支持时
返回共享 `Status`。

多进程 cache/index writer 只需合作式整文件协调时，可锁住同一个 `FileHandle` owner：

```cpp
#include <scope>

HYLE_RETURN_IF_ERROR(file.lock());  // 默认 exclusive
std::scope_exit unlock{[&]() noexcept { (void)file.unlock(); }};
HYLE_RETURN_IF_ERROR(update_cache(file));
```

shared readers 使用 `FileLockMode::kShared`；`try_lock` 的值 `false` 只表示明确竞争，权限/I/O/closed
handle 仍是错误。portable 合同是 advisory cooperation：POSIX 进程若完全忽略协议仍可 I/O，Windows
对冲突 I/O 更强，network filesystem 还有自身限制。它不是 in-process mutex、range lock 或可靠的
distributed lock。同一个 handle 上的并发调用仍需 caller 同步；close/析构会让 OS 释放锁。
duplicate aliases 不能作为两个可移植的独立 lock contender；要测试进程/owner 间竞争，应独立 open。

不要假设大文件必然适合 mapping：普通顺序 I/O 用 `FileHandle`，blocking explicit-offset I/O 用
`RandomAccessFile`，需要共享页、COW 或长期 zero-copy view 时才改用 Mem 的 `FileMapping`。默认优先
使用返回 `Result<size_t>` 的安全 read/write；只有
profile 已确认这层成为瓶颈，才借 `FileHandle::get()` 调 native I/O。`get()` 不转移所有权，不得关闭借出的 raw handle；
需要转移时使用 `release()`。stat/size/权限/symlink 和所有 path 变更都返回 Result/Status：不存在、空文件/0000 权限、
visitor 提前停止不再共享一个 0。跨平台代码优先 `is_readonly/set_readonly`，不要把 Windows
合成的 POSIX mode 当真实 ACL。普通 C++ 代码可从 `fs_convenience.h` 直接取得 owning
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
调 `stat`，platform 不制造持有字符串和分配器策略的高层 directory object。`for_each_entry`
接受 const、mutable、返回 `void` 或 `bool` 的 callable；`bool=false` 是正常提前结束。异常启用时，
callable 抛出的异常会在 native iterator 完成关闭后原样重抛。只有实现 allocation-minimal adapter
时才直接调用 `directory_iterate`；它是 `noexcept` core，传入的 function-pointer callback 不得抛。
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
`make_directory_recursive` 同样先用 4096-byte 栈 scratch，只有更长输入才做一次 checked allocation；
因此 Windows extended path 不会被 HYLE 自己截在 4095 bytes，POSIX 仍以实际内核路径上限为准。
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

文件和 shared-memory mapping 的完整选择、warm-up、residency、flush 与陷阱统一见
[Mem 使用指南](mem.md)。Platform 不再提供第二套 mapping owner；`FileMapping::map` 直接消费这里的
`FileHandle`，所以调用方仍可避免第二次 path lookup，并且不会增加 payload copy。

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

应用想让随后创建的全部 workers 继承统一节能/性能倾向时，在任何 worker 创建前显式设置进程
启动策略：

```cpp
#include "hyle/platform/process.h"

HYLE_RETURN_IF_ERROR(
    hyle::platform::set_process_priority(
        hyle::platform::ProcessPriority::kHigh));
```

Windows 会分别设置 priority class 和 execution QoS，二者没有事务 rollback；Linux 修改调用线程
nice，效果通过随后创建的线程继承；macOS 使用 process nice。提高优先级通常需要权限。HYLE 不提供
realtime 档，也不会在初始化时自动改变宿主。`kNormal` 返回普通/system-managed 档，不是恢复任意
调用前 native state；需要精确保存宿主专有策略的应用应在自身 OS adapter 中完成。已经存在的 worker
仍用 `this_thread::set_priority` 或 `ThreadOpts::priority` 逐线程管理。

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
三者返回共享 `Status`，不另造 errno/TLS 通道。Linux priority 提升通常需要权限；Windows 和 macOS
priority 也映射 execution QoS：低两档请求节能执行，高两档请求 performance-oriented 执行，
`kNormal` 恢复系统管理。只有 optional priority 未设置才完全继承 embedding process 的现有策略；
Windows QoS 是软调度提示，不等于硬选 P-core。macOS 没有公开的硬 CPU pin API，所以 required affinity 返回
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

临时 pin 做校准、NUMA first-touch 或性能测量时，先保存调用线程自己的 mask，再原样恢复；不要
用整个进程的 available set 猜原状态：

```cpp
#include "hyle/platform/thread_convenience.h"

HYLE_ASSIGN_OR_RETURN(auto previous, hyle::platform::this_thread::affinity());
HYLE_RETURN_IF_ERROR(hyle::platform::this_thread::set_affinity(worker_cpu));
HYLE_RETURN_IF_ERROR(run_calibration());
HYLE_RETURN_IF_ERROR(hyle::platform::this_thread::set_affinity(previous));
```

`this_thread::isa_state()` 查询当前线程的动态执行状态，而非第二套 CPU feature detector：
SVE/RVV 的当前 vector length、SME streaming vector length，以及 AMX tile 是否已获许可。Linux AMX
需要每个使用线程显式 `request_amx_tile_permission()`；请求失败不得进入 AMX kernel。这个机制为
SIMD、自定义 kernel 与诊断代码提供线程事实，不选择 kernel，也不会替调用方进入或退出 SME streaming
mode。

## CPU、cache 与 NUMA

`cpu_features()` 返回无顺序的 `CpuFeatureSet`，回答“这个进程的 CPU/OS 能安全执行哪些具名
扩展”。它不是最高档位，也不能跨架构比较。已有 HYLE SIMD 算法应直接调用 SIMD API；只有自建
kernel/backend 时，才由该实现的 owner 写出其生成 TU **全部**允许使用的 feature：

```cpp
#include "hyle/platform/cpu.h"

using hyle::platform::CpuFeature;
using hyle::platform::CpuFeatureSet;

// 这份 body 使用 hyle_target_simd(... ISA AVX2)，因此 requirements
// 与 HYLE canonical direct-Highway AVX2 的编译 flags 完全一致。
constexpr CpuFeatureSet kHyleAvx2Body{
    CpuFeature::kX86Sse2, CpuFeature::kX86Sse3,
    CpuFeature::kX86Ssse3, CpuFeature::kX86Sse41,
    CpuFeature::kX86Sse42, CpuFeature::kX86Avx,
    CpuFeature::kX86Avx2, CpuFeature::kX86Fma,
    CpuFeature::kX86F16c, CpuFeature::kX86Lzcnt,
    CpuFeature::kX86Bmi1, CpuFeature::kX86Bmi2};

if (hyle::platform::cpu_features().has_all(kHyleAvx2Body)) {
  run_my_avx2_body();
} else {
  run_my_portable_body();
}
```

不要只检查营销名称 `kX86Avx2`，也不要机械复制上例：requirements 必须从**这份 TU 实际使用的编译
flags/provider 合同**导出。`hyle_target_simd(... ISA AVX2)` 同时允许 FMA、F16C、BMI 等指令，编译器
可在整份 TU 中使用它们，所以示例逐项检查；只用另一组 flags 的实现应声明它自己的集合。
`CpuFeatureSet::has/has_all/has_any/set` 都是固定两个机器字上的 O(1) 操作，不分配；
`cpu_feature_name()` 只用于诊断。AMX 的 `*Hardware` facts 仅说明 CPUID hardware capability；Linux
线程进入 AMX kernel 前还必须通过 `this_thread::request_amx_tile_permission()` 获得本线程许可。

`cache_info()` 的 L1/L2/L3 是 cached、O(1) 的保守容量提示，未知项为 0。异构机器不能把它当成
每类 CPU 的 cache：Apple 的全局 `hw.l*cache*` 按系统合同表示最低性能 class。需要为 P/E 核选择
tile、working set 或 shard 时，在初始化/重配点取得动态 class snapshot：

```cpp
HYLE_ASSIGN_OR_RETURN(auto classes, available_cpu_cache_classes());
for (const CpuCacheClassInfo& cpu_class : classes) {
  // 已知 performance_rank 从强到弱；0 表示未知。
  // cpu_class.cache.l2、logical_cpus_per_l2 等未知字段同样为 0。
  configure_tiling(cpu_class.performance_rank,
                   cpu_class.cache.l2,
                   cpu_class.logical_cpus_per_l2);
}
```

同 rank 但容量或 sharing width 不同的 profile 会保留为不同记录。`logical_cpus/physical_cores` 是
当前 process view 中该 class 的数量；`logical_cpus_per_l2/l3` 是 native sharing domain 宽度，可能
包含 process affinity 之外的 CPU。macOS 能给出准确 class profile，却仍没有公开、可信的 per-logical-ID
硬绑映射；因此 class tuning 与 worker pin 是两个问题。该查询涉及 OS topology/cache enumeration，
不应进入每 task 热路径，也不选择 SIMD/provider winner。

`hardware_concurrency()` 是 OS-online 的粗略诊断值，不遵守当前 affinity/cgroup，因此不要拿它决定
worker 数。

`available_cpu_count()` / `available_cpu_ids(...)` 是当前 **placement** 视图：前者是 count-only
便利形状；后者返回完整可用数，最多写 `cap` 个可直接交给
`ThreadOpts::affinity` / `CpuAffinity` 的 id；`dst=nullptr` 可只查询数量。查询刻意不缓存，因为 Linux
affinity/cgroup/cpuset 和 Windows CPU Sets 可在进程运行时改变；Linux 具体读取调用线程的
mask，新线程会继承它。常见 Windows / Linux
拓扑不分配；时间是读取 OS CPU 记录必需的 O(N)，用于 runtime 初始化/重配，不应塞进每个
task 的热路径。

placement 数不等于进程获准的 CPU 时间。容器可能允许在 32 个 logical CPU 上迁移，却通过 cgroup
只给 1.5 CPU 的持续带宽。CPU-bound runtime 的默认上限应读取两者，但不能把这个上限误当硬绑核策略：

```cpp
HYLE_ASSIGN_OR_RETURN(auto capacity, process_cpu_capacity());
const std::size_t default_workers = capacity.parallelism_ceiling();
// capacity.placement_logical_cpus: work 能放置到多少 CPU；
// capacity.hard_quota: OS 有可信 hard bandwidth limit 时才有值。
```

Linux 会解析当前 cgroup namespace 的 mountinfo，折叠 leaf 到可见 mount root 的全部 v1/v2 hard
quota。Windows 的公开 API 只能查询 immediate Job，而嵌套 Job 的 rate 相对父级；没有祖先 Job handles
就无法证明进程的有效 hard quota，所以 Windows 当前保持 empty。shares/weight 是竞争权重，不是 hard
capacity，也保持 empty。`parallelism_ceiling()` 只做保守的
`min(placement, ceil(quota))`；IO overlap、SMT、能效或 latency policy 仍由 Sched/推理引擎决定。

需要优先使用强核时，不要假设“大编号/小编号就是 P-core”。`available_cpu_info()` 返回当前
进程可用 CPU 的 `id / performance_rank / kind`：rank 越大表示 OS 在本机报告的能力档越强，
0 表示无法建立可信的 CPU-id 映射。普通 C++ 初始化代码用 `cpu_convenience.h` 的 owning 有序
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

需要按物理核、SMT sibling、package、NUMA、L2 或 LLC sharing domain 分组时调用
`available_cpu_topology()`。它聚合与 `available_cpu_info()` **同一个动态进程视图**；每颗
`CpuInfo` 上的 domain id 只在该次 snapshot 内比较，`kUnknownCpuTopologyId` 表示 OS 没有给出
可信关系。`physical_cores == 0` 是“未知”，不是没有物理核。查询成本与可用 CPU/OS relation 数
成正比，应在初始化或重配点调用，不放进每个 task。

`cache_info()` 仍只是保守容量提示；cache-class snapshot 也只报告 OS facts，不替用户决定 tile 或
worker policy。NUMA/cache-aware stealing 属于 sched 策略；platform 只提供可组合的 CPU 事实、排序、
topology 与 affinity 机制。

频率同样只是一组 OS facts。`available_cpu_frequencies()` 与当前 available id space 对齐，分别报告
hardware maximum、当前 policy maximum 和 current observation；任一未知字段为 0。Windows 使用
processor-power records，Linux 使用 CPUFreq sysfs，macOS 只读取存在的公开全局 sysctl（Apple
Silicon 通常可能全部为 0）。它不参与 `performance_rank`、ISA gate 或默认 worker 数：DVFS/turbo 会
让 current 值立刻变化，频率也不能代表不同微架构的每时钟工作量。

`numa_current_node()` 与 `numa_node_of_cpu(cpu)` 返回 `Result<unsigned>`。查询失败、无效 CPU、
或 OS 不提供可信映射都不是节点 0；必须处理错误后再决定是否显式退化到 `kNumaAny`。如果 worker
已 required-pin 到某 CPU，先用 `numa_node_of_cpu` 选 arena node；没有硬 affinity 时 current-node
只是瞬时事实，不应缓存成线程永久归属。

`numa_nodes()` 是首次查询时固定的 process-lifetime placement 视图：Windows 仅从
当前 process-available CPU Sets/affinity 归纳 node；Linux 依次使用 mempolicy、
`/proc/self/status` 和 cgroup cpuset。它不是全机 inventory。若嵌入宿主在首次调用后
动态改变进程 CPU/cpuset，需重建该 native runtime/process，不能把旧 span 当成新快照。

NUMA preferred 是软策略：分配成功而 placement 为 `Unavailable` 仍是一块可用内存。Linux
bind 是严格节点集合；interleave/selected interleave 的 `Applied` 只说明 OS 接受 policy，
Windows 和 Linux 都可能在目标节点内存压力下回退，不能把它当成逐页 residency 证明。

先用 `numa_nodes()` 取得真实可用 id；它们可能稀疏，因此不要写
`for (unsigned n=0; n<numa_node_count(); ++n)`。`numa_capabilities()` 适合启动期预检或 UI
报告，实际操作返回的 `Status` 仍是最终结果。`<hyle/core/environment.h>` 的 `HYLE_OS_*` 是
当前 translation unit 的编译目标事实，不是运行时 NUMA capability；用户不能用它代替这里的
查询结果来选择 NUMA 路径：

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

## Host memory、环境与安全随机数

`host_memory_info()` 返回一次动态、可失败的快照。物理总量/可用量在成功时总是存在；commit、
swap、cgroup/process limit 在 OS 无法诚实报告时是 `std::nullopt`，合法的 0 不会被当作未知。
Linux 不假定 `/sys/fs/cgroup` 或只看 leaf：它按 `/proc/self/mountinfo` 解析 cgroup namespace，沿可见
祖先合并最严格的 `limit` 与 `limit-current`。这会正确考虑父组中其它进程占用的预算。
Windows 的公开查询只能看到 immediate Job，不能证明嵌套父 Job 的更紧内存限制或父级其它进程占用，
所以当前 `process_limit_bytes/process_available_bytes` 保持 empty；不要从 empty 推断“没有 Job 限制”。
装载模型或建立 KV cache 前，通常以 `effective_available_bytes()` 作为保守上限，再留出应用自己的
headroom：

```cpp
#include "hyle/platform/host_memory.h"

HYLE_ASSIGN_OR_RETURN(auto host, hyle::platform::host_memory_info());
const std::uint64_t planning_budget = host.effective_available_bytes();
```

这只是快照而非 reservation，下一次 allocation 仍可能因竞争失败。统一内存机器上也不能把 host
与 device budget 当成两个可相加的池；Compute/推理引擎必须结合 backend 的实时预算和 coherence
模型重新裁决。

环境变量的 buffer-fill core 在 `environment.h`，普通 owning 查询在
`environment_convenience.h`。unset 返回 `NotFound`，已设置为空字符串是成功；修改进程环境不是
并发配置机制，只应在程序启动或已同步的重配阶段进行。

`secure_random(span<byte>)` 使用 OS entropy，承诺完整填满或返回 `Status`，空 span 直接成功。
它适合 nonce、密钥材料和不可预测 seed；统计采样与可复现实验仍使用 Util 的显式 seed RNG，不能
因为接口相似而互换。

## 错误与所有权

资源构造/查询返回 `Result<T>`，资源变更返回 `Status`；core 成功态不隐藏分配，owning
convenience 的分配是调用形状本身的显式选择，完整原生诊断只在失败路构造。
`exists/is_directory/is_symlink` 把不存在表达为成功的 false，但不会吞掉权限或
I/O 错误。platform 的公开可恢复失败全部直接携带共享 error，不存在另一套 TLS last-error。

POSIX 信号可能让系统调用返回 `EINTR`。Platform-fs 会重启尚未产生可观察副作用、且可安全重启的
查询/control；不会盲目重试 `close/closedir` 或可能已经完成 rename/link/unlink/symlink 的调用。
因此收到后一类错误时，调用方必须按业务协议重新观察资源状态，不能假定操作一定未发生。短
read/write 仍遵守已有短传输合同，由 exact/all helper 继续剩余部分。

所有 RAII owner 都是 move-only。借出的 `data()`、symbol pointer、raw handle 或 unparker
视图不得活过 owner；`release()` 明确把回收责任转给调用方。文件 `duplicate()` 只复制 native
ownership，不复制 cursor/lock state；`RandomAccessFile` 的无 cursor 公共形状也不意味着 duplicate
aliases 自动线程安全。

`DynLib::open(path)` 加载新引用；`get(path)` 只附着到已经加载的 image，但成功后同样持有/pin
一份 loader reference，所以不会与另一个 owner 的 unload 竞争。`symbol_as<Pointer>` 只接受指针
类型，函数签名仍由调用方负责；指针在 `close`/析构后失效。相对路径和搜索名属于进程加载信任
边界，外部输入不得未经验证直接传入。需要观察 unload 诊断时显式检查 `close()`，析构只做兜底。

性能证据和仍存在的平台差异见
[`benchmarks-platform.md`](../implementation/evidence/performance/benchmarks-platform.md)，不要把单个平台点值外推到
另外两个 OS。
