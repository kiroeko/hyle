# HYLE platform/ 设计

> **Native FFI 状态校正（2026-08-25）**：HYLE 没有 native-C 客户，也没有已发布的 managed
> binding。本文关于 C/C++、C/FFI 或 C 用户的旧表述只记录待复审的 ABI 候选，不是当前跨语言
> 证书或兼容承诺；首发前可破坏性调整。先从真实 foreign-runtime 任务裁决 FFI 是否为 `N/A`；
> 只有保留 direct FFI 时，platform 才必须通过 Java FFM + .NET P/Invoke、真实导出、资源所有权/
> 回调与 batch 旅程后冻结该边界。

> 跨操作系统机制基础层。本页是 platform 的长期边界、公共类型形状和维护约束的设计
> 真源。用户如何选择见 [`../guide/platform.md`](../guide/platform.md) 与
> [`../guide/c-platform.md`](../guide/c-platform.md)，性能结论只见
> [`../performance/benchmarks-platform.md`](../performance/benchmarks-platform.md)，施工历史只见
> [`../milestones/platform-remaining.md`](../milestones/platform-remaining.md)。

## 1. 定位与依赖

platform 把 Windows、Linux 和 macOS 的不同系统调用收敛成可组合的**机制**，供 HYLE
内部地基层和需要直接控制 OS 资源的库用户共同使用。它位于共享 `error` 之上、
`mem/simd/container/sched/io/compute` 之下；`time` 是不经过 platform 的平行底层分支，
其 native 核只借 core 头，C adapter 依赖 error：

```text
error -------------------------------+------ time
  |                                  |  clock/sleep/timer semantics
platform  -- CPU/OS facts, thread, wait, VM, file, dynlib
  |\                                         |
  | +------ simd/container        runtime ISA and cache facts
  +-------- mem                  VM, huge page, NUMA, guarded stack backing
  +-------- sched <---------------- timed deadlines
  +-------- io/compute           native file/plugin/backend foundations
```

platform 不拥有调度策略、通用堆、异步 completion 或设备内存。它回答“OS 能做什么、怎样
可靠地做”，上层回答“何时做、为哪个 workload 做”。这是避免底层 API 被某个 benchmark
或单一内部消费者扭曲的首要边界。`Parker::park_timeout` 提供等待机制，但 clock/sleep 与
deadline 的时间语义仍归 time/sched；不能因为 OS 同时实现二者就伪造一条 `platform → time`
构建依赖。

`hyle/platform.h` 只是跨编译器的 `HYLE_OS_*`/属性宏头；包含它不等于链接 platform 模块。
`FileInfo::modified_time` 与 C `hyle_file_info` 复用 time 的自包含 `UnixTimestamp` /
`hyle_unix_timestamp` 值头，以免 OS metadata 与时间转换产生两种表示；它不取得 clock、sleep、
deadline 或 timer 行为，因此是公开 header-only 边，不是 `hyle_platform -> hyle_time` target 边。
只有直接调用本模块符号的实现才声明 private 依赖，公开头若命名 `CpuOrder`、`ThreadPriority`、
`VmRegion` 等 platform 类型则声明 public 依赖。不得依靠 mem/sched 的传递链接掩盖直接消费，
也不得因一个预处理宏虚构二进制依赖。

## 2. 功能面与公共形状

| 能力族 | C++ 公共形状 | 常用场景 | 关键合同 |
|---|---|---|---|
| CPU 与 cache | `CpuFeatures`、`Isa`、`CacheInfo`、`CpuTopology`、`CpuInfo` | ISA 派发、false-sharing 对齐、runtime 初始化、异构核选择 | 静态硬件事实可缓存；进程可用 CPU 动态查询；未知不猜测 |
| OS thread | `Thread`、`ThreadOpts`、`CpuAffinity`、`this_thread::*` | runtime worker、长寿命服务线程、必须控制栈或归属的线程 | move-only；默认 best-effort，required 在 entry 前报告配置失败 |
| 地址等待 | `detail::wait_on_address*`、`Parker`、`Unparker` | mutex/worker parking/waker 地基 | 裸等待允许伪唤醒；Parker 保存一个 token，解决 unpark-before-park |
| 虚拟内存 | `VmRegion` 与 `vm_*` | staged commit、guard page、大块地址保留、自定义 owner | 页粒度；reserve 与 commit 分离；RAII 优先，裸函数供底层组合 |
| NUMA/大页 | `numa_nodes/capabilities`、`vm_*numa*`、huge-page advice | 大块 arena、per-node/selected-node backing、THP 提示 | node id 可稀疏；只有 Linux bind 是严格节点集合；interleave/preferred 是 OS placement policy；建议失败不破坏 mapping |
| 文件系统与环境 | `FileInfo`、`FileHandle`、无 owning 结果的 visitor、buffer-fill 查询 | 配置、资源、目录扫描、临时文件、部署环境 | UTF-8 路径；短读写正常；打开 owner 不阻塞原子替换路径；长路径 scratch OOM 显式返回 |
| 文件映射 | `MappedFile` + `Advice`/`prefetch`/`lock` | 大文件随机访问、权重/数据集、COW、显式 warm-up/驻留 | RO/RW/COW；任意 offset；hint 与 residency 不隐式发生 |
| 动态库 | `DynLib` | 插件、用户 backend、FFI 宿主符号 | open/get/current 语义分开；符号签名由调用方承担 |

这些能力共同遵循“窄机制 + 明确 owner + 可恢复错误”的形状。platform 不提供一个巨型
`PlatformApi` 对象，也不把无关系统调用堆进单一 namespace facade；能力按头文件和资源类型
模块化，调用方只引入所需合同。

## 3. 系统事实、进程事实与策略必须分开

CPU API 有三类不同生命周期，不能混用：

1. `cpu_features()` / `best_isa()` 是当前进程可安全执行的指令能力。x86 同时检查 CPU 位和
   OS 扩展寄存器保存能力；探测 TU 必须保持 baseline ISA。构建系统把该 TU 独立放入关闭 IPO
   的 private object target，其余 platform 源仍可接受父工程 LTO，不能用 target-wide 关 LTO
   换取探测安全。
2. `cache_info()` / `cpu_topology()` 是缓存的系统硬件视图，适合诊断、容量提示和长期策略。
   它们不等于容器/cgroup 给当前进程的 CPU 配额。
3. `available_cpu_count/ids/info()` 是动态进程视图，必须尊重 affinity、cpuset/cgroup 和
   Windows CPU Sets。返回 id 可能稀疏，调用方不得用 `[0, count)` 推导。

`performance_rank` 只采用 OS 提供的可信 CPU-id 映射。没有证据时返回 unknown，不按频率、
编号或营销名称猜 P/E 核。`CpuOrder` 只对完整可用集合排序，不偷偷删核；“只用强核”、
worker 数、轮转、超订和 NUMA-aware stealing 都是 sched 或应用策略。

## 4. Thread 只表达 OS 线程机制

`Thread` 是 move-only OS handle，不用 `std::thread` 暴露 ABI，也不接受 `std::function`。
入口使用 function pointer + context，既能服务内部 runtime，也能在任务成立时映射到 Native FFI 候选。

`ThreadOpts` 只承载 OS 在创建边界可执行的配置：name、stack size、单 CPU/CPU 集合 affinity、
portable priority 和配置策略。view 参数在 `spawn()` 返回前同步复制；用户无需为短期配置
制造长期 owner。

- `kBestEffort` 是可移植默认值：不支持或无权限的附加配置不能阻止线程运行。
- `kRequired` 是强合同：配置必须在用户 entry 前完成，失败则返回 `Status` 且 entry 不运行。
- `this_thread::set_*` 用于已进入线程后的逐项控制，失败直接返回共享 `Status`。
- 析构仍 joinable 是生命周期 bug，fail-fast；platform 不把忘记 join 静默变成 detach。

大量短任务、worker placement 和结构化并发归 `sched`。platform 不提供第二个线程池，也不把
“performance first”偷设为所有用户的默认行为。macOS 没有公开的硬 CPU pin，必须诚实返回
不支持；强核软偏好使用 OS QoS/priority，而不是伪造 affinity。

## 5. 等待层：最小原语与可直接使用的 Parker

地址等待是 futex、`WaitOnAddress` 与 Apple address-wait 的最小公分母。它服务已经拥有原子
状态字的同步原语，允许伪唤醒，调用方必须循环检查谓词。它位于 `detail`，避免普通用户把
一次 wake 错当成业务条件成立。

`Parker` 在其上实现经过验证的三态 token 协议：unpark 可以先于 park，重复 unpark 合并成
一个 token。它适合 worker 睡眠与外部线程唤醒，但不是计数 semaphore、condition variable
或 fiber-aware wait；这些带队列/取消/期限语义的类型归 `sched`。timed park 是 single-shot，
调用方在自己的 deadline loop 中重新检查条件并计算剩余时间。

## 6. VM、mem、io 与 compute 的边界

匿名虚拟内存位于 platform，因为 reserve/commit/protect/decommit/release 是 OS 地址空间
机制。普通小对象和通用容器 backing 归 `mem` allocator，不应直接 mmap/VirtualAlloc。

- `VmRegion::allocate_rw`：常用的一步可读写 mapping；不要求用户为 benchmark 人工拆成
  reserve + commit。
- `VmRegion::reserve`：真正需要稳定地址和按段提交的 arena、JIT 或大模型加载器。
- `reserve_guarded`：显式 guard page，供 fiber stack 等越界必须尽早失败的资源。
- 裸 `vm_*`：供实现自有 owner 的基础设施；普通用户优先 RAII，避免 base/size 配错。

`MappedFile` 是同步、单进程文件映射。普通顺序文件 I/O 用 `FileHandle`；异步 read/write、
注册文件和 completion 归 `io`；跨进程 shared memory 的命名、权限和同步也不伪装成普通
mapping。device VRAM、pinned host memory、coherence 与 GPU completion 归 `compute`，不能
用 `VmRegion` 或 `mem::Allocator` 抹掉 backend 生命周期。

对权重、索引和数据集，映射 owner 与访问策略分开：`Advice` 只给 kernel 访问模式提示，
`prefetch` 是 will-need 便利入口，`lock/unlock` 才明确购买驻留。它们都作用于已有 mapping，
不在 `open()` 时暗中 fault 全文件或消耗 memlock 配额。POSIX 使用 `madvise/mlock`；Windows
只对 `WillNeed` 有 `PrefetchVirtualMemory` 对等能力，其它 hint 诚实返回 `Unimplemented`，
residency 使用 `VirtualLock`。unsupported hint 不影响 mapping 的正确性。

NUMA preferred 与 huge-page advice 是策略提示，返回值必须让用户看见是否 applied、not
needed 或 unavailable。Linux bind 是严格节点集合；interleave/selected interleave 表示 OS 已
接受条带 policy，Windows 和 Linux 都可能在目标节点内存压力下回退，不能把它宣传成物理页
驻留证明。软 preferred 降级后 mapping 仍然有效。

公共 NUMA 面同时提供拓扑、运行期能力与分配机制：`numa_nodes()` 返回稳定的实际 node id
列表，id 可能稀疏，不能从 `numa_node_count()` 推导 `[0,count)`；`numa_capabilities()` 区分
topology、preferred reserve、existing-range bind/interleave 和 selected interleave。用户代码应按
能力或操作返回的 `Status` 决策，不直接围绕 `HYLE_OS_*` 编译分叉；后者只用于 platform 内部选择
系统实现。

`vm_allocate_numa_interleaved(size,nodes,bytes_per_chunk)` 拥有一段连续、已提交的 RW 地址区：
空 node span 表示全部可用节点；显式 id 必须可用，可以重复，重复项像 oneTBB 一样表达 weighted
placement；0 chunk 表示一页，条带遵循调用方给定顺序。拓扑缓存同时保存 id 列表与 membership
位图，因此节点查询 O(1)、输入验证 O(nodes)。Linux 的单 node 与普通逐页递增条带只需一个惰性
`mbind`。自定义 chunk/order 在条带数不大时仍保留惰性 range policy；条带数过大时改用固定批次
first-touch + `move_pages`，以 O(pages) 一次性工作换掉无界 O(stripes) 持久
VMA policy；syscall 为 O(pages/batch)，scratch 为常量。它优于 oneTBB 为全部 page 分配
page/node/status 数组再一次 `move_pages` 的 O(pages) 临时空间。Windows 只在
`VirtualAllocExNuma` 可用时宣称 selected-interleave：先保留一个普通的连续地址区，再在同一
reservation 内按条带 NUMA-aware commit，因此 rollback、raw `vm_release`、`VmRegion` 析构都只
释放一个 owner。`VirtualAlloc2` 的 placeholder replacement 会把各条带变成必须分别释放、恢复并
合并的独立 allocation，与公开的一次整段 release 合同不相容；它可以继续服务单 owner 的
preferred reservation，但不能单独启用 `kSelectedInterleave`。
同一 node 的 weighted 退化会合并成一个 commit。两边普通内存访问都没有 HYLE metadata/dispatch。

没有多节点 primitive 的平台不伪造成功：多节点放置请求返回 `Unimplemented`。macOS 当前只暴露
逻辑 node 0，因此同一 API 正常返回 `NotNeeded` 的普通 allocation。selected allocation 不允许
decommit/recommit 子区间，因为 Windows 重新提交会丢失 placement；使用 `VmRegion` 时会返回
`FailedPrecondition`，裸 API 只与整段 `vm_release` 配对。

## 7. 文件系统只提供跨平台基础资源

platform 文件面覆盖真实基础库常用需求：stat/存在性、目录与文件变更、权限的可移植子集、
link、目录 visitor、cwd/temp/home/env、同步 open/read/write 和原子临时文件。

设计约束如下：

- Windows 路径在边界从 UTF-8 转宽字符并使用 Unicode API；公开面不泄漏 `wchar_t`。
- 查询使用 `Result<T>`，所以合法的 0、空文件、0000 权限不会与失败混淆。
- 变长字符串使用 buffer-fill：返回完整长度，允许先 count-only，再由调用方选择 allocator。
- 目录遍历使用 callback + context，不构造 owning entry；普通路径无逐 entry heap allocation，
  但无界路径/编码 scratch 允许 checked allocation 并以 `ResourceExhausted` 失败。C++ lambda 只是
  inline trampoline。
- entry type 描述目录项自身且不跟随 symlink；`stat` 报 target 信息并另外保留 path 是否 link。
- 递归删除在调用方拥有且遍历期间不被并发替换的目录树中绝不跟随 symlink/junction；这是数据
  安全合同，不是实现细节。普通 path API 无法把逐项检查与后续删除做成跨平台原子事务，敌对方
  能并发改名/替换 entry 的目录不属于这个合同，应由调用方先隔离或使用平台专用 handle-relative
  sandbox 机制。后序遍历使用 resumable native iterator frame 而非递归调用栈：每层保留一个
  OS iterator，单个可增长 path buffer 表示当前路径，普通文件/symlink 立即删除，同层兄弟从不
  排队。辅助内存为 `O(depth + max_path)`，但 OS iterator handle 同样为 `O(depth)`，极深树可能
  命中进程 handle/fd 上限并返回 `ResourceExhausted`；不会因目录宽度线性持有完整路径。
- `create_temp_file` 把 prefix 限为单个 basename 片段，保证结果直接位于指定目录；它原子创建并
  返回已打开且不跨 exec 继承的 handle，保留合法 prefix，避免“生成名字再打开”的 TOCTOU
  窗口和静默名称替换。显式目录不能为空；dynamic checked scratch 避免库自身的 MAX_PATH/1024
  ceiling，native filesystem 上限仍是最终边界。POSIX `read_symlink` 把
  `n==scratch capacity` 视为可能截断并扩容重试，保持 buffer-fill 的完整长度合同。
- `copy_file` 只把 regular-file copy 作为成功合同。POSIX 在 open 后按 descriptor 分类并按 EOF
  驱动，不能依赖 `st_size`；若 non-regular destination 在 open 阶段先以 `ENXIO` 等失败则保留
  native error。Windows 的 reparse follow/replace 行为由 `CopyFileW` 定义，不伪造 POSIX 等价。
- `FileHandle` 的短读/短写是成功；需要观察 close/flush 失败时必须显式 close。
- Windows `FileHandle`/`MappedFile` 以 `FILE_SHARE_DELETE` 打开，使 rename/delete 与 POSIX
  一样只替换目录项、已有 owner 继续持有文件对象；这也是原子模型/缓存更新的必要合同。

platform 不拥有 path 字符串算法、glob、watcher、高层 directory object、async stream 或虚拟
文件系统。纯词法 path 操作属于 container/string，变化通知和异步数据流属于 io。

## 8. 错误、所有权与 Native FFI 候选

platform 可以依赖最底层 error。资源创建和有值查询返回 `Result<T>`，状态变更返回
`Status`；错误保存 HYLE code、operation 和原生系统诊断。成功热路径不分配，模块不维护
私有 TLS last-error，也不让 errno/GetLastError 越过调用边界后才被读取。

POSIX fs、MappedFile、VM 与 NUMA 的 generic errno code 必须来自 error 的
`status_from_errno`；Win32 generic OS code 必须来自 error 的 `status_from_win32`，COM/WinRT
返回值必须来自 `status_from_hresult`。Windows CRT 直接返回 errno 值的 API（例如 `_wputenv_s`）
仍走 errno 入口。只有系统调用
合同更具体时才由 platform 局部覆盖（例如路径分量不是目录）。CUDA/Vulkan 依赖 SDK 类型与
SDK-specific recovery，仍由 compute owner 保留 translator；不能硬套 errno，也不能让 platform
再次复制通用 Win32 taxonomy/message table。

C++ 资源 owner（`Thread`、`VmRegion`、`FileHandle`、`MappedFile`、`DynLib`）均为 move-only
RAII；借出的 pointer、symbol、native handle 或 view 不得活过 owner。需要转移回收责任时必须
使用显式 release/close 语义。

现有 `include/hyle/c/platform.h` 与 `c_platform.cc` 只是 C-syntax ABI 候选账本，不是独立的
native-C 产品面，也不承诺与 C++ API 做 1:1 parity。先以真实 Java/.NET 等任务决定整块边界是
保留、缩小还是 `N/A`；只有保留 direct FFI 时，下面的形状才成为待验合同：

- 小型 POD facts 和 fixed-width tag/flag 可直接值传递；tag 使用 `int32_t`/`uint32_t` typedef +
  integer constants，不让 C `-fshort-enums` 改变参数、返回或 POD 布局；资源用 opaque handle
  或一字宽 native handle。
- fixed typedef 必须贯穿导出参数/返回、callback 与 POD 字段；`hyle_thread_opts` 在
  `struct_size` 后把旧 padding 显式命名为必须为 0 的 `reserved`，业务字段全部可选；保留到
  `config_policy` 为止的当前候选顺序，数组 affinity 只追加在尾部。当前 64-bit fixture 断言 size 72、
  priority/policy/array/count offset 44/48/56/64，mandatory readable size 8；满足显式
  `reserved=0` 合同的字段前缀 56 可读。`hyle_cpu_info` 当前断言 size 12、kind offset 8；这些
  数字在首个 managed binding 发布前不是兼容承诺。
  裸 `int` 只保留 C bool/callback truth 与显式 sentinel，裸 `unsigned` 只保留 CPU id/count。
- 会随新 ISA 扩展的 `hyle_cpu_features` 不按值跨 ABI：调用方用 INIT 提供
  `struct_size + reserved` 版本前缀和可写容量，adapter 只写已知前缀、保留 future tail，且在
  参数错误时也先把可写的已知 feature 字段初始化为 false。已知短前缀可继续使用，旧库也能
  接受更大的未来结构；这类大型可演进 snapshot 统一返回 owning Status + out。
- 所有可恢复失败返回 opaque `hyle_status` + out 参数；`NULL` 是 OK，每个非 NULL owning
  handle 由调用方恰好一次传给 `hyle_status_destroy`。
- callback 使用 function pointer + `void*` context；字符串/数组使用 pointer + count/view。
- C thread opts 以 `struct_size` 开头并提供初始化宏/setter；全零不冒充有效初始化。桥接先以
  当前默认值构造本地结构，再只 memcpy 调用方声明且本库已知的前缀：过短必需前缀拒绝、
  缺失尾字段取默认、未知未来尾部忽略。pointer+count 等成对扩展仅在两字段都完整可读时启用，
  防止部分字段复制后形成伪造 view 或越界读取。
- 不机械复制 C++ 模板/RAII 名称。只有已选中的 foreign-runtime 任务要求能力可达；未选中的
  编译期、RAII 或同工具链能力可以明确不导出，不能为 parity 制造符号。

C++ 在同一个无分配 core 上提供两种显式便利形状：`Thread::spawn(callable, opts)` 拥有 capturing/
move-only callable；`platform/convenience.h` 以 header-only overload 返回 owning
`std::string/std::vector`。后者负责 count/fill、动态视图重试与 STL 分配异常到 `Status` 的
adapter 转换，但不进入 DLL/direct FFI 候选，也不让 `fs.h/cpu.h` 的 core 被迫引入 owning 类型。这样普通
C++ 使用不重复样板，allocator/arena/FFI 与热调用者仍能控制每次分配。

候选符号和精确所有权映射记录在 [`../guide/c-platform.md`](../guide/c-platform.md)。纯 C 编译、
声明引用和动态导出一致性只证明 schema/compiler/export hygiene；若保留 direct FFI，还必须由
Java FFM 与 .NET P/Invoke 从精确 shared-library 路径完成 ABI bootstrap、调用和生命周期旅程。

## 9. 性能架构

platform 的 SOTA 标准不是“比系统调用快”，而是同合同下不引入可避免的额外工作：

- `cpu_features/cache/topology/page_size` 等稳定事实探测一次后 O(1) 读取。
- 动态 CPU availability 保持 O(N) OS 查询，不能为 O(1) 缓存而返回过时 cpuset；只用于
  初始化/重配控制面，不进入每 task 热路径。Windows 默认 CPU Set membership 先排序默认 id，
  再对每个系统 set 二分查找，时间 O(D log D + N log D)、空间 O(D)，避免旧的 O(ND) 嵌套扫描，
  也不按可能稀疏且很大的 opaque id 分配 bitmap。
- Thread 默认路径不为未请求的强保证增加握手/refcount；required 才支付同步配置成本。
- callable 线程入口只在用户选择 owning 便利重载时多一次冷路径 heap 状态分配；OS thread 创建仍是
  主成本，runtime/sched 与 C adapter 继续使用 allocation-minimal `entry + void*` core。
- `wait_on_address` 直接落 OS 原语，Parker 只加一个原子状态机，不增 mutex/heap allocation。
- `allocate_rw` 使用最少 OS mapping 操作；staged reserve+commit 的额外 syscall 是不同合同，
  不能拿来宣称普通分配性能。
- directory visitor 常见路径不逐项 stat、不分配；只有文件系统不给类型时才补查询。
- VM、MappedFile、DynLib 与 FileHandle 的成功生命周期无异常和隐藏堆分配；错误消息只在
  冷失败路构造。
- `MappedFile::advise`、`lock`、`unlock` 是 O(1) 参数检查 + 至多一个 native call，零 heap
  allocation；成本由 range 大小导致的 kernel page work 决定，不伪装成纳秒用户态原语。

基准必须比较相同语义：HYLE wrapper 对 raw OS、目录 visitor 对同等 metadata 的 std/OS
形状、MappedFile 同时对 buffered stream 和 raw mmap。点值、样本和平台输赢只维护在
[`../performance/benchmarks-platform.md`](../performance/benchmarks-platform.md)。

## 10. 平台差异必须显式保留

统一 API 不等于虚构相同 OS：

- Windows affinity 集合的硬 mask 受 processor group 限制；跨组 placement 由上层拆 worker。
- macOS 无公开硬 pin；priority 是 QoS 软偏好。逐 logical CPU 的 P/E 映射未知时保持 unknown。
- macOS 的公开 address-wait 实现依赖 14.4 起可用的 `os_sync_wait_on_address*`。构建 platform
  component 时 deployment target 必须 ≥14.4；HYLE 顶层原生 macOS 构建未显式指定时默认 14.4，
  被父工程嵌入时若父工程选择更低 target 则 configure 失败，不得静默产出会在旧系统缺符号的库。
- Linux process availability 受调用线程 affinity/cpuset 影响，新线程继承该视图。
- POSIX permission bits 与 Windows readonly attribute 只能提供一个诚实的可移植子集。
- 环境变量的空值行为、symlink 权限和 huge-page/NUMA 能力存在系统差异，返回值与 guide 必须
  讲明，而不是静默伪装成功。

新增平台实现时，优先保持合同和错误可观察性；不允许为了让一个测试“全平台同值”而抹掉
真实系统差异。

## 11. 有意不属于 platform

| 能力 | 归属 | 理由 |
|---|---|---|
| allocator/arena/pool/fiber stack owner | `mem` | 分配策略和对象生命周期，不是 OS syscall |
| timer queue/deadline policy | `time` / `sched` | platform 只提供 timed wait 机制 |
| thread pool/task/fiber/coroutine | `sched` | placement、取消、stealing 与结构化生命周期是调度策略 |
| async file/network、watcher、shared-memory channel | `io` | completion、backpressure 与跨进程同步 |
| VRAM/pinned/staging/coherence | `compute` | device/backend/event 生命周期 |
| path lexical/string owner | `container` | 不需要 OS，且 platform 不能反向依赖容器 |
| process/subprocess sandbox | 独立未来模块或真实消费者驱动 | 不能因“OS 也有”就无限扩大基础面 |

最后一行是扩展纪律：platform 可以很强，但不能成为“所有和 OS 有关东西”的杂物间。只有
多个基础库用户反复需要、能形成跨平台诚实合同、且归不到已有高层生命周期 owner 的机制，
才进入 platform 公共面。

## 12. 维护闸门

改变 platform 公共形状时必须同时回答：

1. 这是可复用 OS 机制，还是某个上层 workload 的策略？
2. 系统事实、动态进程事实和用户偏好是否被不同类型表达？
3. 是否存在真实 foreign-runtime 任务？没有时是否明确 `FFI=N/A`；保留 direct FFI 时，Java/.NET
   是否都能完成该任务，且所有权和失败可见？
4. 新调用是否给默认热路径增加 syscall、分配、握手或间接层？
5. Windows/Linux/macOS 的能力差异是显式结果，还是被静默吞掉？
6. 是否有真实内部或外部消费者，而非只为 benchmark 制造形状？

稳定设计更新本页；用户选择和坑更新 guide；实测更新 performance；一次性验收和提交记录
更新 milestone。
