# HYLE platform/ 设计

> 跨操作系统机制基础层。本页是 platform 的长期边界、公共类型形状和维护约束的设计
> 真源。用户如何选择见 [`../guide/platform.md`](../guide/platform.md) 与
> [`../guide/c-platform.md`](../guide/c-platform.md)，性能结论只见
> [`../performance/benchmarks-platform.md`](../performance/benchmarks-platform.md)，施工历史只见
> [`../milestones/platform-remaining.md`](../milestones/platform-remaining.md)。

## 1. 定位与依赖

platform 把 Windows、Linux 和 macOS 的不同系统调用收敛成可组合的**机制**，供 HYLE
内部地基层和需要直接控制 OS 资源的库用户共同使用。它位于共享 `error` 之上、
`mem/simd/container/sched/io/compute` 之下；`time` 是独立的零 HYLE 依赖根：

```text
error                                      time
  |                               clock/sleep/timer semantics
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
| NUMA/大页 | `NumaPlacement`、`vm_bind_numa*`、huge-page advice | 大块 arena、per-node backing、THP 提示 | preferred 是软结果；bind/interleave 是硬合同；建议失败不破坏 mapping |
| 文件系统与环境 | `FileInfo`、`FileHandle`、无分配 visitor、buffer-fill 查询 | 配置、资源、目录扫描、临时文件、部署环境 | UTF-8 路径；短读写正常；不存在与权限/IO 错误分开 |
| 文件映射 | `MappedFile` | 大文件随机访问、权重/数据集、COW | RO/RW/COW 明确；任意 offset；move-only RAII |
| 动态库 | `DynLib` | 插件、用户 backend、FFI 宿主符号 | open/get/current 语义分开；符号签名由调用方承担 |

这些能力共同遵循“窄机制 + 明确 owner + 可恢复错误”的形状。platform 不提供一个巨型
`PlatformApi` 对象，也不把无关系统调用堆进单一 namespace facade；能力按头文件和资源类型
模块化，调用方只引入所需合同。

## 3. 系统事实、进程事实与策略必须分开

CPU API 有三类不同生命周期，不能混用：

1. `cpu_features()` / `best_isa()` 是当前进程可安全执行的指令能力。x86 同时检查 CPU 位和
   OS 扩展寄存器保存能力；探测 TU 必须保持 baseline ISA 且禁止 LTO 内联高级指令。
2. `cache_info()` / `cpu_topology()` 是缓存的系统硬件视图，适合诊断、容量提示和长期策略。
   它们不等于容器/cgroup 给当前进程的 CPU 配额。
3. `available_cpu_count/ids/info()` 是动态进程视图，必须尊重 affinity、cpuset/cgroup 和
   Windows CPU Sets。返回 id 可能稀疏，调用方不得用 `[0, count)` 推导。

`performance_rank` 只采用 OS 提供的可信 CPU-id 映射。没有证据时返回 unknown，不按频率、
编号或营销名称猜 P/E 核。`CpuOrder` 只对完整可用集合排序，不偷偷删核；“只用强核”、
worker 数、轮转、超订和 NUMA-aware stealing 都是 sched 或应用策略。

## 4. Thread 只表达 OS 线程机制

`Thread` 是 move-only OS handle，不用 `std::thread` 暴露 ABI，也不接受 `std::function`。
入口使用 function pointer + context，既能服务内部 runtime，也能直接映射到 C/FFI。

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

NUMA preferred 与 huge-page advice 是策略提示，返回值必须让用户看见是否 applied、not
needed 或 unavailable。依赖精确放置时使用 bind/interleave 的硬合同并处理失败；软降级后
mapping 仍然有效。

## 7. 文件系统只提供跨平台基础资源

platform 文件面覆盖真实基础库常用需求：stat/存在性、目录与文件变更、权限的可移植子集、
link、目录 visitor、cwd/temp/home/env、同步 open/read/write 和原子临时文件。

设计约束如下：

- Windows 路径在边界从 UTF-8 转宽字符并使用 Unicode API；公开面不泄漏 `wchar_t`。
- 查询使用 `Result<T>`，所以合法的 0、空文件、0000 权限不会与失败混淆。
- 变长字符串使用 buffer-fill：返回完整长度，允许先 count-only，再由调用方选择 allocator。
- 目录遍历使用 callback + context 的无分配 core；C++ lambda 只是 inline trampoline。
- entry type 描述目录项自身且不跟随 symlink；`stat` 报 target 信息并另外保留 path 是否 link。
- 递归删除绝不跟随 symlink/junction；这是数据安全合同，不是实现细节。
- `create_temp_file` 原子创建并返回已打开 handle，避免“生成名字再打开”的 TOCTOU 窗口。
- `FileHandle` 的短读/短写是成功；需要观察 close/flush 失败时必须显式 close。

platform 不拥有 path 字符串算法、glob、watcher、高层 directory object、async stream 或虚拟
文件系统。纯词法 path 操作属于 container/string，变化通知和异步数据流属于 io。

## 8. 错误、所有权与 C ABI

platform 可以依赖最底层 error。资源创建和有值查询返回 `Result<T>`，状态变更返回
`Status`；错误保存 HYLE code、operation 和原生系统诊断。成功热路径不分配，模块不维护
私有 TLS last-error，也不让 errno/GetLastError 越过调用边界后才被读取。

C++ 资源 owner（`Thread`、`VmRegion`、`FileHandle`、`MappedFile`、`DynLib`）均为 move-only
RAII；借出的 pointer、symbol、native handle 或 view 不得活过 owner。需要转移回收责任时必须
使用显式 release/close 语义。

C ABI 是独立但等价的产品面：

- POD facts 和 named enum 直接值传递；资源用 opaque handle 或一字宽 native handle。
- 所有可恢复失败返回拥有型 `hyle_status` + out 参数，调用方统一 `hyle_status_free`。
- callback 使用 function pointer + `void*` context；字符串/数组使用 pointer + count/view。
- C thread opts 以 `struct_size` 开头并提供初始化宏/setter；全零不冒充有效初始化。
- C 不机械复制 C++ 模板/RAII 名称，但必须能覆盖同一用户任务，不得出现 C 无法完成的
  platform 能力。

C++ 在同一个无分配 core 上提供两种显式便利形状：`Thread::spawn(callable, opts)` 拥有 capturing/
move-only callable；`platform/convenience.h` 以 header-only overload 返回 owning
`std::string/std::vector`。后者负责 count/fill、动态视图重试与 STL 分配异常到 `Status` 的
adapter 转换，但不进入 DLL/C ABI，也不让 `fs.h/cpu.h` 的 core 被迫引入 owning 类型。这样普通
C++ 使用不重复样板，allocator/arena/FFI 与热调用者仍能控制每次分配。

导出符号和精确所有权以 [`../guide/c-platform.md`](../guide/c-platform.md) 为用户信源；机械
声明/纯 C 引用/动态库导出一致性是发布闸门，不在 architecture 复制易过时的符号计数。

## 9. 性能架构

platform 的 SOTA 标准不是“比系统调用快”，而是同合同下不引入可避免的额外工作：

- `cpu_features/cache/topology/page_size` 等稳定事实探测一次后 O(1) 读取。
- 动态 CPU availability 保持 O(N) OS 查询，不能为 O(1) 缓存而返回过时 cpuset；只用于
  初始化/重配控制面，不进入每 task 热路径。
- Thread 默认路径不为未请求的强保证增加握手/refcount；required 才支付同步配置成本。
- callable 线程入口只在用户选择 owning 便利重载时多一次冷路径 heap 状态分配；OS thread 创建仍是
  主成本，runtime/sched 与 C adapter 继续使用 allocation-minimal `entry + void*` core。
- `wait_on_address` 直接落 OS 原语，Parker 只加一个原子状态机，不增 mutex/heap allocation。
- `allocate_rw` 使用最少 OS mapping 操作；staged reserve+commit 的额外 syscall 是不同合同，
  不能拿来宣称普通分配性能。
- directory visitor 常见路径不逐项 stat、不分配；只有文件系统不给类型时才补查询。
- VM、MappedFile、DynLib 与 FileHandle 的成功生命周期无异常和隐藏堆分配；错误消息只在
  冷失败路构造。

基准必须比较相同语义：HYLE wrapper 对 raw OS、目录 visitor 对同等 metadata 的 std/OS
形状、MappedFile 同时对 buffered stream 和 raw mmap。点值、样本和平台输赢只维护在
[`../performance/benchmarks-platform.md`](../performance/benchmarks-platform.md)。

## 10. 平台差异必须显式保留

统一 API 不等于虚构相同 OS：

- Windows affinity 集合的硬 mask 受 processor group 限制；跨组 placement 由上层拆 worker。
- macOS 无公开硬 pin；priority 是 QoS 软偏好。逐 logical CPU 的 P/E 映射未知时保持 unknown。
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
| timer wheel/deadline policy | `time` / `sched` | platform 只提供 timed wait 机制 |
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
3. C++ 与纯 C 是否都能完成同一真实任务，所有权和失败是否可见？
4. 新调用是否给默认热路径增加 syscall、分配、握手或间接层？
5. Windows/Linux/macOS 的能力差异是显式结果，还是被静默吞掉？
6. 是否有真实内部或外部消费者，而非只为 benchmark 制造形状？

稳定设计更新本页；用户选择和坑更新 guide；实测更新 performance；一次性验收和提交记录
更新 milestone。
