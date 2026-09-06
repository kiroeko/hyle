# HYLE platform/ 设计

> 本页是 platform 当前设计依据。模块只提供 C++23 source/static/shared；Native FFI 当前为
> `N/A`，早期 C schema、bridge、动态导出和 fixture 已删除。验收状态与一次性事实由
> implementation 记录维护，不写进架构真相。

> 用户当前选型见 [user/platform](../user/platform.md)，当前基础证据见
> [platform evidence](../implementation/evidence/performance/benchmarks-platform.md)，v9 定向重开见
> [platform v9 evidence](../implementation/evidence/performance/benchmarks-platform-v9.md)，施工历史见
> [platform tasks](../implementation/tasks/platform/)。

## 1. 定位与依赖

platform 把 Windows、Linux 和 macOS 的不同系统调用收敛成可组合的**机制**，供 HYLE
内部地基层和需要直接控制 OS 资源的库用户共同使用。面向用户只有一个 Platform 产品；内部按
依赖成本和生命周期拆成三个质因数，namespace 仍统一为 `hyle::platform`：

```text
core + error
     |
     +-- platform -------- host facts, raw VM/NUMA, dynlib, env, entropy
     |      |\
     |      | +-- platform_thread -- PUBLIC core/error/time; thread + wait
     |      +---- platform_fs ----- PUBLIC core/error/time; synchronous fs
     |
     +-- Platform product facade = all three factors (consumer only)

platform + platform_fs + mem
     |
     +-- mem_vm --------- mapping/residency/guarded storage owner
```

host factor 公开依赖 `core/error`；`platform_thread` 与 `platform_fs` 各自公开依赖
`core/error/time`，并只为实现复用私有依赖 host factor。`Parker::park_for(time::Duration)` 和
`FileInfo::modified_time` 分别形成真实的 public Time 边。内部模块链接自己直接使用的 factor，绝不
链接产品 facade；source/static/shared 用户选择 Platform 产品时 facade 再聚合三者。这样既允许并行
开发和最小依赖，又不把内部拆分泄漏成普通用户的选型负担。

platform 不拥有调度策略、通用堆、storage mapping、异步 completion 或设备内存。它回答“OS 能
做什么、怎样可靠地做”，上层回答“何时做、为哪个 workload 做”。这是避免底层 API 被某个
benchmark 或单一内部消费者扭曲的首要边界。clock/sleep 与 deadline 的时间语义仍归 time/sched，
文件 mapping/residency 生命周期归 Mem；依赖 Time 或 Mem 不授权 Platform 接管它们的策略。

`<hyle/core/environment.h>` 的 `HYLE_OS_*` 等数值宏只描述当前 translation unit 的编译目标，
归 header-only Core；包含它不等于链接 platform binary。Platform 只拥有运行时 OS/CPU/cache/NUMA
事实与资源机制，不再与根级“platform”宏头共享一个容易误解的名字。
`FileInfo::modified_time` 复用 time 的 `UnixTimestamp`，以免 OS metadata 与时间转换产生两种表示；
这既是公开 header include，也是 `hyle_platform_fs -> hyle_time` 的真实 `PUBLIC` target 边。
只有直接调用某 factor 符号的实现才声明 private 依赖；公开头命名另一个 factor 的类型时声明 public
依赖。不得依靠 product facade 或上层模块的传递链接掩盖直接消费，也不得因一个预处理宏虚构
二进制依赖。

## 2. 功能面与公共形状

| 能力族 | C++ 公共形状 | 常用场景 | 关键合同 |
|---|---|---|---|
| CPU、cache、capacity 与 topology | `CpuFeatureSet`、`CacheInfo`、`CpuInfo`、`CpuCacheClassInfo`、`CpuFrequencyInfo`、`ProcessCpuCapacity`、动态 topology/cache/frequency snapshot | implementation-owned target matching、false-sharing 对齐、异构 class tiling、物理核/SMT/cache-domain placement、quota-aware worker planning | 静态/coarse facts 可缓存；placement、hard bandwidth、P/E class 与瞬时频率是不同事实；动态查询可失败；未知不猜测 |
| 进程启动策略 | `ProcessPriority`、`set_process_priority` | 服务/推理进程在创建 workers 前显式选择节能或性能倾向 | 五档、无 realtime；默认不调用；提升可能因权限失败；Windows 两项 control 非事务 |
| OS thread | `Thread`、`ThreadOpts`、`CpuAffinity`、`this_thread::*` | runtime worker、长寿命服务线程、必须控制栈或归属的线程 | move-only；默认 best-effort，required 在 entry 前报告配置失败 |
| 地址等待 | `detail::wait_on_address*`、`Parker`、`Unparker` | mutex/worker parking/waker 地基 | 裸等待允许伪唤醒；Parker 保存一个 token，解决 unpark-before-park |
| 虚拟内存 | `VmRegion` 与 `vm_*` | staged commit、guard page、大块地址保留、自定义 owner | 页粒度；reserve 与 commit 分离；RAII 优先，裸函数供底层组合 |
| NUMA/大页 | `numa_nodes/capabilities`、`numa_node_memory_info`、`vm_*numa*`、huge-page advice | node-local model/KV/worker budget、大块 arena、per-node/selected-node backing、THP 提示 | node id 可稀疏；node memory 是动态物理事实而非 reservation；只有 Linux bind 是严格节点集合；interleave/preferred 是 OS placement policy；建议失败不破坏 mapping |
| Host memory | `HostMemoryInfo`、`host_memory_info` | model/KV allocation 前的 host budget | 动态 fallible snapshot；optional constraint 不用 0 冒充 unknown；不是 reservation |
| 文件系统 | `FileInfo`、`FileHandle`、`RandomAccessFile`、无 owning 结果的 visitor、buffer-fill 查询 | 配置、资源、顺序/显式 offset 读取、目录扫描、临时文件、缓存/索引控制 | UTF-8 路径；opened-object identity、duplicate alias 与 cursor 语义显式；短读写与 exact/all 并存；显式 sync/close |
| 环境与 OS entropy | `get/set/unset_env_var`、`secure_random` | 启动配置、不可预测 seed/nonce | unset 与 empty 分开；环境写只在同步控制面；entropy 完整填满或失败 |
| 动态库 | `DynLib` | 插件、可选系统库、用户 native backend | open/get/current 语义分开；`get` 持有 loader 引用；显式 close 可观察错误；符号签名由调用方承担 |

这些能力共同遵循“窄机制 + 明确 owner + 可恢复错误”的形状。platform 不提供一个巨型
`PlatformApi` 对象；能力按 factor、头文件和资源类型模块化，调用方只引入所需合同。文件、匿名
VM 和线程 owner 在 Platform；file/shared/anonymous storage mapping 的唯一用户级 owner 在 Mem。

## 3. 系统事实、进程事实与策略必须分开

CPU API 有三类不同生命周期，不能混用：

1. `cpu_features()` 返回当前进程可安全执行的无顺序具名 facts。x86 同时检查 CPU 位和 OS
   扩展寄存器保存能力；探测 TU 必须保持 baseline ISA。Platform 不提供 `best target`：SIMD、
   Container 或用户 kernel 各自由唯一实现 owner 把 facts 与其生成 TU 的完整 requirements 匹配。
   构建系统把探测 TU 独立放入关闭 IPO 的 private object target，其余 platform 源仍可接受父工程
   LTO，不能用 target-wide 关 LTO 换取探测安全。
2. `cache_info()` / `hardware_concurrency()` 是缓存的粗粒度系统事实，适合通用容量提示与诊断。cache
   初始化优先聚合 OS 公布的 system cache hierarchy；异构 class 取各已知 level 的保守非零下界，只有
   OS view 整体不可用才退回 architecture probe。不能从首次调用时的动态 affinity/cpuset view 生成终身
   缓存，否则后续 placement 扩大时可能高估。后者也不等于 affinity/cpuset placement 或 cgroup/job
   CPU 配额。Apple 官方全局 cache key 表示最低性能 class，不能拿来调整 performance-core working set。
3. `available_cpu_cache_classes()` 是动态异构 cache-class 视图，保留 class 的 rank/kind、L1/L2/L3
   容量、line size、共享宽度和当前 process-visible CPU/core 数。它按可信 rank 从强到弱返回，同 rank
   但 cache profile 不同也不合并；未知字段为 0。它提供 tiling/sharding 的事实，但不选择 SIMD target
   或 worker policy。
4. `available_cpu_count/ids/info/topology()` 是动态进程视图，必须尊重 affinity、cpuset/cgroup 和
   Windows CPU Sets。返回 id 可能稀疏，topology domain 只在一次 snapshot 内有效；调用方不得用
   `[0, count)` 或旧 snapshot 推导当前 placement。
5. `process_cpu_capacity()` 把 placement logical CPU 数与可选 hard bandwidth quota 保持为两个字段。
   Linux 解析 cgroup namespace/mount root 并折叠全部可见祖先 quota；shares/weight 不具 hard ceiling，
   不能换算成虚假 CPU 数。Windows 的公开查询只暴露 immediate Job，而嵌套 CPU rate 相对父 Job，缺少
   ancestor handles 时无法证明最终有效上限，因而保持 absent。`parallelism_ceiling()` 是 O(1) 的保守
   事实推导，不拥有 Sched/engine policy。
6. `available_cpu_frequencies()` 只转录 OS 对当前 available id 公布的 hardware/policy/current Hz。
   current 是瞬时值，0 是 unknown；频率不用于猜 P/E kind、ISA availability 或跨微架构强弱。

`performance_rank` 只采用 OS 提供的可信 CPU-id 映射。没有证据时返回 unknown，不按频率、
编号或营销名称猜 P/E 核。`CpuOrder` 只对完整可用集合排序，不偷偷删核；“只用强核”、
worker 数、轮转、超订和 NUMA-aware stealing 都是 sched 或应用策略。

x86 子扩展只有在其父扩展与所需 OS state 同时成立时才进入 set；例如 AVX-512 子项不能脱离
AVX-512F/ZMM，`kX86Avx10_2_512` 同时编码 version、512-bit width 与 ZMM 可用性。macOS x86 的
AVX-512 还采用系统规定的 `hw.optional.avx512f` 处理 lazy XCR0 与 Rosetta 边界。ARM/RISC-V 只使用
各 OS 的稳定 capability ABI；未知键或旧 OS 返回 absent，不依据 CPU 名称、编译器宏或参考库 target
表猜测。

系统级 AMX/SME/SVE/RVV feature 与调用线程实际可用状态也必须分开。`this_thread::isa_state()`
返回动态 vector width/permission，`request_amx_tile_permission()` 是显式线程配置；两者都不选择 SIMD
kernel，也不把失败静默改成较弱语义。临时 affinity 调整通过 query + set 保存和恢复调用线程的
真实 mask，不能用 process-wide available set 猜原值。

### Host resource snapshot 与 entropy

`host_memory_info()` 是动态、fallible、刻意不缓存的 planning snapshot。Linux cgroup 位置来自
`/proc/self/mountinfo` 与 `/proc/self/cgroup`，不能硬编码 mount；从 leaf 到 visible mount root 的每个
hard memory limit/current 都参与保守下界。物理量在成功时存在；
commit/swap/process-or-container limit 使用 optional 表达 OS 是否提供可信值，合法 0 不充当 sentinel。
`effective_available_bytes()` 取已知约束的保守下界，但不预留资源，也不把 UMA device budget 与 host
budget 重复相加。真正的 allocation failure 仍必须由 Mem/Compute consumer 处理。

`numa_node_memory_info(node)` 与上述 whole-process snapshot 是互补事实。它只接受同一
`numa_nodes()` 稀疏 id space 中的 node，并返回该 node 当前的物理可用量，以及 OS 能证明时的物理
总量。Linux 从 node meminfo 组合 free、file-backed 与 reclaimable slab；该 sysfs 没有全局
`MemAvailable` 的 watermark 修正，因此这是公开局限，不冒充 reservation。多节点 Windows 的公开 API
只给 free+zero pages，total 保持 absent；逻辑单节点 Windows/macOS 复用 host snapshot。cgroup、Job、
commit 与 device/UMA budget 都是全局约束，planner 必须只组合一次，Platform 不把它们平均或复制到
每个 node。

Windows 的公开 Job query 只有 immediate Job，父 Job 可能施加更紧的 process/job commit limit，并把
同级进程占用计入父级预算。没有 ancestor handles 时，Platform 不把 immediate 值冒充 effective budget，
所以 `process_limit_bytes/process_available_bytes` 在 Windows 当前保持 absent；系统 physical/commit facts
仍正常返回。将来若 owner 显式持有完整 Job chain，可另行准入带 handle 的查询形状。

`secure_random()` 是无隐藏 PRNG state 的 OS entropy mechanism，成本与输出 bytes/native chunk 成
正比；partial read 与 interrupt 在实现内完成，只有完整填充或错误两种外部结果。它不替代 Util 的
可复现统计 RNG。environment 查询和 mutation 与 filesystem source 分离：环境属于 host process
state，unset 与 empty 有不同结果，mutation 只适合已同步的启动/重配控制面。

## 4. Thread 只表达 OS 线程机制

`Thread` 是 move-only OS handle，公开类型不包含 `std::thread` 或 `std::function`。入口使用
function pointer + context 服务 allocation-minimal 的内部 runtime，C++ 用户使用拥有
capturing/move-only callable 的重载。两者都是当前 C++ 产品面，不暗示跨语言 ABI。

`ThreadOpts` 只承载 OS 在创建边界可执行的配置：name、stack size、单 CPU/CPU 集合 affinity、
portable priority 和配置策略。view 参数在 `spawn()` 返回前同步复制；用户无需为短期配置
制造长期 owner。

- `kBestEffort` 是可移植默认值：不支持或无权限的附加配置不能阻止线程运行。
- `kRequired` 是强合同：配置必须在用户 entry 前完成，失败则返回 `Status` 且 entry 不运行。
- `this_thread::set_*` 用于已进入线程后的逐项控制，失败直接返回共享 `Status`。
- 析构仍 joinable 是生命周期 bug，fail-fast；platform 不把忘记 join 静默变成 detach。

priority 是显式 portable scheduling/quality hint，而不是硬核型选择。Windows 的低两档同时启用
execution-speed throttling（EcoQoS），高两档显式关闭该 throttling（HighQoS），`kNormal` 把 QoS
控制交还系统；macOS 映射原生 QoS，Linux 映射当前线程 nice。`ThreadOpts::priority` 为空时完全不做
这些调用，因此嵌入宿主的策略不会被默认覆盖。这个映射让 performance-oriented 推理 worker 真正
表达其需求，但仍不承诺 OS 必须把它放到某颗 P-core。

大量短任务、worker placement 和结构化并发归 `sched`。platform 不提供第二个线程池，也不把
“performance first”偷设为所有用户的默认行为。macOS 没有公开的硬 CPU pin，必须诚实返回
不支持；强核软偏好使用 OS QoS/priority，而不是伪造 affinity。

进程策略与线程策略是两个真实作用域，不合并成一个含糊 enum。`set_process_priority` 是应用启动期
显式机制：Windows 同时设置 process priority class 与 execution-speed QoS，macOS 使用 process nice，
Linux 的 `setpriority(PRIO_PROCESS, 0, ...)` 实际作用于调用线程并由随后创建的线程继承，因此完整
进程效果要求在创建 workers 前调用。`kNormal` 表示回到 HYLE 的普通/system-managed 档，不是保存并
恢复任意宿主原始 native policy 的事务。持久 worker 的逐线程配置仍使用 `ThreadPriority`；提升档位
可能因权限失败，库不会静默降级，也有意不提供可能饿死系统工作的 realtime 档。

## 5. 等待层：最小原语与可直接使用的 Parker

地址等待是 futex、`WaitOnAddress` 与 Apple address-wait 的最小公分母。它服务已经拥有原子
状态字的同步原语，允许伪唤醒，调用方必须循环检查谓词。它位于 `detail`，避免普通用户把
一次 wake 错当成业务条件成立。

`Parker` 在其上实现经过验证的三态 token 协议：unpark 可以先于 park，重复 unpark 合并成
一个 token。它适合 worker 睡眠与外部线程唤醒，但不是计数 semaphore、condition variable
或 fiber-aware wait；这些带队列/取消/期限语义的类型归 `sched`。timed park 是 single-shot，
调用方传 `time::Duration`，在自己的 deadline loop 中重新检查条件并计算剩余时间；非正 duration
是纯用户态非阻塞 token 检查。一个 Parker 只允许一个等待消费者，多个线程同时 park 不在合同内。

## 6. VM、mem、io 与 compute 的边界

匿名虚拟内存位于 platform，因为 reserve/commit/protect/decommit/release 是 OS 地址空间
机制。普通小对象和通用容器 backing 归 `mem` allocator，不应直接 mmap/VirtualAlloc。

- `VmRegion::allocate_rw`：常用的一步可读写 mapping；不要求用户为 benchmark 人工拆成
  reserve + commit。
- `VmRegion::reserve`：真正需要稳定地址和按段提交的 arena、JIT 或大模型加载器。
- `reserve_guarded`：显式 guard page，供 fiber stack 等越界必须尽早失败的资源。
- 裸 `vm_*`：供实现自有 owner 的基础设施；普通用户优先 RAII，避免 base/size 配错。

普通顺序文件 I/O 用 Platform `FileHandle`；caller-synchronized 的 blocking explicit-offset 访问用
`RandomAccessFile`；并发/异步 read/write、注册文件、direct/vectored I/O 和 completion 归 `io`。
模型权重、索引和 shared storage 的 mapping、advice、populate、residency、partial release 与 range
flush 由 `mem_vm` 的 `FileMapping/SharedMapping` 唯一公开拥有；其 `VirtualMapping` 名字直接 alias
Platform `VmRegion`，不是第二个 owner。`FileMapping::map` 消费 Platform `FileHandle`，因此实现可
直接复用 native handle。这个边界避免两套 mode/advice/lock enum、两份实现和无法判断该选谁的 API。

Platform 仍提供 `vm_reserve/commit/protect/advise/lock` 等 raw anonymous-address mechanisms 和
`VmRegion`，供 Mem 与自定义 arena/JIT owner 组合；它不提供 file/shared mapping 生命周期。
mapping 的访问策略、同步 populate、monotonic residency prefix、惰性 hole/fragment metadata 等合同见
[Mem 设计](mem-design.md)。device VRAM、pinned host memory、coherence 与 GPU completion 归
`compute`，不能用 `VmRegion` 或 `mem::AllocatorRef` 抹掉 backend 生命周期。

NUMA preferred 与 huge-page advice 是策略提示，返回值必须让用户看见是否 applied、not
needed 或 unavailable。Linux bind 是严格节点集合；interleave/selected interleave 表示 OS 已
接受条带 policy，Windows 和 Linux 都可能在目标节点内存压力下回退，不能把它宣传成物理页
驻留证明。软 preferred 降级后 mapping 仍然有效。

公共 NUMA 面同时提供拓扑、运行期能力与分配机制：`numa_nodes()` 返回稳定的实际 node id
列表，id 可能稀疏，不能从 `numa_node_count()` 推导 `[0,count)`；`numa_capabilities()` 区分
topology、preferred reserve、existing-range bind/interleave 和 selected interleave。用户代码应按
能力或操作返回的 `Status` 决策，不直接围绕 `HYLE_OS_*` 编译分叉；后者只用于 platform 内部选择
系统实现。

节点级容量同样沿这份 id authority 查询：调用方遍历 `numa_nodes()` 后逐项调用
`numa_node_memory_info()`，不需要另一份 owning vector 或固定大小表。它是初始化/低频重规划的动态
控制面，返回值不能证明随后 allocation 成功，也不负责 rank、GPU、利用率或 eviction policy。

推理 worker 若要按 NUMA node 分组，不需要第二套固定大小 topology：枚举动态
`available_cpu_ids()`，对每个 id 调 `numa_node_of_cpu()`，再把每组构造成 `CpuAffinity`；arena
则用对应 node 的 preferred/bind 或 selected interleave。这个 O(CPUs) 过程属于初始化/重配，
具体 worker 数、跨 node stealing 和 tensor/KV placement 仍由 sched/model owner 决定。

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

platform filesystem factor 覆盖真实基础库常用需求：stat/存在性、目录与文件变更、权限的可移植
子集、link、目录 visitor、cwd/temp/home、同步 open/read/write 和原子临时文件。进程环境属于 host
factor，mapping 属于 Mem。

设计约束如下：

- Windows 路径在边界从 UTF-8 转宽字符并使用 Unicode API；公开面不泄漏 `wchar_t`。
- 查询使用 `Result<T>`，所以合法的 0、空文件、0000 权限不会与失败混淆。
- 变长字符串使用 buffer-fill：返回完整长度，允许先 count-only，再由调用方选择 allocator。
- 目录遍历使用 callback + context，不构造 owning entry；普通路径无逐 entry heap allocation，
  但无界路径/编码 scratch 允许 checked allocation 并以 `ResourceExhausted` 失败。C++ lambda 只是
  inline trampoline。compiled `directory_iterate` 是 noexcept、allocation-free core，直接 callback
  不得抛；`for_each_entry` 在 consumer TU 捕获 callback exception、让 native iterator 先关闭，再
  原样重抛，并保留 callable 的 const qualification。
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
- recursive mkdir 与 recursive remove 同样采用 stack-first 可增长 path scratch；普通 4096-byte 内
  路径不分配，Windows extended path 不再被 HYLE 固定 ceiling 提前拒绝，最终仍以 native path 和
  component 限制为准。
- `copy_file` 只把 regular-file copy 作为成功合同。POSIX 在 open 后按 descriptor 分类并按 EOF
  驱动，不能依赖 `st_size`；若 non-regular destination 在 open 阶段先以 `ENXIO` 等失败则保留
  native error。Windows 的 reparse follow/replace 行为由 `CopyFileW` 定义，不伪造 POSIX 等价。
- `FileHandle::read/write` 的短传输是成功；普通完整传输使用零分配的
  `read_exact/write_all`，共享 cursor 的位置使用 `seek/position`。durability 需要显式 `sync`，
  需要观察关闭失败时显式 `close`。
- `FileHandle::duplicate()` 创建同一 opened object 的第二个 RAII owner，而不是重新 open path。
  duplicate 在 POSIX/Windows 都共享 native open-file cursor，并可能共享/关联锁状态；它解决 owner
  lifetime，不提供独立 position，也不能当作两个并发 lock contender。`FileHandle::info()` 直接查询
  opened identity，rename/unlink 后仍描述原对象；因为 handle 已经解析 link，`is_symlink` 为 false。
- `RandomAccessFile` 是一字、move-only、caller-synchronized 的 blocking explicit-offset owner。
  `read_at/write_at` 保留短传输，`read_exact_at/write_all_at` 完成传输并检查 offset 加法；它不公开
  seek/position，也不接受 append。POSIX 使用 `pread/pwrite`；Windows offset I/O 可能更新底层 native
  cursor，但该状态在此类型中不可观察。多线程并发、async、direct 或 vectored 访问仍归 `io::File`，
  不为迎合测试在 owner 内加入 mutex、event 或 heap state。
- `FileHandle::size/resize/preallocate` 作用于已经打开的文件对象，不重新解析 path；这在目标名字被
  原子替换时比 path API 更诚实。`resize` 只改变 logical EOF，扩容可以是 sparse；`preallocate`
  只在文件小于 minimum 时请求 backing 并增长 EOF，不缩小、不去稀疏化既有大文件，也不承诺连续
  extent。预留与 EOF 更新在所有 OS 上都不是统一事务，后一步失败可能留下不可见的已分配空间。
- `lock/try_lock/unlock` 是 cooperating processes/handles 的 whole-file shared/exclusive coordination。
  POSIX `flock` 是 advisory，Windows full-range `LockFileEx` 对冲突 I/O 更强；公共合同取两者交集，
  不是线程 mutex、record/range lock 或 distributed-filesystem consensus。一个 `FileHandle` 的并发调用
  仍由 caller 同步，close/析构依 OS 释放锁，HYLE 不为追踪锁状态增加常态 metadata。
- Windows `FileHandle` 以 `FILE_SHARE_DELETE` 打开，使 rename/delete 与 POSIX 一样只替换目录项、
  已有 owner 继续持有文件对象；消费 handle 的 Mem `FileMapping` 保留该文件对象生命周期。这也是
  原子模型/缓存更新的必要合同。

platform 不拥有 path 字符串算法、glob、watcher、高层 directory object、async stream 或虚拟
文件系统。纯词法 path 操作属于 container/string，变化通知和异步数据流属于 io。

## 8. 错误、所有权与交付边界

platform 可以依赖最底层 error。资源创建和有值查询返回 `Result<T>`，状态变更返回
`Status`；错误保存 HYLE code、operation 和原生系统诊断。成功热路径不分配，模块不维护
私有 TLS last-error，也不让 errno/GetLastError 越过调用边界后才被读取。

POSIX fs、VM 与 NUMA 的 generic errno code 必须来自 error 的
`status_from_errno`；Win32 generic OS code 必须来自 error 的 `status_from_win32`，COM/WinRT
返回值必须来自 `status_from_hresult`。Windows CRT 直接返回 errno 值的 API（例如 `_wputenv_s`）
仍走 errno 入口。只有系统调用
合同更具体时才由 platform 局部覆盖（例如路径分量不是目录）。CUDA/Vulkan 依赖 SDK 类型与
SDK-specific recovery，仍由 compute owner 保留 translator；不能硬套 errno，也不能让 platform
再次复制通用 Win32 taxonomy/message table。

C++ 资源 owner（`Thread`、`VmRegion`、`FileHandle`、`RandomAccessFile`、`DynLib`）均为 move-only RAII；借出的
pointer、symbol 或 native handle 不得活过 owner。需要转移回收责任时必须使用显式 release/close
语义。Mem `FileMapping::map(FileHandle, ...)` 消费已验证的 file owner，避免第二次 path lookup；
mapping 的 view 与 access-mode 合同由 Mem 拥有。
`DynLib::get` 在每个平台都取得/pin 一份 loader reference，避免借用 handle 与并发 unload 竞争。

platform 当前只交付 C++23 source/static/shared。BUILD-03 已删除早期 C-syntax schema、bridge、
动态导出和 fixture；Native FFI 为 `N/A`，没有 C ABI layout 或符号兼容承诺。HYLE 模块和同工具链
用户直接使用 typed facts、`Status`/`Result<T>` 与 move-only RAII owner，不绕 opaque C handle。
将来只有已命名的 foreign-runtime consumer 通过
[Native FFI 准入策略](native-ffi-policy.md)后，才按任务设计窄的 runtime 类型擦除 adapter；不为
C++ API parity 预埋 POD、callback 或资源句柄 schema。

C++ 在同一个 buffer-fill/allocation-minimal core 上提供显式便利形状：`Thread::spawn(callable,
opts)` 拥有 capturing/move-only callable；factor-owned 的 `cpu_convenience.h`、
`thread_convenience.h`、`fs_convenience.h` 与 `environment_convenience.h` 用 header-only overload
返回 owning `std::vector/std::string`。adapter 负责 count/fill、动态视图重试与 STL 分配异常到
`Status` 的转换，也不让核心 header 被迫引入 owning 类型或其它 factor。普通 C++ 不重复样板，
allocator/arena 与热调用者仍能控制每次分配。

## 9. 性能架构

platform 的 SOTA 标准不是“比系统调用快”，而是同合同下不引入可避免的额外工作：

- `cpu_features/cache/page_size` 等稳定事实探测一次后 O(1) 读取；coarse cache 优先一次聚合 system-wide
  OS cache hierarchy，只有 OS facts 整体不可用才退回 architecture probe。Windows 为
  O(relation bytes)，Linux 为 O(system CPUs × exposed cache indices)，且都只发生一次。
- 动态 CPU availability/topology 保持 O(N + R) OS 查询，不能为 O(1) 缓存而返回过时 cpuset；只用于
  初始化/重配控制面，不进入每 task 热路径。Windows 把 OS relation 一次索引到 process-available
  CPU，避免对每颗 CPU 重扫全部关系；Linux 从同一次 affinity view 读取 per-CPU sysfs topology，
  不为每个字段重复枚举 mask。聚合 distinct domain 使用有界 scratch，不引入树或常态 metadata。
- cache-class snapshot 在 Windows/Linux 复用同一次 process-available CPU view，再读取 OS cache
  relations/sysfs profile，并按 profile 排序聚合：O(P log P + relation bytes) 或 O(P*C + P log P)。
  常见不超过 256 CPU 的主机使用 stack scratch，更大 view 允许一次可失败 allocation；macOS 直接枚举
  `hw.nperflevels`，O(classes)。既有 `cache_info()` 仍是初始化一次、随后 O(1)。
- process capacity 的 placement 查询为既有 O(N) 动态视图；Linux hard quota 只沿 cgroup 可见祖先深度
  O(depth)。Windows 不为一个无法证明的嵌套 Job 有效值增加伪查询。frequency discovery 为
  O(available CPUs)，属于初始化/诊断，不缓存 DVFS observation，也不进入 task dispatch。
- host-memory snapshot 在 Linux 是 O(1 + cgroup visible depth) 个有界 OS/file 查询；Windows 不做无法
  证明有效值的 Job 查询。单次 node-memory snapshot 在 Windows/macOS 为 O(1) 有界 native 查询，在
  Linux 为 O(node meminfo bytes) 的一次顺序解析；它不缓存、不分配结果集合，也不进入 allocation
  热路径。`secure_random` 是
  O(bytes/native chunk)，不维护 TLS PRNG state。
- Thread 默认路径不为未请求的强保证增加握手/refcount；required 才支付同步配置成本。
- Windows 显式 priority 固定调用一次 QoS control 和一次 native priority control；未请求 priority
  仍是零调用。两者都是线程创建/重配冷路径，不混入 Parker 或 task dispatch。
- 显式 process priority 同样只在调用时做固定次数 native control；默认路线没有初始化、查询或
  全局状态。Windows 的 QoS 与 priority class 是两次 OS 调用，不能为追求一个点值假装成事务。
- callable 线程入口只在用户选择 owning 便利重载时多一次冷路径 heap 状态分配；OS thread 创建仍是
  主成本，runtime/sched 继续使用 allocation-minimal `entry + void*` core。
- `wait_on_address` 直接落 OS 原语，Parker 只加一个原子状态机，不增 mutex/heap allocation。
- `allocate_rw` 使用最少 OS mapping 操作；staged reserve+commit 的额外 syscall 是不同合同，
  不能拿来宣称普通分配性能。
- directory visitor 常见路径不逐项 stat、不分配；只有文件系统不给类型时才补查询；C++ 异常桥只在
  callback 真正抛出时进入冷失败路，正常每项仍只有原有一次 callback。
- VM、DynLib、FileHandle 与 RandomAccessFile 的成功生命周期无异常和隐藏堆分配；错误消息只在
  冷失败路构造。
- `FileHandle::size/resize/lock/unlock` 是固定次数 OS control；`try_lock` 只把明确竞争映射为
  `false`。`preallocate` 的物理工作由 requested growth 与 filesystem 决定，阻塞 lock 的时间由
  contention 决定，二者都不能用纳秒点值冒充算法 SOTA。
- handle duplicate/info 和一次 random-access transfer 各只做一个对应 native control；exact/all 仅在
  native short transfer 后继续，没有 payload copy、隐式 seek、mutex 或 allocation。
- POSIX restart-safe filesystem controls 的无信号成功路径仍只有一次 syscall；`EINTR` 才通过同一
  private helper 重启。close-like 或可能已经产生外部副作用的 path mutation 明确不重试。

基准必须比较相同语义。host primary 是一次 process-available topology + host-memory snapshot，
只与完成同一动态事实合同的 direct-OS/peer route 比较；cached scalar getter 只能作为独立 holdout。
thread primary 是持久 worker 的 Parker request/response handoff，对等 peer 必须同样保留 token、
使用同样两颗物理核和恢复 affinity。filesystem primary 是 warmed regular-file open + 完整顺序 read +
close，setup、bytes 与 teardown 都计费。mapping/residency 性能证据迁到 Mem。raw OS 和微基准只可
解释机制，不能替代消费者结论。点值、样本和平台输赢只维护在
[`../performance/benchmarks-platform.md`](../implementation/evidence/performance/benchmarks-platform.md)。

## 10. 平台差异必须显式保留

统一 API 不等于虚构相同 OS：

- Windows affinity 集合的硬 mask 受 processor group 限制；跨组 placement 由上层拆 worker。
  `numa_nodes()` 也只从当前 process-available CPU ids 归纳 node，不把全机不可达
  node 当成可用 placement。多 node 时 `GetNumaAvailableMemoryNodeEx` 只证明 free+zero pages，不能
  从它推导 per-node total；单 node 才复用完整 host snapshot。
- Windows processor-power records 提供 MHz 级 OS observation，但只在返回记录编号与 available CPU id
  一致时采用。公开 Job 查询只返回 immediate Job，无法折叠嵌套的相对 CPU rate，因此 Windows quota
  当前始终 absent；CPU Set placement 仍独立、可信。
- macOS 无公开硬 pin；priority 是 QoS 软偏好。逐 logical CPU 的 P/E 映射未知时保持 unknown；
  `hw.perflevelN.*` 仍可诚实提供 class-level cache facts。全局 `hw.l*cache*` 是最低性能 class，
  `cache_info()` 因而不能替代 `available_cpu_cache_classes()` 做异构 tuning。公开 frequency sysctl
  不存在时逐 CPU 频率保持 0，不能维护 Apple 型号表。
- macOS 的公开 address-wait 实现依赖 14.4 起可用的 `os_sync_wait_on_address*`。构建 platform
  component 时 deployment target 必须 ≥14.4；HYLE 顶层原生 macOS 构建未显式指定时默认 14.4，
  被父工程嵌入时若父工程选择更低 target 则 configure 失败，不得静默产出会在旧系统缺符号的库。
- Linux CPU placement 受调用线程 affinity/cpuset 影响，新线程继承该视图；CPU quota 与 memory budget
  还必须解析 cgroup namespace mount 并折叠可见祖先。NUMA 只采用 mempolicy、process status 或 cgroup
  cpuset 的 process-allowed view；这些都失败时不能退回 system-online nodes 冒充可用。node meminfo
  没有 `MemAvailable` 的 watermark 估算，公开的 node available 采用 free + file-backed + reclaimable
  slab 并限制在 node total 内；全局 cgroup 限额由 planner 另行只合并一次。
- process priority 的 portable contract 是显式启动期五档 hint：Windows 同时调整 class/QoS，
  macOS 调整 process nice，Linux 调整调用线程 nice 并依赖随后创建的线程继承；不是跨 OS 完全相同
  的 scheduling guarantee。
- Linux 独有 `MLOCK_ONFAULT`；macOS/Windows 对该 lock mode 返回 `Unimplemented`。允许调整锁页
  限额时，POSIX 只提升 soft `RLIMIT_MEMLOCK` 到既有 hard limit 内，Windows 调整当前进程
  working-set minimum/maximum；两者都不会在 unlock 时自动恢复。
- POSIX permission bits 与 Windows readonly attribute 只能提供一个诚实的可移植子集。
- POSIX `pread/pwrite` 不移动 file offset；Windows 在同步 handle 上带 `OVERLAPPED` offset 的
  `ReadFile/WriteFile` 仍可能更新 native file pointer。`RandomAccessFile` 通过不公开 cursor 给出共同
  语义，而不是承诺 Windows 不移动它；需要可并发 completion 的调用者使用 IO owner。
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
| subprocess lifecycle、sandbox 与 IPC channel | 独立未来模块或真实消费者驱动 | 当前进程启动 priority 已是窄机制；spawn/wait/kill/quoting/sandbox 是另一套生命周期 |

最后一行是扩展纪律：platform 可以很强，但不能成为“所有和 OS 有关东西”的杂物间。只有
多个基础库用户反复需要、能形成跨平台诚实合同、且归不到已有高层生命周期 owner 的机制，
才进入 platform 公共面。

## 12. 维护闸门

改变 platform 公共形状时必须同时回答：

1. 这是可复用 OS 机制，还是某个上层 workload 的策略？
2. 系统事实、动态进程事实和用户偏好是否被不同类型表达？
3. 当前是否仍保持 `Native FFI=N/A`；若出现已命名 foreign-runtime 任务，是否先按全局准入策略
   单独立项，而不是在 platform 修改中顺手恢复旧 C schema？
4. 新调用是否给默认热路径增加 syscall、分配、握手或间接层？
5. Windows/Linux/macOS 的能力差异是显式结果，还是被静默吞掉？
6. 是否有真实内部或外部消费者，而非只为 benchmark 制造形状？
7. 性能证据是否完整对齐作用对象、logical/physical effect、blocking、cleanup 与 failure？path 与
   handle、sparse resize 与 preallocation、try 与 blocking 等不同合同必须标 `NOT_COMPARABLE`，不得
   进行苹果对橘子式排名。

稳定设计更新本页；用户选择和坑更新 guide；实测更新 performance；一次性验收和提交记录
更新 implementation task/evidence。
