# platform 实测铁证

> 本页是 [performance 总账](./README.md) 的 **platform 模块**详节。  
> 源：`bench/platform/platform_bench.cc` → `hyle_platform_bench`。  
> 方法：`scripts/bench-stat.py`，信任 **[sign] 配对符号检验**。  
> **2026-07-24 当前 SOTA 修正（实现 `75f3a45`）**：Win MSVC 19.50、WSL GCC 14.2、
> Apple M4 AppleClang 21 的 zero-timeout、MappedFile、FileHandle 与持久随机访问均为
> **n=15**；其它 OS 阻塞段沿用下文标明的新鲜证书。
> 旧 2026-06-30 双平台证书作废（源码 7 月大幅改过）。  
> sched 同步/fiber 不在本页 → [benchmarks-sched.md](./benchmarks-sched.md)。

## 墙钟陷阱（本轮血泪，流程已改）

| 段 | 问题 | 处置 |
|---|---|---|
| 旧 `park_timeout(0)` × 100k × 多 epoch | 旧实现每次真进内核（Win ~µs，Linux futex ETIMEDOUT ~数十 µs）；WSL 单遍曾 **70s+** | `75f3a45` 改纯用户态 load，恢复 100k iterations；三平台 n=15 见总表 |
| raw-wait 乒乓 `minEpochIterations(30)` | Linux ~20µs/round × 20k × 30 epoch → 单段数分钟 | minEpoch **30→5** |
| 双份阻塞乒乓（vs CV + vs raw） | 同形状跑两遍 | 保留（语义不同），接受 WSL n=5 |

**纪律**：OS 主导段必须先估 `iters × 单次 syscall × epoch × N`；亚 µs 算法段的 n=15 不能照搬到 futex 乒乓。

## 三平台总表（中位数 ns/op，除非注明）

| 操作 | Win n=15 | Lin n=5 | M4 n=15 | 判定（对标） |
|---|---:|---:|---:|---|
| **Parker 真阻塞乒乓** vs std CV | 341 / 638 → **1.75x WIN** | 22.5k / 23.5k → **1.05x 平** | 2053 / 2130 → **sign WIN 1.05x** | ⚠️ **平台翻转**：Win 大赢；Lin 平；M4 微赢 |
| **Parker** vs raw wait 地板 | 356 / 310 → TIE | 22.2k / 21.2k → **0.95x 平** | 2035 / 2062 → **sign 近平** | ➖ 三态机税在噪声/个位数 % |
| **Parker 快路径** unpark+park | 9.71（floor 4.68） | 10.64（floor 4.63） | 1.99（floor 0.90） | ➖ 只报绝对值；atomic **非公平对手** |
| **Thread spawn+join** vs std::thread | 43.4k / 44.4k → **TIE** | 33.56k / 33.56k → **TIE** | 10.6k / 10.7k → **TIE** | ➖ 平（OS 主导，证零封装税） |
| **Thread owning callable** vs raw `entry+ctx` core | 40.529k / 40.081k → **TIE** | 32.787k / 33.511k → **TIE** | 10.510k / 10.538k → **TIE** | ➖ n=15 三平台平；便利状态分配淹没于 OS create/join |
| **directory_iterate** vs std::filesystem | 40.23k / 39.76k → **TIE 0.981x**（n=15） | 7.05k / 14.97k → **WIN 2.135x**（n=15） | 16.46k / 18.25k → **WIN 1.109x**（n=15） | ⚠️ 平台差：Win 平；Lin/M4 赢 |
| **vm_allocate_rw+release** vs raw 一步 | 885 / 880 → **TIE**（n=5） | 860.9 / 853.8 → **TIE 0.978x**（n=5） | 309.8 / 305.0 → **TIE 0.982x**（n=5） | ➖ 三平台当前正常用户形状近 OS 地板 |
| **vm_reserve+commit+release** vs raw 一步 | 1144 / 877 → **sign LOSE 0.77x** | 1086.3 / 853.8 → **0.779x 方向**（n=5） | 720.5 / 305.0 → **spread LOSE 0.421x**（n=5） | ❌ **契约差**：两步 vs 一步；见下 |
| **vm_protect** vs raw | 446 / 428 → **sign LOSE 0.96x** | 390.0 / 383.6 → **TIE 0.957x**（n=5） | 675.7 / 693.8 → **paired TIE 0.996x**（n=5） | ➖ 近平；小封装税 |
| **MappedFile** vs ifstream（64 KiB 打开+全扫） | 45.048k / 79.425k → **WIN 1.763x** | 11.479k / **8.723k → LOSE 0.760x** | 13.314k / 20.111k → **WIN 1.510x** | ⚠️ **选型翻转**：Win/M4 赢；Linux 小文件顺扫用 buffered read |
| **MappedFile** vs 同合同 raw mmap | 45.048k / 44.766k → **TIE 0.992x** | 11.479k / 11.219k → **sign LOSE 0.982x** | 13.314k / 13.146k → **TIE 0.992x** | ➖ 同样 UTF/path、size/range 验证后距地板 ≤1.8% |
| **FileHandle** vs raw open/read/close | 34.414k / 35.006k → **TIE 1.006x** | 7.641k / 7.486k → **sign LOSE 0.976x** | 11.904k / 11.620k → **sign LOSE 0.978x** | ➖ 安全 Result/RAII 路径距裸地板 ≤2.4%；极端路径可借 `get()` |
| **持久 MappedFile 随机页访问** vs ifstream seek/read | 2.309 / 2585.654 → **WIN 1125x** | 2.532 / 420.476 → **WIN 168x** | 0.647 / 1527.861 → **WIN 2355x** | ✅ 16 MiB warm page-cache、1024 个非顺序 page probe；真实 mapping 甜点 |
| **park_timeout token 快路径** | 9.215 ns | 9.818 ns | 1.850 ns | ➖ token consume 的 acquire RMW 绝对值 |
| **park_timeout(0) 无 token 轮询** | **1.690 ns** | **0.564 ns** | **1.346 ns** | ✅ 纯用户态 load；旧 3.7 µs / 59 µs / 3.7 ns 全部作废 |

### 2026-07-23 `Result<MappedFile>` 成功路径复证

DynLib/MappedFile 错误接口迁入共享 `Status` 后，benchmark 已改为用户实际写法：
`auto mapped = MappedFile::open(...); if (mapped) mapped->data()`。Windows MSVC 19.50
n=5 得到 HYLE **43.37 µs**、raw mapping **41.82 µs**、ifstream **58.50 µs**；
配对中位比分别 **0.955x** 与 **1.327x**。它与上表旧 n=15 的
43.7/42.0/62.3 µs 落在同一统计带，未观察到 `Result` 成功态附加税。本段只保留接口迁移
历史；raw 行当时缺少 HYLE 必做的 file-size/range validation，当前胜负以总表的
2026-07-24 同合同 n=15 证书为准。

WSL GCC 14 的完整 platform suite 仍不适合作为一次性命令；2026-07-24 已按实际场景拆段
复跑：VM/protect n=5、directory/MappedFile n=15。当前数字见总表与下方最终复证记录，
不再保留“新接口待复证”状态。

### 2026-07-23 VM `Result/Status` 成功路径复证

VM/NUMA 删除 TLS last-error 后，benchmark 使用当前用户形状：reserve 取
`Result<void*>`，commit/protect/release 取 `Status`。Windows MSVC 19.50 的定向 smoke
run 得到一页 reserve+commit+release：raw **897.42 ns**、HYLE **1167.11 ns**（0.769x）；
RW↔RO protect pair：raw **438.51 ns**、HYLE **459.52 ns**（0.954x）。它们分别落在旧证书
1144/877 ns 与 446/428 ns 的同一带内，没有观察到结构化成功态带来的额外回退；两步
VM 的合同差和 protect 的约 5% 薄封装税仍按下节原判定。该单进程 run 只作接口迁移
回归哨兵；Linux/macOS 当前接口已由下方 2026-07-24 定向证书补齐。

### 2026-07-24 一步 RW mapping

面向普通“拿一块立即可写的页内存”场景增加 `vm_allocate_rw` /
`VmRegion::allocate_rw`；Windows 直接调用一次 `VirtualAlloc(MEM_RESERVE|MEM_COMMIT)`，
POSIX 直接调用一次 `mmap(PROT_READ|PROT_WRITE)`。Windows MSVC 19.50 Release 的
5 进程配对中位数为 raw **879.84 ns**、HYLE **885.24 ns**，HYLE/raw op/s 比
**0.987**（每轮 0.985–0.994）；按 n=5 纪律判统计平。旧 staged 形状为
**1130.69 ns**，继续服务“先保留地址、以后分批 commit”的不同合同，不再被当成普通
分配形状的性能代表。Linux/macOS 后续定向复证同样把一步形状判为统计平，见下。

### 2026-07-24 filesystem `Result/Status` 成功路径

目录遍历现在返回 `Result<bool>`：完整遍历为 true，visitor 正常提前停止为 false，OS 失败
才是 error。benchmark 按真实用户写法保留并检查 Result。Windows MSVC 19.50 Release
n=15 得到 HYLE **40.23 us**、std::filesystem **39.76 us**，配对 op/s 中位比
**0.981**（7/15 在 1 以上，sign p=1.0），spread/sign 都判统计平。它取代旧 Win
“sign 输 0.93x”记录；扫描的当前目录内容会影响绝对值，因此只采用同进程配对结论。
Linux/macOS 当前接口已在同日 n=15 复跑并更新总表。

### 2026-07-24 cwd / home / env 错误收敛

这些入口是进程级控制面，不是数据热路径：cwd/temp/home/env buffer-fill 必须至少读取并
复制 N 字节，因此时间 O(N) 是渐近下界。常见目录使用栈缓冲；仅超长 POSIX cwd/passwd
记录或 Windows UTF 转换暂存才几何增长到 O(N) 空间。Windows env 长度竞态最多重试 4 次，
不可能无界自旋；set/unset 仍是一次 CRT/OS 环境更新。

本轮不为它们新增纳秒 microbenchmark：数值主要测 OS、进程环境锁和字符串长度，且会把
接口再次拉向 benchmark 形状。性能验收采用复杂度/分配路径审查，并由 Release/ASan 的
真实 buffer-fill、UTF-8 round-trip、unset/empty 分离和长 HOME 场景覆盖。结构化 Status
只在失败时构造诊断，buffer-fill 成功 Result 是栈上值，不增加成功路堆分配。opt-in
`convenience.h` owning 重载明确购买一次 `std::string` 分配与第二次 fill，并在视图增长时最多
重试 4 次；这是初始化控制面形状，不由不需要 owner 的 core/FFI 用户承担。

### 2026-07-24 系统拓扑与进程 CPU 配额分离

`cpu_topology()` 保留一次探测、缓存读取的系统硬件视图；实际 worker 初始化新增
`available_cpu_count()` / `available_cpu_ids()`，直接读取 Linux 调用线程（新线程继承）的 affinity/cgroup/cpuset
或 Windows 进程 CPU Sets，避免
容器/受限进程按整机核数过度创建线程。查询时间为不可规避的 O(OS CPU 记录数)，输出为
O(可用 CPU 数)；常见 Windows（CPU Set 记录 ≤4 KiB、默认 set ≤64）与 Linux（CPU id
<1024）路径使用栈缓冲，无堆分配，超大机器才扩到 heap/dynamic CPU mask。

这是一条 runtime 初始化/重配控制面，且其正确合同要求观察运行期 affinity 变化，因此不做
会掩盖变化的全局 cache，也不新增以纳秒数塑造 API 的 microbenchmark。验收用 Release/ASan
真实约束测试：临时把进程 affinity 缩为一个现有 CPU 后查询必须立即返回 1，再恢复原 mask；
buffer-fill 另覆盖 count-only、截断返回完整长度和 id 唯一性。性能结论是与原生查询同阶、
buffer-fill 常见路径零堆分配，owning convenience 明确购买一个 `std::vector`；不把 OS 拓扑
枚举时间冒充 HYLE 算法性能。macOS 没有等价硬 affinity
API，fallback 的 AppleClang/arm64 构建、CPU invariant 与纯 C 路径已在 M4 真机通过；运行期
mask 变化测试按合同跳过，不把“不支持硬绑核”伪装成已验证能力。

依赖方回审采用同一进程视图：sched C 默认 worker 数、mem `SharedPool` 默认 shard 数和
container `StringBuf` 的 shard 档位都已从整机 `hardware_concurrency` 切换；显式用户配置不变。
Windows/Linux 约束到单 CPU 的测试分别断言 scheduler 默认 1 worker、SharedPool 默认 1 shard，
避免“platform API 正确但 HYLE 自己仍过度并发”的纸面收口。

### 2026-07-24 线程配置失败可观察

`ThreadOpts` 保留默认 best-effort，同时增加 opt-in required：新线程在调用用户 entry 前完成
name/affinity/priority；Windows 由创建者配置 thread handle 后打开入口 gate，POSIX 由新线程
配置并通过一次地址等待握手交回 `spawn()`；失败时 entry 不运行。
required 是线程创建控制面；name/priority/单 CPU 是 O(1) 加所请求的 OS 调用，N 个 CPU
集合需要 O(N) 填 mask（Linux 另需清零到最大 id 的原生 mask）。空间仍是一份
`ThreadStart` 尾随数组分配，不因集合再增加第二次 heap allocation。默认 spawn+join
基准路径没有等待、没有额外引用计数 RMW，仍是上表与
`std::thread` 统计平的同一形状；不能用 required 的同步保证去对比异步 best-effort。

易用性收口新增 `Thread::spawn(callable, opts)`：它拥有 capturing/move-only callable，并在
spawn/config 失败时销毁而不调用。该 opt-in 形状比 raw `entry + void*` core 多一份 callable
状态分配，因此用同一个真实 create→trivial entry→join 合同单独复证。Win/WSL/M4 n=15
分别为 callable/raw **40.529/40.081、32.787/33.511、10.510/10.538 µs**；配对 sign 均 TIE
（Win 8/15，WSL 7/15，M4 12/15，M4 p=0.035 仍未过 0.01 门槛）。callable 对
`std::thread` 也三平台 TIE。结论仅是 OS thread 创建尺度下无可测便利税；sched/runtime 仍走
allocation-minimal core，不把每 task 并发改造成 OS thread。

Linux affinity mask 改用动态 `CPU_ALLOC`，不再把 ≥1024 CPU 静默跳过；明显越界 id 在分配
mask 前先按 configured CPU 数拒绝，避免错误输入制造超大控制面分配。macOS priority 从
无区分度的 `SCHED_OTHER` priority 0 映射改为系统 QoS；硬 affinity required 明确返回
`Unimplemented`。`this_thread::set_*` 同样返回共享 Status，只有失败路构造诊断。

sched 的 worker placement 也已从 `affinity_cpu=i` 改为消费 `available_cpu_ids()` 的真实 id；
WSL 回归把调用线程临时限制到两个非 dense CPU，断言两个 worker 各得到正确的 singleton
mask。显式超额 worker 不伪造 CPU id，多出的 worker 保持 OS 可迁移。这些都是创建期 OS
控制路径，不新增纳秒 microbenchmark；正确性、复杂度与默认热路径无回归是对应验收。

集合 affinity 补齐的是实际隔离/分区能力，不是 benchmark 形状：marl 的 Affinity、Seastar
的 `cpu_set_t` thread attr、ggml 的 thread cpumask 都允许多个 CPU。HYLE 对外保留扁平
logical CPU id 数组，避免泄漏 `cpu_set_t`/`GROUP_AFFINITY`；Windows 对跨 processor-group
集合明确拒绝，与 marl 的硬 mask 限制一致，不用 CPU Sets 的软选择假装严格约束。单核仍是
便利形状；默认无 affinity 和单核路径都只有既有的一次启动载荷分配。

### 2026-07-24 强核优先是初始化控制面，不是任务热路径

`available_cpu_info()` 把 oneTBB `core_type` / hwloc `cpukinds` 同类的“拓扑事实”留在
platform；`SchedulerOptions::worker_cpu_order` 才是 sched 的 opt-in placement 策略。默认
`available_cpu_ids()` 仍直接走旧查询，不读性能能力、不排序。performance/efficiency-first
保留所有 CPU：Windows 扫一次 CPU Set 记录，Linux 对当前可用 id 读取 `cpu_capacity`，之后
只在已知 rank 确有差异或已知/未知混合时做 O(N log N) 排序；同质或全未知机器保持 OS 顺序。
常见 ≤256 CPU 使用栈数组，超大机器才有一次临时 heap allocation。

这条路径只在 scheduler 构造或用户显式查询时发生，不进入 task submit/steal/dispatch 热路，
因此新增纳秒 microbenchmark 会主要测 sysfs/Windows 拓扑枚举并扭曲 API。验收锁住可观察行为：
Windows 与 WSL/Linux 用单 worker performance-first scheduler，worker 实际运行 CPU 必须等于
排序结果的第一个 id；纯 C 同时覆盖 buffer-fill、无效 enum 与 fallback。Apple M4 因系统不
公开稳定的 logical-id→perflevel 映射而返回 unknown、保持 OS 顺序；想表达软强核偏好的用户
使用 `SchedulerOptions::worker_priority = ThreadPriority::kHigh`（QoS），不伪造硬 pin。
priority 只在线程创建时写入 `ThreadOpts`，不进入任务热路。性能结论是默认路径无回归，
opt-in 初始化路径与原生能力查询同阶；它不宣称改变 OS 最终调度质量。

### 2026-07-24 Linux / M4 当前接口最终复证

为避免完整 suite 的 OS 阻塞段再次占机十分钟以上，本轮按用户场景拆开运行；源码均为当前
`Result/Status` 接口。WSL GCC 14.2 与 Apple M4 AppleClang 21 的 VM/protect 各跑 n=5，
directory/MappedFile 各跑 n=15。n=5 的 sign 最低 p=0.0625，所以只在 spread 足够时判胜负，
其余写 TIE 或方向；n=15 使用配对 sign 结论。

- 一步 `vm_allocate_rw`：WSL HYLE/raw **860.9/853.8 ns，0.978x TIE**；M4
  **309.8/305.0 ns，0.982x TIE**。与 Windows 0.987x TIE 合并后，三平台普通形状均已证近
  OS 地板。
- 分阶段 reserve+commit：WSL **1086.3/853.8 ns，0.779x**，5/5 同方向但 n=5 不升级统计
  lose；M4 **720.5/305.0 ns，spread LOSE 0.421x**。这是明确购买延迟 commit 合同的第二次
  OS 操作，不是该用普通一步形状时的性能缺口。
- protect：WSL 配对 **0.957x TIE**；M4 配对 **0.996x TIE**。两者都是原生调用主导，未见
  接口迁移回退。
- directory：WSL **7.05/14.97 µs，2.135x WIN，15/15**；M4
  **16.46/18.25 µs，1.109x WIN，15/15**。
- 本段 MappedFile 数字已由同日稍后的同合同 raw baseline 与持久随机访问证书取代，见总表；
  保留结论仍是 Linux 64 KiB 顺扫优先 buffered read，持久/随机访问才是 mapping 的战场。

## 性能边界 / 需诚实标价

### 1. staged VM 对 raw 一步分配的旧“输”不是同合同对比

```text
raw Win:  VirtualAlloc(MEM_RESERVE|MEM_COMMIT)   // 1 syscall
HYLE:     vm_reserve(MEM_RESERVE) + vm_commit(MEM_COMMIT)  // 2 syscalls
raw POSIX: mmap(PROT_READ|WRITE)                 // 1
HYLE:      mmap(PROT_NONE) + mprotect/commit     // 2
```

`vm_reserve` + `vm_commit` 有意表达「先占地址空间、以后按需提交」（fiber 栈、稀疏大
arena），两次 OS 操作就是其合同成本。普通立即使用场景应选 `vm_allocate_rw`，它和 raw
基线同为一次 OS 操作且 Windows 已统计打平。保留旧行是为了标价 staged 合同，不再把它
列为普通 platform 分配的性能缺口。

### 2. `directory_iterate` 的平台差异

- Linux/M4 当前 n=15 证书仍赢（d_type/原生目录枚举 + visitor 无分配）。
- Win 底层都是 `FindNextFileW` 量级，HYLE 还付 UTF-16→UTF-8 visitor 转换；当前 n=15
  是统计平，不宣称结构性赢。旧“sign 输 0.93x”已被当前接口复测取代。
- 不把任一 OS 的方向外推到另外两个平台。

### 3. Parker vs CV 平台翻转（预期内）

- Win：CV = SRWLOCK+CONDITION_VARIABLE+谓词 → Parker 大赢。  
- Linux：glibc CV 已是薄 futex → 平。  
- M4：微赢。  
- **禁止**只用 Win 2.99x 写总账“全面赢”。

### 4. Parker 快路径不设“输赢分母”

atomic flag 无 unpark-before-park 安全；binary_semaphore 快路径仍 futex。只报 ~2–10 ns 绝对值。

### 5. `park_timeout(0)` 已从 OS 路径收回用户态

旧实现错误地把“非阻塞检查”送入 zero-deadline OS wait，Windows 约 3.7 µs、WSL 约
59 µs，且最初的 x86 CAS 修法在 M4 反而要 12.7 ns。当前利用 Parker 单 consumer 合同：
无 token 只做 relaxed load，看到 token 才做 acquire RMW。三平台为
**1.690 / 0.564 / 1.346 ns**（Win/WSL/M4），没有 syscall、没有发布 PARKED，也不丢竞态 token。

### 6. Linux 上 MappedFile 对 64 KiB ifstream 仍输，但不是封装失速

Win/M4 赢 ifstream；Linux 当前为 **11.479 vs 8.723 µs（0.760x）**。同合同 raw mmap 是
11.219 µs，HYLE 只多 1.8%，所以根因仍是「小文件 open+mmap+fault+close」不比 buffered
read 便宜。持久 16 MiB 非顺序页 probe 中 MappedFile 在三平台赢 seek/read **168–2355x**。
总账不得写“MappedFile 无条件赢 ifstream”，guide 应按访问形状选型。

### 7. FileHandle 的安全便利路径距 raw 2.4% 内

`open_file + FileHandle::read + close` 在 Windows 与 raw TIE；WSL/M4 分别有 2.4%/2.2%
的 paired sign LOSE。绝对差是每次 64 KiB 完整 open/read/close 的 0.15/0.28 µs，购买参数
校验、EINTR、`Result<size_t>`、统一原生诊断与 RAII。这已处于共享基础库的 raw-floor 带，
不通过把 OS 头/错误翻译塞进 public inline API 来换数字。若实际 profile 证明这段成为瓶颈，用户
仍可借一字 owner 的 `FileHandle::get()` 调 native I/O，所有权不丢。

## 复跑

```bash
cmake --build <build> --target hyle_platform_bench
# 定向 smoke，避免运行无关的阻塞/线程段：
HYLE_PLATFORM_BENCH_ONLY=vm-cycle,vm-protect \
  <build>/bin/hyle_platform_bench
# OS 段建议 n=5~15；单遍先确认墙钟 <~60s
scripts/bench-stat.py <build>/bin/hyle_platform_bench -n 5 \
  --pair "std::condition_variable ping-pong" "HYLE Parker ping-pong" \
  --pair "std::thread spawn+join" "HYLE platform::Thread spawn+join" \
  --pair "HYLE platform::Thread spawn+join" "HYLE platform::Thread callable spawn+join" \
  --pair "std::filesystem::directory_iterator" "HYLE directory_iterate (visitor)" \
  --pair "raw OS (mmap+munmap / VirtualAlloc+Free)" "HYLE vm_allocate_rw+release" \
  --pair "raw OS (mmap+munmap / VirtualAlloc+Free)" "HYLE vm_reserve+commit+release" \
  --pair "raw wait_on_address+wake floor" "HYLE Parker ping-pong" \
  --pair "raw mprotect / VirtualProtect" "HYLE vm_protect" \
  --pair "std::ifstream read+sum (copies)" "HYLE MappedFile open+sum (zero-copy)" \
  --pair "raw validated mmap+sum floor" "HYLE MappedFile open+sum (zero-copy)" \
  --pair "raw OS open+read+close" "HYLE FileHandle open+read+close" \
  --pair "persistent ifstream seek+read" "persistent HYLE MappedFile direct access"
```
