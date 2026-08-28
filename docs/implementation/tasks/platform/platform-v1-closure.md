# platform-v1 重构与收口记录

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`AUDITED`。本文是 platform 当前轮唯一 closure card 与施工记录；既有 platform task、
> architecture、guide 和性能页只作审计输入，不继承旧 `SIGNED`、C ABI 或 SOTA 结论。权威机械状态见
> [`module-status.json`](../../module-status.json)。

## Closure card（`SCOPED`）

- component / owner / round ID：`platform` / platform owner / `platform-v1`
- task kind：`REFACTOR`
- baseline commit / tree：`780bf5b69249cf17df0344b1e40ceb7cf8797c6a` /
  `4f77224063ec669b16caeb844b7079f55ab630ec`
- 进入原因与 demand/defect ID：`PLATFORM-CAP-01`（按基础库完整性标准重新枚举 OS/CPU 能力族）、
  `PLATFORM-NORTHSTAR-01`（量化模型加载、CPU dispatch、worker placement/idle 与 NUMA backing）、
  `PLATFORM-API-01`（C++23 owner/view、错误、默认值和可发现性）、`PLATFORM-ADOPT-01`（下游是否绕过
  platform 或误用 error/time）、`PLATFORM-PERF-01`（旧微基准未通过当前消费者与仪器资格）、
  `PLATFORM-DOC-01`（current 文档仍标 `NOT_REVIEWED_V2` 且夹有旧 C ABI/历史证书叙事）、
  `PLATFORM-DELIVERY-01`（当前中央组件图和 source/static/shared 交付复签）。
- 真实 consumer task / normal workload：
  - scheduler/condition-variable/GPU poller 的长寿命 worker：进程可用 CPU 查询、线程创建配置、
    token 不丢失的 park/unpark、有限 deadline park 和 shutdown/join；
  - 本地量化推理模型加载：只读持久 mapping、显式 prefetch/advice、随机页访问与正常 unmap；小型配置/
    pipeline cache 走 `FileHandle` 短读写；
  - mem 的大块 VM、guarded stack、huge-page/NUMA backing；simd/container 的一次 CPU/ISA/cache 事实读取；
  - 工程计算留出：大数据集持久映射和固定 worker 集合，不引入模型/token 专用策略。
- 范围：`include/hyle/platform/**`、`src/platform/**` 的 CPU/cache/topology、thread/affinity、address wait/
  Parker、anonymous VM、NUMA、filesystem/environment/file handle、mapped file、dynamic loader；全部 public
  surface 反向追踪；直接依赖和代表性直接消费者；platform-only C++ source/static/shared；当前
  architecture/user/performance 文档；统一仪器所需的窄 platform consumer contract。
- 非目标：scheduler/thread-pool/executor 策略、fiber/coroutine wait、模型/KV placement、async I/O、
  filesystem watcher/VFS/glob/path-string 算法、device/registered/pinned memory、跨进程 shared-memory 协议、
  Native FFI/C API、稳定 C++ ABI、Formal release certification，以及为了 benchmark 点值加入常态 metadata、
  cache、padding 或公共入口。
- 直接依赖 / 直接消费者：`PUBLIC core error time`；代表性直接消费者为 `mem`、`simd`、`container`、
  `sched`、`compute`，以及通过这些组件形成的 `io` 路径。本轮只验证直接 platform seam，不替未收口
  的下游签署整体结论。
- tier 与理由：`P1`；`Parker` 的 park/unpark/token path 位于 scheduler 同步与 worker idle 路径，持久
  mapping 是模型加载与随机访问的正常运行时机制。CPU/NUMA/VM/thread/fs/env/dynlib 的初始化、资源创建
  或 syscall 主导路径分别按 P0 结构/复杂度门审查，不能池化成一个“platform 总分”。
- material margin / resource budget：W1-WIN 同合同消费者 replay 使用 5% material/non-inferiority margin；
  Parker 无 heap/mutex、token 快路 O(1)，address wait 至多一次对应 native wait/wake；稳定 CPU/cache facts
  O(1) cached read，动态 process availability 与拓扑枚举同原生输入规模；VM/file/mapping owner 成功路径
  不增加隐藏 allocation/lock，普通一步 RW mapping 不多于一次 native mapping；不为未请求能力支付常态
  metadata 或 syscall。
- 总调查预算 / 性能预算：能力/API/依赖/实现审计最多 240 分钟；统一性能取证最多 90 分钟，只允许一次
  probe、一次双 order/适用 placement 正式采样，以及协议允许的一次 material 异常机制诊断与一次产品
  重跑。产品 `FIX` 集在 `AUDITED` 冻结，预计实现超过一个工作日时必须拆分或进入具名 blocker，不能
  默默扩面。
- 最多复现、机制诊断、产品修改次数：每个异常一次反序复现、一次 codegen/profile 机制检查、最多一个
  同时拥有机制和消费者证据的产品修改及重跑；不得在本轮发明第五种 benchmark 姿势或修改中央统计规则。
- machine profile：`W1-WIN` 为 P1 engineering 裁决机；`W1-WSL` 与 `M1-MAC` 承担适用 correctness、
  sanitizer、OS/backend 与交付复核，不跨平台池化性能。多 NUMA node 真机当前未登记，因此只允许结构/
  capability 结论，不产生跨 socket 带宽声明。
- instrument profile：P1 使用 `closure-perf-v1`；correctness/delivery 使用
  `closure-correctness-v1` 与现有 component/package 合同。
- strong peers（每项 capability 最多两个，均为本地钉死源码）：
  - wait/Parker/thread：Folly `423fd7f8d97fd7bf52fc4c986dfd20320c4de8f9` 的 Baton/ParkingLot/thread
    utilities，与 oneTBB `e349c15263bbb1d6b48f2b514c06201e1b90eea3` 的 address waiter/thread monitor；
  - CPU/topology/NUMA：oneTBB 同 commit 的 `info`/NUMA allocation，与 Folly 同 commit 的
    HardwareConcurrency；
  - file/mapping/VM：Folly 同 commit 的 `File`/`MemoryMapping`，与 Boost
    `df8eeb90846648464e3c8e378e312f0cec772e83` 的 Interprocess mapped-region/Filesystem；
  - dynlib：Boost.DLL 同 commit 与 oneTBB 同 commit 的 dynamic-link implementation。
  当前消费者形状另以 llama.cpp `7584430716ee229751771ed0d6bbcb780d105eeb` 和 ggml
  `d99724f2b141fed9e1a9f402213506c081433465` 的 model mmap/NUMA/thread paths 取证；它们不冒充
  已有 HYLE 生产消费者。
- 无同合同 peer 时的预算及禁止声明：NUMA selected-node/chunk 在单节点 W1 上只做复杂度、syscall/
  scratch、rollback 和 single-node floor；CPU hybrid rank、macOS QoS、filesystem UTF/path 与 unsupported
  OS operation 按真 backend correctness/结构裁决。不得声明“所有 platform 操作跨平台 SOTA”、物理 NUMA
  placement/bandwidth、或把 raw syscall 不同语义的点值写成产品输赢。
- correctness gates：normal；null/empty/zero；path/size/offset/alignment/overflow/native-limit；invalid UTF-8/
  flags/node/CPU/symbol；OS/OOM/permission/unsupported；move/release/close/join/destructor；unpark-before-park、
  timeout/wake race、spurious wake；environment/global-state concurrency 限制；recursive-delete symlink safety；
  property/differential/stress；ASan/UBSan/TSan（按合同适用）；Windows/Linux/macOS 真 backend 与 public-header
  self-containment。
- delivery gates：中央 `platform PUBLIC core error time` 及 include-owner 对账；platform-only source build；
  relocated static/shared C++23 SDK consumer；代表性 mem/simd/container/sched/compute seam；产物/安装树无
  C schema、bridge、Native FFI symbol 或伪 option。
- performance gates：冻结两个独立 family，不互相池化：`scheduler-worker-idle-handoff` 使用正常 Parker
  token/blocked lifecycle 对 faithful same-contract peer，通过 A/A、5%/10% delta、floor、双 order 和 5%
  NI；`model-loader-mapped-probe` 使用持久只读 mapping、固定页序列与完整 open/access/close 计费，对
  Folly/等价 raw mapping 同合同 replay 通过相同资格与 5% NI。Thread/VM/FileHandle 等 syscall/control
  paths若没有 material profile，只做结构与 raw-operation-count diagnostic，不进入 P1 aggregate。
- 预期输出：capability-family map、public-surface trace、有限 gap ledger 与冻结 `FIX`；immutable candidate；
  correctness、delivery、usability/docs、red-team 和 P1 evidence；四项 LOC、known limitations 与 reopen
  triggers。
- 停止条件 / 可接受限制：所有本轮 `MUST` 为 `COMPLETE/ROUTED`、适用门通过且 P1 两个 family 分别 PASS
  则 `CLOSED`；若 correctness/security、同合同 5% NI、交付或 material MUST 在冻结预算内无法裁决，则
  点名 blocker/owner/resume condition 进入 `BLOCKED`。无具名消费者的 watcher/shared-memory/loader flags、
  多节点真机证书、Native FFI 和上层 placement policy 是可接受 negative space，不因 peer API 数量扩面。

## `SCOPED` 审计输入与待证假设

- 当前实现约 6,264 行，公开面分为 CPU/cache、thread、wait/Parker、VM、NUMA、fs/env/FileHandle、
  MappedFile、DynLib 八个族；既有 architecture/user 文档已经描述很多能力，但仍是旧设计输入，不能直接
  抄成 `COMPLETE`。
- 语法消费已确认 sched/compute 使用 Parker/Thread，mem 使用 VM/NUMA/thread，simd/container 使用
  CPU facts，compute 使用同步 file API；本轮还要扫描语义重复和错误的 OS/error/time 绕行。
- 初查发现的假设包括：公开头仍有已删除 C adapter/FFI 和历史 milestone 注释；`for_each_entry` 的
  forwarding-reference/lvalue callable 形状需要编译验证；DynLib 的保留理由和完整 loader journey 需由
  当前基础需求而非旧 FFI 铺垫重裁决；MappedFile 的 const/view、range、flush 与 owner 旅程需同 peer
  对账。这些都只是待证假设，只有 `AUDITED` ledger 才能冻结为 `FIX/KEEP/ROUTED/REJECT`。

## `AUDITED` capability-family coverage map

本表从用户任务出发，不按当前文件名凑功能数。`MUST` 是本轮已准入能力；`GENERAL` 只留下 owner 和
触发条件。`PARTIAL` 行全部在下方冻结为 `FIX`，不存在 `PARTIAL + KEEP`。

| family | admitted journeys / variants | lifecycle、boundary 与当前路线 | coverage / missing edge |
| --- | --- | --- | --- |
| 可执行 CPU 与进程 CPU 视图 | 当前 simd/container 的安全 ISA dispatch；sched/mem 的 process-available CPU、P/E 排序和 worker/shard sizing；诊断用 cache/system topology | 一次 baseline-ISA probe 后 O(1) cached feature/tier；进程 affinity/cpuset 是 live buffer-fill；cache/topology 是 best-effort system/首次调用线程快照，不是 per-core placement policy | `COMPLETE`（当前已发货 kernel/worker 旅程）；精确 physical-core/cache-domain、AMX/SME/SME2 enablement 为 `GENERAL`，见 negative space |
| 原生线程 owner 与配置 | move-only spawn/join；capturing/move-only callable；name/stack/单核或集合 affinity/priority；best-effort 或 entry 前 required setup | spawn 同步复制 view；required 失败不运行 entry；joinable 析构/覆盖是程序不变量并 fail-fast；无 detach | `COMPLETE`；公开注释仍声称存在 C adapter，文档 `FIX` |
| worker idle handoff | sched/condition-variable/GPU poller 的 token 不丢失 park/unpark、zero poll、有限等待；多个 waker、一个 parker | O(1)、无 heap/mutex；blocking path 至多一次 native wait/wake；spurious wake 由 consumer deadline loop 处理 | `PARTIAL`：算法完整，但公共有限等待裸 `uint64_t ns` 易传错单位，且 single-consumer 陷阱不够可发现；`FIX` 为 `time::Duration` 形状 |
| anonymous VM 与 guarded region | mem 大块 backing、fiber stack、一步 RW、staged reserve/commit/decommit/protect、guard、soft THP hint、RAII/adopt | 普通一步 mapping 一次 native call；staged 形状按合同增加调用；owner move/release；selected NUMA region 禁止破坏 placement 的 decommit | `PARTIAL`：`large_page_size()` 在 Linux/macOS 硬编码并不等于 OS 暴露值；其余 `COMPLETE` |
| NUMA facts 与 placement | 当前 mem 的 preferred node；推理大块 backing 的 bind/all-node/selected-node/chunk/weighted placement；能力查询和实际 node IDs | cached process-lifetime node set；单节点正常退化；validation O(nodes)，普通 Linux policy O(1)，自定义布局 O(stripes/pages) 且 bounded scratch | `PARTIAL`：current-node/CPU-node 查询把失败和无效 CPU 静默变成首节点，会把 placement 错误伪装成成功；改为 `Result<unsigned>` |
| 同步 file owner | 小配置/pipeline cache、非-mmap 模型 loader 的 seek→exact read、完整写入、显式 durability、move/close；短读写仍可观察 | 一字 OS owner；read/write 是一次最多-N 操作；exact/all 循环无 allocation；seek 是单 cursor、并发 positional/async 明确归 `io` | `PARTIAL`：缺 seek/position、exact/all 和 sync，compute 已出现本地完整读写循环；本轮补齐，不复制 async proactor |
| path/filesystem/environment | stat/exists、目录和递归生命周期、copy/rename/remove/link/permissions/temp/cwd/home/env、visitor 与 opt-in owning convenience | UTF-8 public path；buffer-fill/visitor core，owning wrapper 明确购买标准容器；递归删除不跟随 link 且要求 caller 排除并发替换 | `PARTIAL`：`for_each_entry` 对 lvalue callable 形成 pointer-to-reference，现有测试只覆盖 temporary lambda；空 open path 也未统一拒绝 |
| 持久 file mapping | 模型/数据集 read-only persistent map、任意 offset/range、RO/RW/COW、const view/checked writable view、range flush、advice/prefetch、whole lock、close | move-only；path 或已打开 file owner；mapping 成功后 POSIX 可关 fd、Windows owner 留到 reset；访问 O(1)，控制操作至多一次相应 native call（durable Win flush 另含 file flush） | `PARTIAL`：只有 path 构造、`void* data() const` 绕过 RO/const、flush 只能 whole mapping；补 owner 互操作、span view 和 range flush |
| 进程内动态 backend 装载 | 推理/device backend 的 open-local-now→typed symbol→使用→close；attach 已加载 image；current-image lookup | move-only loader ref；symbol borrow 到 close；签名由 caller 承担；不声称 foreign-runtime FFI | `PARTIAL`：Windows `get()` 不 pin，存在并发 unload/reuse race；显式 close 丢失错误；typed lookup 未约束 pointer type；旧 FFI/out-of-process 理由错误 |
| owning query convenience | 初始化/控制路径自然取得 `std::string`/`std::vector`，仍可选 caller storage | 状态变化最多四次重试；allocation/length 失败进入 error；不进入热路径 | `COMPLETE` |
| async/positional I/O、shared memory、JIT/process/watch policy | proactor file offsets、跨进程协议、可执行代码权限、process lifecycle、watcher/VFS/glob | 分别由 `io`、未来 JIT/compute、process/tooling owner 承载；当前 platform 不伪造半项包装 | `OUT_OF_SCOPE` / `ROUTED`（仅 `io::File` 已有 async positional route）；不是本轮 FIX |

### Public-surface trace

| public header | surface → capability family | audit decision |
| --- | --- | --- |
| `cpu.h` | `CpuFeatures`、`Isa`、feature/tier/name、cache facts、available CPU ids/info/order/count、system topology → CPU 事实 | `KEEP`；修正文档对 cache/per-core 精度的边界，不为未发货 kernel 预建 feature museum |
| `thread.h` | `CpuAffinity`、`ThreadOpts`、`Thread`、hardware concurrency、`this_thread` → thread owner/config | `KEEP`；删除不存在的 C adapter/ABI 叙事 |
| `wait.h` | detail address wait/wake、`Parker`、`Unparker` → idle handoff | detail core `KEEP`；公共 timed shape `FIX` |
| `virtual_memory.h` | page geometry、raw VM、THP advice、`VmRegion` → anonymous VM | owner/raw dual surface `KEEP`；large-page query `FIX` |
| `numa.h` | capabilities/nodes/query/bind/reserve/selected interleave → NUMA | placement `KEEP`；两个 fallible fact query `FIX` |
| `fs.h` | `FileHandle`、metadata/path/link/temp/env/visitor → sync file + filesystem | path/env core `KEEP`；file lifecycle与 lvalue visitor `FIX` |
| `mapped_file.h` | `MappedFile::{Mode,Advice,open,data,flush,advise,prefetch,lock,unlock,reset}` → persistent mapping | mapping mechanism `KEEP`；owner/view/range lifecycle `FIX` |
| `dynlib.h` | extension、open/get/current/symbol/symbol_as/close → dynamic backend loader | capability `KEEP`；pin/error/type contract `FIX` |
| `convenience.h` | owning string/vector queries → owning convenience | `KEEP`；allocation 明码标价 |

语法消费之外的语义扫描结论：mem 正确把 VM/NUMA 交给 platform；sched/compute 正确消费 Thread/Parker；
simd/container 正确消费 CPU probe。`sched::Mutex` 的 Windows SRWLOCK、time 的 clock backend、error 的
native-code translation、obs 的 terminal write 和 util 的 demangler 都是各自语义 owner 的直接 OS 机制，
不能为了“都走 platform”制造循环或较慢的通用包装。compute 的 pipeline-cache `read_all/write_all` 是
本轮 file-owner 缺口的证据，但 compute/graph 仍是半成品，本轮不越界重构它；其未来收口时迁移到新方法。

## Capability / gap ledger 与冻结 `FIX`

| ID | demand / exact evidence | owner、normal shape、failure / complexity | audit decision / verification |
| --- | --- | --- | --- |
| `PLATFORM-FIX-01` | 定义性 sync-file journey；llama.cpp pinned loader 的 seek/full read；Folly pinned `readFull/writeFull`；仓内 Vulkan cache 重复循环 | platform semantic/artifact；single-cursor `seek/position`、`read_exact/write_all`、`sync`；短 read/write 保留；O(bytes/native-limit)、零 heap；并发 positional 归 io | `MUST / PARTIAL -> FIX`；Win/POSIX round-trip、EOF/zero/large offset/closed/invalid-origin、durability smoke；`for_each_entry` named lvalue 与 empty path 回归 |
| `PLATFORM-FIX-02` | 北极星 model mapping；Folly `MemoryMapping(File)` + const/writable ranges；Boost mapped-region range flush | platform；consume `FileHandle` owner，path 是 convenience；const bytes + checked RW/COW span；range overflow/alignment；open/access/reset O(1)，无常态 metadata | `MUST / PARTIAL -> FIX`；RO/RW/COW、moved owner、range flush、invalid writable/offset/EOF、source/static/shared tests |
| `PLATFORM-FIX-03` | 公开合同缺陷：header 声称 OS-exposed/0，Linux/mac 实现却按 arch 硬编码；Linux THP 暴露真实 PMD size，mac 无本轮 hint backend | platform；一次 cached OS query；unknown 0；alignment helper 仍 fallback base page；O(1) cached、冷路径有限 file parse | `MUST / PARTIAL -> FIX`；真实 host invariant + injected parser/unsupported tests，不声称 THP 实际 promotion |
| `PLATFORM-FIX-04` | 北极星 worker→node placement 与现有 public promise；libnuma/OS node queries 可失败，当前 silent-first-node 会错误放置 | platform；`Result<unsigned>`；single-node current 正常值，invalid/unavailable CPU 明确 error；控制路径可 O(available CPUs/nodes)，无热路径 metadata | `MUST / PARTIAL -> FIX`；invalid CPU、single-node、真实 backend；mem soft fallback 显式选择 `kNumaAny` |
| `PLATFORM-FIX-05` | 定义性 loader lifecycle；Boost.DLL/OS loader pin semantics；Windows GetModuleHandle race 是 correctness defect | platform；`get` 在两端都持有 loader ref；explicit close 返回 Status；symbol borrow invalidated at close；typed API 只接受 pointer | `MUST / PARTIAL -> FIX`；open/get/current/symbol/move/close/missing/invalid UTF，Windows ref pin smoke |
| `PLATFORM-FIX-06` | 当前 sched/compute timed park；time 已冻结 `Duration`；裸 ns 是跨模块单位陷阱 | platform semantic，time type contract；`park_for(Duration)`，non-positive poll，正值按现有 native bound 转换；算法/state 不变、零新 metadata/syscall | `MUST / PARTIAL -> FIX`；race/stress/zero/negative/large；direct consumers compile/test；P1 replay复用同一生命周期 |
| `PLATFORM-FIX-07` | current docs/public comments 含已删除 C adapter、FFI/out-of-process、clay milestone、cache 精度与旧证书叙事 | architecture/user/public docs；只写 current C++23 source/static/shared、选择指南、trap/negative space | `MUST / DOC -> FIX`；docs checker、header self-containment、全 public surface 可发现性人工清单 |
| `PLATFORM-FIX-08` | 本轮 P1/交付缺证；旧 platform benchmark 混合不同合同与历史点值 | platform + build/perf evidence；冻结 Parker handoff 与 persistent mapping 两 family，不能池化；直接组件 seam 和 platform-only 三交付 | `MUST / EVIDENCE -> FIX`；统一仪器资格、5% NI、correctness/delivery manifest；不因点值修改 API |
| `PLATFORM-FIX-09` | pinned Boost Interprocess 先包含 Windows loader 头后，公开 `time/clock.h` 的 `Duration::max()` 被 SDK `max(a,b)` 宏改写，platform 依赖头无法组合 | time public dependency + platform convenience；采用标准 parenthesized static-member 调用，不要求消费者控制 include order 或全局定义 `NOMINMAX` | `MUST / CORRECTNESS -> FIX`；保留模拟 Windows min/max 的独立组合 TU，并重跑 time/platform public-header 与 package consumer |

### `GENERAL` / negative-space ledger

- **AMX/SME/SME2 与 per-thread enablement：`DEFER`，owner=platform+simd backend。** llama.cpp/oneDNN
  证明它们对推理有价值，但 HYLE 当前没有对应 kernel artifact；只加 CPUID/HWCAP bool 不能完成用户旅程，
  AMX 还涉及 per-thread OS permission。触发条件是具名 HYLE kernel/backend 被准入，届时 probe、enable、
  isolated TU 和 dispatch 必须同轮交付，不能提前报告“可用”。
- **physical core/SMT/cache-domain 与 per-CPU cache：`DEFER`，owner=platform facts + sched/engine policy。**
  oneTBB/Folly 证明该问题域存在，但当前 user 可显式 worker 数/CPU set，尚无 profile 证明自动 SMT policy；
  触发条件是 engine/sched 的具名 placement 旅程需要选择每 core 一个 thread 或 cache-local shard。当前
  `cache_info()` 只作 best-effort representative hint，不得用于安全/ABI 或冒充 hybrid per-core fact。
- **MappedFile partial unmap、range mlock/on-fault、prefault/collapse：`DEFER`。** llama 的 specialized loader
  使用这些，但 HYLE 已能 subrange map、`kDontNeed` advice 和 whole lock；第二个同语义消费者或端到端
  residency/profile 证据出现时重开，不为微基准加入 range-state metadata。
- **direct I/O、scatter/gather、concurrent positional sync I/O：`ROUTED/DEFER` 到 `io::File` 或未来专用
  loader。** 本轮同步 single-cursor API覆盖启动加载；不得把 OVERLAPPED/proactor 状态塞进一字
  `FileHandle`。真实 direct-I/O loader 被准入时单独设计 alignment/fallback owner。
- **dynlib lazy/global/deep-bind/nodelete/isolated search flags：`DEFER`。** 当前 backend contract 固定
  eager+local；只有具名 plugin dependency/search problem 才扩 flags。相对路径的信任边界必须由 guide 标价。
- **executable VM/JIT、named/cross-process shared memory、process spawn、filesystem watch/VFS/glob：
  `REJECT` 本模块。** 分别归 compute/JIT、io protocol、tooling/process 与 higher-level filesystem owner。
- **Native FFI/C API：`REJECT`。** 当前无 foreign-runtime consumer；source/static/shared 都是 C++ 产品形状。

### 冻结施工边界

产品代码只允许实现 `PLATFORM-FIX-01..06` 与审计后由真实 peer 组合触发的 correctness blocker
`PLATFORM-FIX-09`；`FIX-07..08` 只改 current 文档、测试、交付与统一性能证据。
若实现中发现新的 correctness/security blocker，可按协议点名重开 ledger；命名偏好、单个 peer feature、
新 benchmark 点值或“顺便完整 hwloc”不得扩面。完成上述 FIX 后冻结 candidate，再做三平台和 P1 裁决。

## Candidate

进入 `CANDIDATE` 时填写。

## Verification manifest

候选冻结后按 closure card 的适用矩阵填写，不把未运行项写成 PASS。

## Closure decision

尚未裁决。
