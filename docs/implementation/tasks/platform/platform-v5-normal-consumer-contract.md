# Platform v5：正常消费者合同复检

<!-- MODULE CLOSURE RECEIPT: CLOSED -->

> 本页按状态机顺序记录本轮七问、能力超集、质因数依赖和“先冻结自然 API、再看性能”的完整回放；
> 前段的只读、`SCOPED`、`AUDITED` 与 `CANDIDATE` 文字是当时阶段事实，最终裁决以文末 receipt 为准。

## 只读 replay

- reviewed product：`platform` host factor、`platform_thread`、`platform_fs` 及聚合产品；
- baseline：`1cb204e79e11532252053971dd5ff97dfe9648d7`；
- 当前 north-star：llama.cpp remote `4d9176092d00586775af140581bb0b558ddc4389`，随仓 ggml tree
  `f4ed259b5708475fb9a853998591f2d4d2624c4f`；standalone ggml
  `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`；
- current peers：Folly `65749da412d92fe27d87ae45a8ef670720712858`、oneTBB
  `e00c68dcac0e83947c9b54f6d5304c0616c9639f`。oneTBB 自 v3 锁定版本在 `src/include`
  没有相关变更；Folly 相关变化在 async IO/atomic/hazptr，不改变本轮同步文件和线程 QoS 裁决；
- current product identity：Platform v4 candidate
  `d015b0de1cf6d5899da8336fef94104974bcd90e` / tree
  `1456eea7575fa72ce48626a9db13a80f0aa31c09`；
- review 输出：host factor 为 `CONFIRMED`；thread 与 fs 各有可复现 `PRODUCT_TRIGGER`，只重开受影响
  factor，不重开 CPU facts、NUMA、VM、DynLib、environment、entropy、Parker 或 FileHandle I/O。

### 能力覆盖回放

| family | normal consumer / owner | verdict | negative space / disposition |
|---|---|---|---|
| CPU/OS feature facts、cache、process-available topology、P/E order | backend gate、worker sizing、shard/placement fact；policy 归 SIMD/Sched/engine | `COMPLETE` | v4 已移除 target authority；device facts 仍归 Compute |
| host memory budget | model/context fit planning | `COMPLETE` | device/UMA 聚合归 Compute/engine；snapshot 不是 reservation |
| NUMA 与 raw anonymous VM | allocator backing、first-touch/selected-node/chunk placement、guard stack | `COMPLETE` | mapping/storage owner 归 Mem；placement policy 归 consumer |
| DynLib、environment、secure entropy | backend discovery、startup config、安全 temp seed | `COMPLETE` | plugin ABI/search trust policy 不属于 Platform |
| native Thread、affinity、priority、current ISA state | persistent runtime worker | `PARTIAL` | Windows explicit performance priority 没有关闭 execution throttling；见 `PLAT5-T01` |
| address wait / Parker | scheduler worker mailbox | `COMPLETE` | counting/multi-waiter/cancel/fiber-aware primitive 归 Sched |
| synchronous FileHandle / namespace / temp / replace | model/config/cache 同步加载与控制面 | `PARTIAL` | low-level I/O 完整；C++ visitor 与 Windows 长路径合同见 `PLAT5-F01/F02` |
| file mapping、residency、shared mapping | model weights / shared storage | `ROUTED` | `mem_vm` 是唯一公开 storage owner |
| positional/vectored/direct/async/watch | loader/proactor | `ROUTED` | IO；不扩张一字同步 `FileHandle` |
| process priority、detach、file lock/preallocate/resize | application policy 或未来具名机制 | `GENERAL/DEFER` | process priority 跨 OS 作用域不同且现有 worker priority 已承载 runtime；没有准入本轮 |

### 七问 replay 直答

1. **第三方产品路径：**三个 factor 都只使用 C++23、compiler builtin 与 Windows/Linux/macOS ABI，
   没有第三方产品依赖；llama.cpp/ggml/Folly/oneTBB 只是需求和 peer。`CONFIRMED`。
2. **推理能力：**CPU facts、topology、NUMA/VM、memory budget、mapping route、loader/fs 均覆盖当前
   north-star；但最新 ggml 的持久 CPU worker 在 Windows 显式关闭 execution-speed throttling，而
   HYLE high priority 只调用 `SetThreadPriority`。该具名旅程构成 thread 产品触发。
3. **内部复用：**Sched/Container/Mem/SIMD 均消费 Platform 的已公开机制；Mem/IO 的 raw mapping 或
   proactor syscall 是其 owner mechanism，不是重复 Platform policy。未发现缺失的低层 HYLE 加速设施。
4. **性能：**v4 CPU fact 查询保持缓存 O(1)；v3 的同合同 host/Parker/warm-file receipts 未被 peer
   更新推翻。本轮三个 FIX 都是 P0 结构问题：显式线程配置最多增加一个 OS 控制调用，未请求 priority
   不变；visitor 正常路径仍每项一次 callback、零 owning allocation；短路径 mkdir 保持栈 scratch。
5. **旧 C ABI / Native FFI：**公开面只有 C++ namespace/types；没有 C header/schema/bridge/managed
   sample/FFI option 或未装饰 C export。OS/private `extern "C"` 不构成产品 FFI。`CLEAN`。
6. **交付：**当前 host/thread/fs factor 和聚合 Platform 的 source/static/shared 已有三平台 receipt；
   product change 后必须重放 exact factors、聚合产品和 Sched/Mem/IO 直接 consumer，当前不能继承为新
   candidate 的 `PASS`。
7. **易用性与文档：**host/VM/NUMA/Thread/FileHandle 的 owner、失败和默认值自然；但
   `for_each_entry` 对 const callable 编译失败，且未标 `noexcept` 的包装会让 callback 异常穿过
   `directory_iterate(...) noexcept` 并终止。现有 guide 也没有说明 Windows priority 的 QoS 映射。

## Reopen evidence（`REOPENED`）

### `platform_thread`

- trigger ID：`PLATFORM-THREAD-WINDOWS-QOS-2026-09-05`；
- latest ggml `ggml-cpu.c::ggml_thread_apply_priority` 的正常 threadpool worker 路线在 Windows 11
  对非低优先级调用 `SetThreadInformation(ThreadPowerThrottling)`，明确避免 execution-speed
  throttling 导致核心停放/worker 聚集；HYLE `set_priority_impl` 目前只调用 `SetThreadPriority`；
- 这不是增加 benchmark API：既有 `ThreadOpts::priority` / `this_thread::set_priority` 已承诺 portable
  priority/QoS，Sched 也把它作为用户显式的 performance-oriented placement；Windows 实现没有完成该
  既有语义；
- 初步范围：只完善 Windows 显式 priority 的 execution QoS 映射、错误传播、测试与文档。未设置
  optional priority 时保持 OS 默认；不增加 process-wide priority、realtime、affinity 或 scheduler policy；
- replay verdict：`PRODUCT_TRIGGER`，初步 tier `P0`。

### `platform_fs`

- trigger ID：`PLATFORM-FS-CPP-VISITOR-LONG-PATH-2026-09-05`；
- `for_each_entry(path, const_callable)` 的当前 MSVC compile-only 复现失败于把
  `const Callable*` 传给 `void*`；普通 throwing callable 又会跨过 compiled
  `directory_iterate(...) noexcept`，触发 `std::terminate`，与包装自身没有 `noexcept`、普通 C++
  algorithm 风格和文档直觉均冲突；
- `make_directory_recursive` 另有固定 `char[4096]`，在 Windows 的 UTF-8/extended-path 机制仍可处理
  更长合法路径时提前返回 HYLE 自造的 4095-byte 上限；同一文件的 temp/remove 路径已经使用可增长
  scratch；
- 初步范围：让便利 visitor 接受 const callable，并在启用异常的 consumer TU 内捕获后于 compiled
  noexcept core 返回时原样重抛；no-exception/core 路线保持 no-throw；recursive mkdir 使用短路径栈
  scratch、超长路径才 fallible heap。底层 visitor、FileHandle、syscall 数和普通短路径不变；
- 不增加 lock/preallocate/resize、path class、watch、mapping 或 async I/O；
- replay verdict：`PRODUCT_TRIGGER`，初步 tier `P0`。

下一步只允许为上述两个 factor 冻结 closure card、有限 `FIX` 与验证矩阵；host factor 保持
`CLOSED`，不得借本轮重写 v4 CPU/target 边界或扩建性能仪器。

## Closure card（`SCOPED`）

- component / owner / round / kind：`platform_thread + platform_fs` / Platform product /
  `platform-v5-normal-consumer-contract` / `REOPEN`；
- baseline 与 trigger：沿用上述 immutable baseline 及两个 trigger ID；host factor 只回放、不变更；
- 真实消费者：显式选择 performance-oriented worker 的 CPU inference runtime；以普通 const/throwing
  callable 扫描 model/cache directory 的 C++ loader；Windows 合法 extended UTF-8 cache/model path；
- 范围：
  1. Windows `ThreadPriority` 在调用者显式配置时，同时设置 native scheduling priority 与 execution QoS：
     `kLowest/kLow -> EcoQoS`，`kNormal -> system-managed`，`kHigh/kHighest -> HighQoS`；optional 未设置
     时完全不调用；
  2. `for_each_entry` 使用 caller-TU trampoline context 支持 const callable；启用异常时捕获 callback
     exception、令 noexcept compiled iterator 完成清理，再原样重抛；无异常 TU 和 low-level core 保持
     no-throw；
  3. recursive mkdir 以 4096-byte stack scratch 为常用路线，只有更长输入才 checked heap；OS 自身的
     component/path failure 原样转 Status，不再由 HYLE 固定拒绝；
- 非目标：registry 中两个 factor 的 non-goals；尤其不新增 process priority、realtime、detach、文件
  lock/preallocate/resize、path owner、mapping/async IO，也不改 CPU facts、Parker 或 FileHandle I/O；
- 直接依赖 / consumer：依赖保持 thread `PRIVATE platform; PUBLIC core,error,time`、fs
  `PRIVATE platform; PUBLIC core,error,time`；直接回放 Platform 聚合、Sched priority、Mem/IO fs consumer；
- 内部设施：使用现有 Error `Status/Result`、Core environment facts 和现有 stack-first malloc scratch；
  不需要 Container/Mem allocator（否则会倒置底层依赖），不引入第三方产品库；
- tier：两个 factor 都是 `P0 / closure-correctness-v1`。QoS 只在显式 cold setup 多一个 native call；
  callback 每项仍一次 indirect call且零 owning allocation，异常分支为冷路；4096-byte 内短路径没有 heap；
- 性能边界：不重跑未改变的 host/Parker/open-read-close P1。若编译器显示 visitor 正常路径增加 allocation、
  额外 directory syscall 或每项额外动态派发，则不得进入 candidate；
- machine：`W1-WIN / W1-WSL / M1-MAC`；Windows 是 QoS 与 >4095-byte path 的语义机，另两平台证明
  conditional code/header 和既有行为；
- peers / authority：Windows supported API contract、current ggml worker route；标准 C++ exception/callable
  语义和三平台 OS filesystem ABI。Folly/oneTBB 只回放未改 families；
- provenance：`REAL_CONSUMER`（QoS）、`NORTH_STAR_JOURNEY`（visitor/long model path）、`STANDARD`
  （C++ callback semantics）；
- correctness gates：normal/invalid enum、required/best-effort、configured/unconfigured、const/void/bool/throwing
  visitor、early stop、cleanup、short/long/OOM path、三平台 tests、WSL ASan/UBSan；
- delivery gates：两个 exact factor + Platform source/static/shared relocation，公开头自含，Sched/Mem/IO
  selected direct consumers，旧 C/FFI symbol/header/option scan；
- investigation budget：90 分钟；验证预算 60 分钟；最多一次机制诊断、两次产品修正；若需改公开
  ownership、因子依赖、Parker/FileHandle 热路或通用仪器则本轮 `BLOCKED`，不得扩大；
- 预期输出：两个小型实现修复、定向测试、architecture/user guide 校准、完整七问/红队与 terminal receipt；
- 停止条件：有限 `FIX` 全通过即关闭；跨平台 OS 不支持本来就不可能成立的相同值不构成 blocker。

当前状态为 `SCOPED`。下一步只允许完成现有 public surface 双向追踪、签署有限 `FIX` 和 API freeze；
在 `AUDITED` 前仍不修改产品源码。

## Audit verdict（`AUDITED`）

双向追踪覆盖三个 factor 的全部公开 headers/compiled sources/tests、中央组件图、Platform product façade、
Sched/Mem/IO/SIMD/Container 直接引用，以及 latest llama.cpp/ggml 的 host/thread/fs 调用。没有发现另一套
CPU detector、target selector、NUMA/VM owner、thread wait、filesystem mapping owner 或错误系统；host
factor 因此维持 `CONFIRMED`。两个受影响 factor 的有限 ledger 如下：

| ID | class | exact defect / demand | frozen natural shape | complexity / cost | action / verification |
|---|---|---|---|---|---|
| `PLAT5-T01` | `MUST` | Windows explicit priority 没有 execution QoS，performance-oriented worker 语义不完整 | optional 未设置保持继承；low 两档 EcoQoS，normal 恢复 system-managed，high 两档 HighQoS，并保留对应 native thread priority | 显式 cold setup 固定两个 native controls；无常态 metadata/allocation；默认零新增调用 | `FIX` 实现 + 子线程 native readback + invalid/required/best-effort |
| `PLAT5-F01` | `MUST` | const callback compile failure；throwing callback 穿越 noexcept core | caller-TU context 保留 cv；异常启用时捕获、停止 iterator、完成 close 后原样重抛；异常关闭时只保留 no-throw route | 每项仍一个 callback，无 heap/entry collection/额外 stat；exception path 冷 | `FIX` compile/runtime tests + early-stop/error precedence |
| `PLAT5-F02` | `MUST` | recursive mkdir 的 4095-byte HYLE ceiling 低于 Windows extended-path mechanism | 4096-byte stack first，超长 checked malloc；OS 限制通过 Status 可观察 | O(path bytes + ancestors)，短路径零 heap；长路径一次 fallible allocation | `FIX` >4095 Windows path + short/invalid/OOM boundary |
| `PLAT5-D01` | `MUST` | 行为合同变化后旧 guide/delivery receipt 不可直接继承 | architecture/user/performance 边界同步；两个 exact factor、Platform 三交付和直接 consumers | 无产品热路成本 | `FIX` docs/scans/builds/red team |

全部现有 public entries 反向分类为 `COMPLETE`、`ROUTED` 或以上三个 `PARTIAL -> FIX`；不存在
`MISSING + KEEP`。`PLAT-A-16` detach 保持 lifecycle safety 非目标；`PLAT-A-18` process priority 仍是
application-wide policy/general backlog：Windows process class、Linux current-thread nice 与 macOS process
行为不能伪装成同一 scope，本轮持久 worker 已由逐 thread API 完整承载；`PLAT-A-26` file
lock/preallocate/resize 在当前 north-star 同步基础旅程没有缺口，positional/direct/async 继续归 IO。

### API / usability freeze

- 状态：`FROZEN`（2026-09-05）；公开函数、类型、对象 layout、target/include path 均不新增或改名；
- `ThreadPriority` 表示显式的 portable scheduling/quality hint，而不是 CPU hard pin。缺省 optional 才是
  完全继承；`kNormal` 明确恢复系统管理，high/low 是显式 performance/efficiency QoS；
- `directory_iterate` 保持 allocation-free、noexcept type-erased core，直接 callback 不得抛；普通 C++
  用户首选 `for_each_entry`，它在 caller TU 内提供标准 callable 异常传播，且不会把异常编码成 Status；
- recursive mkdir 仍是同一个 `Status` API；allocation failure 是 `ResourceExhausted`，native path/component
  上限是带 OS detail 的 Status，不暴露 scratch owner；
- 性能只能在以上接口冻结后验证。若修复要求 thread/process policy 新类型、FileHandle state、额外 factor
  dependency、每项 allocation/syscall 或 benchmark-only API，必须退回 `AUDITED` 重新裁决，本轮不得做。

冻结 `FIX` 集只有 `T01/F01/F02/D01`。从现在起授权这些产品改动；其余 Platform 产品面保持关闭。

## Candidate（`CANDIDATE`）

- product commit / tree：`5818a7ac9b143b980ac43eb39e2a01dac1e5b108` /
  `e0c2a9e25b49350965d1bfb3a39789ece976569e`；
- `T01`：Windows 显式 priority 同时配置 thread priority 与 execution QoS；optional 缺省路径代码未变；
- `F01`：`for_each_entry` 使用 caller-owned context 保留 const，并在异常启用 TU 中捕获、关闭、重抛；
- `F02`：recursive mkdir 改为 4096-byte stack-first + 单次 checked malloc fallback；
- `D01`：architecture/user/performance/watchlist 已同步；CMake factor graph、公开函数与 layout 未改；
- W1-WIN Release：host/thread/fs 三个 test binary `3/3 PASS`；新增 native QoS readback、const callback、
  exception cleanup 与 4.3KB recursive path 全部通过；MSVC const-callback exception/no-exception
  compile-only 均通过；
- P0 结构检查：未配置 priority 零新增调用；显式 priority 固定两次 native control；visitor 每项仍一次
  callback、无 entry allocation/collection/额外 stat；短 mkdir 零 heap、长路径一次 allocation；
- 当前尚未完成 WSL/macOS、sanitizer、exact factor、Platform 三交付、直接 consumers、FFI/symbol scan 与
  最终红队，所以只允许 `CANDIDATE`，不能写 `VERIFIED/CLOSED`。

## Candidate verification（`VERIFIED`）

验证绑定 product candidate `5818a7ac9b143b980ac43eb39e2a01dac1e5b108` / tree
`e0c2a9e25b49350965d1bfb3a39789ece976569e`。candidate 后到本节记录前，Platform 的公开头、实现、
测试、sample、CMake 与 package 路径均无 diff。只读 reference 仓在终验时已 fast-forward 到上述
llama.cpp/embedded ggml、standalone ggml `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`、Folly
`65749da412d92fe27d87ae45a8ef670720712858` 和 oneTBB
`e00c68dcac0e83947c9b54f6d5304c0616c9639f`；新增提交只涉及 SYCL、AsyncSocket zero-copy drain 与
TCM Windows 脚本，没有新的 host/thread/synchronous-fs 产品触发。

### Correctness、sanitizer、consumer 与 delivery

| gate | W1-WIN / MSVC 19.50 | W1-WSL / GCC 14.2 | M1-MAC / AppleClang 21 | verdict |
|---|---|---|---|---|
| selected Release + Werror | exact thread `2/2`、exact fs `2/2`；聚合及新增原生 readback/4.3KB path 通过 | host/thread/fs `6/6` | host/thread/fs `6/6` | `PASS` |
| exception / no-exception header route | const visitor 两种形状均 compile-only 通过 | 两种形状均通过 | `-fno-exceptions` 实例化通过 | `PASS` |
| sanitizer | 本轮 Windows 原生语义机通过 | affected tests：ASan `2/2`、UBSan `2/2` | Release 真机覆盖 arm64 | `PASS` |
| source embedding | Platform `1/1` | Release 聚合闭包通过 | Platform `1/1` | `PASS` |
| relocated shared SDK | Platform `1/1` | 不重复外推 Windows/macOS receipt | Platform `1/1` | `PASS` |
| relocated static SDK | Platform `1/1` | 不重复外推 Windows/macOS receipt | Platform `1/1` | `PASS` |
| direct consumers | `mem_vm + sched + io` 共 `17/17` | Platform 聚合闭包通过 | Platform 聚合闭包通过 | `PASS` |

macOS 真机为 26.6.2 build 25G83，较登记 profile 的操作系统 patch 前移；硬件与 AppleClang contract
仍匹配。本轮没有采集 macOS 性能点值，只用它验证条件实现、公开头、deployment floor 与交付，因此将
该 drift 明示记录而不据此签发新的性能结论。安装后的 dylib `minos=14.4`、依赖均为 `@rpath` 或系统库。

Windows shared SDK 的 `hyle_platform_thread.dll` 只有 22 个 MSVC C++ decorated exports，
`hyle_platform_fs.dll` 只有 36 个 decorated exports；macOS `nm -gU` 同样只有 C++ mangled product
symbols。Platform 产品路径没有 C header/schema/bridge/FFI option 或未装饰 C export，故旧 C ABI / Native
FFI 门为 `CLEAN`。exact factor configure 分别只解析
`core,error,time,platform,platform_thread|platform_fs`；聚合 target 和安装头与中央 manifest 一致。

### Engineering performance

两个 factor 的本轮 tier 均为 `P0 / closure-correctness-v1`，结构门 `PASS`：

- 未设置 `ThreadOpts::priority` 时 `has_priority=false`，不会进入 priority/QoS setter；显式设置只在
  thread cold setup 执行两个有界 OS controls，没有 per-task state、allocation 或热路检查；
- `for_each_entry` 正常路线维持每项一个 type-erased callback；caller-owned context 不收集 entry、不做
  `stat`、不拥有 heap allocation。只有 callback 已经抛异常时才持有 `exception_ptr`，native iterator
  关闭后原样重抛；
- recursive mkdir 仍为 O(path bytes + ancestors)；4096 bytes 以内只用栈，超出时恰好一次 checked
  allocation，不按 component 重复分配；
- host cache、Parker、compiled `directory_iterate`、FileHandle open/read/close 和普通短路径未改变。三份
  v3 P1 manifest 重新 deep-verify 为 `QUALIFIED`、provenance `EXACT`、environment policy `CURRENT`，
  仅继续约束各自未改变的同合同路线，不冒充 v5 新控制路径的计时证书。

### Final red team 与七问

- invalid priority 在任何 OS call 前返回；显式 Windows low/normal/high 的 native priority 与
  throttle control/state 都由子线程 readback 证明。OS 不提供两项 policy 的事务式更新；required
  spawn 在任一失败时禁止 entry，best-effort 保持显式软失败语义，未伪称原子更新；
- callback context 只活在同步调用栈，指针不逸出；异常由 noexcept trampoline 截获，底层完成 native
  close 后才越过 C++ wrapper。callback exception 优先于同时发生的 close Status，符合普通 C++
  callback failure 的可观察语义；bool early-stop 仍是成功的 `Result(false)`；
- length `+1` 先做 overflow 检查，heap 失败为 `ResourceExhausted`，OS path/component 上限仍通过共享
  Error Status 返回；没有用 Container/Mem allocator 倒置底层依赖；
- 第三方路径、推理能力、内部复用、性能、FFI 清洁、三交付、易用性/文档七问均为 `PASS`。公开 API
  没有为了 benchmark 增加类型或 metadata；四项 `MUST` 全部 `COMPLETE`，无 `PARTIAL + KEEP` 或
  medium-or-higher finding。

七项 aggregate gate（scope、architecture、correctness、delivery、usability/docs、red-team、engineering
performance）全部 `PASS`。`CANDIDATE -> VERIFIED` 前 registry snapshot 位于
`C:\public\project-hyle\_validation\platform-v5-replay\module-status-before-verified.json`，SHA-256
`3E8167F0E16139F39C59C3C494D3B1B1A9B98F672E70DB3F295885DA83AAD1ED`。当前阶段为 `VERIFIED`；
Formal 仍独立 `PENDING`，不阻止 P0 工程收口。

## Closure decision 与 terminal receipt

本轮 terminal verdict 为 **`CLOSED`**。Platform 的三个内部 factor 现在各有单一职责：host prime 提供
CPU/拓扑/内存预算/NUMA/raw VM/DynLib/environment/entropy，thread factor 提供 native thread、affinity、
priority/QoS、current-thread ISA state 与 wait/Parker，fs factor 提供同步 handle 与 namespace 操作。
SIMD target policy、Mem storage mapping 和 IO async/positional/direct 能力仍由各自 owner 承担。四项
`FIX`、七项 aggregate gate、三平台及适用的三交付/直接 consumer 均为 `PASS`；没有未决 blocker。

已知限制：

- `ThreadPriority` 是 portable scheduling/QoS hint，不是 CPU hard pin、实时保证或 process-wide policy；
  Linux 提升 nice 可能需要权限，macOS 没有公开 hard-affinity API，Windows process priority class 由应用
  所有；
- Windows native priority 与 execution QoS 是两个 OS control，不具备事务式原子更新；required spawn
  保证配置失败时不进入用户 entry，best-effort 只承诺尽力应用；
- `for_each_entry` 在 callback 抛异常时优先重抛该异常；直接调用 allocation-free
  `directory_iterate` 的 callback 不得抛。同步 filesystem 不承诺 lock/preallocate/resize/watch 或
  async/direct/positional/vectored I/O；
- 本轮 P0 只签冷配置、异常桥与长路径 scratch 的结构成本。v3 P1 只继续约束未改变的 host/Parker/
  warm-file workload；跨机器 Formal、端到端推理和所有 OS/filesystem 的普遍 SOTA 均未声明。

只有以下新事实可以重开：

1. 具名真实 consumer 证明当前 Platform-owned host/thread/fs 能力无法表达必需且有稳定跨平台或具名平台
   语义的机制；
2. 支持 OS/CPU/filesystem 上复现 priority/QoS、callback cleanup/exception、长路径或现有 host/VM/NUMA
   能力的 correctness/security 缺陷；
3. 公开 API/layout/语义、factor ownership/依赖、正常复杂度或 source/static/shared delivery materially
   改变；
4. HYLE 其他模块重复实现本应由 Platform 拥有的机制，或 Platform 再次开始选择 SIMD/provider、拥有
   mapping storage 或吸收 async IO；
5. 成熟 same-contract peer 或真实 inference profile 证明存在更优复杂度档，或出现超过 5% 且可复现的
   normal-path material regression。

candidate LOC self-check（snapshot `5818a7ac9b143b980ac43eb39e2a01dac1e5b108`）：全库代码
443,621；自实现包含测试 112,651；自实现不包含测试 71,198；host/public Platform boundary 5,010、
`platform_thread` 实现 857、`platform_fs` 实现 2,073，三者互斥合计 7,940 行。LOC 只记录规模，不证明
功能或性能。product candidate/tree 保持 `5818a7ac9b143b980ac43eb39e2a01dac1e5b108` /
`e0c2a9e25b49350965d1bfb3a39789ece976569e`；后续只按本节 trigger 重开。

`VERIFIED -> CLOSED` 前 registry snapshot 位于
`C:\public\project-hyle\_validation\platform-v5-replay\module-status-before-closed.json`，SHA-256
`2E30C11CB61733471B4460840EAC1D669201863F5FD399F23F0EE6323782C000`。
