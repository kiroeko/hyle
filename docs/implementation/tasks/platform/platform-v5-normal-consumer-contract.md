# Platform v5：正常消费者合同复检

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 本轮按当前七问、能力超集、质因数依赖和“先冻结自然 API、再看性能”的标准回放整个
> Platform 产品。当前只登记只读复检和重开证据；进入 `SCOPED/AUDITED` 前不授权修改产品源码。

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
