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
