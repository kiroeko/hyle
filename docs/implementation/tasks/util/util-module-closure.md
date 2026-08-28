# util 两层严格收口（HISTORICAL，2026-07）

> **历史记录，不是当前 API 或证书。** 本页忠实记录 2026-07 当时存在的 util-core、util-runtime
> 与纯 C adapter；BUILD-03 后旧 C schema/bridge/动态导出已经删除。当前 `util_core` v1 的结论见
> [`util-core-v1`](util-core-v1-closure.md) 与
> [`capability replay`](util-core-capability-replay-v1.md)，当前 `util` runtime 记录见
> [`util-runtime-v1`](util-runtime-v1-closure.md)。唯一当前状态只看
> [`module status`](../../status.md)。下文的 C ABI、动态符号和旧跑分不得作为当前使用说明。

## 当前状态（2026-07-30）

| 审查单元 | 结论 |
|---|---|
| `util-core`：Ref/ScopeGuard/Rng/type-name/SpinLock | ✅ `3c32be6`：独立组件、成熟形状、三平台正确性/TSan/性能收口 |
| `util-runtime`：Registry/demangle/纯 C adapter | ✅ `1b148ef`/`70fa4e8`：fallible/raw 双形状、共享 error/mem、完整 C ABI、三平台正确性/TSan/性能与 package 收口 |

当时完整 util 曾按旧方法暂时结束关注；这不是当前结论。当前方法见
[`module-closure-methodology.md`](../../../developer/review/module-closure-methodology.md)。
长期边界见 [`../architecture/util-design.md`](../../../architecture/util-design.md)，用户选型见
[`user/util.md`](../../../user/util.md)，Native FFI 当前准入边界见
[`native-ffi-policy.md`](../../../architecture/native-ffi-policy.md)，
三平台性能证据见 [`../performance/benchmarks-util.md`](../../evidence/performance/benchmarks-util.md)。

## 为什么分两层

`util-core` 是零 HYLE binary 依赖的 `hyle::util_core` INTERFACE component，供低层 mem/sched/
graph 和普通 C++ 用户消费。Registry 的形状依赖 container SwissTable/string view 和 mem，
因此 Registry、demangle 与 C adapter 属于 `hyle::util` runtime。中央 component graph 与真实
target link 都表达 `util -> error + mem + container + util_core`，没有制造 container↔util 环。

## util-core 结论

- ScopeGuard 与 Abseil/Folly 同形：栈内、move-only、一次执行、dismiss/invoke；析构 callback
  抛异常会 terminate，guide 明确优先资源自己的 RAII。
- Ref 是单指针 intrusive strong ownership，补齐异构 identity/hash、const conversion 和三类
  pointer cast；无 weak/aliasing 的选择边界公开。
- Rng 是 xoshiro256++ 标准 URBG，Lemire bounded draw、iterator shuffle、jump/long_jump 与
  TLS 非安全便利流完整；安全随机明确不在范围内。
- SpinLock 是 1-byte TTAS/backoff。一次 acquire-only RMW 令 M4 8-thread handoff 跌到 oneTBB
  约 0.47x，`3c32be6` 恢复 acquire-release 后双 QoS 回到同档；高争用和长临界区不选 spin。
- mem LeakTracker/ConcurrentVector 等锁内会分配或执行用户代码的误用已改 mutex；真正短固定
  状态/指针转移的 mem/graph 路径保留 SpinLock。

## util-runtime 发现与修复

### 面向消费者的 Registry

- 定位为 startup-registration/service-dispatch named factory，而不是 benchmark map 或动态模块
  unload manager。拥有 key/help；priority 决定覆盖，避免链接/静态初始化顺序成为隐藏策略。
- mutable 阶段线程安全；creator/visitor 都在锁外调用且可重入。mutable 枚举分配时间点快照，
  frozen 枚举无锁、无分配。
- `reserve` 后使用一张 Swiss index；`freeze()` 原地 release-publish，单向、幂等、O(1)、零分配。
  普通 Registry 读取保留一次生命周期 acquire；高频 dispatch 缓存非 owning `FrozenView`，直接
  查表。creator 内联于 lookup value，key 与冷 help/priority 元数据分区。
- 明确不提供 unregister：动态库、created object 和并发 reader 的 quiescence 属于 module
  manager。Registry 析构前由 owner 排空操作，且必须活过 FrozenView。
- factory 返回值贴近消费错误模型：raw nullable 的 missing/NULL 保持最短路径；
  `Registry<Result<T>>` 把 missing 映射为 `NotFound` 并原样传播 creator error。两者表布局相同，
  没有为了 benchmark 添加每项元数据；default creator 同样支持 `Result<Ref<T>>`。

### C ABI 与底层模块接缝

- 删除历史 `hyle_util_last_error`，所有可失败控制面统一返回 owning `hyle_status`；Ref、Registry、
  SpinLock 的 heap handle 都接受 `hyle_allocator`，不私自发明错误/分配协议。
- 当前为 47 个动态符号 + 10 个 header-inline 便利入口。纯 C smoke 覆盖全部入口；字符串使用
  length-aware view 并提供 cstr inline；RNG 有 POD/scalar/batch/discard/jump/shuffle，Ref/Registry/
  SpinLock 以 opaque handle 擦除。
- 删除 C heap ScopeGuard。C 没有词法析构，手动 destroy 的 heap handle 没有提供 guard 的安全
  语义；C 用 goto-cleanup，宿主语言用自身 defer/finally。
- SpinLock 同时支持 allocator-backed heap 与 caller-owned aligned storage；C creator 直接返回
  对象并接收可选 status。checked Registry create 传播 owning error，nullable hot create 传 NULL
  让 creator 不构造诊断，诊断成本不污染热路。
- demangle 保持 util binary 的单个冷路径符号；为一个符号再拆 component 不产生有意义的独立
  能力族。

## 验证证据

- Windows MSVC 19.50 Release：util 4/4；installed shared/static 的纯 C 与 C++ consumer 四组通过。
- WSL GCC 14 Release + `HYLE_WERROR=ON`：util 4/4；ASan/UBSan 4/4；TSan 在 WSL 需禁用 ASLR，
  28 个非 death C++ case + C ABI test 无 race。
- Apple M4 / AppleClang 21 Release + `HYLE_WERROR=ON`：util 4/4；performance 分
  user-interactive/utility 两档。
- C Registry 自有字符串、allocator failure、freeze race、可重入 factory/visitor、并发读取、
  checked/hot 分流与 callback 生命周期均有回归。

## 性能结论

runtime benchmark 使用 Registry N=256 的真实 lookup+factory call。Windows/Linux/M4 上 mutable
Registry 对 `mutex+std::unordered_map` 全部 WIN；推荐 FrozenView 对 Abseil immutable map 为
WIN/WIN/TIE，多线程为 TIE/WIN/TIE；同错误合同的 `Registry<Result<void*>>` 为 WIN/WIN/TIE。
C nullable frozen create 对 std 单/多线程三平台全胜。

M4 utility 普通 Registry 因保留 lifecycle atomic 单线程约慢 Abseil 4.5%，absolute TIE；明确
缓存 FrozenView 后回到 TIE。Linux 11-byte raw hash 比 Abseil 慢约 14%，但它只是 container
原语诊断，完整 Registry 路径仍胜；没有为孤立微基准扭曲产品 API 或增加常态元数据。

## 代码量快照

`cloc`（tracked workspace；自实现语言取 C++/C header/C/CUDA）：全库代码 **281,579**；
自实现含测试 **136,674**；自实现不含测试 **93,690**；当前 util-runtime 实现
（Registry/demangle/C adapter，不含测试/benchmark/文档）**1,614** 行。

## 重新打开本模块的触发器

- 出现需要动态装卸 shared library 的真实消费者：新增独立 module manager，不给 Registry 偷加
  无法证明安全的 unregister。
- Registry 规模/读写生命周期明显偏离 init-heavy/read-heavy，或 FrozenView 生命周期在真实 API
  中难以管理。
- 新 ABI 增删、allocator/error 约定变化、RNG 状态布局变化，或 sanitizer/三平台 peer 证书回退。
- 下游模块收口发现 util 接缝的真实缺口；修复后必须回跑受影响闭包和 installed-package probes。
