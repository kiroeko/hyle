# util 模块全面收口（2026-07-19）

| 字段 | 结论 |
|---|---|
| 范围 | Ref/ScopeGuard/type_name/Registry/Rng/SpinLock/demangle + C ABI |
| 状态 | ✅ 功能、红队、C ABI、测试、三平台与性能边界收口 |
| C ABI | 44 个动态导出 + 2 个 header-inline；纯 C smoke 全入口覆盖 |

## 功能与红队修复

1. `Ref<T>` 增加批量计数的零值、溢出、下溢 fail-fast；新增 `RefCompatible`，编译期拒绝 `Derived : RefCounted<Base>` 且 Base 非虚析构这一类最终 `delete Base*` 的危险形状。
2. `Registry` 从“注册静止后读无锁”改为全部表访问同步；creator 指针在锁内复制、锁外调用，因此可重入。修复 move-only 参数未 forward、`Ref` default creator 实际不可构造，以及 GNU `##__VA_ARGS__` 非标准宏腿。
3. `Rng` 增加状态 snapshot/restore、reseed、原生 u32 Lemire、`long_jump()` 和极端有限 double range。固定 seed 输出被测试钉死。`thread_rng` 改用进程唯一序列，修复 OS 复用线程栈时默认流重复。
4. Windows ARM 的 `cpu_relax` 从无条件 `_mm_pause` 修为 ARM `__yield`；MSVC 显式包含 intrinsic 头。
5. Windows demangle 对超过 1 KiB 的合法 `?` decorated symbol 做有上限扩容重试；buffer 合同统一 fail-fast。
6. 下游审计确认 SpinLock 被 mem/container/sched/graph 实际消费；Rng/Ref/Registry/ScopeGuard 主要是用户能力，不为“消费率”强塞进不适合的热路径。调度器 victim 选择保留单字 xorshift：它只是 idle hot-path hint，不需要用户级随机质量。

## C ABI 能力覆盖

按 `c-abi-design.md` / `c-abi-authoring-guide.md` 做能力擦除：

| 原生能力 | C 通路 |
|---|---|
| demangle | buffer-fill |
| Rng | 256-bit POD + scalar/batch/range/shuffle/jump/TLS |
| Ref<T> | opaque payload + destroy callback strong handle |
| ScopeGuard<Fn> | opaque callback+ctx guard |
| Registry<Ptr,Args...> | opaque runtime registry + factory/visitor |
| SpinLock / cpu_relax | opaque lock / header-inline hint |
| type_name<T>/type_hash<T> | 有意 C++ only；运行时无法恢复编译器静态类型 |

机械核对：C 头共 46 个 `hyle_util_*` 可调用入口，纯 C smoke 46/46 均引用；其中 44 个为动态符号，`rng_clone` 与 `cpu_relax` 为 header-inline。

## 验证

- Windows x64 MSVC：`hyle_util_test` 73/73，纯 C smoke 绿；DLL 44/44 动态符号。
- WSL Linux GCC 14.2，本地 `/root/hyle` ext4：旧 70-case 全量 death run 也通过，但 WSL 的 gtest death 子进程收割约耗 103 秒；当前 TSan、ASan、UBSan 下 68 个非 death case + 纯 C smoke 全绿。death contract 另在 Windows/macOS 快速真跑，不为缩短 WSL 仪器时间改变全局 trap 语义。
- Apple Silicon / AppleClang 21：73/73 + 纯 C smoke；ARM64 `cpu_relax`/SpinLock 真编真跑；dylib 44/44 动态符号。
- 并发：原生 Ref/Registry/SpinLock 与 C Ref/Registry/SpinLock 均有多线程压力；TSan 零 race。
- 内存：callback 生命周期、registry 自有字符串、opaque 句柄经 ASan+LeakSanitizer；零泄漏/UAF。

GCC `-Werror` 全库门仍会被既有 SIMD 头的 `ignored-attributes` / unused typedef 警告挡住；util 自身不是报错源。普通 warning 配置、三类 sanitizer 与 AppleClang 均通过。该跨模块 warning 债不伪装成 util 功能缺陷。

## 三平台性能结论

单位 ns/op；单次收口复测用于确认量级和边界，不把高方差争用行宣传成精确 SOTA。

| 项 | Windows | Linux | Apple M4 | 结论 |
|---|---:|---:|---:|---|
| Rng next / mt19937_64 | 1.59 / 2.32 | 1.38 / 1.44 | 2.26 / 3.47* | Win/Mac 赢；GCC 基本平（M4 std 行高方差） |
| runtime-varying range / std | 2.09 / 2.86 | 1.47 / 2.08 | 2.17 / 7.68 | 三平台赢 1.37–3.54x |
| C scalar / C++ scalar | 2.47 / 1.52 | 1.84 / 1.40 | 2.35 / 2.26 | 单值 ABI seam 约 0.09–0.95ns |
| C fill / C++ fill | 4.99 / 5.16 | 0.67 / 0.68 | 0.65 / 0.65 | batch 摊薄后打平 |
| SpinLock / mutex，无争用 | 3.71 / 10.94 | 4.79 / 2.48 | 1.01 / 4.19 | 平台翻转；Linux mutex 更快 |
| SpinLock / mutex，8 threads | 50.17 / 50.59 | 60.44 / 39.08 | 271.88 / 20.98 | 高争用不该用 spinlock，M4 尤其明显 |
| Ref / shared_ptr，fan-out | 10.13 / 8.09 | 8.33 / 11.12 | 3.19 / 3.22 | Windows 输、Linux 赢、M4 平；不作无条件“更快”宣传 |

边界比漂亮倍数更重要：Rng 的 batch C ABI 是成功的；SpinLock 只用于极短低争用区；Ref 的价值是侵入式所有权/少一次控制块分配和稳定对象模型，不是保证每个工具链的原子拷贝都更快。

## 附带修复的 benchmark 仪器债

`scheduler_ext_bench` 仍调用已经删除的 `(Allocator, stack_size, ...)` Fiber 构造器，收口时首次重建即失编译，已迁到 guarded-stack 构造器。原 2,000-round work-stealing shape 又被串行 VM stack setup 淹没，1→8 曲线只剩约 1.45x；A/B 证明与 Rng/xorshift 无关。将计算量提高到 20,000 rounds 后，Win/Linux/M4 的 1→2→4→8 曲线分别为
`1/1.78/2.85/3.95`、`1/1.66/2.47/3.11`、`1/1.74/2.95/4.34`，重新能测调度扩展而不是栈构造。
