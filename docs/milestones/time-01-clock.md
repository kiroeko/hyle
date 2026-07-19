# time-01：单调钟、墙钟、deadline 与 sleep

| 字段 | 值 |
|---|---|
| 阶段 | P2 |
| 模块 | time/ |
| 依赖 | core 宏/头；无编译期 HYLE 模块依赖 |
| 状态 | ✅ 2026-07-19 收口；Windows MSVC + WSL GCC14 + Apple M4 Clang 真测 |

## 交付能力

- `monotonic_ns()`：唯一可用于 elapsed/deadline/timeout 的稳态时钟。
- `realtime_ns()`：Unix epoch ns 墙钟，只用于日志/时间戳。
- `fast_monotonic_ns()` + `fast_monotonic_available()`：x86 invariant-TSC
  profiling 快路径；不支持时回退，绝不用于 deadline。
- `fast_realtime_ns()`：Windows shared-data / Linux
  `CLOCK_REALTIME_COARSE` 的低分辨率日志时间戳。
- `sleep_for_ns/us/ms()`、`sleep_until_ns()`。
- `saturating_add_ns()`、`deadline_after_ns()`、`remaining_ns()`：统一消除
  `now + timeout` 溢出和过期减法下溢。
- `Instant`：C++ 轻量 elapsed 快照；未来 instant 会钳到 0。

纯 C/FFI 对应入口在 `include/hyle/c/time.h`，指南见
[`docs/guide/c-time.md`](../guide/c-time.md)。

## 收口修整

1. C++ `monotonic_ns`/`realtime_ns` 改为 header-inline 的 `std::chrono`
   标准钟。此前共享构建中调用路径是 caller → `hyle_time.so` →
   libc/libstdc++，Linux 15-run 配对测出约 6–7.5% 真接缝损失；内联后只剩
   标准钟本身的一跳。C ABI 仍由 DLL 导出稳定 wrapper。
2. deadline 与单位乘法全部饱和；`Instant::elapsed_ns` 不再在未来 instant
   上无符号下溢。
3. Windows `Sleep(DWORD)` 分块处理超长时长，避开 `INFINITE` 特殊值和
   强转截断；POSIX `nanosleep` 只重试 `EINTR`，其他错误 fail-fast。
4. `sleep_until_ns` 每次醒来重读单调钟，容忍系统早醒。
5. fast realtime 的 epoch 乘法饱和，POSIX clock 调用失败不再静默使用
   未初始化 timespec。

## 正确性边界

- `fast_monotonic_ns` 无全局 max 钳制，跨核/恢复/VM 迁移可能产生毛刺；
  它的价值是高频 profiling，而不是“更快的 deadline 时钟”。
- `fast_realtime_ns` 值的单位仍是 ns，但分辨率通常只有 1–15ms。
- Windows `Sleep` 本身是毫秒级且可能被调度器延迟；接口保证不主动向下
  取整，不保证精准唤醒。
- 所有 deadline 均使用 absolute monotonic ns；墙钟跳变不得进入调度路径。

## 验收

- 16 个 clock 测试：单调性、epoch、sleep 下限、coarse 对齐、TSC 校准/
  fallback、deadline 饱和、过去/未来 sleep-until、未来 Instant。
- 纯 C smoke 覆盖每个时钟与便利入口。
- sched/io 已改用 `deadline_after_ns`/`remaining_ns`，不再各自手写易溢出算术。
- clock benchmark 与 `std::chrono` 比较完全相同的“转换为非负 uint64 ns”
  工作，而不是拿 MSVC system_clock 的原生 100ns count 充 ns 基线。

## 参考

- `std::chrono::steady_clock/system_clock`
- MSVC `__msvc_chrono.hpp` QPC 频率特判
- Abseil TSC 频率配对校准思路
- Tracy invariant-TSC CPUID 探测
