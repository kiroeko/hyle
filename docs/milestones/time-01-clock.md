# time-01：单调钟 + 墙钟 + sleep（ns）

| 字段 | 值 |
|---|---|
| **阶段** | P2 |
| **模块** | time/（新建，零 HYLE 依赖，纯 syscall）|
| **状态** | ✅ 完成（Win x64 MSVC + Linux x64 GCC14.2 双平台；time 0→8 case）|

> **cloc**：**生产 103 行**（clock.h 30 + clock.cc 73）+ **测试 62 行**。
> **clay 对比 ~159（time.h 46 + time.cc）→ HYLE 103**:精简**但功能更全**——加了 clay 没有的**单调钟**,用 ns 统一替掉 clay 的 float us/ms/secs 三个 sleep 变体 + 一堆 tick 别名。8 测试(单调性 10万次/epoch 边界/sleep 下限/Instant)证明非 stub。

## 核心增量：单调钟（clay 缺失的关键能力）

clay 的 time.h **只有墙钟**(`SYS_time_since_epoch_ms`)——而墙钟会被 NTP/管理员**回拨**,拿它算超时/deadline 会静默出错(经典 bug)。time-01 加 **monotonic_ns()**(稳态、不回退):sched 的 timer wheel(time-02)、超时、elapsed 全要它。两钟分明,不混用。

## clay API diff + 决策

- **monotonic_ns()**:clay 无,**新增**。Linux/mac `clock_gettime(CLOCK_MONOTONIC)`;Windows `QueryPerformanceCounter` + 缓存 freq,split 缩放避溢出。
- **realtime_ns()**:覆盖 clay since_epoch_ms(ns 精度替 ms)。Windows `GetSystemTimePreciseAsFileTime` + 1601→1970 偏移转换。
- **sleep_for_ns/us/ms**:覆盖 clay sleep_us/ms/secs。ns 基底(clay 用 float,精度丢失);POSIX nanosleep(EINTR 重入)、Windows Sleep(向上取整,不 under-sleep)。
- **Instant**:clay 无,新增(tokio 式 elapsed_ns/until_ns,header-inline 无 DLL hop)。
- **ns 精度决策**:clay 的 `float us` 在大值丢精度;ns(uint64)是 std::chrono/tokio 的现代单位,也是 syscall 原生给的。

## 对抗式自审（六类）

- **① 正确性**:Windows QPC `ticks*1e9` 在 ~9s 溢出 64 位→用 whole+frac split(已注释);FILETIME 1601→1970 偏移 116444736000000000(100ns ticks)正确(测试断言 realtime 落在 2020-2100,抓 epoch bug);POSIX nanosleep EINTR 用 rem 重入,信号不缩短睡眠。
- **② 性能**:monotonic = 单 syscall(QPC/clock_gettime),qpc_freq static 缓存;rdtsc 更快但有 invariant-TSC/核迁移坑,**标注未来可选**,默认稳妥 syscall。
- **③ 规范**:.cc 用 windows.h/nanosleep/std::this_thread(内部,不出头);公开头只 cstdint。
- **④ clay 覆盖**:全覆盖 + 加单调钟/Instant。**简化登记**:clay sleep_secs(float) 未单列(sleep_for_ms 够;秒级罕见)。
- **⑤ 残留脚手架**:无。
- **⑥ 过度复杂**:无。
- **已知局限**:Windows `Sleep` 粒度粗(~15ms),sleep_for_ns(10ms) 实际可能睡 ~15ms(测试容差已含)——**亚毫秒精确等待是 sched-04(Waker+deadline)的事**,非本基础层。

## 交付物

- `include/hyle/time/clock.h`（monotonic_ns/realtime_ns/sleep_for_* + Instant）
- `src/time/clock.cc`（POSIX clock_gettime / Windows QPC+FILETIME）+ CMake wire
- `tests/time/clock_test.cc`（8）

## 参考

- clay `sys/time.h`(46)。roadmap P2 门槛"单调性"。tokio Instant(elapsed 语义)。
