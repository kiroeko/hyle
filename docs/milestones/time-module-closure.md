# time 模块收口审计（2026-07-19）

## 结论

time 的功能、红队正确性、下游接入、测试、性能仪器、纯 C ABI 与用户文档已经
收口。已知未解决项不是功能洞：`CommonTimeoutQueue` 在 Apple M4 的同形
TAILQ 对拍中 arm+expire 只有约 0.57x；该项保留为明确平台限制，接入 io 前需用
真实 fixed-timeout workload 再验。

## 1. 功能与下游利用

- clock：monotonic/realtime、opt-in TSC/coarse、相对/绝对 sleep、`Instant`。
- deadline：饱和加法、生成 deadline、剩余时间；sched drain/sleep/CV 与 io
  timeout 测试已统一使用，不再手写可溢出的 `now + timeout`。
- timer：稳定地址 `TimerEntry`、静态 `TimerSource`、8×64 `TimerWheel`、
  fixed-duration `CommonTimeoutQueue`。
- sched 已用 wheel 承载 fiber sleep；io timed recv/accept/connect 经 sched
  wheel 做 timer-vs-IO 仲裁。
- Common queue 尚无生产消费者。这是明确就绪度，不把“实现存在”写成“已服役”。

## 2. 红队修复

| 问题 | 修复 |
|---|---|
| 6×64 horizon 不能覆盖 `uint64_t ns`，远期 timer 可提前触发 | 扩为 8×64；补 former-horizon 与 `UINT64_MAX` 测试 |
| `now + timeout`、us/ms 乘法可能 wrap | 统一饱和算术 |
| bool scheduled 不能识别 wrong source | 精确 owner identity；跨 source fail-fast/C ABI 返回错误 |
| 侵入节点可 move/copy 导致悬空 | 删除 copy/move；析构检查 source/output-list 泄漏 |
| common reschedule 拒绝可能丢旧 arm | `try_schedule` 事务化 |
| source 非空析构与 fired-list 悬挂无诊断 | 生命周期 fail-fast |
| `container_of` 依赖标准布局，但 `TimerEntry` 混合访问控制后不再满足 | owner identity 包装为不可伪造的标准布局成员；core 与 timer 两层 `static_assert` |
| Windows 超长 sleep 截断或撞 `INFINITE` | 有限 DWORD 分块 |
| POSIX sleep 对所有错误盲重试 | 只重试 `EINTR` |
| time → container 假依赖 | intrusion primitive 下沉 `core/` |
| benchmark 用 lazy heap、list+hash 假对手 | indexed eager heap、直接节点 TAILQ |
| benchmark 反复构造节点/可能融合阶段 | epoch 外预分配复用 + compiler memory barrier |

曾实验 null-ended、低位 membership tag 的链表以处理 M4 common expire lose。
三平台正确性虽绿，但 M4 expire 无改善、cancel 从约 1.10x 退到约 0.94x，故撤回；
最终保留简单圆形 sentinel，不把无收益复杂度带进 core。

## 3. C ABI

六件套齐全：

- `include/hyle/c/time.h`
- `src/time/c_time.cc`
- `tests/time/c_time_test.c`（真 C 编译）
- `docs/guide/c-time.md`
- CMake 接线
- guide 索引与 close-01 登记

面包含 34 个动态导出函数和 4 个 header-inline 便利入口。类型只使用整数、
`size_t`、C callback 和 opaque handle；没有 C++ 类布局、STL、模板、异常或
module API 表。timer/wheel/queue 的分配与释放都在同一 HYLE 模块内。

纯 C 测试覆盖时钟、所有 handle 生命周期、NULL 默认、跨 source 错误、顺序拒绝、
callback 中 re-arm 和所有 inline 便利入口。Windows `dumpbin` 已核对 34/34
`hyle_time_*` 动态符号。

## 4. 验收矩阵

| 环境 | time C++ | time C | container/sched/io 回归 |
|---|---:|---:|---:|
| Windows x64 / MSVC Release | 48 pass + 1 TSC-fallback skip | pass | pass / pass / pass |
| WSL `/root/hyle` / GCC14 Release | 48 pass + 1 TSC-fallback skip | pass | pass / pass / pass |
| Apple M4 / AppleClang Release | 49/49 pass | pass | pass / pass / pass |
| WSL GCC14 ASan | 48 pass + 1 skip | pass | pass / pass / pass |
| WSL GCC14 UBSan | 48 pass + 1 skip | pass | pass / pass / pass |

测试构成：1 个 self-contained header、16 clock、19 wheel、13 common queue，
共 49。包含两组百万次 arm/cancel oracle、death/lifecycle、完整 deadline 域、
纯 C callback re-arm。

## 5. 性能证据

N=4096、每平台 15 个独立进程；括号是 HYLE throughput / rival：

| 负载 | Windows | WSL | Apple M4 |
|---|---:|---:|---:|
| wheel arm+cancel | 7.24 vs 16.04 ns（2.23x） | 5.81 vs 6.72（1.14x） | 4.92 vs 8.13（1.65x） |
| wheel arm+expire | 15.37 vs 48.27（3.16x） | 23.37 vs 41.36（1.76x） | 13.54 vs 22.51（1.66x） |
| wheel 70% cancel churn | 11.44 vs 20.16（1.75x） | 14.74 vs 15.34（1.05x） | 8.02 vs 10.57（1.31x） |
| common arm+expire | 3.60 vs 4.25（1.18x） | 3.13 vs 3.94（1.25x） | 4.86 vs 2.76（**0.57x**） |
| common arm+cancel | 2.88 vs 3.14（1.10x） | 1.90 vs 2.59（1.34x） | 3.17 vs 3.09（**0.98x**） |

TimerWheel 三平台三形状全赢真实 indexed eager-erase heap。Common queue 的
M4 lose 不能被 Windows/Linux 结果抵消，也不能用复杂度符号掩盖。

clock 的标准路径与相同工作量 `std::chrono` 三平台统计同档。x86 opt-in
fast monotonic 约 1.8x；M4 无 TSC 正常回退。fast realtime 在 Windows/Linux
约 4.94x/3.36x，macOS 无 coarse 原语而同档回退。

完整命令、仪器判定和局限见
[`benchmarks-sched.md`](../performance/benchmarks-sched.md)。

## 6. 保留边界

- TimerWheel / CommonTimeoutQueue 都是 owner-thread 数据结构，不是并发容器。
- wheel 公开 ns，执行粒度 1ms；保证不早触发，不保证亚毫秒唤醒。
- fast monotonic 只用于 profiling。
- Common queue 只接收单一 duration 形成的非递减 deadline；混合时长用 wheel。
- Common queue 尚未接入 io，且 M4 expire 有明确 lose。
- 全库 `-Werror` 仍会被既有 SIMD/IO 警告阻塞；本轮 Apple 编译发现并修正
  `TimerEntry` 的 struct/class 前向声明不一致，time 自身没有新增 warning。
