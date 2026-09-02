# Time v5：系统时间互操作补全

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`REOPENED`。本页先保存击穿 v4 能力边界的新事实；进入 `SCOPED`、完成能力审计并冻结有限
> `FIX` 集以前，不写 Time 产品代码、不新增 benchmark。权威状态见
> [`module-status.json`](../../module-status.json)。

## Reopen trigger

- component / owner / round：`time` / Time owner / `time-v5-interoperability`；
- trigger ID：`TIME-INTEROP-COMPLETENESS-2026-09-02`；
- baseline：`1f18f8fc26298b39524420735c7cc4713b1ee729`；
- 新事实：owner 明确要求补审 Time 互操作。当前 public surface 只把 `UnixTimestamp` 与 fixed-ns
  system time point、`time_t`、绝对 `timespec` 和 raw Windows FILETIME ticks 互转；它没有覆盖成熟 Time
  库与真实网络/推理 host 都反复使用的 `timeval`，也没有区分 POSIX struct 作为绝对 timestamp 与相对
  duration/timeout 的两种语义；常见有符号 Unix s/ms/us/ns 和直接 Win32 `FILETIME` 仍要求调用者手写
  易溢出、易错负 epoch 或平台字段代码；
- 受影响旧结论：Time v3 “Unix/OS interchange 无单向缺口”以及 architecture/user guide 的完整性描述；
  v4 已签的 host clock、deadline、sleep、ManualClock 与 timer_queue correctness/P1 evidence 不受无界重审，
  但相关公共头、交付和下游 Platform/IO 编译必须作为回归门。

锁定审计输入：

- Abseil `cc05267c15127e41d497ddfc985328e59992ea53`：Unix unit、chrono、`time_t`、timespec/timeval 的
  absolute/duration 双路线；
- Folly `176c088840724715b2c3c6211b2b347611676002`：checked chrono↔timespec/timeval、溢出与规范化；
- Boost `78505eab9b203e3e26e83893d0685f160133fa0f`：不污染公共头的 Win32 FILETIME 表达；
- libevent `d82464a277d0f42703702c4dfd9af6af38595a83` 与 llama.cpp
  `b96806d96061049a5b574269b049bf6241d63d46`：`timeval` timeout/network consumer；ggml
  `d4716378882593333721eb33f153144b6885caf2`：紧凑 monotonic ms/us 只作需求差异输入。

## 进入 `SCOPED` 前的有界问题

1. 绝对 timestamp：是否应补 signed Unix s/ms/us/ns、`timeval` 与 direct `FILETIME`，同时保持
   `UnixTimestamp` 为唯一宽范围 canonical value？
2. 相对 duration/timeout：timespec/timeval 转换应如何区分一般数值转换与“不早醒”的 timeout rounding，
   而不制造含糊 API？
3. Header cost：POSIX/Winsock/Windows 类型应放在显式 opt-in adapter header，不能把 `windows.h` 或
   Winsock include-order 成本施加给普通 clock consumer。
4. 路由：timezone/calendar/RFC 文本、filesystem clock、device clock、scheduler wait/cancel 与 C/Native
   FFI 是否都有明确 owner，不能借“互操作”无限扩张？
5. 性能：新增纯值转换若未被 profile 证明为热点，应以 P0 的 O(1)、constexpr/inline、零分配/锁/syscall
   与 code shape 收口，不重新计时已冻结 host clock。

下一步只允许冻结 closure card 与 capability/public-surface ledger；产品实现必须等到 `AUDITED`。
