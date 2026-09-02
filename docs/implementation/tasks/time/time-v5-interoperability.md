# Time v5：系统时间互操作补全

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`SCOPED`。本页已保存击穿 v4 能力边界的新事实并冻结本轮决策合同；完成能力审计并冻结
> 有限 `FIX` 集以前，不写 Time 产品代码、不新增 benchmark。权威状态见
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

## Closure card（`SCOPED`）

- component / owner / round / kind：`time` / Time owner / `time-v5-interoperability` / `REOPEN`；
- baseline / trigger：`1f18f8fc26298b39524420735c7cc4713b1ee729` /
  `TIME-INTEROP-COMPLETENESS-2026-09-02`；
- 使用原型：
  - `NORTH_STAR_JOURNEY`：模型服务、trace/metadata/protocol 在 signed Unix s/ms/us/ns 与宽
    `UnixTimestamp` 间 checked round-trip，包括 pre-epoch；
  - `REAL_CONSUMER`：libevent/llama.cpp-shaped socket/select timeout 把 typed `Duration` 转为 `timeval`，
    正 duration 不得因精度降低而早醒；
  - `PROVISIONAL_ARCHETYPE`：gettimeofday/stat 等绝对 `timeval` 与 canonical timestamp 双向转换；
  - `REAL_CONSUMER`：Windows 文件/IO metadata 的 `FILETIME` 与 timestamp 互转，不手写 high/low packing；
- 范围：完整审计并在冻结后实现 common signed Unix unit、absolute timeval、relative timespec/timeval
  与 direct Win32 FILETIME 适配；明确一般 duration 与 timeout rounding；新 native adapter header 隔离
  OS include cost；更新 Time package sample、architecture/user guide 和三种 C++ 交付；
- 非目标：calendar/timezone/locale/RFC 文本、filesystem/device clock、socket API 本身、scheduler
  wait/cancel、timer_queue、host clock/deadline/sleep/ManualClock 调优、Platform/IO 已关闭产品迁移、C 或
  Native FFI、稳定 C++ ABI、公共仪器和 Formal；
- 直接依赖：保持 `PUBLIC core`，无新第三方产品依赖；标准 chrono、POSIX/Winsock/Win32 类型只是边界
  mechanism。Platform/IO 只编译回归，不跨其状态轮修改；
- tier / instrument：`P0` / `closure-correctness-v1`。这些是未被 profile 证明为热点的 fixed-work pure
  conversion；本轮签 O(1)、无 allocation/lock/syscall/dynamic dispatch、可内联值路线或单 ABI call 的
  结构成本，不运行 timing benchmark，也不重放未改 host clock 的 v4 P1；
- API 档位：`UnixTimestamp` 继续是唯一宽范围 canonical absolute value；精度降低必须在名称/文档中明确
  floor/trunc/timeout-ceil 语义；invalid/noncanonical/overflow 返回 `optional` 或 output+`bool`，不饱和成
  另一个时刻；native struct 不是 HYLE wire/FFI layout；
- correctness gates：epoch 两侧、每种 unit 边界/overflow、invalid subsecond、lossy rounding、Duration
  min/max、timespec/timeval absolute-vs-relative、FILETIME 1601/1970/UINT64 边界、constexpr/value
  properties、Windows include order、三平台 compiler 与 sanitizer；
- delivery gates：精确 `time` factor；新增 header owner；source/static/shared relocated Time consumer；
  Windows shared/static 符号与 POSIX/macOS 头闭包；Platform/IO compile regression；
- 文档/易用性：先按“绝对时刻、一般 duration、timeout”选入口，明确 Windows 需要提供 native type
  definition、负 epoch、精度损失、optional/output-bool 和不能把 realtime 当 deadline；
- 调查/实现预算：本轮最多四小时；每个能力族最多两个 mature peer；P0 不得因点值扩建 benchmark。
  若 native type 的跨平台头/ABI 无法形成自然合同，记录一个具名 blocker 并停止；
- 停止条件：所有 `MUST` 为 `COMPLETE/ROUTED`、七项适用门有 verdict 后关闭；material correctness、
  delivery 或 API 语义未决进入 `BLOCKED`，不得用继续研究替代裁决。

下一步只完成 capability-family map、现有 public-surface reverse trace、gap ledger 与 API/cost freeze；
未到 `AUDITED` 不实现。
