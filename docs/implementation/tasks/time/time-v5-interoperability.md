# Time v5：系统时间互操作补全

<!-- MODULE CLOSURE RECORD: ACTIVE -->

> 当前阶段：`CANDIDATE`。八项冻结 `FIX` 已落在不可变产品候选；后续只运行 closure card 预声明的
> correctness、delivery、P0 code-shape、文档与红队验证，不新增 API 或 benchmark。权威状态见
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

以下 closure card 与能力审计回答这些问题；旧结论只在列出的互操作面被击穿。

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

## Capability-family coverage map（`AUDITED`）

| Family | normal journey / peer | current route | coverage | decision / negative space |
|---|---|---|---|---|
| canonical absolute value | 文件/协议/metadata；Abseil Time | signed seconds + canonical nanos | `COMPLETE` | 保持唯一 `UnixTimestamp`，不造第二种 epoch value |
| chrono / `time_t` | ordinary C++、stat/HTTP date | fixed-ns system time point 与 `time_t` 双向 checked | `COMPLETE` | 任意 clock/domain 继续由 typed chrono 明示，不跨 clock 猜 epoch |
| signed Unix integer units | DB/trace/JSON/protocol；Abseil | caller 手写或先压到 ns | `MISSING` | 补 s/ms/us/ns；to 方向 checked，精度降低向负无穷取整 |
| absolute POSIX structs | clock/gettimeofday/stat；Abseil/Folly | timespec 完整，timeval 无 | `PARTIAL` | 补 strict canonical timeval；不把 invalid subsecond 静默 normalize |
| relative POSIX structs | nanosleep/futex/select；Abseil/Folly/libevent | OS owner 重复拆 fields | `MISSING` | 补 Duration↔timespec、timeval trunc 与 timeout-ceil；不拥有 syscall |
| Windows FILETIME | Platform/IO file metadata；Boost/Win32 | raw 100ns ticks 完整，caller 重复 high/low packing | `PARTIAL` | 补 native struct adapter；raw tick codec 保留为 portable core |
| civil/timezone/text | calendar、RFC3339/HTTP-date | std::chrono / protocol owner | `ROUTED` | 不复制 tzdb/parser/formatter；HTTP date 属协议层 |
| filesystem/device/native clock IDs | file clock、GPU events、clockid/itimerspec | Platform/Compute/IO + std clock conversion | `ROUTED` | epoch/queue/handle policy不同，不收入 Unix adapter |
| scheduling / FFI | wait/cancel/periodic、foreign runtime | Sched/IO 与 Native FFI admission | `ROUTED` | native structs 不是稳定 C ABI；本轮无 opaque runtime object |

### 现有 public-surface 与内部重复反向追踪

- `unix_timestamp.h` 的 value、valid/factory、fixed-ns time point、`time_t`、absolute timespec 与 raw
  FILETIME ticks 全部有当前 Platform/IO/clock/tests 或普通 C++ journey；保留，不改 layout。
- `clock.h` 的 typed/raw precise/coarse/profiling/CPU、Duration/deadline/Instant/provider/sleep 不负责 native
  struct serialization；v4 P1 身份不变，本轮不以互操作 helper 重写读钟路径。
- Platform/IO 三处 Win32 file metadata 重复 high/low packing，证明 direct FILETIME adapter 是真实边界；
  但两个 owner 未处于本轮，保持产品不变，只做 compile regression，后续自己的 reopen 才迁移。
- Platform futex 与 IO kqueue 的 `uint64_t timeout_ns` 还包含超出 signed `Duration` 的范围或
  `UINT64_MAX` sentinel，不能为提高内部采用率强迁到 typed adapter；这是合法 domain specialization。
- ggml 的 compact monotonic ms/us 是内部 elapsed convenience；HYLE 已有 typed chrono 与 compact ns，
  增加 raw ms/us 会扩散单位混用，因此 `REJECT`。llama.cpp/libevent 的 timeval timeout 才是本轮相同合同。
- 没有旧 C ABI、native export schema 或 benchmark-only public seam；新增 compiled 函数只形成普通 C++
  decorated Time symbols，不承诺跨 toolchain ABI。

## Capability / gap ledger 与冻结 `FIX`

| ID | class | owner / public shape | cost / semantics | action | verification |
|---|---|---|---|---|---|
| `TIME5-CANON-01` | `MUST` | `UnixTimestamp` layout/validity | 16-byte natural C++ value；canonical nanos | `KEEP` | layout/value/invalid regression |
| `TIME5-UNIT-01` | `MUST` | signed Unix s/ms/us/ns factories + checked extractors | constexpr O(1) integer div/mul；to floors toward -infinity | `FIX` | epoch±1、unit edges、INT64/range property |
| `TIME5-ABS-TV-01` | `MUST` | absolute timeval adapter | strict `[0,1e6)`；timestamp→us floors；failure leaves output unchanged | `FIX` | invalid/range/pre-epoch/round-trip |
| `TIME5-DUR-TS-01` | `MUST` | relative Duration↔timespec | exact ns；strict canonical；overflow checked | `FIX` | Duration min/max、invalid/range |
| `TIME5-DUR-TV-01` | `MUST` | timeval→Duration + `duration_to_timeval_trunc` | exact input；ns→us toward zero，名称暴露 loss | `FIX` | ±1ns/±1001ns、micro-aligned round-trip |
| `TIME5-TIMEOUT-01` | `MUST` | `timeout_to_timeval` | positive ceil to us；nonpositive `{0,0}`，避免早醒 | `FIX` | zero/negative/fraction/max/range |
| `TIME5-FT-01` | `MUST` | Win32 `_FILETIME` adapter | O(1) field pack over existing raw codec；to checked | `FIX` | 1601/1970/max、100ns floor、field order |
| `TIME5-HEADER-01` | `MUST` | `<hyle/time/native_interop.h>` + compiled implementation | Windows forward declarations；普通 clock 不取得 Windows/Winsock headers | `FIX` | self-containment/include order/symbol owner |
| `TIME5-DELIVERY-01` | `MUST` | Time sample/docs/source/static/shared | no new dependency or artifact kind | `FIX` | relocated package + Platform/IO compile |
| `TIME5-CIVIL-01` | `GENERAL` | timezone/calendar/text | std/protocol owner | `ROUTED/REJECT` | guide route only |
| `TIME5-NATIVE-01` | `GENERAL` | file/device clocks、itimerspec、clockid | Platform/Compute/IO owner | `ROUTED/DEFER` | named consumer must prove common value contract |

冻结 `FIX` 只有 `UNIT/ABS-TV/DUR-TS/DUR-TV/TIMEOUT/FT/HEADER/DELIVERY` 八项。任何 clock backend、timer
algorithm、timezone/text、FFI、Platform/IO migration 或计时工作都不进入候选。

## API / usability / cost freeze

- `unix_timestamp.h` 新增 `unix_timestamp_from_unix_{seconds,milliseconds,microseconds,nanoseconds}` 与
  `unix_timestamp_to_unix_{seconds,milliseconds,microseconds,nanoseconds}`。from 接受 `int64_t` 且总可
  canonical 表示；to 对 invalid/overflow 返回 `optional<int64_t>`，低精度结果 floor，全部 `constexpr`。
- 新 `<hyle/time/native_interop.h>` 包含 `clock.h`；POSIX 目标取得完整 timeval，Windows 只前置声明
  `timeval` 与 `_FILETIME`，不 include `winsock2.h/windows.h`。调用者实际构造 native struct 时负责先
  include 对应 OS header；声明头本身在三平台 self-contained。
- compiled `HYLE_TIME_API` 固定为：
  - `duration_from_timespec` / `duration_to_timespec`；
  - `unix_timestamp_from_timeval` / `unix_timestamp_to_timeval`；
  - `duration_from_timeval` / `duration_to_timeval_trunc` / `timeout_to_timeval`；
  - Windows-only `unix_timestamp_from_windows_filetime` /
    `unix_timestamp_to_windows_filetime`，参数使用 Win32 tag `_FILETIME`，所以用户的 `FILETIME` typedef
    自然匹配。
- timeval/FILETIME 的 to 方向使用 output reference + `bool`，因为 Windows 声明头不能以
  `optional<incomplete-native-type>` 返回；失败不得修改 output。from 方向返回 value/optional；invalid
  native subsecond 返回 `nullopt`，不偷偷 normalize。
- timestamp→timeval 与 Unix低精度 extractor floor；一般 duration→timeval 的名字明确 `trunc`；
  timeout 单独 ceil positive 并 clamp nonpositive。三种语义不得合并为 runtime rounding enum 或含糊默认。
- 所有新路线 fixed-work O(1)、无 heap/lock/atomic/syscall/dynamic dispatch/global state。header-only unit
  converter 可常量求值；native incomplete-type adapter只付一次普通 C++ ABI call，不进入已冻结 clock
  hot path，也没有计时声明。
- 新增头由 `time` factor 自动拥有，`timer_queue` split 清单不变；产品用户仍只链接 `hyle::time`。
  公开 value/API 只面向 C++23；Native FFI 保持 `N/A`。

## Product candidate

- product commit：`680c9941d7af787067afae904a8ba886c14ba851`；
- product tree：`b09b704304d943adb8d6442c3790fbe02a3048de`；
- `TIME5-UNIT-01`：四种 signed Unix unit factory/extractor 已为 `constexpr`；extractor 使用无符号幅值
  分解规避 `INT64_MIN` 的可表示最终值被中间乘法误判 overflow；
- `TIME5-ABS-TV-01`、`DUR-TS-01`、`DUR-TV-01`、`TIMEOUT-01`：strict absolute、exact relative ns、
  named trunc 与 positive ceil/nonpositive poll 已分成独立入口，失败不修改 output；
- `TIME5-FT-01`、`HEADER-01`：Win32 `_FILETIME` 只做 field pack 后复用既有 raw codec；Windows public
  header 只 forward-declare native structs，普通 Time include 不取得 Winsock/Windows headers；
- `TIME5-DELIVERY-01`：新增 compiled source 由 `hyle_time` 拥有，sample 真实调用 native adapter；中央
  factor/product graph 与依赖集合不变。

候选前置验证只用于确认可进入验证阶段，不冒充最终 gate：W1-WIN MSVC 19.50 Release/Werror 的 Time 与
timer_queue tests、header self-containment、macro composition、codegen audit 全部通过；selected Time 的
source embedding、relocated shared/static consumer 各 1/1；当前 Platform/IO shared targets 重新编译通过。
待完成 W1-WSL sanitizers、M1-MAC、最终三交付、P0 code-shape/符号与独立红队后才能进入 `VERIFIED`。

任何 API/semantic/cost material 变化必须退回 `AUDITED`，不能在候选后静默修补。
