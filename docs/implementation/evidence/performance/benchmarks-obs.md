# Obs 工程性能证据

<!-- MODULE CLOSURE RECORD: CURRENT -->

> **obs-v4：candidate 前验证中。** 本页不再把 obs-v3 的 `16 slots x 256 ticks / 3-second`
> 合成 replay 冒充 llama.cpp 当前真实消费。v4 只接受当前 Abseil 普通调用点、当前 llama.cpp
> model-load state callback，以及受 `mem_reclaim` 迁移影响的一个 host-adapter registry 回归。

稳定边界、用户选型和本轮施工事实分别见
[Obs architecture](../../../architecture/obs-design.md)、[Obs guide](../../../user/obs.md) 与
[obs-v4 task](../../tasks/obs/obs-v4-structured-event.md)。本页在 immutable candidate 资格化完成前
不签发 `PASS`，也不从手工单次运行推导性能结论。

## 复杂度与资源合同

| 操作 | 当前合同 |
|---|---|
| 编译期 floor 以下的 log、trace OFF | O(0)；参数、condition、period、clock 均不求值，无 backend |
| runtime level drop | O(1)；一次 relaxed threshold load，控制状态不推进 |
| approximate every-N / power-of-two | O(1)；一个 relaxed atomic load/store，无 allocation、产品 mutex、syscall |
| exact every-N | O(1)；一个 relaxed RMW，无 allocation、产品 mutex、syscall |
| first-N | 无竞争 O(1)；relaxed CAS 在竞争下重试，单次重试次数不承诺有界；到达上限后退化为 load |
| injected-time interval | 无竞争 O(1)；一个 relaxed load，边界时 CAS；竞争下重试次数不承诺有界；无 clock、heap、产品 mutex、syscall |
| clock-owning interval | 上述判断加一次 `steady_clock::now()`；CAS 复杂度边界相同 |
| `log_record` / exact `string_view` | producer dispatch O(S)，S 为同步 sink 数且 S<=64；短 `mem_reclaim` guard、每 sink active ref；不扫描 view、不分配、不拿 mutation mutex |
| typed adapter consumption | O(S+A)，A 为 adapter 实际读取的 attributes/correlation/value bytes；HYLE 不做隐藏 schema lookup 或 copy |
| `log_message(const char*)` / format | O(M+S) / O(formatted bytes+S)；format 使用固定 1 KiB caller 栈缓冲 |
| add/remove/flush | O(S) 冷控制路径；snapshot publish 等短 grace period，remove 再等目标 callback 排空 |
| trace ON | 直接展开为相同 Tracy native event，无 HYLE runtime adapter |

`LogValue` 是 constant-size、trivially-copyable tagged scalar；普通 record 不自动捕获 clock、thread、
task context 或 owning string。typed descriptor 只在调用方显式选择 structured record 时存在。
emit 的 O(S) 是同步 callback fan-out 的必要下界；普通应用和当前参考 consumer 都通过一个 process
adapter 进入宿主 logger，后续队列、文件与网络 fan-out 不属于 HYLE producer 热路。

## v4 冻结性能输入

- consumer manifest：`bench/obs/consumer_shapes_v2.json`，SHA-256
  `317d984ccddd0692e604bab733d5491fc7e5d597c36f55c229342f2eec67f652`；
- ordinary contract：`bench/contracts/obs-ordinary-p1-v1.json`，lock
  `5d2d7210782f1bae55c14bd42246f2485ce02ce6b74cf2e1f5bd8e20c8a33759`；
- structured contract：`bench/contracts/obs-structured-p1-v1.json`，lock
  `b35d0c0d81980ba39bc59a8c70048962fd9625fe179632ca801ed34f310594f0`；
- concurrent registry contract：`bench/contracts/obs-concurrent-p1-v2.json`，lock
  `3d527efed8d509b18207b80a7c0f556f908a11fb4ee9f23ffde0df000b32eb34`；
- benchmark sources：ordinary `bench/obs/control_bench.cc` SHA-256
  `d5725d8bf3b50adae62428461c1676379ad5b82b1dcbb4e94482c9385743e004`；structured
  `bench/obs/structured_bench.cc` SHA-256
  `958099d25f88219e6b1c4b817dc2a74555500f55aaabe623209fafc138e1c1d0`；
- current peers/consumer：Abseil `9e15d3d6cc19ae4703f3ad7fa413a789580ec1fc`；llama.cpp
  `8b4b3558f1459c13e4aa38d5c94d306a00dc6acd`；registry baseline
  `f305b9d3fed87a065d300145e5d9e8de3f5a77a0`；
- W1-WIN，MSVC 19.50、Windows SDK 10.0.26100.0、x64 Release/static、LTO OFF、portable ISA、fresh
  build；点值不与 WSL/macOS 池化。每项必须经过 runner 的独立进程样本、A/A、known-delta、timer-floor、
  build/artifact identity 与对应 semantic/resource validator。

### 1. 普通调用点

同一 repeated diagnostic journey：双方执行 4096 次 level/frequency check，每第 64 次按自己的公开
自然 API 格式化并投给一个 synchronous adapter。HYLE 使用 `HYLE_LOG_INFO_EVERY_N`；Abseil 使用
`ABSL_LOG_EVERY_N(...).ToSinkOnly`。这是 natural-route aggregate，不是孤立原子排名。

当前状态：`PENDING QUALIFICATION`。

### 2. 推理 model-load structured progress

锁定 llama.cpp commit 的 `load_progress_callback` 有三个 owner：`text_model`、`mmproj_model`、
`spec_model`。每个 owner 首条、每 200 ms 和末条必发，并向 host state callback 提供
`stages/current/value`。冻结 replay 为每 stage 256 个 50 ms callback，共 768 次判断与 195 条输出。

两条路线构造完全相同的 HYLE typed record：category/event、三个重复 stage attribute、current、double
value、一个 host correlation、source、canonical wall timestamp；经过相同 registry，再由同一个 adapter
遍历所有字段和 value kind。唯一差异是 HYLE `LogIntervalLimiter` 与 source-derived single-owner manual
gate。llama.cpp 只提供真实 consumer control，并不被宣传为 structured logger 性能 peer；完整路径同时
受绝对 consumer budget 与 5% gate 保护，不能用 getter/union 微基准改变 API。

当前状态：semantic validator `PASS`（两条路线各 195 条、390 次完整消费，primary/placement artifact
不同）；正式资格为 `PENDING QUALIFICATION`。

### 3. Registry 机制回归

Obs v4 把重复的 snapshot reader epoch 换为已冻结的 `mem_reclaim` guard，但 per-sink active ref 仍保护
任意 callback。性能门只取正常的八 producer/一个 host adapter，分别裁决 throughput 和 256-emit chunk
p99，peer 是相同源码、相同编译器链接的 immutable pre-v4 HYLE registry。4/16 independent sinks 仍可
在 executable 中运行以观察容量/争用，却是 `DIAGNOSTIC`，不进入 v4 contract，也不能阻挡或签发产品。

当前状态：`PENDING QUALIFICATION`。

## v3 证据如何保留

- v3 ordinary Abseil 与 limiter 结果是历史事实；v4 改了 record layout、registry 与直接依赖，必须用
  新 candidate 重跑，不能靠旧 tree 自动续证；
- v3 `16 slots x 256 ticks / 3-second` 并非锁定 llama.cpp 的精确 source journey，现降级为历史合成
  诊断。它不再出现在当前 contract、当前结论或 user/architecture 的性能声明中；
- 4/16 independent sink 仍是固定 registry 容量与 contention 诊断；没有 named normal consumer 时不
  升为产品门；
- `trace.h` 与 direct Tracy seam 未在 v4 修改；它的旧同事件结构证据可支持“无 runtime adapter”这个
  结构结论，trace ON/OFF correctness 仍需在当前三平台交付复验。它不证明所有 Tracy 事件全面 SOTA。

## 当前可声明与禁止外推

candidate 资格化前只能声明复杂度、资源合同与 semantic validator 结果。不得声明 obs-v4 P1 已通过、
全面 SOTA、跨平台固定纳秒、异步 logger 性能、所有 sink fan-out 或 end-to-end token latency。最终页面
只会记录 runner 深验证过的 process-median/CI 和 bundle identity；Formal 认证独立保持 `PENDING`。
