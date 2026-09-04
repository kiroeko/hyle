# Obs 工程性能证据

<!-- MODULE CLOSURE RECORD: CURRENT -->

> **obs-v4：不可变 candidate 的 P1 工程资格已完成。** 本页不再把 obs-v3 的
> `16 slots x 256 ticks / 3-second` 合成 replay 冒充 llama.cpp 当前真实消费。v4 只接受当前
> Abseil 普通调用点、当前 llama.cpp model-load state callback，以及受 `mem_reclaim` 迁移影响的一个
> host-adapter registry 回归。

稳定边界、用户选型和本轮施工事实分别见
[Obs architecture](../../../architecture/obs-design.md)、[Obs guide](../../../user/obs.md) 与
[obs-v4 task](../../tasks/obs/obs-v4-structured-event.md)。以下结论只绑定 candidate
`14c82a18f8f00b892479ae16abba4d8320d5915c` / tree
`7594a2f13271ba7956c3e685e24c6302a91f015b` 和三份 retained bundle；不从手工单次运行推导结论。

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

资格结果：`PASS`。15 个独立进程样本下，四个 execution-order/placement strata 的 combined exact
verdict 全为 `WIN`，5% non-inferiority 全为 `PASS`。HYLE process median 为
`345.69–349.60 M op/s`，Abseil 为 `61.85–63.24 M op/s`；candidate/peer median ratio 为
`5.5077–5.6112`，四层 99.26% CI 的最低下界为 `5.4030`。这证明冻结的自然 repeated-diagnostic
journey 明显胜出，不外推其他 Abseil logging 能力。

### 2. 推理 model-load structured progress

锁定 llama.cpp commit 的 `load_progress_callback` 有三个 owner：`text_model`、`mmproj_model`、
`spec_model`。每个 owner 首条、每 200 ms 和末条必发，并向 host state callback 提供
`stages/current/value`。冻结 replay 为每 stage 256 个 50 ms callback，共 768 次判断与 195 条输出。

两条路线构造完全相同的 HYLE typed record：category/event、三个重复 stage attribute、current、double
value、一个 host correlation、source、canonical wall timestamp；经过相同 registry，再由同一个 adapter
遍历所有字段和 value kind。唯一差异是 HYLE `LogIntervalLimiter` 与 source-derived single-owner manual
gate。llama.cpp 只提供真实 consumer control，并不被宣传为 structured logger 性能 peer；完整路径同时
受绝对 consumer budget 与 5% gate 保护，不能用 getter/union 微基准改变 API。

资格结果：`PASS`。semantic validator 确认两条路线各 195 条输出、390 次完整消费，且
primary/placement artifact 不同。结果因接近边界按计划扩展到 31 个独立进程样本；四层 combined exact
verdict 均为 `PARITY`，5% non-inferiority 均为 `PASS`。HYLE process median 为
`7.4185–7.4309 M op/s`，source-derived control 为 `7.5758–7.6304 M op/s`；median ratio 为
`0.9715–0.9821`，全部 98.93% CI 落在预声明等价带内。点值约慢 `1.8–2.85%`，属于完整 typed
record、registry 与 adapter 都相同之后 limiter 实现的非 material 差异，不触发产品改写。

### 3. Registry 机制回归

Obs v4 把重复的 snapshot reader epoch 换为已冻结的 `mem_reclaim` guard，但 per-sink active ref 仍保护
任意 callback。性能门只取正常的八 producer/一个 host adapter，分别裁决 throughput 和 256-emit chunk
p99，peer 是相同源码、相同编译器链接的 immutable pre-v4 HYLE registry。4/16 independent sinks 仍可
在 executable 中运行以观察容量/争用，却是 `DIAGNOSTIC`，不进入 v4 contract，也不能阻挡或签发产品。

资格结果：`PASS`。31 个独立进程样本下，两种 execution order 的 throughput 与 chunk-p99 rate 均通过
5% non-inferiority；combined exact verdict 均为 `INCONCLUSIVE`，不宣传确定胜出。throughput 的 HYLE
median 为 `16.34–16.62 M op/s`，pre-v4 为 `15.38–15.42 M op/s`，median ratio
`1.0519–1.0536`；chunk-p99 rate 分别为 `1.572–1.610 M op/s` 与 `1.505–1.537 M op/s`，ratio
`1.0460–1.0475`。置信区间跨越 exact-win 决策边界，但没有暴露 `mem_reclaim` 迁移造成的 material 回退。

### Retained bundle identity

三份 manifest 均由 `hyle-perf verify` 深验证为 `QUALIFIED`、`EXACT`、`CURRENT`，且
`paths_verified=true`：

| Contract | manifest SHA-256 | summary SHA-256 | controls SHA-256 | observations SHA-256 |
|---|---|---|---|---|
| ordinary | `54381e39ed8a390146aea4bd9761f51bfa33d6c4284c01430e93ade6a7ec0681` | `f36d99bb3bf345aafc765e18b92a2553d927586d0ea7e8f3edbe83eb6e8d2af1` | `236926f56256698f88b651c2534048f2dba33503d2b4c4b5985d24a85422de0e` | `c513b740e3b9c3babd37329b77b96721b3339edf5a757be03df6c4e42c423b71` |
| structured | `22bb9d5416f0f0947318e4bd1ce045dfafc4ed6c48c228d6fbdb5d15e65e6d01` | `7cbf305d6ba7603e9eb5e0921accd9ed8e77db9bcc18e671367962b2c4073475` | `9b8c4d406c8500592e29899084da0c1e38fcfee31a47dcfbe8491e058c753a5d` | `fd26298c5aa6b0cd1bf51d6860c6341690f35cc1a73c116a71a83b20ac24f16b` |
| concurrent | `cd42b1879493007a0dc7a8027c3e882847dce06e89197f9c35a0a889710b0b0c` | `8ca4ce843068d9e9a87538236ad7496bc2acca5fca6156e7d92323909d63fb14` | `7430713a523c3566caae2b66650a57308e7113bb200fc8d874323db75dac14d3` | `f55b016e368a7bf0299ab6c501d945978cd51d6c94d46702acfd0556c49b78ce` |

本机 retained roots 分别为 `C:\hyle-scratch\obs-v4-qual-ordinary-evidence`、
`C:\hyle-scratch\obs-v4-qual-structured-evidence` 与
`C:\hyle-scratch\obs-v4-qual-concurrent-evidence`。它们是工程审计输入，不是发布包内容。

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

可以声明 obs-v4 candidate 在上述三个冻结的正常 journey 上通过 W1-WIN 5% P1 工程门：ordinary
Abseil 路线明确胜出，完整 structured progress 路线等价，受影响 registry 路线非劣。不得据此声明全面
SOTA、跨平台固定纳秒、异步 logger 性能、所有 sink fan-out 或 end-to-end token latency；4/16 sink
仍只是诊断。Formal 认证独立保持 `PENDING`。
