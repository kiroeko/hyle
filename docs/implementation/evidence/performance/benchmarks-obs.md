# Obs 工程性能证据

<!-- MODULE CLOSURE RECORD: CURRENT -->

> **obs-v3：`QUALIFIED / PASS`。** 当前 candidate
> `21c07d15c4eac477c7b9a768b6475850af87f463` / tree
> `3db40042b12c25a5f9cf562d92de42caa077c057` 在 W1-WIN 的普通限频日志和完整推理进度路径上通过
> 5% P1 engineering non-inferiority gate。结论只覆盖下述一个宿主 adapter 的真实消费形状，不是
> Formal 证书，也不宣称异步 logger、4/16 独立 sink 或所有未消费组合全面 SOTA。

稳定边界、用户选型和本轮施工事实分别见
[Obs architecture](../../../architecture/obs-design.md)、[Obs guide](../../../user/obs.md) 与
[obs-v3 task](../../tasks/obs/obs-v3-closure.md)。obs-v2 的多 sink 聚合结论已经成为历史压力诊断，
不再冒充普通推理拓扑或当前 blocker。

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
| `log_record` / exact `string_view` | O(S)，S 为同步 sink 数且 S≤64；不扫描 view、不分配、不拿 mutation mutex |
| `log_message(const char*)` / format | O(M+S) / O(formatted bytes+S)；format 使用固定 1 KiB caller 栈缓冲 |
| add/remove/flush | O(S) 控制路径；remove 等待旧 reader/callback 排空 |
| trace ON | 直接展开为相同 Tracy native event，无 HYLE runtime adapter |

控制状态只在使用对应宏或显式对象时存在；没有给每条 `LogRecord`、sink snapshot 或普通 emit 增加
metadata。emit 的 O(S) 是逐 callback fan-out 的必要下界；正常应用和 llama.cpp 都通过一个 process
adapter 进入自己的 logger，后续队列/文件/网络 fan-out 不属于 HYLE producer 热路。

## 冻结消费合同

- contract：`bench/contracts/obs-control-p1-v1.json`，SHA-256
  `e97c70dc41b63dfc0c3e238db746fccd15cba43f4ecc45c03f4ac6945116e9cc`；
- consumer manifest：`bench/obs/consumer_shapes_v1.json`，SHA-256
  `c614a48f792237b7ae243f8ae645969548fb6906539fbdb6816dc113b24fb514`；
- benchmark source：`bench/obs/control_bench.cc`，SHA-256
  `d5725d8bf3b50adae62428461c1676379ad5b82b1dcbb4e94482c9385743e004`；
- ordinary peer：Abseil `81c993af4fb88b3a51829a796df02251fa27102f`；
- inference consumer/control：llama.cpp `7584430716ee229751771ed0d6bbcb780d105eeb`；
- W1-WIN，MSVC 19.50、Windows SDK 10.0.26100.0、x64 Release/static、LTO OFF、portable ISA、fresh
  build；每个 stratum 15 个独立进程样本；
- primary/placement × candidate-first/peer-first 四个 strata；A/A、5%/10% known delta、timer floor、
  artifact/build identity、consumer provenance 与 semantic/resource oracle 全部 `PASS`。

### 普通调用点：4096 次 check，64 次自然格式化和一个 adapter

双方都使用自己的公开自然形状：HYLE `HYLE_LOG_INFO_EVERY_N`，Abseil
`ABSL_LOG_EVERY_N(...).ToSinkOnly`。数字是每次 check 的绝对中位数；ratio > 1 有利于 HYLE。

| Stratum | HYLE ns/check | Abseil ns/check | ratio | 98.93% CI | 5% NI |
|---|---:|---:|---:|---:|---|
| primary / candidate-first | 2.6947 | 16.0670 | 5.9526 | [5.9009, 6.0029] | `PASS` |
| primary / peer-first | 2.6794 | 15.6439 | 5.8361 | [5.7518, 6.0125] | `PASS` |
| placement / candidate-first | 2.7045 | 16.1370 | 5.9789 | [5.9292, 5.9951] | `PASS` |
| placement / peer-first | 2.6872 | 15.8233 | 5.8891 | [5.8660, 5.9903] | `PASS` |

这是同语义 aggregate 的工程非劣/优势证据，不外推到 Abseil 的 category/configuration、fatal 或异步
能力。差距主要包含双方自然 record/format 实现，不是孤立原子指令排名。

### 推理进度：16 slots × 256 ticks，80 次实际输出到一个 adapter

每个 slot 自己持有状态；每轮调度共享一次注入的 50 ms monotonic tick，三秒 cadence 共 4096 次判断
与 80 条 preformatted record。peer 是 llama.cpp 当前手写 per-slot elapsed-time gate；输出路径、adapter
与观察计数完全共享，只比较把手写状态换成 HYLE 精确 limiter 后的完整消费者成本。

| Stratum | HYLE ns/check | manual ns/check | ratio | 98.93% CI | 5% NI |
|---|---:|---:|---:|---:|---|
| primary / candidate-first | 1.3501 | 1.3482 | 1.0032 | [0.9916, 1.0143] | `PASS` |
| primary / peer-first | 1.3444 | 1.3481 | 1.0020 | [0.9902, 1.0206] | `PASS` |
| placement / candidate-first | 1.3741 | 1.4542 | 1.0612 | [1.0244, 1.0899] | `PASS` |
| placement / peer-first | 1.3611 | 1.4494 | 1.0635 | [1.0381, 1.0898] | `PASS` |

最慢 HYLE stratum 约 5.63 us/完整 4096-check replay，已经**包含** 80 条 adapter record，仍远低于冻结
的 100 us aggregate budget。四种姿势全部通过 5% NI，故不是靠一个执行顺序或地址布局挑出的胜负。

## 绝对成本诊断，不参与 SOTA 排名

`LogIntervalLimiter` 提供的精确并发 at-most-one 语义比 llama.cpp 单 owner 的普通 bool/time state 更强；
二者单独判断的成本又落在亚纳秒量级，因此 limiter-only 行在合同中预先标成
`NOT_COMPARABLE / diagnostic`，不能改变 API 或决定产品 performance gate。

| 诊断 | W1-WIN 观察 | 冻结预算 | 裁决 |
|---|---:|---:|---|
| injected-time exact limiter，4096 checks | 0.5034–0.5536 ns/check；2.06–2.27 us/aggregate | ≤25 ns/check；≤100 us/aggregate | `PASS` |
| clock-owning suppressed limiter | 5 个独立进程 12.59–12.98 ns/check | ≤100 ns/check | `PASS` |
| 同一 probe 的 `steady_clock::now()` | 11.85–12.05 ns/read | 仅解释成本来源 | diagnostic |

clock-owning probe 使用 candidate public header、4096-call batch 和 nanobench；source / executable
SHA-256 分别为
`BA27389CA244DB999818B3365E17541E8F3DD790325AD34B463E27BD69C20C2B` /
`B5991ECA7832D62FD19683BA2A0E1EFBED0A3443403328FC772C137A49948DC5`。它证明宽松绝对预算，未经过
P1 相对 peer qualification，因而不用于“更快”宣传。

## 未变化路径的 retained 证据

obs-v3 没有修改 `src/obs`、registry/record layout、基础 `HYLE_LOG_<LEVEL>` 展开或 `trace.h`。因此下列
exact implementation identity 证据仍支持这些路径，但不会替代上面的 v3 新能力合同：

- W1-WIN 单 adapter preformatted 42.6–43.9 ns/emit（≤100 ns）、`std::format` 93.8–97.8 ns/emit
  （≤150 ns），相对冻结 HYLE baseline 的双顺序 5% NI `PASS`；
- direct Tracy v0.14.1 counter/named-zone/annotation 共 12 个 strata 全部 5% NI `PASS`；HYLE 只宣称
  native seam 无材料级 adapter tax，不宣称比 Tracy event 更快；
- 8 producer × 1 sink 的 throughput/tail 5% NI `PASS`。旧 4/16 sink 的 n=31 区间未决仍保留为容量/
  contention 事实，但 consumer manifest 已证明它不是普通 HYLE/llama.cpp 产品拓扑，所以不进入 v3
  product gate。若出现真实多独立 sink 消费者，必须用新 round 和它自己的贡献/KPI 重开。

历史 bundle 与完整点值保留在 `obs-v1`/`obs-v2` task，不在本页重复易腐表格。

## 可复验证据与当前裁决

资格化 bundle：

```text
C:\hyle-scratch\obs-v3\qual-evidence-21c07d15\manifest.json
```

- manifest SHA-256：`77A4DBAC0E8883046759D36585B6CA8631E14517CA63D35FC27639D5072CE547`；
- summary SHA-256：`0A5B04FFE639243AC0C5BB8646F1A88EF399339FBC02584E118AC40C7723CB9E`；
- controls SHA-256：`CC92B4DFABB24D6A372E1F7471C67736FD0C385C985B22C517EEC73EEB4D6EAC`；
- observations SHA-256：`017F3D7AAB47C7F63D6A5C2C0405DEE6B5EED243CC7493920BE0478FDFD67650`。

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/obs-control-p1-v1.json `
  --manifest C:\hyle-scratch\obs-v3\qual-evidence-21c07d15\manifest.json
```

当前 verifier 返回 `qualification=QUALIFIED`、`paths_verified=true`、
`provenance_compatibility=EXACT`、`environment_policy_compatibility=CURRENT`、
`current_certification_policy_eligible=true`；qualification gate 与 product performance gate 都是
`PASS`。因此 obs-v3 的 P1 engineering performance 可签 `PASS`。Formal 仍为 `PENDING`；candidate、
consumer/provenance、peer、工具链、算法/API/layout 或直接依赖发生 material 变化时必须重新资格化，
不能从本页点值自动续证。
