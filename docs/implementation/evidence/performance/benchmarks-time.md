# time 工程性能证据

<!-- MODULE CLOSURE RECORD: HISTORICAL -->

> **time-v1：`QUALIFIED / PASS`。** 当前 exact product candidate 在 W1-WIN 的真实 scheduler timer
> 生命周期合同上，对 faithful Seastar peer 的 8 个 required strata 全部取得 `PARITY` 和 5%
> non-inferiority `PASS`。这是具名消费者的 P1 engineering 结论，不是跨平台 Formal 证书，也不
> 宣称每一种 clock、sleep、timer cardinality 或未消费能力都普遍更快。

稳定设计、用户选型和施工事实分别见 [time architecture](../../../architecture/time-design.md)、
[time guide](../../../user/time.md) 和 [time-v1 closure](../../tasks/time/time-v1-closure.md)。本页只保存
当前可复验性能命题；旧 C ABI、旧 candidate、microbenchmark N sweep 和易腐点值可从历史 task/git
追溯，不再冒充当前证书。

## 可证明命题

`TimerQueue` 使用 65-bucket radix：

- arm/cancel/next 为 O(1)，正常路径零分配；
- advance 为 O(触碰桶 + fired/rebucketed 节点)；
- caller-owned stable node、single-owner thread、detached ready list；
- callback/data/owner metadata 和完整 arm/cancel/advance/drain 生命周期都进入工作量。

强 peer 固定为 Seastar commit `fd71a5b4c71b91abd4e93c9d20f88e1fd7465976`、tree
`98e2e8095e099f52dea9bd70dc30b1e778679618` 的 `timer_set` 算法形状。benchmark 是对该成熟实现
的 faithful local port，使双方具有相同 node ownership、deadline/cancel 序列、callback/data/owner
状态和 ready-list drain；不是线性扫描、heap 或被削弱的假对手。

因此当前能成立的结论是：HYLE 已达到成熟 radix timer 的同一复杂度等级，并在下述真实消费者合同
上通过冻结的 5% 非劣门。不能从该结论外推“所有 time API 都比所有库快”。

## 冻结消费者 workload

`scheduler-timer-lifecycle-64` 从当前 `src/sched/scheduler.cc` 的实际使用反向构造：

1. 64 个独立分配、地址稳定且散布的 timer owner/node；
2. 固定分布的 absolute deadline 和约 70% cancel；
3. 每个 event-loop tick 逐步 advance，而不是一次把所有节点冲掉；
4. 最终 drain ready list，观察 callback/data/owner 和 checksum；
5. allocation/input construction 在 epoch 外，计费包含完整 queue 生命周期操作。

这是一段 deterministic source-consumer replay，不冒充生产 trace。修改 replay、计费边界、consumer
provenance、ownership topology 或首席 scheduler 使用形状，都必须使用新 contract ID 重签。

## time-v1 W1-WIN 结果

- candidate commit `0ef2544ceece030bb522857854285d280e368d36`，tree
  `104a4b6dc904e5b261010ace820ec2c474374c31`，clean；
- contract `bench/contracts/time-source-consumer-v1.json`，semantic SHA-256
  `ab4223b7066e8540f15bbc6ad586ef7e37cf678feb877f408162e7d08bec2356`；
- benchmark source SHA-256
  `9d345e9d3bd046cdaf79ef73b3c1c6e78902767e0685efd70cdea2d430783b90`；
- W1-WIN，MSVC `19.50.35728.0`，Windows SDK `10.0.26100.0`，x64 Release/static、portable
  ISA、LTO off，fresh build；
- `closure-perf-v1`，5% equivalence/non-inferiority margin；primary/placement × candidate/peer
  execution order × candidate/peer allocation order，共 8 strata；
- fresh n=15 只因 exact/NI 的置信区间未能裁决而按冻结停止规则升级到一轮独立 n=31；两阶段不
  合并，最终表只使用 n=31 的 independent-process paired samples。

`advantage ratio > 1` 有利于 HYLE。每行的 98.93% distribution-free median CI 都落在冻结的
equivalence band `[0.95238, 1.05]` 内，并且 CI lower 高于 5% non-inferiority limit。

| Layout / execution / allocation order | Median advantage ratio | 98.93% CI | Exact | 5% NI |
|---|---:|---:|---|---|
| primary / candidate-first / candidate-allocation-first | 0.9784 | [0.9722, 0.9933] | PARITY | PASS |
| primary / candidate-first / peer-allocation-first | 0.9761 | [0.9597, 0.9828] | PARITY | PASS |
| primary / peer-first / candidate-allocation-first | 0.9798 | [0.9657, 1.0081] | PARITY | PASS |
| primary / peer-first / peer-allocation-first | 0.9674 | [0.9572, 0.9765] | PARITY | PASS |
| placement / candidate-first / candidate-allocation-first | 1.0119 | [0.9974, 1.0283] | PARITY | PASS |
| placement / candidate-first / peer-allocation-first | 1.0070 | [0.9899, 1.0255] | PARITY | PASS |
| placement / peer-first / candidate-allocation-first | 1.0116 | [1.0016, 1.0301] | PARITY | PASS |
| placement / peer-first / peer-allocation-first | 1.0000 | [0.9888, 1.0129] | PARITY | PASS |

主布局四种姿势的中位数约低 2.0%–3.3%，placement 从持平到约高 1.2%；因此“只跑一个地址姿势”
会给出相反叙事。冻结判定没有把这种布局效应解释成产品缺陷，也没有用平均值掩盖某个姿势：8/8
分别通过同一个门，combined exact 才是 `PARITY`。

## 仪器资格控制

| Control | 结果 | 关键观察 |
|---|---|---|
| A/A，candidate-first | PASS / PARITY | ratio 0.9994，CI [0.9959, 1.0018] |
| A/A，peer-first | PASS / PARITY | ratio 0.9992，CI [0.9960, 1.0019] |
| known delta 5% | PASS | 两顺序 ratio 1.0489 / 1.0495，CI 正确包围预埋差异 |
| known delta 10% | PASS | 两顺序 ratio 1.1003 / 1.0993，且与 5% 单调 |
| batch-matched timer floor | PASS | workload 约为保守空白 floor 的 4,881x |
| primary + placement | PASS | 两种 code/data placement 都实际执行 |
| artifact/build/source identity | PASS | fresh build、candidate/peer/工具链/产物 hash 均锁定 |

这些控制说明仪器既能把 A/A 判成平，也能识别 5%/10% 的已知差异；当前 PARITY 不是计时器分辨率
不足。独立进程与双顺序的代价较高，但避免把地址布局、注册顺序和进程历史伪装成亚纳秒产品差异。
n=31 是合同允许的唯一功效升级；完成后无论 PASS/FAIL/UNRESOLVED 都停止。

## 可复验证据

本机 retained engineering bundle：

```text
C:\hyle-scratch\time-v1-qualified-evidence\manifest.json
```

- manifest SHA-256:
  `95621addc344f917e8d4b89a356cb89276b24897f380636bb9e6d1e8a755d2d6`
- summary SHA-256:
  `8ca6ee341b5490f85048c8fb0916f14dd795ce4aa0c3071775653a336cd724a8`
- observations SHA-256:
  `9c9a59a8a14224b9489d679dadc719106060b9960802412a282fc4d2ad5ac0bf`

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/time-source-consumer-v1.json `
  --manifest C:\hyle-scratch\time-v1-qualified-evidence\manifest.json
```

当前 verifier 返回 `qualification=QUALIFIED`、`paths_verified=true`、
`current_certification_policy_eligible=true`、`product_gate_policy=consumer-evidence-v1`。这是
本机可深重放的工程证据，不是发布 artifact 或 Formal certificate；live bundle 丢失且需要续签时必须
重新 qualification，不能只从本页点值恢复证书。

## 未进入产品聚合的能力

| 能力 | 当前裁决边界 |
|---|---|
| precise clock | header-inline 到对应 `std::chrono` primitive；以 code/semantic/correctness 门保证不多一层动态调用 |
| coarse clock | native availability 与 fallback 语义已验证；没有具名产品 workload，不用旧点值签胜负 |
| profiling clock | raw/ordered 是不同合同；只作 mechanism diagnostic，不与 std 双读混判 |
| CPU accounting | 一次 native read、O(1)、零分配；busy-vs-sleep 和单调性属于正确性门 |
| CommonTimeoutQueue | O(1) arm/cancel/next、O(fired) advance；尚无产品 consumer，不冒充 scheduler 收益 |
| blocking sleep | 证明不早醒和 OS 路由；wake latency 受 OS/QoS 影响，不进入 timer 容器聚合 |
| Unix/FILETIME conversion | O(1)、零分配、checked boundary；属于结构/正确性，不靠纳秒表证明 |

旧证书曾混入 C ABI 成本、不同 cardinality、clock/sleep 和不同语义 peer。它们对机制调查曾有价值，
但不再是当前产品结论。任何新消费者、算法/API/layout、编译器、交付形状或直接依赖变化都重开；
单次点值、亚纳秒摆动或某一 placement/order 翻转不能单独驱动产品修改。
