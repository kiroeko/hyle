# Time / timer_queue 工程性能证据

> **time-v4 / timer-queue-v1：`QUALIFIED / PASS`。** Time candidate
> `a9b9ff3117ab647581a6b846318a8254a6cd148f` / tree
> `6b5026fa1bbc0e916a0f81fa707ed6f8bbe75301` 在 W1-WIN 上通过 host
> phase/deadline 5% non-inferiority gate；timer_queue 仍绑定其独立 candidate
> `381eb4b195973ed35ed23ca63c05f9605a132f9b` / tree
> `a5d48a259161c1a60e19450e42b4a6eac9a1a712`，scheduler lifecycle 对当前
> Seastar peer 为 `WIN`。

这是冻结 consumer、一台工程裁决机和 5% 非劣边界下的可复验结论，不是跨机器
Formal 证书，也不外推为所有 clock、sleep、timer cardinality 或未消费能力全面 SOTA。
设计、用户选型和施工事实分别见 [Time architecture](../../../architecture/time-design.md)、
[Time guide](../../../user/time.md)、[time task](../../tasks/time/time-v4-msvc-instant-codegen.md)
和 [timer_queue task](../../tasks/timer_queue/timer-queue-v1-factor.md)。

## 两个不可混合的性能合同

| Factor | 冻结正常形状 | peer | 本轮结论 |
|---|---|---|---|
| `time` | 64 个 runtime phase；每个两次 precise monotonic read，再做 elapsed、饱和 deadline 和 remaining | Seastar precise route 实际采用的同 TU `std::chrono::steady_clock` 等价路线 | exact 为 3 `PARITY` + 1 `INCONCLUSIVE`；5% NI 4/4 `PASS` |
| `timer_queue` | Scheduler N=64 完整 lifecycle：64 arm、45 cancel、64 advance tick、19 delivery/callback-state observation | Seastar `timer_set` faithful local port | 8/8 order/placement/allocation strata `WIN`，5% NI 8/8 `PASS` |

host 合同不拿 HYLE 的包装形状对比一个减少读钟次数的假 peer；timer 合同也不拿完整
owner/cancel/drain 语义对比轻量单操作。两者的 input construction 和分配在 epoch 外，但所有
消费者必需的状态变换和 checksum sink 都计费。

## Host Time：5% non-inferiority `PASS`

- contract：`bench/contracts/time-host-consumer-v1.json`；SHA-256
  `707029c26eb677dcecfc1478cca207d6a2a5d64b305eb04c43391f14d0cb158e`；
- peer：Seastar `991cf9a908482a3f11ca2fc34a1b3970a06d8622`，其 precise timer clock 是
  `std::chrono::steady_clock`；
- W1-WIN，MSVC `19.50.35728.0`，x64 Release/static、portable ISA、LTO off、fresh build；
- n=15 独立进程 paired samples 已能裁决，未升级到 n=31；advantage ratio > 1 有利于 HYLE。

| Stratum | HYLE op/s | peer op/s | median ratio | 99.26% CI | exact / 5% NI |
|---|---:|---:|---:|---:|---|
| primary / candidate-first | 35.714 M | 35.632 M | 1.0016 | [0.9839, 1.0273] | PARITY / PASS |
| primary / peer-first | 35.278 M | 35.513 M | 0.9869 | [0.9617, 1.0149] | PARITY / PASS |
| placement / candidate-first | 35.307 M | 35.015 M | 1.0016 | [0.9789, 1.0562] | INCONCLUSIVE / PASS |
| placement / peer-first | 35.470 M | 35.140 M | 1.0084 | [0.9915, 1.0306] | PARITY / PASS |

四个独立 strata 的 CI lower 都高于 5% 非劣限制 `1/1.05 = 0.95238`，因此产品 gate
为 `PASS`。中位点随 order/placement 在 -1.3% 到 +0.8% 间翻转；其中一个 placement CI
越过 exact 等价判定边界，所以聚合 exact ranking 是 `INCONCLUSIVE`，不能宣称固定胜出。

v3 对象检查曾发现 MSVC `/O2 /Ob2` 的 `Instant::now()` phase 比直接 chrono 多两次
外联 `steady_clock::now()` 壳；v4 用局部 `[[msvc::flatten]]` 消除了这个可复核机制。
maintenance-only Release probe 现在证明两条 phase body 均为 143 条规范化指令、逐项相同且
无 wrapper edge。这个结构事实与上述正常 consumer 的非劣结果共同支持修复；没有把亚纳秒点值
变成新 API、常态 metadata、全局 `/Ob3` 或“全面 SOTA”声明。

## Timer queue：`WIN`

- contract：`bench/contracts/timer-queue-consumer-v1.json`；SHA-256
  `48e636cb9e59d578b1148ef0af091cb04b6daeb38097581b6e315b9f1eb760fe`；
- peer：Seastar `991cf9a908482a3f11ca2fc34a1b3970a06d8622` 的 `timer_set`；
- 同一 W1-WIN/toolchain/build 合同，n=15 已裁决；advantage ratio > 1 有利于 HYLE。

| Placement / execution / allocation order | HYLE op/s | peer op/s | median ratio | 99.26% CI |
|---|---:|---:|---:|---:|
| primary / candidate-first / candidate-first | 62.797 M | 53.422 M | 1.1815 | [1.1308, 1.2328] |
| primary / candidate-first / peer-first | 63.079 M | 55.924 M | 1.1293 | [1.0911, 1.1664] |
| primary / peer-first / candidate-first | 63.099 M | 55.923 M | 1.1338 | [1.1045, 1.1597] |
| primary / peer-first / peer-first | 63.594 M | 55.091 M | 1.1678 | [1.1305, 1.1878] |
| placement / candidate-first / candidate-first | 64.625 M | 54.387 M | 1.1754 | [1.1310, 1.2195] |
| placement / candidate-first / peer-first | 63.659 M | 54.389 M | 1.1898 | [1.1534, 1.2167] |
| placement / peer-first / candidate-first | 63.167 M | 53.537 M | 1.1758 | [1.1478, 1.2163] |
| placement / peer-first / peer-first | 63.746 M | 55.028 M | 1.1740 | [1.1353, 1.2026] |

八个 strata 全部在 5% margin 外胜出，中位优势约 12.9%--19.0%，最低 CI lower 仍为
1.0911。结论只签当前 scheduler lifecycle：65-bucket radix queue 的 arm/cancel/next O(1)，
advance 按 touched buckets + delivered/rebucketed nodes 计费，caller-owned node 且零分配/lock/atomic/syscall。

## 仪器、身份与可复验证据

两份 bundle 的 A/A、预埋 5%/10% known delta、batch-matched timer floor、primary/placement、
artifact identity、build contract、consumer provenance 和 semantic/resource validator 全部 `PASS`。
deep verifier 均返回 `QUALIFIED`、`paths_verified=true`、`provenance=EXACT`、
`environment=CURRENT` 和 `current_certification_policy_eligible=true`。

Timer-queue retained manifest 仍对自己的 immutable candidate source 返回 `EXACT`。当前 canonical
Sched consumer 相对冻结 provenance 只有 17 处 `mem::Allocator` → `mem::AllocatorRef` 词汇迁移；
timer ownership、arm/cancel/advance/drain、全部 charged phases、算法和 delivery 均未改变。该精确
`9fc1b77d... -> 32ec5084...` 边已由
[`Timer-queue provenance review`](provenance-reviews/timer-queue-scheduler-lifecycle-9fc1-to-32ec.md)
裁决为 `NON_MATERIAL / ENGINEERING_REPLAY_ONLY` 并登记到中央 renewal ledger。因此既有工程结论仍
表示同一消费命题，不重跑采样，也不把它升级为当前 Sched 性能或 Formal 结论。

| Bundle | retained manifest | manifest / summary / controls / observations SHA-256 |
|---|---|---|
| host | `C:\hyle-scratch\time-v4-perf-evidence\manifest.json` | `4207D5E656E34C8017B3944CE7B908CC5D09D44EA06BBE4225F54A34227E125E` / `901508FC76615C5AD7471C8DB77793D49053DFC4622CECF7189A244EB1A24DC8` / `7B45ECD630A554A20CA15C184376743DAF34B73ADFAC819F137E47C13F589319` / `078EE44C7472D61FB25220917309FC079CBFD31A576C904D81A58DF9AA638F24` |
| timer | `C:\hyle-scratch\time-v3-timer-perf-evidence\manifest.json` | `B7D23B40DB8EE3D5B77CB0CD6EC7B6F5F5810F4BBA609A1B92BF605FBCE0FB90` / `2C2D986359FCB43725850EEBCF9F352D11465D7B4D9CDE3ADD1D22EDC33AF124` / `9BA1CF2D4DC6EADB478A5943A2DCE27079E1995F4484D34098BA887B78DCD2D5` / `5F36C2DBC4346F0765B2A9373313EA70F853E5ED773FEAF4C10BE8A445D2A52C` |

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/time-host-consumer-v1.json `
  --manifest C:\hyle-scratch\time-v4-perf-evidence\manifest.json

python scripts/hyle-perf.py verify `
  --contract bench/contracts/timer-queue-consumer-v1.json `
  --manifest C:\hyle-scratch\time-v3-timer-perf-evidence\manifest.json
```

bundle 是本地可深重放工程证据，不是发布 artifact 或 Formal certificate。candidate、peer、consumer
hash、workload/计费、toolchain、machine profile 或 instrument policy 变化而仍要续签声明时，必须新建
合同或重新 qualification，不得抄旧点值。

## 不使用 P1 排名的能力

| 能力 | 当前性能边界 |
|---|---|
| `ManualClock` | 一个 always-lock-free 64-bit atomic state；`now()` 一次 load，write 在无竞争时 O(1)，竞争时 CAS retry 次数随竞争而变；零 waiter/分配/系统调用 |
| `MonotonicClockRef` | 两字借用引用、一次显式间接 `now()`、零分配；只在用户选择 runtime type erasure 时付费 |
| coarse/profiling/CPU clock | 一次 native mechanism、O(1)、零分配；由 availability/monotonic/domain correctness 裁决，无具名正常 consumer 时不伪造排名 |
| blocking sleep | 一次 native wait 并处理 interrupt；只签不早醒与路由，wake latency 受 OS/QoS 支配 |
| Unix/FILETIME conversion | constexpr/O(1)、零分配、checked boundary；属正确性/结构门 |
| `CommonTimeoutQueue` | arm/cancel/next O(1)、advance O(delivered)；目前无 HYLE 产品 consumer，只签 correctness/complexity，不冒充 scheduler 性能收益 |

任何新 consumer、算法/API/layout、直接依赖或三种交付形状的 material 变化都是重开信号；
单次点值、亚纳秒摆动或某一 placement/order 翻转不能单独驱动产品修改。
