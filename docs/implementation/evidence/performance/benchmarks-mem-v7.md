# Mem v7 逻辑区间分配工程性能证据

> 本页只证明候选 `d3f732979f396aa5651e5eb81c228fdc73b7c166` / tree
> `16ac17e2fb501abeaeac623785ccf9daca1b29de` 的一个冻结正常任务，不把它外推为 Mem
> 全能力、全 alignment、全平台或端到端推理的普遍 SOTA。能力与边界见
> [Mem architecture](../../../architecture/mem-design.md)，用户选型见
> [Mem guide](../../../user/mem.md)，完整裁决见
> [Mem v7 closure record](../../tasks/mem/mem-v7-logical-range-suballocator.md)。

## 命题与同合同边界

正常任务是在一个 512 MiB opaque backing 内维持 256 个 live range，并按确定的资源型大小分布完成
512 次 release + allocate + offset observation + move replacement。HYLE 与 peer 都在计时前建立固定
metadata 和初始 working set，计时内都承担：

- fallible `Result`、move-only RAII lease、owner/lease 独立寿命和 generation invalidation；
- exact allocated/live/peak/handle counters；
- 相同的 512 次成功路径、相同 slot/size trace、无锁、无 syscall、无 payload access；
- 32-byte lease，以及足够处理相同 live 上限的 `2*M+1` 固定 node metadata。

peer 是 source-pinned OffsetAllocator
`3610a7377088b1e8c8f1525f458c96038a4e6fc0`，外加上述零热路分配的安全适配器。其原生裸 handle
路线没有这些生命周期、失败和统计语义，开发诊断中约快 14%--20%，但属于较弱合同，不能反向塑造
HYLE API，也不进入本页排名。冻结合同为
[`mem-v7-logical-range-p1-v1.json`](../../../../bench/contracts/mem-v7-logical-range-p1-v1.json)，
只在双方交集的 `alignment=1`、32-bit capacity 上计时；64-bit capacity 与变化 alignment 由正确性
holdout 证明。

## W1-WIN 结果

W1-WIN、MSVC 19.50、Release、static source consumption、LTO off；每个 stratum 15 个独立进程
样本。`advantage ratio > 1` 有利于 HYLE：

| placement / execution / owner allocation | HYLE median (M op/s) | peer median (M op/s) | median advantage | 99.26% CI | exact | 5% NI |
|---|---:|---:|---:|---:|---|---|
| primary / candidate-first / candidate-first | 24.982 | 24.101 | 1.0332 | [1.0196, 1.0627] | `INCONCLUSIVE` | `PASS` |
| primary / candidate-first / peer-first | 24.863 | 24.122 | 1.0383 | [1.0183, 1.0524] | `INCONCLUSIVE` | `PASS` |
| primary / peer-first / candidate-first | 24.759 | 23.953 | 1.0310 | [1.0218, 1.0554] | `INCONCLUSIVE` | `PASS` |
| primary / peer-first / peer-first | 24.949 | 24.050 | 1.0366 | [1.0265, 1.0553] | `INCONCLUSIVE` | `PASS` |
| perturbed / candidate-first / candidate-first | 25.054 | 24.032 | 1.0381 | [1.0214, 1.0496] | `PARITY` | `PASS` |
| perturbed / candidate-first / peer-first | 24.873 | 23.976 | 1.0411 | [1.0134, 1.0615] | `INCONCLUSIVE` | `PASS` |
| perturbed / peer-first / candidate-first | 24.822 | 23.861 | 1.0432 | [1.0204, 1.0575] | `INCONCLUSIVE` | `PASS` |
| perturbed / peer-first / peer-first | 25.010 | 23.991 | 1.0412 | [1.0261, 1.0625] | `INCONCLUSIVE` | `PASS` |

八个点估计都朝 HYLE 方向，约为 3.1%--4.3%；多数置信区间同时跨过 parity/WIN 的 5% 决策边界，
所以聚合 exact verdict 诚实保留为 `INCONCLUSIVE`，不签发普遍胜出。需要回答的工程问题是 material
regression：8/8 strata 的 5% non-inferiority gate 全部 `PASS`，因此冻结正常任务没有固定可修复劣化。

A/A 两种顺序均为 `PARITY`（median `0.9982/0.9949`）；5% known-delta 测得
`1.0467/1.0459`，10% known-delta 测得 `1.0982/1.1015`；timer-floor、placement、artifact identity
与 build contract 全部 `PASS`。资格器和 deep verifier 分别给出
`qualification_gate=PASS`、`performance_gate=PASS`、`QUALIFIED`、`paths_verified=true`、
source/provenance `EXACT`。

## 身份与边界

- evidence bundle：
  `C:\public\project-hyle\_validation\mem-v7-perf-manifest-d3f-run1.json`；
- contract / manifest / summary / observations SHA-256：
  `87AAABE3CDFF3ADB03BAC9DBC2DF26D5DAE910A7737CA8F56F3E9AA648F8AE02` /
  `0729718183BD745D87ECBE4EB7725405E26E381B2EE1220592CD42A95F2C526E` /
  `970C8832F66BBEDDE6BF15B629575067767C38DADDBD289C2B7FF4DE7E6FA25E` /
  `15F729B3AB1B3999B16CF14B3EEBACB9977512E586D9A96413E885070E3A7EEF`；
- candidate 与 peer checkout 均为 clean，peer tree 为
  `c2111cd59011fd0d28767012eea440be4118730b`；
- HYLE common allocate lookup 为两级 bitmap O(1)，地址相关完整性 fallback 最坏 O(F)，individual
  free/coalesce O(1)，reset O(M)，热路不增长 metadata、不锁、不 syscall；本页不把 aligned/64-bit
  的额外能力计成 peer 性能胜负。

## 复现

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/mem-v7-logical-range-p1-v1.json `
  --manifest C:\public\project-hyle\_validation\mem-v7-perf-manifest-d3f-run1.json\manifest.json
```

