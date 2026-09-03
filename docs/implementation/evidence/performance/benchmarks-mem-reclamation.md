# Mem reclamation / Container table lifetime 工程性能证据

> **`QUALIFIED / PASS`。** 本页只签署 2026-09-03 的两个冻结 P1 consumer：短期不可变
> snapshot read，以及稳定 node registry 的 warmed hit。它不是 Formal 证书，也不外推为全部
> Mem、Container、RCU 写侧、rehash 或所有平台的 SOTA 排名。

产品 candidate 为 `0f6ef8e83ee290ab565c0d53081a0acb75a46675`，tree
`8cb5ba0024cdb20af65347144f09e16f26147632`。统一仪器从 clean harness commit
`c2ccc06f7c29fe19f67423b1ec3ef07d1d1cad79` 采样；candidate 到 harness 只改变 closure 状态、
benchmark runtime staging 和 run-nonce telemetry，`include/`、`src/`、两个 package consumer 与
component product graph 均无差异。后续 `f457babf` 只把 Container sample 的 factor 断言从错误的
`util_core` 校正为实际 direct factors `container_intrusive,mem_reclaim`，没有改变 timed product。

## 1. Immutable snapshot read：same-contract `WIN`

- contract：`bench/contracts/mem-reclaim-p1-v2.json`；SHA-256
  `8935256a3d9cfcab6319aa442d6904db4189dc5d9c8192147c3f4a593653d340`；
- task：warmed 4096 次 immutable pointer read；两侧都在 read guard 内 acquire-load 同一个
  caller-owned payload 并累计相同 checksum；registration、retire、synchronize 与 teardown 不计入
  read epoch；
- peer：Folly `rcu_domain`，tag `v2026.02.23.00`、commit
  `46afdbec8f58fc21445741278f14d53ec7144676`、tree
  `903da6ce26e31a50f3992aa3608ee3f53498a2ea`；官方 vcpkg package
  `2026.02.23.00#2`；
- comparison：`SAME_CONTRACT`；5% non-inferiority；advantage ratio > 1 有利于 HYLE；
- W1-WIN，MSVC `19.50.35728.0`，x64 Release/static、portable ISA、LTO off；n=15 独立
  paired processes 已裁决，不升级到 n=31。

| Stratum | HYLE op/s | Folly op/s | median ratio | 99.26% CI | exact / 5% NI |
|---|---:|---:|---:|---:|---|
| primary / candidate-first | 80.507 M | 55.767 M | 1.3812 | [1.3441, 1.4960] | WIN / PASS |
| primary / peer-first | 79.046 M | 55.453 M | 1.3777 | [1.3535, 1.4900] | WIN / PASS |
| placement / candidate-first | 77.222 M | 55.037 M | 1.3799 | [1.3542, 1.5319] | WIN / PASS |
| placement / peer-first | 79.551 M | 56.314 M | 1.3649 | [1.3358, 1.4053] | WIN / PASS |

四个 strata 的置信下界都在 5% margin 之外胜出。该结论证明当前短 guard/read 实现对冻结的
Folly 同合同路线没有 adoption regression；不证明 writer-heavy、超过 64 个 live reader、长
critical section 或 callback work 的相同倍率。

## 2. Stable node registry hit：natural-route `WIN`

- contract：`bench/contracts/container-node-table-p1-v1.json`；SHA-256
  `5d232e0495ace4f109c6233e1839048d7b37f39c4a97be1750cecd656fa2bc95`；
- task：预热 4096-entry registry 后完成同样的 successful lookup 与稳定 value observation；
- peer：oneTBB `concurrent_hash_map` commit
  `843162db8cf4b2531a4a6147b6ea3fd860d44668`、tree
  `d69c763c4dcb956c88f363e75cbd0f2009cf3171`；
- comparison：`NATURAL_ROUTE`。oneTBB 使用 segmented lazy rehash，HYLE 使用稳定 node +
  stop-the-table swap，因此只能回答完整用户路线，不能把差值冒充底层算法同合同排名；
- 同一 W1-WIN/toolchain/build 合同，n=15，5% adoption non-inferiority。

| Stratum | HYLE op/s | oneTBB op/s | median ratio | 99.26% CI | exact / 5% NI |
|---|---:|---:|---:|---:|---|
| primary / candidate-first | 93.065 M | 66.235 M | 1.4028 | [1.3707, 1.4169] | WIN / PASS |
| primary / peer-first | 91.076 M | 65.750 M | 1.3911 | [1.3607, 1.4113] | WIN / PASS |
| placement / candidate-first | 92.216 M | 65.948 M | 1.3921 | [1.3583, 1.4153] | WIN / PASS |
| placement / peer-first | 92.532 M | 65.805 M | 1.4022 | [1.3539, 1.4197] | WIN / PASS |

公开默认 `kMapDestruction` 路线不进入 reclamation guard；持续增长且要求服务期释放旧表时显式选择
`kAfterReaders`。两档的资源合同有意不同，reclaiming-vs-retained holdout 只用于解释内存/延迟取舍，
不以点值增删 API。

## 3. 仪器、correctness 与交付

两份 bundle 的 artifact/build identity、A/A、预埋 5%/10% known delta、batch-matched timer floor、
primary/placement、consumer provenance 和语义/resource validator 均为 `PASS`。当前 verifier 返回
`paths_verified=true`、`provenance=EXACT`、`environment=CURRENT`、
`current_certification_policy_eligible=true`。

| Bundle | retained manifest | manifest / summary / controls / observations SHA-256 |
|---|---|---|
| reclamation | `C:\hyle-scratch\mem-reclaim-formal-out-r4\manifest.json` | `59a806070e45844499f8fcc5f8062861503890ef0a72c733fa3923a1d675fcdd` / `6e177306951bb032a281d6a7928db762e9bc71a08ec262ff0c3d45ef8164b95f` / `59f0b1536cb8dc130f309e8d71c41495bf22583259c26cb19a177c857e380d44` / `3234277d5e7495bc8fc10e3240140322819410f71d7a8055622bf6652d7cd644` |
| node registry | `C:\hyle-scratch\container-node-formal-out-r1\manifest.json` | `3f7708ad72a56170e1a54d55736256ea86761b685de24038e75e3bde3b55f1db` / `e50e84b29da8bbe2b1633e3af166fec680f1f56d61504ee14a5571da1688f039` / `8cd4aadfce88702aa0c9e77d5f2b60009029e58de7c1a35a0d9a93f1b696897c` / `5f233812604dda4c4d4f3ecca3c719a4be566c8fff4d6d4b62672058b423be49` |

产品候选还通过：W1-WIN Release 定向测试与 Mem/Container 三交付；W1-WSL GCC 14 Release、ASan、
UBSan、TSan、50 轮并发 stress 与两产品三交付；M1-MAC AppleClang 21 arm64 Release 定向测试与
两产品三交付。`hyle_mem_reclaim.dll` 只有 7 个 MSVC 修饰 C++ export，没有未修饰 HYLE C/Native
FFI surface。

```powershell
python scripts/hyle-perf.py verify `
  --contract bench/contracts/mem-reclaim-p1-v2.json `
  --manifest C:\hyle-scratch\mem-reclaim-formal-out-r4\manifest.json

python scripts/hyle-perf.py verify `
  --contract bench/contracts/container-node-table-p1-v1.json `
  --manifest C:\hyle-scratch\container-node-formal-out-r1\manifest.json
```

candidate、timed consumer、peer、workload/计费、toolchain、机器或仪器策略发生 material 变化后必须
重新 qualification。bundle 是本地可深重放工程证据；`formal=false`，不能称为发布级跨平台证书。
