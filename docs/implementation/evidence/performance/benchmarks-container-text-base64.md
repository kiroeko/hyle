# Container Text Base64 工程性能证据

> 本页只签署 G01 的两个冻结正常 consumer shape。它证明 HYLE 公开 caller-buffer adapter 在
> W1-WIN 上相对同 revision simdutf 的相同语义路线通过 5% P1 engineering non-inferiority；不声明
> 跨平台绝对 SOTA，也不把 1%--4% 的 placement/order 点值写成稳定胜负。当前模块状态只看
> [`module-status.json`](../../module-status.json)。

设计、用法与施工边界分别见
[`container-design`](../../../architecture/container-design.md)、
[`Container 用户指南`](../../../user/container.md)和
[`container-text-v3-base64`](../../tasks/container/container-text-v3-base64.md)。

## 不可变身份与合同

- HYLE candidate：`e50a2d94f601ba0d9b7d87c854c868126ca4e596`，tree
  `b6d00764459b8c9d9825a5e20b652e19aee9c585`，clean detached source；
- contract：`bench/contracts/container-text-base64-p1-v1.json`，SHA-256
  `4d2ab20dec63f3ff488f4ff6fef49a5a282bde01820a84099809f2b167ccf5f6`；
- peer：simdutf v9.1.0 commit `2d4e249dcfd978b39dd567b156f20a43db97377b`，tree
  `a39108d83b5487400dc59fb9b842bf1153c12df4`；外部 checkout 提供精确声明，双方链接产品内字节相同的
  provider implementation；
- W1-WIN、MSVC 19.50、Release/static source、portable ISA、LTO off；primary/placement ×
  candidate-first/peer-first 四个 strata；
- n=15 后按合同唯一允许的精度升级取得 n=31 独立进程最终样本；candidate、workload、peer、margin 和
  计费阶段均未改变。

两个 pair 都计入相同 options/span/overlap/capacity、严格 whitespace policy、safe provider 调用、typed
result/progress 与 observable checksum。大 payload 不含 JSON/data-URI/allocation，短 encode 不含 SHA-1、
HTTP parser 或 socket；这些阶段不属于 codec owner，双方都不计费。

## 结果

`advantage_ratio > 1` 有利于 HYLE。每格区间为 98.93% distribution-free median CI；裁决使用预声明的
5% reciprocal band。

| Workload / stratum | HYLE / peer median | ratio / CI | exact / 5% NI |
|---|---:|---:|---|
| 256 KiB strict decode / primary candidate-first | 51.831 / 50.158 M 64-byte units/s | 1.0160 / [1.0078, 1.0571] | INCONCLUSIVE / PASS |
| strict decode / primary peer-first | 51.303 / 49.984 M | 1.0318 / [1.0229, 1.0429] | PARITY / PASS |
| strict decode / placement candidate-first | 51.026 / 51.687 M | 0.9896 / [0.9739, 1.0048] | PARITY / PASS |
| strict decode / placement peer-first | 51.255 / 51.668 M | 0.9952 / [0.9813, 1.0101] | PARITY / PASS |
| 20-byte encode batch / primary candidate-first | 75.112 / 72.537 M records/s | 1.0449 / [1.0318, 1.0513] | INCONCLUSIVE / PASS |
| encode / primary peer-first | 75.496 / 72.827 M | 1.0329 / [1.0119, 1.0508] | INCONCLUSIVE / PASS |
| encode / placement candidate-first | 75.424 / 72.777 M | 1.0397 / [1.0290, 1.0486] | PARITY / PASS |
| encode / placement peer-first | 75.394 / 72.915 M | 1.0387 / [1.0252, 1.0533] | INCONCLUSIVE / PASS |

两个 workload 的 combined exact verdict 都是 `INCONCLUSIVE`，combined non-inferiority gate 都是
`PASS`。这里的 `INCONCLUSIVE` 是区间跨 5% decision boundary，不是产品失败；八格没有 `LOSE`，也没有
任何格越过 5% 非劣下界。默认 strict decode 的 HYLE 额外语义扫描在 peer 中逐阶段复现，因此不是
forgiving-vs-strict 的苹果对橘子比较。

A/A 双顺序、5%/10% known delta、batch-matched timer floor、placement、artifact/build identity、
consumer provenance 与 Base64 semantic/call validator 全部 `PASS`。机械重放返回
`paths_verified=true`、`qualification=QUALIFIED`、`provenance_compatibility=EXACT`、
`current_certification_policy_eligible=true`。

## Retained engineering bundle

本机 bundle 位于：

```text
C:\hyle-scratch\g01-base64-p1-evidence-e50a2d94\manifest.json
```

关键 SHA-256：manifest
`4f71128055a9c0a4a13f7d61069a3ae1263f486800c40103f2a5b8d049627ce6`；summary
`62faf17006998c66342c01a67b2aa35d6a18d66b1ed1c9138a87b6780cb6eb0a`；controls
`7be6fb92da76bebc1b958520dea1bedde8b26a47a105dd3a37e609b9963250b5`；observations
`9b447d9498399de453ee3375a6fe0272f421feff3391af8a8ef3339be0304350`；raw index
`a33b8eba13f4cba8a7cd516be158bc23eac277e6f37a2f6aa889fa04197cbd10`。

```powershell
python scripts/hyle-perf.py verify `
  --contract C:\hyle-scratch\g01-base64-p1-source-e50a2d94\bench\contracts\container-text-base64-p1-v1.json `
  --manifest C:\hyle-scratch\g01-base64-p1-evidence-e50a2d94\manifest.json
```

bundle 是本机可深重放的工程收口证据，不是跨平台发布证书。candidate、simdutf revision、公开语义、
正常 workload/ownership、编译器/ISA 或统一仪器发生 material 变化后必须重新 qualification；bundle 丢失时
不能只凭本页数字续签新结论。
