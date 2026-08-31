# Container v2 工程性能证据

本页只解释 `container-v2` 的有限工程性能结论。旧 `container-v1` 的五行结果和聚合
`INCONCLUSIVE` 属历史 candidate，不能续签当前产品；需要追溯时从 Git 历史读取。

产品 candidate 是 `336793d63dd7cf08fde9c1be74c6439472335381`，tree 是
`968d169bafe0eab270d397ab1614f80c1a98079f`。正式 bundle 从 clean
`0ce6168c94ccdf008dfce7337354f437fe82f67b` 构建；candidate 到该提交只改变任务/状态文档，
没有改变 Container 公开头、实现、正常 workload 或 peer。后续测试提交同样不改变产品 candidate。

## 裁决

`container-v2` 的 P1 engineering verdict 为 **PASS**。这个结论严格限定为下面五个当前正常
consumer/replay 任务，在 W1-WIN、source-primary static Release 形状、5% material margin 下，相对
锁定同合同 peer 的 non-inferiority。它不是“Container 全操作/全平台 SOTA”。

比值定义为大于 1 有利于 HYLE；范围是八个 placement × execution-order × allocation-order strata
各自的 process-median advantage ratio，独立样本数均为 15。

| 正常任务 | 同合同 peer | 八 strata 中位比值范围 | 最差置信下界 | exact / 5% NI |
|---|---|---:|---:|---|
| 4096-element 短 sequence inline+spill 完整生命周期 | Abseil `InlinedVector` | 1.522–1.754 | 1.464 | `WIN / PASS` |
| 32768-entry metadata flat index 的 4096 次 50% hit/miss lookup | Abseil `flat_hash_map` | 1.187–1.212 | 1.156 | `WIN / PASS` |
| 两个不同物理 P-core 间、1024-slot ring 的 4096-item bounded MPMC handoff | rigtorp `MPMCQueue` | 1.403–1.450 | 1.341 | `WIN / PASS` |
| 262144-bit dense active mask 的 4096-block count | Boost `dynamic_bitset<uint64_t>` | 2.556–2.613 | 2.498 | `WIN / PASS` |
| 256-key warm set 上 4096 次 varied 64-byte seeded table hash | xxHash `XXH3_64bits_withSeed` | 1.028–1.040 | 1.007 | `INCONCLUSIVE / PASS` |

hash 的点估计只有约 3%，在预声明 5% practical margin 内，因此不写成 HYLE 胜出；但八个 strata
都通过 non-inferiority。五项分别裁决，不平均成一个“Container 分数”，sequence 的优势也不能抵偿
另一项未来可能出现的 material regression。

## 为什么只有这五项计时

P1 计时只覆盖已经绑定当前真实消费 provenance、会反复执行且存在同合同 peer 的五个机制：短 batch、
metadata lookup、跨核 handoff、dense mask scan 和中等长度 table hash。其他 Container 能力仍接受
复杂度、allocation/copy/lock/syscall/metadata、correctness 和 direct-consumer 门，但没有 profile
事实时不制造逐 overload 微基准：

- `CircularBuffer` steady push/pop/index 是 O(1)、满时覆盖且不分配；capacity change 是 O(n)。
- iterator erase 是 amortized O(1)，range/`erase_if` 是 O(n)，没有新增 per-entry metadata。
- pair/tuple/array semantic hash 对每个字段访问一次，不读取 padding，不动态分发。
- `StringView` 是 16-byte trivial borrowed value，无 allocation/lock/动态分发；查找成本仍按输入长度。
- const interner lookup 保持 expected O(1) 和一个 shard lock，不新增 allocation。

这些是 P0 结构 verdict，不是计时胜负。若真实推理 profile 把 ring、text、interner、ordered map 或其他
操作提升成端到端热点，再按具名 workload 定向重开；不能用本页五行提前替它们签 SOTA。

## 资格、身份与资源

冻结合同是 [`container-p1-v2.json`](../../../../bench/contracts/container-p1-v2.json)，SHA-256：
`50323f331be34fa1528dceb2d2ffbf07058a3f5593e121580d1f6ed65e532941`。锁定 peers：

- Abseil `2c004366e983c5be8334ac1ea3d4420e8fbcbea7`；
- Boost `78505eab9b203e3e26e83893d0685f160133fa0f`；
- rigtorp `b9808ede08f26fa9df4df4e081d19cace8f6c6ea`；
- xxHash `c0b5ea995d66691734b1a79ad89e73a0d2fd5a53`。

bundle 位于
`C:\hyle-scratch\container-v2-perf-qualified-336793d6-evidence`，关键摘要为：

- manifest SHA-256：`44a4d2e9aaa3726e74282df16987c5f407ecfffa7fbf7ded5ed4c04625f2ca94`；
- summary SHA-256：`d4dba9b8a22c50bba1b29a9240007b27c55ef7327bb931ede1e56a70f1eda921`；
- observations SHA-256：`dddc79f13b2bcaa471c5b9eebe8bf78f64af0a8a5dede971ff734811524d6864`；
- `qualification=QUALIFIED`、`paths_verified=true`、provenance `EXACT`、当前工程策略 eligible；
- MSVC 19.50、Release、static、LTO off、portable ISA、fresh build、W1-WIN 注册 P-core pair；
- artifact/build/semantic/resource validator 和 A/A、5%/10% known delta、timer floor、placement、
  双执行顺序、双 allocation 顺序全部通过；n=15 已给出终局，没有看结果追加到 31。

资源合同把两边的语义、逻辑规模、ownership 和计费阶段锁在一起：sequence 的 owner 构造、spill、遍历
与销毁都计费；map build/reserve 在两边都属于 setup；两种 queue 都预分配同容量并使用同一 persistent
producer/consumer；bitset 都拥有 262144 logical bits/32768 payload bytes；hash 两边读取同一完整
64-byte key 和 seed schedule。吞吐没有通过更弱 lifetime、隐藏 allocation、额外稀疏容量或少读 bytes
换取。

## 平台与声明边界

W1-WSL 和 M1-MAC 只验证 GCC/AppleClang correctness、sanitizer、真实受支持 ISA 和交付，不与
W1-WIN 点值池化。Windows 直接差分执行 scalar/SSE4/AVX2；Apple M4 直接差分执行 scalar/NEON。
AVX-512、SVE、RVV 等未在本轮真硬件执行的路径没有被写成 PASS。

Formal 状态是 `NONE`。本页不能外推：

- 普通长 sequence、string/build/erase、ordered/concurrent map 的全部操作；
- SPSC/sharded/work-steal、many-producer scalability 或 blocking/backpressure；
- sparse set、ring、UTF-8、interner、path 的逐操作计时胜负；
- 稳定 fingerprint、wire hash、安全 MAC，或跨平台固定纳秒。

## 复现

从 clean Windows 权威 checkout 和新的独占 scratch 执行：

```powershell
python scripts/hyle-perf.py validate `
  --contract bench/contracts/container-p1-v2.json

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/container-p1-v2.json `
  --machine W1-WIN `
  --peer abseil=C:\public\project-hyle\abseil-cpp `
  --peer boost=C:\public\project-hyle\boost `
  --peer rigtorp=C:\public\project-hyle\rigtorp-MPMCQueue `
  --peer xxhash=C:\public\project-hyle\xxHash `
  --build C:\hyle-scratch\container-v2-perf-qualified-build `
  --output C:\hyle-scratch\container-v2-perf-qualified-evidence

python scripts/hyle-perf.py verify `
  --contract bench/contracts/container-p1-v2.json `
  --manifest C:\hyle-scratch\container-v2-perf-qualified-evidence\manifest.json
```

`probe` 只用于资格预检；只有 clean `qualify-platform` bundle 和成功 deep verify 能携带本页 verdict。
