# container-v1 工程性能证据

本页只记录当前 `container-v1` 的工程性能结论。2026-07 的混合 microbenchmark 表、旧 C
surface 和不同 API/容量/并发语义的点值已经从 current 文档删除；它们仍可从 Git 历史追溯，
但不能参与本轮裁决。

产品 candidate 为 `19a1e9b515522ad74872fb963a3b3260cf915bab` / tree
`68d456382352d2d81fdd41e4d1c725440424336a`。当前性能 harness revision 为
`9f3fa43adad6afc14cd71c4d7792f55b7924d619` / tree
`221a9ad72d126ad2a4c23e43e088d31ee8c43b0d`；harness、contract 和安装消费示例没有改变
container 产品实现 identity。

## 当前裁决

当前五项 W1-WIN consumer/replay slice 为 `PASS`；`container-v1` 冻结卡的聚合 P1 门仍为
`INCONCLUSIVE`。原因不是下表存在性能失败，而是 closure card 预声明的若干正常轴尚未进入
`closure-perf-v1`：未运行项不能写成 `PASS`。因此 container 本轮必须保持 `BLOCKED`，不能由五项
代表性结果外推为整模块 SOTA。

| workload | 正常消费形状 | source-pinned peer | 8 个分层的 advantage-ratio 中位数范围 | 最差置信下界 | verdict |
|---|---|---|---:|---:|---|
| inline + spill sequence | 4096 个 element lifecycle，混合 256 个 8-element inline owner 与 32 个 64-element spill owner | Abseil `InlinedVector` | 1.557030–1.723319 | 1.510363 | `WIN`；NI `PASS` |
| metadata flat lookup | 32768-entry table，4096 个 50% hit/miss integral query | Abseil `flat_hash_map` | 1.198815–1.209000 | 1.158363 | `WIN`；NI `PASS` |
| bounded MPMC handoff | 两个不同物理 P-core 间，通过 1024-slot strict bounded ring 传 4096 个 `uint64_t` | rigtorp `MPMCQueue` | 1.402940–1.462899 | 1.361199 | `WIN`；NI `PASS` |
| dense active-mask count | 4096 个 `uint64_t` block / 262144 bits | Boost `dynamic_bitset<uint64_t>` | 5.144917–5.336622 | 5.104358 | `WIN`；NI `PASS` |
| varied 64-byte metadata hash | 256-key warm working set，4096 次带 seed table hash | xxHash `XXH3_64bits_withSeed` | 1.033094–1.039218 | 0.999593 | `PARITY`；NI `PASS` |

比值大于 1 有利于 HYLE。每项独立裁决，不汇总成一个 container 分数。严格胜负采用合同中的
5% practical margin；hash 虽然点估计约快 3%，仍如实写 `PARITY`。

冻结卡中仍缺资格化证据的有限集合是：

1. strict fixed/non-trivial sequence lifecycle；
2. string flat hit/miss/build/erase、read-mostly sorted lookup/bulk 与 ordered mutation/expiry；
3. SPSC/sharded/work-steal、concurrent append、flat/stable-node concurrent map 的正常读写/rebuild；
4. bitset algebra/iteration、SparseSet lifecycle、object/byte ring staging；
5. short/long byte hash、UTF-8 validate/count、borrowed search/case 与 interner 的单线程/正常并发
   lifecycle。

没有同合同 peer 的轴仍须按冻结 card 对 immutable baseline 做 5% source-primary NI，并守住零额外
常态 allocation/hash scan/lock/syscall。恢复轮可以把这些轴拆成有限 contract，但不能改变 candidate、
margin、正常 ownership/concurrency 语义，或用旧 2026-07 点值代签。

## 资格与资源门

不可变合同是
[`container-p1-v1.json`](../../../../bench/contracts/container-p1-v1.json)，SHA-256
`f46e4e4af0d46ab530d62ae82894c6875b8fe13f9fa6453e55933b59ea22f3a4`。clean fresh
qualification bundle 位于
`C:\hyle-scratch\container-v1-p1-qualify-9f3fa43-output`：

- manifest SHA-256：`7a7acb9aed353d5d58275cddd6fa41395c15ccc6db260e055c4c22afc8366db4`；
- summary SHA-256：`ddb456f7f55b0f09d876d8b9fecadd3951caf6751d8ba9ed98e7283efdc5e72d`；
- runner/verifier：`QUALIFIED`、`paths_verified=true`、current engineering policy eligible；
- source clean、MSVC 19.50、Release/static、LTO off、registered W1-WIN、fresh build；
- artifact identity、build contract、module semantic/resource oracle 全部 `PASS`；
- A/A、已知 5%/10% 差值、timer floor、primary/placement、双执行顺序和双 allocation
  顺序全部 `PASS`；n=15 已给出终局，没有选择性追加样本。

资源 oracle 同时冻结了比较形状，避免只看吞吐：

| resource fact | HYLE | peer | 解释 |
|---|---:|---:|---|
| inline sequence owner object | 88 B | 72 B | HYLE 更快但 owner 大 16 B；不能隐去这个 layout trade-off |
| reserved flat-map capacity | 65536 | 65535 | 等价规模，没有用更稀疏表换取点值 |
| 1024-slot MPMC backing | 65536 B | 65600 B | 都为 cache-line-spaced strict bounded ring |
| 262144-bit bitset payload | 32768 B | 32768 B | 相同逻辑位数与 block payload |

这五条路径均无额外 per-entry benchmark metadata、动态派发、常态 lock 或隐藏 allocation。
P0 结构审计另外守住：连续/关联结构达到对应的 amortized O(1)、expected O(1)、O(log N) 或 O(N)
下界；固定/队列热路径预分配；text/path/hash 按输入 bytes 线性；没有为本表改变公开 API。P0
结构门不能替代上面已冻结的未运行 P1 replay。

## MPMC 仪器纠错

第一版 adapter 让 candidate 与 peer 各自创建 worker，并在每个计时 callback 内使用
`atomic::wait/notify` 和 `std::this_thread::yield`。两边因此都随机落入 Windows scheduler 的慢模态：
candidate 约 25M/115M、peer 约 23M/80M operations/s；31 个独立样本后仍有两个分层的 5% NI
置信区间未决。100 ms 计时外预热没有消除该模态，证明问题不是普通冷启动。

源码审计发现这个 adapter 与合同不符：所谓被 4096 items 摊薄的 batch boundary 实际进入了 OS
park/yield。最终 adapter 复用已经资格化的 mem 双核形状：两个 queue 按受控 allocation 顺序构造，
同一条常驻、固定到 worker core 的线程处理 candidate/peer，32 个 paired warmup batch 在计时外完成，
callback 只用 busy-pause generation 和相同 full/empty retry。修正后的 diagnostic 8 个分层为
candidate 115.73–118.24M、peer 80.58–85.60M；随后正式 n=15 得到上表 `PASS`。

这次修正只移除了测量器的 OS 调度噪声，没有改变 `BoundedQueue`、容量、slot layout、memory order、
API 或常态 metadata。旧 bundle 只能说明旧仪器未决，不能被挑选为产品证据。

## 声明边界

- sequence 只证明冻结的短 inline + 偶发 spill 生命周期；普通长寿命 sequence 仍优先
  `std::vector`/PMR，不能从该行宣称所有 vector 操作胜出。
- map 行是 integral metadata mixed lookup，不替代 string、build、erase、ordered 或 concurrent map
  的逐操作证书；这些能力的复杂度、正确性和资源门已审查，未被真实 profile 提升为独立 P2。
- MPMC 行是持续的 1 producer / 1 consumer strict bounded handoff。它不代表 SPSC、sharded、
  work-stealing、blocking/backpressure 或 many-producer scalability。
- bitset 行证明 dense count；SparseSet 的价值是 O(1) clear + O(active) iteration，不应为 dense
  membership benchmark 增加第三份 bitmap。
- hash 是进程内 table hash，不是稳定 fingerprint、wire format 或安全 MAC；不可信输入使用显式
  `HardenedStringHash` 与调用方 secret。
- Linux/macOS 承担 correctness、sanitizer 与 ISA fallback，不和 W1-WIN 点值池化。Formal 保持
  `PENDING`。

## 复现

从 clean checkout 使用唯一 scratch 路径：

```powershell
python scripts/hyle-perf.py validate `
  --contract bench/contracts/container-p1-v1.json

python scripts/hyle-perf.py probe `
  --contract bench/contracts/container-p1-v1.json `
  --machine W1-WIN `
  --peer abseil=C:\public\project-hyle\abseil-cpp `
  --peer boost=C:\public\project-hyle\boost `
  --peer rigtorp=C:\public\project-hyle\rigtorp-MPMCQueue `
  --peer xxhash=C:\public\project-hyle\xxHash `
  --build C:\hyle-scratch\container-p1-probe-build `
  --output C:\hyle-scratch\container-p1-probe-output

python scripts/hyle-perf.py qualify-platform `
  --contract bench/contracts/container-p1-v1.json `
  --machine W1-WIN `
  --peer abseil=C:\public\project-hyle\abseil-cpp `
  --peer boost=C:\public\project-hyle\boost `
  --peer rigtorp=C:\public\project-hyle\rigtorp-MPMCQueue `
  --peer xxhash=C:\public\project-hyle\xxHash `
  --build C:\hyle-scratch\container-p1-qualify-build `
  --output C:\hyle-scratch\container-p1-qualify-output

python scripts/hyle-perf.py verify `
  --contract bench/contracts/container-p1-v1.json `
  --manifest C:\hyle-scratch\container-p1-qualify-output\manifest.json
```

`probe` 只检查映射、资源、timer floor 和预计成本；只有 clean independent-process
`qualify-platform` manifest 与成功 deep replay 能携带工程 verdict。
